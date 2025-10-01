"""
jaxpr -> ASCII graph

Usage: call ascii_graph_from_jaxpr(jaxpr_text) which returns a string.
"""

import re
from math import ceil

# ----------------------
# Parsing helpers
# ----------------------
def parse_jaxpr(text):
    """
    Parse a jaxpr-like string into a structure:
      - params: list of parameter names
      - nodes: dict varname -> {"op": opname, "inputs": [names_or_consts], "raw": rhs}
      - outputs: list of output variable names
    Note: if a rhs produces "_" (unused), we still create a node keyed by a synthetic name.
    """
    # Find lambda parameter line
    params = []
    m = re.search(r"lambda\s*;\s*([^\.]+)\.", text)
    if m:
        params_part = m.group(1).strip()
        # tokens that look like var:... or var
        param_tokens = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", params_part)
        if not param_tokens:
            # fallback: tokens without : (rare)
            param_tokens = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)", params_part)
        params = param_tokens

    # Find outputs in "in (i,)" or "in (o,)" etc
    outputs = []
    m = re.search(r"in\s*\(\s*([^)]+?)\s*\)", text)
    if m:
        out_tokens = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)", m.group(1))
        outputs = out_tokens

    # Find let lines: 'let <lhs>:type = <rhs>'
    nodes = {}
    synthetic_unused_count = 0

    for line in re.split(r"[\r\n]+", text):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("let"):
            continue
        # remove leading "let"
        rhs_part = line[3:].strip()
        # match lhs like 'b:f32[] = sin a' or '_:f32[] = mul b d'
        m = re.match(r"([A-Za-z0-9_]+)\s*:[^=]+=\s*(.+)", rhs_part)
        if not m:
            continue
        lhs = m.group(1)
        rhs = m.group(2).strip()

        # Separate op name (maybe op[...]) and rest inputs
        m2 = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]+\])?\s*(.*)", rhs)
        if m2:
            opname = m2.group(1)
            rest = m2.group(3).strip()
        else:
            # fallback: take first token
            toks = rhs.split()
            opname = toks[0]
            rest = " ".join(toks[1:])

        # Extract input tokens: variable names and numeric constants
        # We'll capture names and decimal numbers
        inputs = []
        # find float or int constants
        consts = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", rest)
        # find name tokens
        names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rest)
        # But names may repeat opname; remove initial opname if caught
        names = [n for n in names if n != opname]

        # Merge names and constants preserving order by scanning rest
        token_scan = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[-+]?\d+\.\d+|[-+]?\d+", rest)
        for t in token_scan:
            if re.match(r"[-+]?\d+\.\d+|[-+]?\d+", t):
                inputs.append(t)
            else:
                # treat 'np' and 'int64' parts as noise if they appear chained like np.int64(0)
                if t == "np" or t in ("int64",):
                    continue
                # skip opname (already removed), but just in case
                if t == opname:
                    continue
                inputs.append(t)

        # If lhs is '_' create synthetic name because it may still be referenced (rare).
        if lhs == "_":
            synthetic_unused_count += 1
            lhs = f"_unused_{synthetic_unused_count}"

        nodes[lhs] = {"op": opname, "inputs": inputs, "raw": rhs}

    # Add parameter nodes so they appear in nodes as producers
    for p in params:
        nodes.setdefault(p, {"op": "param", "inputs": [], "raw": p})

    # add constants seen in inputs (1.0 etc) as const nodes
    # find all inputs across nodes
    all_inputs = set()
    for v in nodes:
        for inp in nodes[v]["inputs"]:
            all_inputs.add(inp)
    for token in sorted(all_inputs):
        if re.match(r"[-+]?\d+\.\d+|[-+]?\d+", token):
            # constant node
            const_name = f"const_{token}"
            # but references in inputs use the literal token, so map that token to const_name by creating a node keyed by the token literal
            nodes.setdefault(token, {"op": "const", "inputs": [], "raw": token})

    return {"params": params, "nodes": nodes, "outputs": outputs}


# ----------------------
# Build dependency graph & compute depths
# ----------------------
def build_graph(parsed):
    nodes = parsed["nodes"]
    # Determine producers: nodes are producers keyed by name
    # compute depth (distance from param/const)
    depth = {}
    visiting = set()

    def get_depth(name):
        if name in depth:
            return depth[name]
        if name not in nodes:
            # unknown token — treat as param/const at depth 0
            depth[name] = 0
            return 0
        op = nodes[name]["op"]
        if op in ("param", "const"):
            depth[name] = 0
            return 0
        # prevent recursion cycles
        visiting.add(name)
        ins = nodes[name]["inputs"]
        if not ins:
            d = 0
        else:
            child_depths = []
            for inp in ins:
                # if inp not in nodes, -> depth 0
                if inp == name:
                    child_depths.append(0)
                else:
                    child_depths.append(get_depth(inp))
            d = max(child_depths) + 1
        depth[name] = d
        visiting.discard(name)
        return d

    for n in list(nodes.keys()):
        get_depth(n)

    # group nodes by depth (but keep only meaningful nodes: ignore param/const if desired)
    by_depth = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)

    return {"nodes": nodes, "depth": depth, "by_depth": by_depth}


# ----------------------
# Layout and ASCII render
# ----------------------
def layout_and_render(graph, max_width=120, col_spacing=4, row_spacing=1):
    nodes = graph["nodes"]
    by_depth = graph["by_depth"]
    depths = sorted(by_depth.keys())

    # Build labels for nodes: show op and name for ops; show param/const differently
    labels = {}
    for name, info in nodes.items():
        op = info["op"]
        if op == "param":
            labels[name] = name
        elif op == "const":
            labels[name] = info["raw"]
        else:
            # label like "sin\nb" but we'll show in one line if short
            lab = f"{op}({name})"
            labels[name] = lab

    # Determine column widths = max label len in that depth + padding
    col_widths = []
    for d in depths:
        maxlen = 0
        for n in by_depth[d]:
            maxlen = max(maxlen, len(labels[n]))
        col_widths.append(max(6, maxlen + 2))  # minimum width

    # Compute x positions for each column (left coordinate)
    col_x = []
    x = 0
    for w in col_widths:
        col_x.append(x)
        x += w + col_spacing
    total_width = x - col_spacing
    total_width = max(total_width, max_width)
    # Determine rows per column (stack nodes vertically)
    col_rows = []
    max_col_rows = 0
    for d in depths:
        count = len(by_depth[d])
        col_rows.append(count)
        max_col_rows = max(max_col_rows, count)

    # Decide a row height (lines per node) and spacing
    line_per_node = 1  # keep label single-line to simplify
    row_height = line_per_node + row_spacing

    total_height = max_col_rows * row_height + 4

    # Create canvas
    canvas = [[" " for _ in range(total_width)] for __ in range(total_height)]

    # For each node, compute its (x_center, y_center)
    node_pos = {}
    for ci, d in enumerate(depths):
        nodes_in_col = by_depth[d]
        # sort nodes for determinism (original insertion order in dict might be fine)
        nodes_in_col = list(nodes_in_col)
        # stack vertically centered
        col_h = len(nodes_in_col) * row_height
        start_y = (total_height - col_h) // 2
        for ri, name in enumerate(nodes_in_col):
            node_x_left = col_x[ci]
            node_w = col_widths[ci]
            # center within column cell
            cx = node_x_left + node_w // 2
            y = start_y + ri * row_height
            node_pos[name] = (cx, y)
            # draw label centered at (cx, y)
            lab = labels[name]
            start_c = cx - len(lab) // 2
            for i, ch in enumerate(lab):
                c = start_c + i
                if 0 <= c < total_width and 0 <= y < total_height:
                    canvas[y][c] = ch
            # draw a small box border left-right (optional)
            left = node_x_left
            right = node_x_left + node_w - 1
            # put vertical bars to visually delimit node area
            if 0 <= left < total_width:
                canvas[y][left] = "|"
            if 0 <= right < total_width:
                canvas[y][right] = "|"

    # Draw edges: for every node (that is an op), connect each input producer to this node center
    for dst_name, info in nodes.items():
        dst_op = info["op"]
        if dst_op in ("param", "const"):
            continue
        dst_x, dst_y = node_pos.get(dst_name, (None, None))
        if dst_x is None:
            continue
        for inp in info["inputs"]:
            # find producer for inp; if literal constant token used, it exists as nodes[token]
            if inp not in node_pos:
                # try const literal or param with same token
                if inp in nodes:
                    src = inp
                else:
                    # create a virtual constant node position at the leftmost column near depth 0
                    # place it at x = 0 and somewhere above
                    src_x = 0
                    src_y = dst_y
                    sx, sy = src_x, src_y
                if inp in node_pos:
                    sx, sy = node_pos[inp]
                else:
                    sx, sy = src_x, src_y
            else:
                sx, sy = node_pos[inp]

            # route: horizontal from (sx, sy) to just before dst_x, then vertical to dst_y at that col, then to dst_x
            # Simplified: horizontal at sy from sx+1 to dst_x-1, then vertical at dst_x-1 from sy to dst_y.
            if sx == dst_x and sy == dst_y:
                continue

            # Horizontal segment
            x0 = min(sx + 1, dst_x - 1)
            x1 = max(sx + 1, dst_x - 1)
            for cx in range(x0, x1 + 1):
                if 0 <= sy < total_height and 0 <= cx < total_width:
                    # don't overwrite node label characters (letters). Only write connector chars on spaces or bars.
                    if canvas[sy][cx] == " ":
                        canvas[sy][cx] = "-"
                    elif canvas[sy][cx] in ("|", "+"):
                        canvas[sy][cx] = "+"
            # Vertical segment at x = dst_x - 1 from sy to dst_y
            vx = dst_x - 1
            y0 = min(sy, dst_y)
            y1 = max(sy, dst_y)
            for ry in range(y0, y1 + 1):
                if 0 <= ry < total_height and 0 <= vx < total_width:
                    if canvas[ry][vx] == " ":
                        canvas[ry][vx] = "|"
                    elif canvas[ry][vx] in ("-", "+"):
                        canvas[ry][vx] = "+"

            # Place arrow head close to dst node
            ax = dst_x - 1
            ay = dst_y
            if 0 <= ay < total_height and 0 <= ax < total_width:
                canvas[ay][ax] = ">"

    # Optionally mark outputs on rightmost column
    outputs = parsed_outputs_from_graph(graph)
    rightmost_x = max(x for x, y in node_pos.values()) if node_pos else total_width - 1
    for oid in outputs:
        if oid in node_pos:
            x, y = node_pos[oid]
            # Put marker to the right of the node
            rx = min(total_width - 2, x + 6)
            if 0 <= y < total_height:
                canvas[y][rx] = "*"
                # write "OUT" vertically or horizontally
                s = "OUT"
                for i, ch in enumerate(s):
                    cx = rx + 1 + i
                    if cx < total_width:
                        canvas[y][cx] = ch

    # Convert canvas to string lines
    lines = ["".join(row).rstrip() for row in canvas]
    # trim empty top/bottom lines
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def parsed_outputs_from_graph(graph):
    # try to return outputs if present in nodes keys; if not, return highest-depth nodes
    parsed = graph
    nodes = parsed["nodes"]
    # try to glean outputs from "in (...)" by looking for nodes that aren't used as inputs anywhere else
    used = set()
    for name, info in nodes.items():
        for inp in info["inputs"]:
            used.add(inp)
    # candidates are nodes that are not used by anyone else (sinks)
    sinks = [n for n in nodes.keys() if n not in used]
    if sinks:
        return sinks
    return list(nodes.keys())


# ----------------------
# Top-level convenience
# ----------------------
def ascii_graph_from_jaxpr(jaxpr_text, max_width=120):
    parsed = parse_jaxpr(jaxpr_text)
    graph = build_graph(parsed)
    # attach parsed for marking outputs (simple)
    graph["parsed"] = parsed
    global parsed_cache  # small helper for output marking
    parsed_cache = parsed
    ascii_art = layout_and_render(graph, max_width=max_width)
    # produce a small header explaining nodes
    header = []
    header.append("ASCII graph (left → right indicates dataflow; 'op(name)' nodes show op and produced var):")
    header.append("")
    return "\n".join(header) + ascii_art


# ----------------------
# Example usage (three jaxpr strings from your message)
# ----------------------
if __name__ == "__main__":
    example1 = """{ lambda ; a:f32[]. let
    b:f32[] = sin a
    c:f32[] = cos a
    d:f32[] = exp a
    _:f32[] = mul b d
    e:f32[] = mul b 1.0
    f:f32[] = mul 1.0 d
    g:f32[] = mul e d
    h:f32[] = mul f c
    i:f32[] = add_any g h in (i,) }"""
    example2 = """{ lambda ; a:f32[]. let b:f32[] = sin a c:f32[] = cos a d:f32[] = exp a e:f32[] = mul b d f:f32[] = log a _:f32[] = mul e f g:f32[] = mul e 1.0 h:f32[] = mul 1.0 f i:f32[] = div g a j:f32[] = mul b h k:f32[] = mul h d l:f32[] = mul j d m:f32[] = add_any i l n:f32[] = mul k c o:f32[] = add_any m n in (o,) }"""
    example3 = """{ lambda ; a:f32[2]. let b:f32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=(1,)] a c:f32[] = squeeze[dimensions=(0,)] b d:f32[1] = slice[limit_indices=(2,) start_indices=(1,) strides=(1,)] a e:f32[] = squeeze[dimensions=(0,)] d f:f32[] = sin c g:f32[] = cos c h:f32[] = exp e _:f32[] = mul f h i:f32[] = mul f 1.0 j:f32[] = mul 1.0 h k:f32[] = mul i h l:f32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] k m:f32[2] = pad[padding_config=((1, np.int64(0), 0),)] l 0.0 n:f32[] = mul j g o:f32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] n p:f32[2] = pad[padding_config=((0, np.int64(1), 0),)] o 0.0 q:f32[2] = add_any m p in (q,) }"""

    print("=== Example 1 ===")
    print(ascii_graph_from_jaxpr(example1))
    print("\n\n=== Example 2 ===")
    print(ascii_graph_from_jaxpr(example2))
    print("\n\n=== Example 3 ===")
    print(ascii_graph_from_jaxpr(example3))
