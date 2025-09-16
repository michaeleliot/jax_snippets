# jax-only multimodal prefix approach (no PyTorch)
# - Whisper (Flax) encodes audio -> jnp array
# - Flax adapter downsamples + projects to LM embed dim
# - Add <audio_*> tokens to tokenizer, expand Flax model embedding matrix
# - Overwrite new token embeddings with projected audio embeddings (JAX arrays)
# - Generate with FlaxAutoModelForCausalLM.generate using updated params

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer

# --- small Flax modules ----------------------------------------------------------
class AudioAdapter(nn.Module):
    """Downsample audio embeddings in time by conv1d (kernel=4, stride=4)."""
    embedding_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (batch, seq_len, embedding_dim) -> use Conv1D across seq dim
        conv = nn.Conv(features=self.embedding_dim, kernel_size=(4,), strides=(4,))
        return conv(x)


class ProjectToEmbed(nn.Module):
    """Project audio adapter outputs to the LM token embedding dimension."""
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, seq_len, in_dim) -> project per-frame to out_dim
        dense = nn.Dense(self.out_dim)
        return dense(x)


# --- whisper encoding ------------------------------------------------------------
def whisper_encode_audio(sample):
    """
    Given a dataset audio sample (dict with 'array' and 'sampling_rate'),
    return whisper last_hidden_state as jnp array (1, seq_len, embed_dim).
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model, whisper_params = FlaxWhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base", _do_init=False
    )
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features
    enc_out = model.encode(input_features=input_features, params=whisper_params)
    last_hidden_state = enc_out.last_hidden_state  # jnp array (1, seq_len, in_dim)
    return last_hidden_state


# --- downsample + project --------------------------------------------------------
def downsample_and_project(audio_embedding: jnp.ndarray, target_embed_dim: int, rng: jax.random.PRNGKey):
    """
    Downsample audio embeddings and project to target_embed_dim.
    audio_embedding: jnp array (1, seq_len, in_dim)
    returns: numpy array shape (seq_len_down, target_embed_dim)
    """
    in_dim = audio_embedding.shape[-1]
    adapter = AudioAdapter(embedding_dim=in_dim)
    proj = ProjectToEmbed(out_dim=target_embed_dim)

    rng1, rng2 = jax.random.split(rng)
    adapter_vars = adapter.init(rng1, audio_embedding)
    adapted = adapter.apply(adapter_vars, audio_embedding)  # (1, seq_len_down, in_dim)

    proj_vars = proj.init(rng2, adapted)
    projected = proj.apply(proj_vars, adapted)  # (1, seq_len_down, target_embed_dim)

    # convert to numpy for easy placement into embedding matrix
    projected_np = np.asarray(projected[0])  # (seq_len_down, target_embed_dim)
    return projected_np


# --- helpers for updating Flax model embeddings -----------------------------------
def find_embedding_key_and_array(params, old_vocab_size):
    """
    Recursively search params dict for a 2D array whose first dim == old_vocab_size.
    Return key path (as tuple) and array.
    """
    # params is a nested dict / FrozenDict; convert to immutable mapping via unfreeze if needed
    def _recurse(d, path=()):
        if isinstance(d, dict):
            for k, v in d.items():
                yield from _recurse(v, path + (k,))
        else:
            # leaf: numpy/jax array / DeviceArray
            try:
                arr = jnp.asarray(d)
                if arr.ndim == 2 and arr.shape[0] == old_vocab_size:
                    yield (path, arr)
            except Exception:
                return

    # work on unfreeze(params) to ensure dict-like traversal
    plain = unfreeze(params)
    found = list(_recurse(plain))
    return found  # list of (path_tuple, array)


def set_in_params(params, path_tuple, new_array):
    """
    Given a (possibly frozen) params mapping, set params[path_tuple] = new_array and return new frozen params.
    """
    mut = unfreeze(params)
    cur = mut
    for k in path_tuple[:-1]:
        cur = cur[k]
    cur[path_tuple[-1]] = jnp.asarray(new_array)
    return freeze(mut)


# --- injection + generation ------------------------------------------------------
def inject_audio_tokens_and_generate_flax(
    flax_model: FlaxAutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    audio_embeddings_np: np.ndarray,
    query_text: str,
    rng: jax.random.PRNGKey,
    max_length: int = 128,
    max_new_tokens: int = 64,
):
    """
    - Add <audio_i> tokens to tokenizer
    - Expand Flax model embedding matrix in params to accommodate them
    - Overwrite newly-added token embeddings with audio_embeddings_np
    - Build input_ids: [audio_tokens..., tokenized(query_text)...]
    - Call flax_model.generate with updated params
    """

    # 1) remember old vocab size then add tokens
    old_vocab_size = len(tokenizer)
    seq_len_audio, embed_dim = audio_embeddings_np.shape
    audio_tokens = [f"<audio_{i}>" for i in range(seq_len_audio)]
    num_added = tokenizer.add_tokens(audio_tokens)  # modifies tokenizer in-place
    new_vocab_size = len(tokenizer)
    if new_vocab_size - old_vocab_size != seq_len_audio:
        # handle mismatch (maybe some tokens existed); adjust audio token ids accordingly
        # find ids via convert_tokens_to_ids
        pass

    # 2) find the embedding array inside flax_model.params (search for shape[0]==old_vocab_size)
    params = flax_model.params
    found = find_embedding_key_and_array(params, old_vocab_size)
    if not found:
        raise RuntimeError("Could not find embedding matrix with first dim == old_vocab_size in model params.")
    # Pick the most plausible candidate: prefer arrays with typical embedding size
    # We'll choose the first match (if multiple, user can refine later).
    path_tuple, emb_array = found[0]
    model_embed_dim = int(emb_array.shape[1])
    if model_embed_dim != embed_dim:
        # If dims mismatch, raise — you should project audio to model embed dim.
        raise ValueError(f"embed dim mismatch: audio_projected {embed_dim} != model embed {model_embed_dim}")

    # 3) Build new embedding matrix (concatenate old + audio embeddings at the end)
    old_emb = emb_array  # DeviceArray
    # Ensure correct dtype
    audio_jax = jnp.asarray(audio_embeddings_np, dtype=old_emb.dtype)
    new_emb = jnp.concatenate([old_emb, audio_jax], axis=0)  # shape (new_vocab_size, embed_dim)

    # 4) set in params (returns a frozen dict)
    new_params = set_in_params(params, path_tuple, new_emb)

    # 5) build input ids: audio tokens first, then query
    # get the ids for the audio tokens (use tokenizer.convert_tokens_to_ids to be robust)
    audio_token_ids = tokenizer.convert_tokens_to_ids(audio_tokens)
    # tokenize the query
    tokenized = tokenizer(query_text, return_tensors="np", padding=False)
    query_ids = tokenized["input_ids"][0].astype("int32")  # numpy 1D

    # compose final input_ids (1, total_len)
    input_ids = np.concatenate([np.array(audio_token_ids, dtype="int32"), query_ids], axis=0)[None, :]
    # create attention_mask
    attention_mask = np.ones_like(input_ids)

    # 6) call Flax generate
    # Note: Flax generate expects jnp arrays or numpy arrays; pass our updated params explicitly
    # Provide a PRNG key. If needed, split externally.
    output = flax_model.generate(
        input_ids=jnp.asarray(input_ids),
        attention_mask=jnp.asarray(attention_mask),
        params=new_params,
        prng_key=rng,
        max_length=max_length,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # output is a GenerationOutput-like object. The sequences attr holds token ids.
    # Convert to numpy and decode
    out_ids = np.asarray(output.sequences)
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    return decoded, new_params  # return new_params so the caller can cache it


# --- top-level wrapper ----------------------------------------------------------
def answer_question_from_audio_sample_flax(sample, query_text, model_dir="mistral-hf-7B-v0.2", rng_seed=0):
    """
    High-level function: encode audio, downsample & project, inject into Flax model params,
    and generate text. Everything is JAX/Flax (no PyTorch).
    """
    # 1) Whisper encode
    audio_last_hidden = whisper_encode_audio(sample)  # jnp (1, seq_len, in_dim)

    # 2) load Flax Mistral and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token  # be safe

    flax_model = FlaxAutoModelForCausalLM.from_pretrained(model_dir, dtype=jnp.float32)  # load into RAM (JAX)
    # get model embed dim by inspecting params (search)
    # find an embedding array (shape: vocab, embed_dim)
    params = flax_model.params
    found = find_embedding_key_and_array(params, old_vocab_size=len(tokenizer))
    if not found:
        # fallback: attempt to find any 2D array and treat its second dim as embed dim
        def _find_any_2d(p):
            for k, v in unfreeze(p).items():
                if isinstance(v, dict):
                    res = _find_any_2d(v)
                    if res is not None:
                        return res
                else:
                    try:
                        arr = jnp.asarray(v)
                        if arr.ndim == 2:
                            return arr
                    except Exception:
                        continue
            return None
        candidate = _find_any_2d(params)
        if candidate is None:
            raise RuntimeError("Could not find any 2D param to infer embed dim.")
        model_embed_dim = int(candidate.shape[1])
    else:
        model_embed_dim = int(found[0][1].shape[1])

    # 3) downsample & project
    rng = jax.random.PRNGKey(rng_seed)
    audio_projected_np = downsample_and_project(audio_last_hidden, target_embed_dim=model_embed_dim, rng=rng)

    # 4) inject & generate
    rng, subrng = jax.random.split(rng)
    outputs, new_params = inject_audio_tokens_and_generate_flax(
        flax_model=flax_model,
        tokenizer=tokenizer,
        audio_embeddings_np=audio_projected_np,
        query_text=query_text,
        rng=subrng,
        max_length=128,
        max_new_tokens=64,
    )

    return outputs, new_params


# ---------------- Example usage -----------------------------------------
if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]  # has keys "array", "sampling_rate"

    out, params = answer_question_from_audio_sample_flax(sample, "What is the point of this text?", model_dir="mistral-hf-7B-v0.2")
    print("Model output:", out)
