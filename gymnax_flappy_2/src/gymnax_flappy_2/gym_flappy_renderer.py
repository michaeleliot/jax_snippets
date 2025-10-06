# gymnax_flappy/pygame_renderer.py
import pygame


def init_renderer(params):
    """Initialize pygame window & surfaces."""
    pygame.init()
    screen = pygame.display.set_mode((params.win_w, params.win_h))
    pygame.display.set_caption("Flappy JAX")
    clock = pygame.time.Clock()
    return screen, clock


def draw_state(screen, state, params):
    """Draw EnvState (bird, pipes, ground)."""
    # Fill background sky blue
    screen.fill((135, 206, 250))

    # Draw pipes
    for pipe in state.pipes:
        px, gap_top = float(pipe[0]), float(pipe[1])
        gap_half = params.pipe_gap // 2

        # Top pipe rect
        top_rect = pygame.Rect(px, 0, params.pipe_w, gap_top - gap_half)
        # Bottom pipe rect
        bottom_rect = pygame.Rect(
            px, gap_top + gap_half, params.pipe_w,
            params.win_h - params.ground_h - (gap_top + gap_half)
        )

        pygame.draw.rect(screen, (0, 200, 0), top_rect)
        pygame.draw.rect(screen, (0, 200, 0), bottom_rect)

    # Draw bird (as a yellow square)
    bird_rect = pygame.Rect(
        params.bird_x,
        int(state.bird_y),
        params.bird_size,
        params.bird_size,
    )
    pygame.draw.rect(screen, (255, 255, 0), bird_rect)

    # Draw ground
    ground_rect = pygame.Rect(0, params.win_h - params.ground_h, params.win_w, params.ground_h)
    pygame.draw.rect(screen, (222, 184, 135), ground_rect)

    # Draw score text
    font = pygame.font.SysFont(None, 36)
    text = font.render(f"Score: {int(state.score)}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    # Flip display
    pygame.display.flip()

