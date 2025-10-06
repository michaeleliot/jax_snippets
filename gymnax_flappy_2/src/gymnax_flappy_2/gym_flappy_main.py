import jax
import pygame
from gymnax_flappy_2.gym_flappy_logic import FlappyEnv
from gymnax_flappy_2.gym_flappy_renderer import init_renderer, draw_state

def main():
    env = FlappyEnv()
    params = env.default_params
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)

    # Initialize pygame
    screen, clock = init_renderer(params)

    running = True
    while running:
        # Handle pygame events (quit, keyboard)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # For demo: flap if space pressed
        keys = pygame.key.get_pressed()
        action = 1 if keys[pygame.K_SPACE] else 0

        # Step environment
        key, subkey = jax.random.split(key)
        obs, state, reward, done, info = env.step_env(subkey, state, action, params)

        print(reward)

        # Draw state
        draw_state(screen, state, params)
        clock.tick(30)

        if bool(done):
            obs, state = env.reset_env(key, params)

    pygame.quit()


if __name__ == "__main__":
    main()
