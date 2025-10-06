# gymnax_flappy/flappy_env.py
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


# ----- State & Params dataclasses -----
@struct.dataclass
class EnvState(environment.EnvState):
    # bird's vertical position (float), velocity (float)
    bird_y: jax.Array
    bird_vy: jax.Array
    # pipes: shape (num_pipes, 2) -> (x_position, gap_top)
    pipes: jax.Array
    score: jax.Array
    time: jax.Array
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    # environment/game parameters
    win_w: int = 400
    win_h: int = 600
    ground_h: int = 100

    bird_x: int = 100
    bird_size: int = 30

    gravity: float = 0.5
    flap_v: float = -9.0
    max_fall_speed: float = 15.0

    pipe_w: int = 70
    pipe_gap: int = 160
    pipe_speed: float = 4.0

    num_pipes: int = 3
    pipe_distance: int = 240  # spacing between respawn x positions

    max_steps_in_episode: int = 10000
    tick_reward: float = 0.1       # reward per frame survived
    ceiling_penalty: float = -0.5  # penalty for hitting ceiling
    pipe_pass_reward: float = 1.0  # reward for passing a pipe

    
    # existing params like win_w, win_h, pipe_speed, etc.


# ----- Environment class -----
class FlappyEnv(environment.Environment[EnvState, EnvParams]):
    """JAX (Gymnax-style) Flappy Bird environment (logic-only)."""

    def __init__(self):
        super().__init__()
        # observation vector length:
        # 2 for bird (y, vy) + 2 per pipe (x, gap)
        # we'll set obs_shape at reset according to params.num_pipes
        self.key = jax.random.PRNGKey(0)  # single PRNG key for the environment
        self._obs_shape = None

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def reset(self, key, env_params):
        """Reset environment for RL agent. Returns flattened obs + state."""
        self.key, subkey = jax.random.split(self.key)
        obs, self.state = self.reset_env(subkey, self.default_params)
        return obs.flatten(), self.state

    def step(self, keys, state, action: int, env_params):
        """Step environment for RL agent. Returns flattened obs + next_state + reward + done."""
        self.key, subkey = jax.random.split(self.key)
        obs, self.state, reward, done, _info = self.step_env(subkey, self.state, action, self.default_params)
        info = {}
        return obs.flatten(), self.state, reward.astype(float), done.astype(bool), info

    def _obs_shape(self, params: EnvParams) -> Tuple[int, ...]:
        return (2 + 2 * params.num_pipes,)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """
        Step the environment.
        action: scalar int: 0 = noop, 1 = flap
        """
        a = jnp.asarray(action).astype(jnp.int32)
        # 1) Apply agent action (flap or noop)
        state, reward_survive = step_agent(a, state, params)

        # 2) Move pipes and respawn offscreen pipes (randomize gap top using RNG)
        key, subkey = jax.random.split(key)
        state, pass_rewards, key = step_pipes_and_score(state, subkey, params)

        # total reward is reward_pass (should be zero, but keep for parity) + pass_rewards
        reward = reward_survive + pass_rewards

        # 3) Update bird physics (gravity & cap)
        state = step_physics(state, params)

        # 4) Check collisions (bird vs pipes, ground/ceiling)
        collided = check_collision(state, params)
        done = jnp.logical_or(collided, state.time >= params.max_steps_in_episode)

        # update score/time/terminal
        state = state.replace(
            time=state.time + 1,
            terminal=done,
        )

        obs = self.get_obs(state, params, key)
        info = {"discount": self.discount(state, params)}

        # important: convert reward/done dtype
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            jnp.asarray(reward).astype(jnp.float32),
            jnp.asarray(done),
            info,
        )

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        """Reset the environment state. Returns (obs, state)."""
        # initial bird in middle, zero velocity
        bird_y = params.win_h // 2
        bird_vy = 0.0

        # initialize pipes positions spaced to the right
        xs = jnp.array(
            [params.win_w + 100 + i * params.pipe_distance for i in range(params.num_pipes)],
            dtype=jnp.float32,
        )
        # sample gap tops
        key, subkey = jax.random.split(key)
        gap_min = 60
        gap_max = params.win_h - params.ground_h - 60 - params.pipe_gap
        gap_max = jnp.maximum(gap_min + 1, gap_max)
        gap_tops = jax.random.randint(subkey, (params.num_pipes,), minval=gap_min, maxval=gap_max + 1)
        pipes = jnp.stack([xs, gap_tops.astype(jnp.float32)], axis=1)  # shape (num_pipes, 2)

        state = EnvState(
            bird_y=jnp.asarray(bird_y, dtype=jnp.float32),
            bird_vy=jnp.asarray(bird_vy, dtype=jnp.float32),
            pipes=pipes,
            score=jnp.asarray(0, dtype=jnp.int32),
            time=jnp.asarray(0, dtype=jnp.int32),
            terminal=False,
        )

        obs = self.get_obs(state, params, key)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams, key: jax.Array | None = None) -> jax.Array:
        """
        Return compact observation vector:
        [bird_y_norm, bird_vy_norm, pipe0_x_norm, pipe0_gap_norm, pipe1_x_norm, pipe1_gap_norm, ...]
        Normalization: positions are scaled to [0,1] by dividing by window dims.
        """
        # bird y normalized (0..1), velocity scaled by something reasonable (div by 50)
        by = state.bird_y / float(params.win_h)
        bvy = state.bird_vy / 50.0
        pipe_x = state.pipes[:, 0] / float(params.win_w)  # (num_pipes,)
        pipe_gap = state.pipes[:, 1] / float(params.win_h)  # (num_pipes,)

        obs_vec = jnp.concatenate([jnp.array([by, bvy], dtype=jnp.float32), pipe_x.astype(jnp.float32).ravel(), pipe_gap.astype(jnp.float32).ravel()])
        return obs_vec.astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return jnp.array(state.terminal)

    @property
    def name(self) -> str:
        return "Flappy-JAX"

    @property
    def num_actions(self) -> int:
        return 2  # noop, flap

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        shape = (2 + 2 * params.num_pipes,)
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "bird_y": spaces.Box(0, params.win_h, jnp.array(0.0)),
                "bird_vy": spaces.Box(-jnp.inf, jnp.inf, jnp.array(0.0)),
                "pipes": spaces.Box(0, params.win_w, jnp.zeros((params.num_pipes, 2))),
                "score": spaces.Discrete(1000000),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
                "terminal": spaces.Discrete(2),
            }
        )


# ----- Helper / step functions (pure functions, JAX-friendly) -----


def step_agent(action: jax.Array, state: EnvState, params: EnvParams) -> tuple[EnvState, jax.Array]:
    """
    Apply action to bird. action: 0 noop, 1 flap.
    Returns updated state and immediate reward (including survival, ceiling penalty).
    """
    flap_cond = action == 1
    # set vy to flap_v if flap_cond else keep existing vy
    new_vy = jax.lax.select(flap_cond, jnp.asarray(params.flap_v, dtype=jnp.float32), state.bird_vy)
    new_state = state.replace(bird_vy=new_vy)

    # Base reward for surviving
    reward = jnp.asarray(params.tick_reward, dtype=jnp.float32)  # e.g., 0.1 per tick

    # Penalty if hitting ceiling
    hit_ceiling = state.bird_y < 0
    reward += jax.lax.select(hit_ceiling, jnp.asarray(params.ceiling_penalty, dtype=jnp.float32), 0.0)

    return new_state, reward


def step_physics(state: EnvState, params: EnvParams) -> EnvState:
    """Apply gravity and clamp fall speed, update bird_y."""
    vy = state.bird_vy + params.gravity
    vy = jnp.minimum(vy, params.max_fall_speed)
    y = state.bird_y + vy
    return state.replace(bird_vy=vy, bird_y=y)


def step_pipes_and_score(state: EnvState, key: jax.Array, params: EnvParams) -> tuple[EnvState, jax.Array, jax.Array]:
    """
    Move pipes, respawn offscreen ones, and award score for passing pipes.
    Returns updated state, reward_from_passing, and updated key.
    """
    pipes = state.pipes
    bird_x = params.bird_x

    # Move pipes
    new_xs = pipes[:, 0] - params.pipe_speed

    # Offscreen pipes
    offscreen = new_xs + params.pipe_w < 0
    max_x = float(params.win_w) + 100.0
    new_xs = jnp.where(offscreen, max_x + params.pipe_distance, new_xs)

    # Sample new gaps for respawned pipes
    gap_min = 60
    gap_max = params.win_h - params.ground_h - 60 - params.pipe_gap
    gap_max = jnp.maximum(gap_min + 1, gap_max)
    sampled_gaps = jax.random.randint(key, (params.num_pipes,), minval=gap_min, maxval=gap_max + 1)
    new_gaps = jnp.where(offscreen, sampled_gaps.astype(jnp.float32), pipes[:, 1].astype(jnp.float32))

    new_pipes = jnp.stack([new_xs, new_gaps], axis=1) # TODO: update this function to update existing pipes array instead of replace and removing

    # Reward for passing pipes
    prev_xs = pipes[:, 0]
    prev_cond = (prev_xs + params.pipe_w) >= bird_x
    now_cond = (new_xs + params.pipe_w) < bird_x
    passed = jnp.logical_and(prev_cond, now_cond)

    reward_pass = jnp.sum(passed.astype(jnp.float32)) * jnp.asarray(params.pipe_pass_reward, dtype=jnp.float32)

    new_score = state.score + jnp.sum(passed.astype(jnp.int32))

    return state.replace(pipes=new_pipes, score=new_score), reward_pass, jax.random.split(key)[0]


def check_collision(state: EnvState, params: EnvParams) -> jax.Array:
    """
    Check AABB collisions between bird and pipes or ground/ceiling.
    Bird bbox: (bird_x, bird_y, bird_size, bird_size)
    Pipe top bbox: (px, 0, pipe_w, gap_top - pipe_gap//2)
    Pipe bottom bbox: (px, gap_top + pipe_gap//2, pipe_w, win_h - ground_h - (gap_top + pipe_gap//2))
    """
    bx = params.bird_x
    by = state.bird_y
    s = params.bird_size

    # ground/ceiling
    hit_ground = by + s > (params.win_h - params.ground_h)
    hit_ceiling = by < 0
    hit_gc = jnp.logical_or(hit_ground, hit_ceiling)

    # check pipes
    def _collide_with_pipe(pipe):
        px = pipe[0]
        gap_top = pipe[1]
        top_h = gap_top - params.pipe_gap // 2
        bot_y = gap_top + params.pipe_gap // 2
        # top rect: (px, 0, pipe_w, top_h)
        # bot rect: (px, bot_y, pipe_w, win_h - ground - bot_y)
        # AABB overlap test:
        # overlap if not (bx+s < px or bx > px+pipe_w or by+s < y1 or by > y2)
        top_overlap = jnp.logical_not(
            jnp.logical_or(
                bx + s < px,
                jnp.logical_or(bx > px + params.pipe_w, jnp.logical_or(by + s < 0, by > top_h)),
            )
        )
        bot_overlap = jnp.logical_not(
            jnp.logical_or(
                bx + s < px,
                jnp.logical_or(bx > px + params.pipe_w, jnp.logical_or(by + s < bot_y, by > params.win_h - params.ground_h)),
            )
        )
        return jnp.logical_or(top_overlap, bot_overlap)

    # vectorize over pipes
    collisions = jax.vmap(_collide_with_pipe)(state.pipes)
    any_pipe_collision = jnp.any(collisions)

    return jnp.logical_or(hit_gc, any_pipe_collision)


# ----- end of file -----
