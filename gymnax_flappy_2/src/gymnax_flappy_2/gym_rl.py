import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pygame
from gymnax_flappy_2.gym_flappy_logic import FlappyEnv
from gymnax_flappy_2.gym_flappy_renderer import init_renderer, draw_state

# --------------------------
# Environment
# --------------------------
env = FlappyEnv()
obs_flat, state = env.reset("", "")
n_actions = env.num_actions
key = jax.random.PRNGKey(0)

# --------------------------
# Networks
# --------------------------
hidden_sizes = [64, 64]

def policy_fn(x):
    mlp = hk.nets.MLP(hidden_sizes + [n_actions])
    return jax.nn.softmax(mlp(x))

def value_fn(x):
    mlp = hk.nets.MLP(hidden_sizes + [1])
    return mlp(x)

policy = hk.without_apply_rng(hk.transform(policy_fn))
value = hk.without_apply_rng(hk.transform(value_fn))

key, subkey = jax.random.split(key)
policy_params = policy.init(subkey, obs_flat)
key, subkey = jax.random.split(key)
value_params = value.init(subkey, obs_flat)

policy_opt = optax.adam(3e-4)
value_opt = optax.adam(3e-4)
policy_opt_state = policy_opt.init(policy_params)
value_opt_state = value_opt.init(value_params)

# --------------------------
# Helpers
# --------------------------
def select_action(params, obs, key):
    probs = policy.apply(params, obs)
    key, subkey = jax.random.split(key)
    action = jax.random.choice(subkey, n_actions, p=probs)
    return int(action), key

def compute_advantage(rewards, values, gamma=0.99, lam=0.95):
    adv = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
        gae = delta + gamma * lam * gae
        adv.insert(0, gae)
    return jnp.array(adv)

# --------------------------
# Training Loop
# --------------------------
episodes = 3000
max_steps = 1000
gamma = 0.99
lam = 0.95

for ep in range(episodes):
    obs, state = env.reset("", "")
    obs = obs.astype(jnp.float32)
    ep_reward = 0.0

    obs_buffer = []
    actions_buffer = []
    rewards_buffer = []
    values_buffer = []

    for step in range(max_steps):
        obs_buffer.append(obs)
        value_pred = value.apply(value_params, obs).squeeze()
        values_buffer.append(value_pred)

        # select action
        action, key = select_action(policy_params, obs, key)
        actions_buffer.append(action)

        # step environment
        obs_next, state, reward, done, _ = env.step("", "", action, "")
        rewards_buffer.append(reward)
        obs = obs_next.astype(jnp.float32)
        ep_reward += reward

        if done:
            break


    # Convert buffers to arrays
    obs_batch = jnp.stack(obs_buffer)
    actions_batch = jnp.array(actions_buffer)
    rewards_batch = jnp.array(rewards_buffer)
    values_batch = jnp.array(values_buffer)

    # Compute advantages and returns
    advantages = compute_advantage(rewards_batch, values_batch, gamma, lam)
    returns = advantages + values_batch

    # --- Policy update ---
    def policy_loss_fn(params, obs, actions, adv):
        logits = policy.apply(params, obs)
        logp_all = jnp.log(logits + 1e-8)
        logp = jnp.take_along_axis(logp_all, actions[:, None], axis=1).squeeze()
        return -jnp.mean(logp * adv)

    grads = jax.grad(policy_loss_fn)(policy_params, obs_batch, actions_batch, advantages)
    updates, policy_opt_state = policy_opt.update(grads, policy_opt_state)
    policy_params = optax.apply_updates(policy_params, updates)

    # --- Value update ---
    def value_loss_fn(params, obs, targets):
        v_pred = value.apply(params, obs).squeeze()
        return jnp.mean((v_pred - targets) ** 2)

    grads = jax.grad(value_loss_fn)(value_params, obs_batch, returns)
    updates, value_opt_state = value_opt.update(grads, value_opt_state)
    value_params = optax.apply_updates(value_params, updates)

    if ep % 10 == 0:
        print(f"Episode {ep}, Reward: {ep_reward:.2f}")

# --------------------------
# Playback trained agent
# --------------------------
screen, clock = init_renderer(env.default_params)
obs, state = env.reset("", "")
obs = obs.astype(jnp.float32)
running = True

print(policy)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    print(obs)
    # Select action greedily from trained policy
    logits = policy.apply(policy_params, obs)
    action = int(jnp.argmax(logits))
    print(action, logits)

    obs, state, reward, done, _ = env.step("", "", action, "")
    obs = obs.astype(jnp.float32)

    draw_state(screen, state, env.default_params)
    clock.tick(30)

    if done:
        obs, state = env.reset("", "")
        obs = obs.astype(jnp.float32)

pygame.quit()
