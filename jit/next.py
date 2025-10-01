import time
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap

# Save current settings
old_settings = np.seterr(divide='ignore', over='ignore', invalid='ignore')

print(jax.devices())

# ------------------------------
# Parameters
# ------------------------------
N_SAMPLES = 100_000       # number of data points
N_FEATURES = 50           # number of features
BATCH_SIZE = 50           # number of datasets for batched gradient descent
SPREAD = 2.5              # noise standard deviation
LR_JAX = 0.001            # learning rate for JAX
ITERATIONS = 500          # gradient descent iterations
SEED = 0


key = random.PRNGKey(SEED)

# ------------------------------
# Generate synthetic multivariate data
# ------------------------------
key, subkey1, subkey2, subkey3 = random.split(key, 4)
X_jnp = random.uniform(subkey1, shape=(N_SAMPLES, N_FEATURES), minval=-10, maxval=10)
true_W = random.uniform(subkey2, shape=(N_FEATURES,))
y_jnp = X_jnp @ true_W + random.normal(subkey3, shape=(N_SAMPLES,)) * SPREAD

# ------------------------------
# JAX Gradient Descent
# ------------------------------
def gradient_step(params, _):
    W = params
    y_pred = X_jnp @ W
    error = y_jnp - y_pred
    grad = (-2 / N_SAMPLES) * (X_jnp.T @ error)
    W = W - LR_JAX * grad
    return W, None

W0 = random.uniform(key, shape=(N_FEATURES,), minval=-1, maxval=1)
def jax_gradient_descent():
    W_final, _ = lax.scan(gradient_step, W0, None, length=ITERATIONS)
    return X_jnp @ W_final

jax_gd_compiled = jit(jax_gradient_descent)
start = time.perf_counter()
y_pred_jax_gd = jax_gd_compiled()
finish = time.perf_counter()
print(f"JAX Gradient Descent (vectorized) {finish - start:0.4f} seconds")


# ------------------------------
# NumPy version: single dataset
# ------------------------------
np.random.seed(SEED)
X_np = np.array(X_jnp)
y_np = np.array(y_jnp)

# Normalize features for numerical stability
X_np = (X_np - np.mean(X_np, axis=0)) / np.std(X_np, axis=0)

# Manual gradient descent
W_np = np.zeros(N_FEATURES)  # initialize small
start = time.perf_counter()
for _ in range(ITERATIONS):
    y_pred = X_np @ W_np
    error = y_np - y_pred
    grad = (-2 / N_SAMPLES) * (X_np.T @ error)
    W_np -= LR_JAX * grad
y_pred_np_gd = X_np @ W_np
finish = time.perf_counter()
print(f"NumPy Gradient Descent (stable) {finish - start:0.4f} seconds")

# ------------------------------
# JAX lstsq
# ------------------------------
def jax_lstsq_multivariate():
    ones = jnp.ones((N_SAMPLES, 1))
    X_aug = jnp.hstack((ones, X_jnp))
    beta, residuals, rank, s = jnp.linalg.lstsq(X_aug, y_jnp, rcond=None)
    return X_aug @ beta

jax_lstsq_compiled = jit(jax_lstsq_multivariate)
start = time.perf_counter()
y_pred_jax_lstsq = jax_lstsq_compiled()
finish = time.perf_counter()
print(f"JAX lstsq (multivariate) {finish - start:0.4f} seconds")

# ------------------------------
# NumPy lstsq
# ------------------------------
start = time.perf_counter()
X_aug_np = np.hstack([np.ones((N_SAMPLES,1)), X_np])
beta_np, _, _, _ = np.linalg.lstsq(X_aug_np, y_np, rcond=None)
y_pred_np_lstsq = X_aug_np @ beta_np
finish = time.perf_counter()
print(f"NumPy lstsq {finish - start:0.4f} seconds")


# ------------------------------
# Optional: Batched gradient descent (multiple datasets)
# ------------------------------
# Generate batch of datasets
key, subkey1, subkey2 = random.split(key, 3)
X_batch = random.uniform(subkey1, shape=(BATCH_SIZE, N_SAMPLES, N_FEATURES), minval=-10, maxval=10)
true_W_batch = random.uniform(subkey2, shape=(BATCH_SIZE, N_FEATURES))
y_batch = jnp.einsum('bni,bi->bn', X_batch, true_W_batch) + random.normal(subkey3, shape=(BATCH_SIZE, N_SAMPLES)) * SPREAD

def batched_gradient_descent(X, y):
    def gradient_step(W, _):
        y_pred = X @ W
        error = y - y_pred
        grad = (-2 / X.shape[0]) * (X.T @ error)
        W = W - LR_JAX * grad
        return W, None

    W0 = jnp.zeros((X.shape[1],))
    W_final, _ = lax.scan(gradient_step, W0, None, length=ITERATIONS)
    return X @ W_final

batched_gd_compiled = jit(vmap(batched_gradient_descent, in_axes=(0,0)))
start = time.perf_counter()
y_pred_batch = batched_gd_compiled(X_batch, y_batch)
finish = time.perf_counter()
print(f"JAX Batched Gradient Descent ({BATCH_SIZE} datasets) {finish - start:0.4f} seconds")


# ------------------------------
# NumPy batched gradient descent (for comparison with JAX batched)
# ------------------------------
X_batch_np = np.random.uniform(-10, 10, size=(BATCH_SIZE, N_SAMPLES, N_FEATURES))
true_W_batch_np = np.random.uniform(-1, 1, size=(BATCH_SIZE, N_FEATURES))
y_batch_np = np.einsum('bni,bi->bn', X_batch_np, true_W_batch_np) + np.random.normal(0, SPREAD, size=(BATCH_SIZE, N_SAMPLES))

# Normalize each batch
X_batch_np = (X_batch_np - np.mean(X_batch_np, axis=1, keepdims=True)) / np.std(X_batch_np, axis=1, keepdims=True)

start = time.perf_counter()
y_batch_pred_np = np.zeros_like(y_batch_np)
for b in range(BATCH_SIZE):
    W = np.zeros(N_FEATURES)
    for _ in range(ITERATIONS):
        y_pred = X_batch_np[b] @ W
        error = y_batch_np[b] - y_pred
        grad = (-2 / N_SAMPLES) * (X_batch_np[b].T @ error)
        W -= LR_JAX * grad
    y_batch_pred_np[b] = X_batch_np[b] @ W
finish = time.perf_counter()
print(f"NumPy Batched Gradient Descent ({BATCH_SIZE} datasets, stable) {finish - start:0.4f} seconds")


# ------------------------------
# Plot predictions (first 1000 samples)
# ------------------------------
plt.figure(figsize=(12,4))
plt.plot(y_np[:1000], y_pred_np_gd[:1000], 'o', alpha=0.3, label='NumPy GD Pred vs True')
plt.plot(y_np[:1000], y_pred_np_lstsq[:1000], 'x', alpha=0.3, label='NumPy lstsq Pred vs True')
plt.plot(y_jnp[:1000], y_pred_jax_gd[:1000], '.', alpha=0.3, label='JAX GD Pred vs True')
plt.plot(y_jnp[:1000], y_pred_jax_lstsq[:1000], '+', alpha=0.3, label='JAX lstsq Pred vs True')
plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.legend()
plt.title("Multivariate Linear Regression Predictions (first 1000 samples)")
plt.show()
