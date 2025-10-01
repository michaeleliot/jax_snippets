import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

old_settings = np.seterr(divide='ignore', over='ignore', invalid='ignore')

print(jax.devices())

# ---------------- NumPy (functional) ----------------
def init_layer_numpy(n_in, n_out):
    W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)  # He init
    b = np.zeros((n_out,))
    return {"W": W, "b": b}

def init_network_numpy(layer_sizes, seed=0):
    np.random.seed(seed)
    return [init_layer_numpy(n_in, n_out) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]

def forward_numpy(params, X):
    x = X
    for layer in params[:-1]:
        x = np.maximum(0, np.dot(x, layer["W"]) + layer["b"])  # ReLU
    last = params[-1]
    return np.dot(x, last["W"]) + last["b"]

def loss_fn_numpy(params, X, Y):
    preds = forward_numpy(params, X)
    return np.mean((preds - Y) ** 2)

def compute_grads_numpy(params, X, Y):
    # Forward pass with storage
    activations = [X]
    preacts = []
    x = X
    for layer in params[:-1]:
        z = np.dot(x, layer["W"]) + layer["b"]
        x = np.maximum(0, z)
        activations.append(x)
        preacts.append(z)
    z_last = np.dot(x, params[-1]["W"]) + params[-1]["b"]
    y_pred = z_last
    activations.append(y_pred)
    preacts.append(z_last)

    grad = 2 * (y_pred - Y) / Y.shape[0]  # MSE gradient

    grads = []
    # Last layer
    dW = activations[-2].T @ grad
    db = np.sum(grad, axis=0)
    grads.insert(0, {"W": dW, "b": db})

    # Hidden layers
    grad = grad @ params[-1]["W"].T
    for i in range(len(params) - 2, -1, -1):
        relu_grad = (preacts[i] > 0).astype(float)
        grad = grad * relu_grad
        dW = activations[i].T @ grad
        db = np.sum(grad, axis=0)
        grads.insert(0, {"W": dW, "b": db})
        if i > 0:
            grad = grad @ params[i]["W"].T

    return grads

def update_numpy(params, grads, lr=0.01):
    new_params = []
    for p, g in zip(params, grads):
        new_params.append({
            "W": p["W"] - lr * g["W"],
            "b": p["b"] - lr * g["b"],
        })
    return new_params



# ---------------- Dataset ----------------
def target_fn(x):
    return jnp.sin(x) + 0.1 * x

key = jax.random.PRNGKey(42)
X = jnp.linspace(-5, 5, 200).reshape(-1, 1)
Y = target_fn(X)

n_train = 150
X_train, Y_train = X[:n_train], Y[:n_train]
X_test, Y_test = X[n_train:], Y[n_train:]


# ---------------- JAX Network ----------------
def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
    b = jnp.zeros((n_out,))
    return {"W": W, "b": b}

def init_network(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [init_layer(k, n_in, n_out) for k, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:])]

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(jnp.dot(x, layer["W"]) + layer["b"])
    last = params[-1]
    return jnp.dot(x, last["W"]) + last["b"]

def loss_fn(params, x, y):
    preds = forward(params, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def update(params, x, y, lr=0.01):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

import time

# ---------------- Train JAX ----------------
key = jax.random.PRNGKey(0)
params = init_network([1, 32, 32, 1], key)

iterations = 1000000

start_time = time.perf_counter()
for step in range(iterations):
    params = update(params, X_train, Y_train, lr=0.01)
    if step % 100000 == 0:
        print(f"[JAX] Step {step}, Train Loss = {loss_fn(params, X_train, Y_train):.4f}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Jax Training Elapsed time: {elapsed_time:.4f} seconds")

# ---------------- Train NumPy ----------------
X_train_np = np.array(X_train)
Y_train_np = np.array(Y_train)
X_test_np = np.array(X_test)
Y_test_np = np.array(Y_test)

start_time = time.perf_counter()

# Example training loop
params_np = init_network_numpy([1, 32, 32, 1], 0)

for epoch in range(iterations):
    grads = compute_grads_numpy(params_np, X_train_np, Y_train_np)
    params_np = update_numpy(params_np, grads, lr=0.01)
    if epoch % 100000 == 0:
        print(f"[NumPy] Epoch {epoch}, Loss={loss_fn_numpy(params_np, X_train_np, Y_train_np):.4f}")


end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Jax Training Elapsed time: {elapsed_time:.4f} seconds")

Y_pred_numpy = forward_numpy(params_np, X_test_np)
Y_pred_jax = forward(params, X_test)

# NumPy loss
loss_numpy = loss_fn_numpy(params_np, X_test_np, Y_test_np)
print(f"Final NumPy Loss on test set: {loss_numpy:.4f}")

# JAX loss
loss_jax = loss_fn(params, X_test, Y_test)
print(f"Final JAX Loss on test set: {loss_jax:.4f}")

# ---------------- Plot ----------------
plt.figure(figsize=(8,5))
plt.scatter(X_train, Y_train, label="Train Data", color="blue", alpha=0.5)
plt.scatter(X_test, Y_test, label="Test Data", color="green", alpha=0.5)
plt.plot(X_test, Y_pred_jax, label="JAX Prediction", color="red", linewidth=2)
plt.plot(X_test, Y_pred_numpy, label="NumPy Prediction", color="orange", linewidth=2, linestyle="--")
plt.legend()
plt.title("NumPy vs JAX Neural Network Approximation of f(x) = sin(x) + 0.1x")
plt.show()
