import jax
import jax.numpy as jnp

# Define a simple 2-layer neural network
def two_layer_nn(params, x):
    W1, b1, W2, b2 = params
    z1 = jnp.dot(x, W1) + b1
    a1 = jnp.tanh(z1)
    z2 = jnp.dot(a1, W2) + b2
    return z2

# Loss function: mean squared error
def loss_fn(params, x, y):
    preds = two_layer_nn(params, x)
    return jnp.mean((preds - y) ** 2)

# Initialize parameters
key = jax.random.PRNGKey(0)
x = jnp.ones((10, 3))         # batch of 10, input dim = 3
y = jnp.ones((10, 2))         # target dim = 2

W1 = jax.random.normal(key, (3, 5))
b1 = jnp.zeros((5,))
W2 = jax.random.normal(key, (5, 2))
b2 = jnp.zeros((2,))
params = (W1, b1, W2, b2)

# Gradient of loss wrt params
grad_loss_fn = jax.grad(loss_fn)

# JAXPR of backpropagation
jaxpr = jax.make_jaxpr(grad_loss_fn)(params, x, y)
print(jaxpr)
