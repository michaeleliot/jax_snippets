import jax.numpy as jnp
from jax import jit
import numpy as np
import jax

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)

norm_compiled = jit(norm)

np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)

print(norm(X))
print(norm_compiled(X))

print(jax.devices())