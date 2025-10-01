import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(x) * jnp.exp(x)

# Get the derivative function
df = jax.grad(f)

# Print the traced computation
print(jax.make_jaxpr(df)(1.0))



def f(x):
    return jnp.sin(x) * jnp.exp(x) * jnp.log(x)

df = jax.grad(f)
print(jax.make_jaxpr(df)(1.0))

import jax
import jax.numpy as jnp

# Define a function of two variables
def f(xy):
    x, y = xy
    return jnp.sin(x) * jnp.exp(y)

# Point to evaluate at
xy0 = jnp.array([1.0, 2.0])

# Gradient (reverse-mode autodiff)
grad_f = jax.grad(f)
print("Gradient:", grad_f(xy0))  # should be [e^y cos(x), sin(x)e^y]

# Jacobian (explicit vector-Jacobian)
jac_f = jax.jacobian(f)
print("Jacobian:", jac_f(xy0))

# Make Jaxpr for the gradient
print("\nJaxpr of grad(f):")
print(jax.make_jaxpr(grad_f)(xy0))


# def mult(functions: ArrayLike, x: float) -> float:
#     results = [fn(x) for fn in functions]
#     return jnp.prod(jnp.array(results))

# u = sin(x)
# v = exp(x)
# F(x) = u * v
# du/dx = cos(x)
# dv/dx = exp(x)
# U(x) = [sin, exp]
# U'(x) = [cos, exp]
# F(U(x)) = F'(U(x)) * U'(x)
# F([sin, exp]) = F'([sin, exp]) * [cos, exp]
# F'(U(x)) = F'([sin, exp])
# F' = [F'(u), F'(v)]
# F' = [v, u]
# F' = [exp(x), sin(x)]
# [exp(x), sin(x)] * [cos(x), exp(x)] = exp(x) * cos(x) + sin(x) * exp(x)





import numpy as np

# Define the function
def f(x):
    return np.sin(x)

# Point where we want the derivative
x0 = np.pi / 4  # 45 degrees

# Step size
h = 1e-5

# Forward difference approximation
forward_diff = (f(x0 + h) - f(x0)) / h

# Backward difference approximation
backward_diff = (f(x0) - f(x0 - h)) / h

# Central difference approximation (more accurate, O(h^2))
central_diff = (f(x0 + h) - f(x0 - h)) / (2 * h)

# True derivative for comparison
true_value = np.cos(x0)

print(f"Point: {x0}")
print(f"Forward difference:  {forward_diff}")
print(f"Backward difference: {backward_diff}")
print(f"Central difference:  {central_diff}")
print(f"True derivative:     {true_value}")

p = 0
res = jnp.cos(p)/p
print(res, 1.0/0.0)