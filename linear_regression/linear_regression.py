import matplotlib.pyplot as plt
# import numpy as np
import time

num_points = 200
spread = 2.5 # Standard deviation of the normal distribution

# Simple LCG (Linear Congruential Generator)
seed = 123456789

def rand():
    global seed
    # Parameters from Numerical Recipes
    a = 1664525
    c = 1013904223
    m = 2**32
    seed = (a * seed + c) % m
    return seed / m  # uniform in [0,1)

def rand_range(low, high):
    return low + (high - low) * rand()

def randn():
    # Box-Muller transform
    u1 = rand()
    u2 = rand()
    r = (-2 * (u1 if u1 > 0 else 1e-10)) ** 0.5   # avoid log(0)
    import math
    z0 = ( -2 * math.log(u1) ) ** 0.5 * math.cos(2 * math.pi * u2)
    return z0

def randn_mu_sigma(mu=0, sigma=1):
    return mu + sigma * rand()

x_values = [rand_range(-10, 10) for i in range(num_points)]
y_values = [x_values[i] + randn_mu_sigma(0, spread) for i in range(num_points)]

start = time.perf_counter()
x_sum = sum(x_values)
y_sum = sum(y_values)
sum_of_x_squared = sum([x_values[i] * x_values[i] for i in range(num_points)])
sum_of_x_times_y = sum([x_values[i] * y_values[i] for i in range(num_points)])

b = (y_sum * sum_of_x_squared - x_sum * sum_of_x_times_y) / ((num_points * sum_of_x_squared) - (x_sum * x_sum))

m = ((num_points * sum_of_x_times_y) - (x_sum * y_sum)) / ((num_points * sum_of_x_squared) - (x_sum * x_sum))

y_line = [m * x_values[i] + b for i in range(len(x_values))]
finish = time.perf_counter()
print(f"Linear Regression Formula {finish - start:0.4f} seconds")

start = time.perf_counter()
[m2, b2] = [rand_range(-10, 10) for i in range(2)]

for _ in range(1000):
  y_estimates = [(m2 * x_values[i] + b2) for i in range(num_points)]
  mse = sum([(y_values[i] - y_estimates[i]) ** 2 for i in range(num_points)])
  alpha = 0.01

  changeInM = (-2/num_points) * sum([x_values[i] * (y_values[i] - y_estimates[i]) for i in range(num_points)])
  changeInB = (-2/num_points) * sum([y_values[i] - y_estimates[i] for i in range(num_points)])

  m2 = m2 - (alpha * changeInM)
  b2 = b2 - (alpha * changeInB)

y2_line = [m2 * x_values[i] + b2 for i in range(num_points)]
finish = time.perf_counter()
print(f"Manual Gradient Descent {finish - start:0.4f} seconds")

plt.subplot(1, 3, 1)

plt.plot(x_values, y_values, 'ro')
plt.plot(x_values, y_line, label='Linear Regression Formula')
plt.plot(x_values, y2_line, label='Manual Gradient Descent')
# plt.show()
plt.legend()



import numpy as np

x_values_np = np.random.uniform(-10, 10, num_points)
y_values_np = x_values_np + np.random.uniform(0, spread, num_points)

start = time.perf_counter()
[m_np, b_np] = np.random.uniform(0, 1, size=2)

for _ in range(1000):
  y_estimates_np = m_np * x_values_np + b_np
  mse = (y_values_np - y_estimates_np) ** 2
  alpha = 0.001

  changeInM = np.sum((-2 / num_points) * (x_values_np * (y_values_np - y_estimates_np)))
  changeInB = np.sum((-2 / num_points) * (y_values_np - y_estimates_np))

  m_np = m_np - (alpha * changeInM)
  b_np = b_np - (alpha * changeInB)

y_np_line = m_np * x_values_np + b_np
finish = time.perf_counter()
print(f"Numpy Gradient Descent {finish - start:0.4f} seconds")

start = time.perf_counter()
m_poly, b_poly = np.polyfit(x_values_np, y_values_np, 1)
y_np_poly_line = m_poly * x_values_np + b_poly
finish = time.perf_counter()
print(f"Numpy Poly Fit {finish - start:0.4f} seconds")

plt.subplot(1, 3, 2)

plt.plot(x_values_np, y_values_np, 'ro')
plt.plot(x_values_np, y_np_line, label='Numpy Gradient Descent')
plt.plot(x_values_np, y_np_poly_line, label='Numpy Poly Fit')
plt.legend()

import jax
import jax.numpy as jnp


# key for randomness
key = jax.random.PRNGKey(0)

# sample data
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
x_values_jnp = jax.random.uniform(subkey1, shape=(num_points,), minval=-10, maxval=10)
y_values_jnp = x_values_jnp + jax.random.uniform(subkey2, shape=(num_points,), minval=0, maxval=spread)

start = time.perf_counter()
m_jnp, b_jnp = jax.random.uniform(subkey3, shape=(2,), minval=0.0, maxval=1.0)

# gradient descent loop
alpha = 0.001
for _ in range(1000):
    y_estimates_jnp = m_jnp * x_values_jnp + b_jnp
    error_jnp = y_values_jnp - y_estimates_jnp

    changeInM_jnp = jnp.sum((-2 / num_points) * (x_values_jnp * error_jnp))
    changeInB_jnp = jnp.sum((-2 / num_points) * error_jnp)

    m_jnp = m_jnp - alpha * changeInM_jnp
    b_jnp = b_jnp - alpha * changeInB_jnp

y_line_jnp = m_jnp * x_values_jnp + b_jnp
finish = time.perf_counter()
print(f"Jax Gradient Descent {finish - start:0.4f} seconds")

start = time.perf_counter()
ones_jnp = jnp.ones((x_values_jnp.shape[0], 1))
X_jnp = jnp.hstack((ones_jnp, x_values_jnp.reshape(-1, 1)))  # shape (n, 2)

# solve least squares
beta_jnp, residuals, rank, s = jnp.linalg.lstsq(X_jnp, y_values_jnp, rcond=None)

# predictions
y_lstsq_jnp = X_jnp @ beta_jnp
finish = time.perf_counter()
print(f"Jax lstsq {finish - start:0.4f} seconds")

plt.subplot(1, 3, 3)
plt.plot(x_values_jnp, y_values_jnp, 'ro')
plt.plot(x_values_jnp, y_line_jnp, label='Jax Gradient Descent')
plt.plot(x_values_jnp, y_lstsq_jnp, label='Jax lstsq')

plt.legend()
plt.show()