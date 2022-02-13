from jax import jit, numpy as jnp, random


def upsample_2d(x, scale):
    x = jnp.repeat(x, scale, axis=1)
    x = jnp.repeat(x, scale, axis=2)
    return x


@jit
def add_batch_std(x):
    shape = x.shape
    mean = jnp.mean(x, axis=0, keepdims=True)
    # Variance over the batch:
    var = jnp.mean(jnp.square(x - mean), axis=0, keepdims=True) + 1e-8
    # Mean of std across the channels and pixels:
    mean_std = jnp.mean(jnp.sqrt(var))
    mean_std = jnp.tile(mean_std, (shape[0], shape[1], shape[2], 1))
    x = jnp.concatenate((x, mean_std), axis=-1)
    return x


@jit
def pixel_norm(x):
    x_2_mean = jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-8
    norm = jnp.sqrt(x_2_mean)
    x = x / norm
    return x


if __name__ == '__main__':
    key = random.PRNGKey(0)
    X = random.normal(key, (32, 10, 10, 3))
