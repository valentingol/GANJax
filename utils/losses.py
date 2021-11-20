import jax
from jax import jit, numpy as jnp
import optax

@jit
def cross_entropy(logits, labels):
    return optax.sigmoid_binary_cross_entropy(logits, labels)

@jit
def wasserstein(logits, labels):
    # Wasserstein Loss without clipping
    # labels: [0, 1] -> [-1, 1]
    labels = 2.0 * labels - 1.0
    return labels * logits
