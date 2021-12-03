from jax import numpy as jnp

class Mean(object):
    """ Compute dynamic mean of given inputs. """

    def __init__(self):
        self.val = 0.0
        self.count = 0
        # Keep the history of **given inputs**
        self.history = []

    def reset(self):
        self.val = 0.0
        self.count = 0

    def reset_history(self):
        self.history = []

    def __call__(self, val):
        if isinstance(val, jnp.ndarray):
            val = val.item()
        # Keep the history of **given inputs**
        self.history.append(val)
        self.val = (self.val * self.count + val) / (self.count + 1)
        self.count += 1
        return self.val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return repr(self.val)

    def __format__(self, *args, **kwargs):
        return self.val.__format__(*args, **kwargs)