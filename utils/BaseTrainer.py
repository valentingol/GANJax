from functools import partial

import haiku as hk
from jax import grad, jit, numpy as jnp, random
import optax

class BaseTrainer(object):
    """
    Base class for all GAN trainers. The not implemented fonctions
    inside should be overwrite for each GAN implementation.
    """

    @hk.transform
    def gen_fwd(self, *args, **kwargs):
        """ (transformed) Forward pass of generator. """
        raise NotImplementedError

    @hk.transform
    def disc_fwd(self, *args, **kwargs):
        """ (transformed) Forward pass of generator. """
        raise NotImplementedError

    def input_func(self, *args, **kwargs):
        """ Input function (of the generator typically). """
        raise NotImplementedError

    def init_gen(self, *args, **kwargs):
        """ Initialize the generator parameters/states
        and its optimizer. """
        raise NotImplementedError

    def init_disc(self, *args, **kwargs):
        """ Initialize the discriminator parameters/states
        and its optimizer. """
        raise NotImplementedError

    def fwd_loss_gen(self, *args, **kwargs):
        """ Computes the loss of the generator
        over one batch. """
        raise NotImplementedError

    def fwd_loss_disc(self, *args, **kwargs):
        """ Computes the loss of the discriminator
        over one batch. """
        raise NotImplementedError

    @jit
    def train_gen(self, *args, **kwargs):
        """ (jit) Update the generator parameters and
        its optimizer over one batch. """
        raise NotImplementedError

    @jit
    def train_disc(self, *args, **kwargs):
        """ (jit) Update the discriminator parameters and
        its optimizer over one batch. """
        raise NotImplementedError

    def cycle_train(self, *args, **kwargs):
        """ Train the generator and the discriminator and update
        the means.
        """
        raise NotImplementedError

    def main(self, *args, **kwargs):
        """ Perform a complete training. """
        # return params_gen, state_gen, params_disc, state_disc, history
        raise NotImplementedError
