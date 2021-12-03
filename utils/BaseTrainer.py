from functools import partial

import haiku as hk
from jax import grad, jit, numpy as jnp, random
import optax

from gan.dcgan.modules import DCGenerator, DCDiscriminator

class BaseTrainer(object):
    """
    Base class for all GAN trainers. The fonctions inside
    should be overwrite for each GAN implementation.
    By default, BaseTrainer is based on DCGAN training
    pipeline with label smoothing.
    """
    def __init__(self, loss_fn=None):
        """ Initialize the loss function (to make it
        easier to change further). """
        self.loss_fn = loss_fn

    def input_func(self, key, batch_size, zdim):
        """ Input of generator ( = "noise" in classic GAN). """
        return random.normal(key, (batch_size, zdim))

    @hk.transform_with_state
    def gen_fwd(self, z, kwargs_gen, is_training):
        """ (transformed) Forward pass of generator. """
        generator = DCGenerator(**kwargs_gen)
        X_fake = generator(z, is_training=is_training)
        return X_fake

    @hk.transform_with_state
    def disc_fwd(self, X, kwargs_disc, is_training):
        """ (transformed) Discriminator pass of generator. """
        discriminator = DCDiscriminator(**kwargs_disc)
        y_pred = discriminator(X, is_training=is_training)
        return y_pred

    def gen_fwd_init(self, key, *args, **kwargs):
        """ Simplify the use of gen_fwd.init """
        return self.gen_fwd.init(key, self, *args, **kwargs)

    def gen_fwd_apply(self, parameter, state, key, *args, **kwargs):
        """ Simplify the use of gen_fwd.apply """
        return self.gen_fwd.apply(parameter, state, key, self,
                                  *args, **kwargs)

    def disc_fwd_init(self, key, *args, **kwargs):
        """ Simplify the use of disc_fwd.init """
        return self.disc_fwd.init(key, self, *args, **kwargs)

    def disc_fwd_apply(self, parameter, state, key, *args, **kwargs):
        """ Simplify the use of disc_fwd.apply """
        return self.disc_fwd.apply(parameter, state, key, self,
                                   *args, **kwargs)

    def init_gen(self, key, config, z, kwargs_gen):
        """ Initialize the generator parameters/states
        and its optimizer."""
        params_gen, state_gen = self.gen_fwd_init(key, z, kwargs_gen,
                                                  is_training=True)
        opt_gen = optax.adam(
            learning_rate=config.lr_gen,
            b1=config.beta1_gen,
            b2=config.beta1_gen
            )
        opt_state_gen = opt_gen.init(params_gen)
        return state_gen, opt_gen, opt_state_gen, params_gen

    def init_disc(self, key, config, X, kwargs_disc):
        """ Initialize the discriminator parameters/states
        and its optimizer."""
        params_disc, state_disc = self.disc_fwd_init(key, X, kwargs_disc,
                                                     is_training=True)
        opt_disc = optax.adam(
            learning_rate=config.lr_disc,
            b1=config.beta1_disc,
            b2=config.beta1_disc
            )
        opt_state_disc = opt_disc.init(params_disc)
        return state_disc, opt_disc, opt_state_disc, params_disc

    def fwd_loss_gen(self, params_gen, params_disc, state_gen, state_disc, z,
                     is_training, kwargs_gen, kwargs_disc):
        """ Computes the loss of the generator over one batch. """
        X_fake, state_gen = self.gen_fwd_apply(
            params_gen, state_gen, None, z, kwargs_gen,
            is_training=is_training)
        y_pred_fake, state_disc = self.disc_fwd_apply(
            params_disc, state_disc, None, X_fake, kwargs_disc,
            is_training=is_training
            )
        loss_gen = self.loss_fn(y_pred_fake, jnp.ones_like(y_pred_fake))
        loss_gen = jnp.mean(loss_gen)
        return loss_gen, (loss_gen, state_gen, state_disc)

    def fwd_loss_disc(self, params_disc, params_gen, state_disc, state_gen,
                      z, X_real, is_training, kwargs_gen, kwargs_disc):
        """ Computes the loss of the discriminator over one batch. """
        X_fake, state_gen = self.gen_fwd_apply(
            params_gen, state_gen, None, z, kwargs_gen, is_training=is_training
            )
        y_pred_fake, state_disc = self.disc_fwd_apply(
            params_disc, state_disc, None, X_fake, kwargs_disc,
            is_training=is_training
            )
        y_pred_real, state_disc = self.disc_fwd_apply(
            params_disc, state_disc, None, X_real, kwargs_disc,
            is_training=is_training
            )

        # Smooth label (+/- 0.1)
        fake_loss = self.loss_fn(y_pred_fake,
                                 jnp.zeros_like(y_pred_fake) + 0.1)
        real_loss = self.loss_fn(y_pred_real,
                                 jnp.ones_like(y_pred_real) - 0.1)
        loss_disc = ((fake_loss + real_loss) / 2.0)
        loss_disc = jnp.mean(loss_disc)
        return loss_disc, (loss_disc, state_disc, state_gen)

    @partial(jit, static_argnums=(0, 5, 8, 9, 10))
    def train_gen(self, params_gen, params_disc, state_gen, state_disc,
                  opt_gen, opt_state_gen, z, is_training, kwargs_gen,
                  kwargs_disc):
        """ (jit) Update the generator parameters/states and
        its optimizer over one batch. """
        grads, (loss_gen, state_gen, state_disc) = grad(
            self.fwd_loss_gen, has_aux=True)(
            params_gen, params_disc, state_gen, state_disc, z,
            is_training, kwargs_gen, kwargs_disc
            )
        updates, opt_state_gen = opt_gen.update(
            grads, opt_state_gen, params_gen
            )
        params_gen = optax.apply_updates(params_gen, updates)
        return params_gen, state_gen, state_disc, opt_state_gen, loss_gen

    @partial(jit, static_argnums=(0, 5, 9, 10, 11))
    def train_disc(self, params_disc, params_gen, state_disc, state_gen,
                   opt_disc, opt_state_disc, z, X_real, is_training,
                   kwargs_gen, kwargs_disc):
        """ (jit) Update the discriminator parameters/states and
        its optimizer over one batch. """
        grads, (loss_disc, state_disc, state_gen) = grad(
            self.fwd_loss_disc, has_aux=True)(
            params_disc, params_gen, state_disc, state_gen, z, X_real,
            is_training, kwargs_gen, kwargs_disc
            )
        updates, opt_state_disc = opt_disc.update(
            grads, opt_state_disc, params_disc
            )
        params_disc = optax.apply_updates(params_disc, updates)
        return params_disc, state_disc, state_gen, opt_state_disc, loss_disc

    def cycle_train(self, X_real, key, params_gen, params_disc, state_gen,
                    state_disc, opt_gen, opt_state_gen, opt_disc,
                    opt_state_disc, kwargs_gen, kwargs_disc, mean_loss_gen,
                    mean_loss_disc, config):
        """ Train the generator and the discriminator and update
        the means (mean_loss_gen and mean_loss_disc).
        """
        X_real = jnp.array(X_real)
        batch_size = X_real.shape[0] # (can change at the end of epoch)
        key, *keys = random.split(key, 1 + config.cylce_train_disc)

        # Train generator
        z = self.input_func(key, batch_size, config.zdim)
        (params_gen, state_gen, state_disc, opt_state_gen,
            loss_gen) = self.train_gen(
                params_gen, params_disc, state_gen, state_disc, opt_gen,
                opt_state_gen, z, True, kwargs_gen, kwargs_disc
                )
        mean_loss_gen(loss_gen)

        # Train discriminator (cylce_train_disc times)
        for k in range(config.cylce_train_disc):
            z = self.input_func(keys[k], batch_size, config.zdim)

            (params_disc, state_disc, state_gen, opt_state_disc,
                loss_disc) = self.train_disc(
                params_disc, params_gen, state_disc, state_gen, opt_disc,
                opt_state_disc, z, X_real, True, kwargs_gen, kwargs_disc
                )
            mean_loss_disc(loss_disc)

        return (params_gen, params_disc, state_gen, state_disc,
                opt_state_gen, opt_state_disc, mean_loss_gen, mean_loss_disc)
