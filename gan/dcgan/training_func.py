from functools import partial

import haiku as hk
from jax import grad, jit, numpy as jnp, random
import optax

from gan.dcgan.modules import DCGenerator, DCDiscriminator
from utils.losses import cross_entropy as loss_fn

def input_func(key, batch_size, zdim):
    return random.normal(key, (batch_size, zdim))


@hk.transform_with_state
def gen_fwd(z, kwards_gen, is_training):
    generator = DCGenerator(**kwards_gen)
    X_fake = generator(z, is_training=is_training)
    return X_fake


@hk.transform_with_state
def disc_fwd(X, kwards_disc, is_training):
    discriminator = DCDiscriminator(**kwards_disc)
    y_pred = discriminator(X, is_training=is_training)
    return y_pred


## Initializations

def init_gen(key, config, z, kwards_gen):
    params_gen, state_gen = gen_fwd.init(key, z, kwards_gen,
                                         is_training=True)
    opt_gen = optax.adam(
        learning_rate=config.lr_gen,
        b1=config.beta1_gen,
        b2=config.beta1_gen
        )
    opt_state_gen = opt_gen.init(params_gen)
    return state_gen, opt_gen, opt_state_gen, params_gen


def init_disc(key, config, X, kwards_disc):
    params_disc, state_disc = disc_fwd.init(key, X, kwards_disc,
                                            is_training=True)
    opt_disc = optax.adam(
        learning_rate=config.lr_disc,
        b1=config.beta1_disc,
        b2=config.beta1_disc
        )
    opt_state_disc = opt_disc.init(params_disc)
    return state_disc, opt_disc, opt_state_disc, params_disc


## Forward pass + loss

def fwd_loss_gen(params_gen, params_disc, state_gen, state_disc, z,
                is_training, kwargs_gen, kwargs_disc):
    X_fake, state_gen = gen_fwd.apply(
        params_gen, state_gen, None, z, kwargs_gen, is_training=is_training)
    y_pred_fake, state_disc = disc_fwd.apply(
        params_disc, state_disc, None, X_fake, kwargs_disc,
        is_training=is_training
        )
    loss_gen = loss_fn(y_pred_fake, jnp.ones_like(y_pred_fake))
    loss_gen = jnp.mean(loss_gen)
    return loss_gen, (loss_gen, state_gen, state_disc)


def fwd_loss_disc(params_disc, params_gen, state_disc, state_gen, z, X_real,
                  is_training, kwargs_gen, kwargs_disc):
    X_fake, state_gen = gen_fwd.apply(
        params_gen, state_gen, None, z, kwargs_gen, is_training=is_training
        )
    y_pred_fake, state_disc = disc_fwd.apply(
        params_disc, state_disc, None, X_fake, kwargs_disc,
        is_training=is_training
        )
    y_pred_real, state_disc = disc_fwd.apply(
        params_disc, state_disc, None, X_real, kwargs_disc,
        is_training=is_training
        )

    # Smooth label (+/- 0.1)
    fake_loss = loss_fn(y_pred_fake, jnp.zeros_like(y_pred_fake) + 0.1)
    real_loss = loss_fn(y_pred_real, jnp.ones_like(y_pred_real) - 0.1)
    loss_disc = ((fake_loss + real_loss) / 2.0)
    loss_disc = jnp.mean(loss_disc)
    return loss_disc, (loss_disc, state_disc, state_gen)


# Training functions

@partial(jit, static_argnums=(4, 7, 8, 9))
def train_gen(params_gen, params_disc, state_gen, state_disc, opt_gen,
              opt_state_gen, z, is_training, kwargs_gen, kwargs_disc):
    grads, (loss_gen, state_gen, state_disc) = grad(fwd_loss_gen, has_aux=True)(
        params_gen, params_disc, state_gen, state_disc, z,
        is_training, kwargs_gen, kwargs_disc
        )
    updates, opt_state_gen = opt_gen.update(
        grads, opt_state_gen, params_gen
        )
    params_gen = optax.apply_updates(params_gen, updates)
    return params_gen, state_gen, state_disc, opt_state_gen, loss_gen


@partial(jit, static_argnums=(4, 8, 9, 10))
def train_disc(params_disc, params_gen, state_disc, state_gen, opt_disc,
              opt_state_disc, z, X_real, is_training, kwargs_gen, kwargs_disc):
    grads, (loss_disc, state_disc, state_gen) = grad(fwd_loss_disc, has_aux=True)(
        params_disc, params_gen, state_disc, state_gen, z, X_real,
        is_training, kwargs_gen, kwargs_disc
        )
    updates, opt_state_disc = opt_disc.update(
        grads, opt_state_disc, params_disc
        )
    params_disc = optax.apply_updates(params_disc, updates)
    return params_disc, state_disc, state_gen, opt_state_disc, loss_disc
