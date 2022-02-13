from functools import partial
from time import time

import haiku as hk
from jax import grad, jit, numpy as jnp, random, value_and_grad as vgrad
import matplotlib.pyplot as plt
import optax
import tensorflow as tf

from gan.progan.modules import ProGenerator, ProDiscriminator
from utils.BaseTrainer import BaseTrainer
from utils.losses import wasserstein
from utils.means import Mean
from utils.plot_img import plot_curves, plot_tensor_images
from utils.save_and_load import save_gan

class ProGANTrainer(BaseTrainer):
    def __init__(self, loss_fn=None):
        """ Initialize the loss function (to make it easier
        to change further). By default Wasserstein loss. """
        super().__init__()
        loss_fn = wasserstein if loss_fn is None else loss_fn
        self.loss_fn = loss_fn

    @hk.without_apply_rng
    @hk.transform
    def gen_fwd(self, z, alpha, kwargs_gen):
        """ (transformed) Forward pass of generator. """
        generator = ProGenerator(**kwargs_gen)
        X_fake = generator(z, alpha=alpha)
        return X_fake

    @hk.without_apply_rng
    @hk.transform
    def disc_fwd(self, X, alpha, kwargs_disc):
        """ (transformed) Discriminator pass of generator. """
        discriminator = ProDiscriminator(**kwargs_disc)
        y_pred = discriminator(X, alpha=alpha)
        return y_pred

    def gen_fwd_init(self, key, *args, **kwargs):
        """ Simplify the use of gen_fwd.init """
        return self.gen_fwd.init(key, self, *args, **kwargs)

    def gen_fwd_apply(self, parameter, *args, **kwargs):
        """ Simplify the use of gen_fwd.apply """
        return self.gen_fwd.apply(parameter, self, *args, **kwargs)

    def disc_fwd_init(self, key, *args, **kwargs):
        """ Simplify the use of disc_fwd.init """
        return self.disc_fwd.init(key, self, *args, **kwargs)

    def disc_fwd_apply(self, parameter, *args, **kwargs):
        """ Simplify the use of disc_fwd.apply """
        return self.disc_fwd.apply(parameter, self, *args, **kwargs)

    def input_func(self, key, batch_size, zdim):
        """ Input of generator ( = "noise" in classic GAN). """
        return random.normal(key, (batch_size, zdim))

    def init_gen(self, key, config, z, kwargs_gen, old_params=None):
        """ Initialize the generator parameters/states
        and its optimizer."""
        params_gen = self.gen_fwd_init(key, z, alpha=1,
                                                  kwargs_gen=kwargs_gen)
        # Merge the old params with the new ones (by the names of layers)
        if old_params is not None:
            old_params = hk.data_structures.to_mutable_dict(old_params)
            conv11 = old_params[f'{config.name_gen}/~/new_conv11']
            old_params[f'{config.name_gen}/~/prev_conv11'] = conv11
            del old_params[f'{config.name_gen}/~/new_conv11']
            old_params = hk.data_structures.to_haiku_dict(old_params)
            params_gen = hk.data_structures.merge(params_gen, old_params)


        opt_gen = optax.adam(
            learning_rate=config.lr_gen,
            b1=config.beta1_gen,
            b2=config.beta1_gen
            )
        opt_state_gen = opt_gen.init(params_gen)
        return opt_gen, opt_state_gen, params_gen

    def init_disc(self, key, config, X, kwargs_disc, old_params=None):
        """ Initialize the discriminator parameters/states
        and its optimizer."""
        params_disc = self.disc_fwd_init(key, X, alpha=1,
                                         kwargs_disc=kwargs_disc)
        # Merge the old params with the new ones (by the names of layers)
        if old_params is not None:
            old_params = hk.data_structures.to_mutable_dict(old_params)
            conv11 = old_params[f'{config.name_disc}/~/new_conv11']
            old_params[f'{config.name_disc}/~/prev_conv11'] = conv11
            del old_params[f'{config.name_disc}/~/new_conv11']
            old_params = hk.data_structures.to_haiku_dict(old_params)
            params_disc = hk.data_structures.merge(params_disc, old_params)

        opt_disc = optax.adam(
            learning_rate=config.lr_disc,
            b1=config.beta1_disc,
            b2=config.beta1_disc
            )
        opt_state_disc = opt_disc.init(params_disc)
        return opt_disc, opt_state_disc, params_disc

    def fwd_loss_gen(self, params_gen, params_disc, z, alpha, kwargs_gen,
                     kwargs_disc):
        """ Computes the loss of the generator over one batch. """
        X_fake = self.gen_fwd_apply(params_gen, z, alpha, kwargs_gen)
        y_pred_fake = self.disc_fwd_apply(params_disc, X_fake, alpha,
                                          kwargs_disc)

        loss_gen = self.loss_fn(y_pred_fake, jnp.ones_like(y_pred_fake))
        loss_gen = jnp.mean(loss_gen)
        return loss_gen

    def fwd_loss_disc(self, params_disc, key, params_gen, z, X_real, alpha,
                      lambd, kwargs_gen, kwargs_disc):
        """ Computes the loss of the discriminator over one batch. """
        X_fake = self.gen_fwd_apply(params_gen, z, alpha, kwargs_gen)
        unif_rd = random.uniform(key, shape=X_real.shape)
        # X_fake = unif_rd * X_real + (1 - unif_rd) * X_fake
        y_pred_fake = self.disc_fwd_apply(params_disc, X_fake, alpha,
                                         kwargs_disc)
        y_pred_real = self.disc_fwd_apply(params_disc, X_real, alpha,
                                          kwargs_disc)

        fake_loss = self.loss_fn(y_pred_fake, jnp.zeros_like(y_pred_fake))
        real_loss = self.loss_fn(y_pred_real, jnp.ones_like(y_pred_real))

        # Apply gradient penalty
        def mean_fwd_data(params_disc, X_fake, alpha, kwargs_disc):
            return self.disc_fwd_apply(params_disc, X_fake, alpha,
                                       kwargs_disc).mean()
        grads_data = grad(mean_fwd_data, argnums=1)(
            params_disc, X_fake, alpha, kwargs_disc
        )
        grad_penalty = (optax.global_norm(grads_data) - 1) ** 2
        loss_disc = fake_loss + real_loss + grad_penalty * lambd
        loss_disc = jnp.mean(loss_disc)
        return loss_disc

    @partial(jit, static_argnums=(0, 3, 7, 8))
    def train_gen(self, params_gen, params_disc, opt_gen, opt_state_gen, z,
                  alpha, kwargs_gen, kwargs_disc):
        """ (jit) Update the generator parameters and
        its optimizer over one batch. """
        loss_gen, grads = vgrad(self.fwd_loss_gen)(
            params_gen, params_disc, z, alpha, kwargs_gen, kwargs_disc
            )
        updates, opt_state_gen = opt_gen.update(
            grads, opt_state_gen, params_gen
            )
        params_gen = optax.apply_updates(params_gen, updates)
        return params_gen, opt_state_gen, loss_gen

    @partial(jit, static_argnums=(0, 4, 10, 11))
    def train_disc(self, key, params_disc, params_gen, opt_disc,
                   opt_state_disc, z, X_real, alpha, lambd, kwargs_gen,
                   kwargs_disc):
        """ (jit) Update the discriminator parameters and
        its optimizer over one batch. """
        loss_disc, grads = vgrad(self.fwd_loss_disc)(
            params_disc, key, params_gen, z, X_real, alpha, lambd, kwargs_gen,
            kwargs_disc
            )
        updates, opt_state_disc = opt_disc.update(
            grads, opt_state_disc, params_disc
            )
        params_disc = optax.apply_updates(params_disc, updates)
        return params_disc, opt_state_disc, loss_disc

    def cycle_train(self, X_real, resolution, key, alpha, lambd, params_gen,
                    params_disc, opt_gen, opt_state_gen, opt_disc,
                    opt_state_disc, kwargs_gen, kwargs_disc, mean_loss_gen,
                    mean_loss_disc, config):
        """ Train the generator and the discriminator and update
        the means (mean_loss_gen and mean_loss_disc).
        """
        # Downsample the images
        X_real = tf.image.resize(X_real, (2 * resolution, 2 * resolution))

        X_real = jnp.array(X_real)
        batch_size = X_real.shape[0]  # (can change at the end of an epoch)
        n_critic = config.cylce_train_disc
        key_gen, *keys_disc = random.split(key, 1 + 2 * n_critic)

        # Train generator
        z = self.input_func(key_gen, batch_size, config.zdim)
        (params_gen, opt_state_gen, loss_gen) = self.train_gen(
                params_gen, params_disc, opt_gen, opt_state_gen, z, alpha,
                kwargs_gen, kwargs_disc
                )
        mean_loss_gen(loss_gen)

        # Train discriminator (cylce_train_disc times)
        for k in range(config.cylce_train_disc):
            z = self.input_func(keys_disc[n_critic + k], batch_size,
                                config.zdim)
            (params_disc, opt_state_disc, loss_disc) = self.train_disc(
                keys_disc[k], params_disc, params_gen, opt_disc,
                opt_state_disc, z, X_real, alpha, lambd, kwargs_gen,
                kwargs_disc
                )
            mean_loss_disc(loss_disc)

        return (params_gen, params_disc, opt_state_gen, opt_state_disc,
                mean_loss_gen, mean_loss_disc)


    def update_block_config(self, kwargs_gen, kwargs_disc):
        """ Add 1 in n_blocks in kwargs_gen and kwargs_disc
        (and change the hash value). """
        kwargs_gen_mutable = hk.data_structures.to_mutable_dict(kwargs_gen)
        kwargs_disc_mutable = hk.data_structures.to_mutable_dict(kwargs_disc)
        if 'n_blocks' not in kwargs_gen_mutable:
            kwargs_gen_mutable['n_blocks'] = 0
            kwargs_disc_mutable['n_blocks'] = 0
        kwargs_gen_mutable['n_blocks'] += 1
        kwargs_disc_mutable['n_blocks'] += 1
        kwargs_gen = hk.data_structures.to_immutable_dict(kwargs_gen_mutable)
        kwargs_disc = hk.data_structures.to_immutable_dict(kwargs_disc_mutable)
        return kwargs_gen, kwargs_disc

    def init_models(self, dataset, resolution, config, key, kwargs_gen,
                    kwargs_disc, old_params_gen, old_params_disc):
        for X_real in dataset.take(1):
            X_real = tf.image.resize(X_real, (2 * resolution, 2 * resolution))
            X_real = jnp.array(X_real)
            batch_size = X_real.shape[0]
        z = self.input_func(key, batch_size, config.zdim)
        opt_gen, opt_state_gen, params_gen = self.init_gen(
            key, config, z, kwargs_gen, old_params=old_params_gen
            )
        opt_disc, opt_state_disc, params_disc = self.init_disc(
            key, config, X_real, kwargs_disc, old_params=old_params_disc
            )
        return (opt_gen, opt_state_gen, params_gen, opt_disc, opt_state_disc,
                params_disc)

    def main(self, dataset, key, config, max_time=None,
         display_step=500, num_images=(10, 10), plot=True,
         save_step=None, model_path=None):
        # max_time = inf iif max_time in {0, None}:
        max_time = max_time or jnp.inf

        kwargs_gen, kwargs_disc = config.get_models_kwargs()

        kwargs_gen, kwargs_disc = self.update_block_config(kwargs_gen,
                                                           kwargs_disc)
        resolution = config.first_resolution
        # Initialize generator and discriminator
        (opt_gen, opt_state_gen, params_gen, opt_disc, opt_state_disc,
                params_disc) = self.init_models(dataset, resolution, config,
                                                key, kwargs_gen, kwargs_disc,
                                                None, None)
        print('Initialization succeeded.')

        mean_loss_gen, mean_loss_disc = Mean(), Mean()
        len_ds = int(dataset.__len__())
        lambd = config.lambda_gp
        itr = 0
        start = time()
        total_epochs = sum(config.growing_epochs) + sum(config.fixed_epochs)
        for step in range(config.n_steps):
            print(f'Iteration {step}, resolution '
                  f'{resolution}x{resolution}')

            # Reinitialize the params using previous ones
            # and update the architectures
            if step > 0:
                kwargs_gen, kwargs_disc = self.update_block_config(kwargs_gen,
                                                                   kwargs_disc)
                (opt_gen, opt_state_gen, params_gen, opt_disc, opt_state_disc,
                params_disc) = self.init_models(dataset, resolution, config,
                                                key, kwargs_gen, kwargs_disc,
                                                params_gen, params_disc)

            growing_epochs = config.growing_epochs[step]
            fixed_epochs = config.fixed_epochs[step]
            n_epoch = growing_epochs + fixed_epochs
            print('-> Growing phase')
            for ep in range(n_epoch):
                if ep == growing_epochs:
                    print('-> Fixed phase')
                print(f' Epoch {ep + 1}/{n_epoch}')
                alpha = ep / growing_epochs if ep < growing_epochs else 1.0

                for i_batch, X_real in enumerate(dataset):
                    t = time() - start
                    eta = t / (itr + 1) * (total_epochs * len_ds - itr - 1)
                    t_h, t_m, t_s = t // 3600, (t % 3600) // 60, t % 60
                    eta_h, eta_m, eta_s = (eta // 3600, (eta % 3600) // 60,
                                           eta % 60)
                    print(f'   batch {i_batch + 1}/{len_ds} - '
                        f'gen loss:{mean_loss_gen: .5f} - '
                        f'disc loss:{mean_loss_disc: .5f} - '
                        f'time: {int(t_h)}h {int(t_m)}min {int(t_s)}sec - '
                        f'eta: {int(eta_h)}h {int(eta_m)}min {int(eta_s)}sec'
                        '    ',
                        end='\r')
                    key, subkey = random.split(key, 2)

                    # Training both generator and discriminator
                    (params_gen, params_disc, opt_state_gen, opt_state_disc,
                     mean_loss_gen, mean_loss_disc) = self.cycle_train(
                        X_real, resolution, key, alpha, lambd, params_gen,
                        params_disc, opt_gen, opt_state_gen, opt_disc,
                        opt_state_disc, kwargs_gen, kwargs_disc, mean_loss_gen,
                        mean_loss_disc, config
                        )

                    itr += 1
                    # Plot images
                    if plot and itr % display_step == 0 and itr > 0:
                        z = self.input_func(subkey,
                                            num_images[0] * num_images[1],
                                            config.zdim)
                        X_fake = self.gen_fwd_apply(params_gen, z, alpha,
                                                    kwargs_gen)
                        plot_tensor_images(X_fake, num_images=num_images)
                        plt.title(f'Step {step} - Epoch {ep + 1}/{n_epoch} - '
                                  f'iteration {itr}',
                                fontsize=15)
                        plt.show(block=False)
                        plt.pause(0.1)

                    if (save_step is not None and model_path is not None
                        and itr % save_step == 0 and itr > 0):
                        save_gan(params_gen, None, params_disc, None,
                                config, model_path + f'/itr{itr}',
                                verbose=False, with_states=False)

                    if time() - start >= max_time:
                        print('\n\nTraining time limit reached.')
                        break

                mean_loss_gen.reset(), mean_loss_disc.reset()
                print()

                if time() - start >= max_time:
                    break

            resolution *= 2

            if time() - start >= max_time:
                    break

        history = {'loss_gen': mean_loss_gen.history,
                   'loss_disc': mean_loss_disc.history}
        if plot:
            z = self.input_func(key, num_images[0] * num_images[1],
                                config.zdim)
            X_fake = self.gen_fwd_apply(params_gen, z, 1.0, kwargs_gen)
            plt.figure(figsize=(40, 20))
            plt.subplot(1, 2, 1)
            plot_tensor_images(X_fake, num_images=num_images)
            plt.title('Final images generation')
            plt.subplot(1, 2, 2)
            plt.title('Loss curves')
            plot_curves(history, total_epochs)
            plt.show()
        return params_gen, {}, params_disc, {}, history
