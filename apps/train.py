from time import time

from jax import numpy as jnp, random
import jaxlib
import matplotlib.pyplot as plt

from gan.dcgan import DCGAN as GAN
from utils.data import load_images_celeba_64 as load_images
from utils.plot_img import plot_curves, plot_tensor_images
from utils.save_and_load import save_gan

Config = GAN().Config
init_gen = GAN().init_gen
init_disc = GAN().init_disc
train_gen = GAN().train_gen
train_disc = GAN().train_disc
input_func = GAN().input_func
gen_fwd_apply = GAN().gen_fwd_apply


class Mean(object):
    """ Compute dynamic mean of given inputs. """

    def __init__(self, init_val=0.0):
        self.init_val = init_val
        self.val = init_val
        self.count = 0
        # Keep the history of **given inputs**
        self.history = []

    def reset(self):
        self.val = self.init_val
        self.count = 0
        self.history = []

    def __call__(self, val):
        if isinstance(val, jaxlib.xla_extension.DeviceArray):
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


# Global training function

def train_loop(dataset, key, config, n_epochs=60, display_step=500,
               num_images=(10, 10), plot=True, save_step=None,
               model_path=None):
    # Initialize generator and discriminator
    kwargs_gen, kwargs_disc = config.get_models_kwargs()
    X_real = jnp.array(dataset.take(1).as_numpy_iterator().next())
    batch_size = X_real.shape[0]

    z = input_func(key, batch_size, config.zdim)
    state_gen, opt_gen, opt_state_gen, params_gen = init_gen(
        key, config, z, kwargs_gen
        )

    state_disc, opt_disc, opt_state_disc, params_disc = init_disc(
        key, config, X_real, kwargs_disc
        )

    print('Initialization succeeded.')
    mean_loss_gen, mean_loss_disc = Mean(), Mean()
    history_loss_gen, history_loss_disc = [], []
    len_ds = int(dataset.__len__())
    itr = 0
    start = time()
    for ep in range(n_epochs):
        print(f'Epoch {ep + 1}/{n_epochs}')
        for i_batch, X_real in enumerate(dataset):
            t = time() - start
            eta = t / (itr + 1) * (n_epochs * len_ds - itr - 1)
            t_h, t_m, t_s = t // 3600, (t % 3600) // 60, t % 60
            eta_h, eta_m, eta_s = eta // 3600, (eta % 3600) // 60, eta % 60
            print(f'  batch {i_batch + 1}/{len_ds} - '
                  f'gen loss:{mean_loss_gen: .5f} - '
                  f'disc loss:{mean_loss_disc: .5f} - '
                  f'time: {int(t_h)}h {int(t_m)}min {int(t_s)}sec - '
                  f'eta: {int(eta_h)}h {int(eta_m)}min {int(eta_s)}sec    ',
                  end='\r')
            X_real = jnp.array(X_real)
            batch_size = X_real.shape[0]
            key, *keys = random.split(key, 1 + config.cylce_train_disc)

            z = input_func(key, batch_size, config.zdim)

            (params_gen, state_gen, state_disc, opt_state_gen,
             loss_gen) = train_gen(
                    params_gen, params_disc, state_gen, state_disc, opt_gen,
                    opt_state_gen, z, True, kwargs_gen, kwargs_disc
                    )
            mean_loss_gen(loss_gen)

            for k in range(config.cylce_train_disc):
                z = input_func(keys[k], batch_size, config.zdim)

                (params_disc, state_disc, state_gen, opt_state_disc,
                 loss_disc) = train_disc(
                    params_disc, params_gen, state_disc, state_gen, opt_disc,
                    opt_state_disc, z, X_real, True, kwargs_gen, kwargs_disc
                    )
                mean_loss_disc(loss_disc)
            itr += 1

            if plot and itr % display_step == 0 and itr > 0:
                z = input_func(key, num_images[0]*num_images[1], config.zdim)
                X_fake, state_gen = gen_fwd_apply(
                    params_gen, state_gen, None, z, kwargs_gen,
                    is_training=False
                    )
                plot_tensor_images(X_fake, num_images=num_images)
                plt.title(f'Epoch {ep + 1}/{n_epochs} - iteration {itr}',
                        fontsize=15)
                plt.show(block=False)
                plt.pause(0.1)

            if (save_step is not None and model_path is not None
                and itr % save_step == 0 and itr > 0):
                save_gan(params_gen, state_gen, params_disc, state_disc, config,
                         model_path + f'/itr{itr}', verbose=False)

        history_loss_gen.extend(mean_loss_gen.history)
        history_loss_disc.extend(mean_loss_disc.history)
        mean_loss_gen.reset(), mean_loss_disc.reset()
        print()

    history = {'loss_gen': history_loss_gen, 'loss_disc': history_loss_disc}
    if plot:
        z = input_func(key, num_images[0]*num_images[1], config.zdim)
        X_fake, state_gen = gen_fwd_apply(
            params_gen, state_gen, None, z, kwargs_gen, is_training=False,
            )
        plt.figure(figsize=(40, 20))
        plt.subplot(1, 2, 1)
        plot_tensor_images(X_fake, num_images=num_images)
        plt.title('Final images generation')
        plt.subplot(1, 2, 2)
        plt.title('Loss curves')
        plot_curves(history, n_epochs)
        plt.show()
    return params_gen, state_gen, params_disc, state_disc, history


if __name__ == '__main__':
    # Global configs
    seed = 0
    save_name = "CelebA-64/dcgan" # None or empty to not save
    batch_size = 128
    n_epochs = 40
    display_step = 100
    save_step = 3000
    num_images = (6, 6)
    plot = False

    # DCGAN configs
    config = Config(
        zdim=100,
        cylce_train_disc=1,
        lr_gen=2e-4,
        lr_disc=2e-4,

        channels_gen=(512, 256, 128, 64, 3),
        ker_shapes_gen=4,
        strides_gen=(1, 2, 2, 2, 2),
        padding_gen=(0, 1, 1, 1, 1),

        channels_disc=(64, 128, 256, 512, 1),
        ker_shapes_disc=4,
        strides_disc=(2, 2, 2, 2, 1),
        padding_disc=(1, 1, 1, 1, 0),
    )

    key = random.PRNGKey(seed)

    dataset = load_images(batch_size=batch_size, seed=seed)

    model_path = f'pre_trained/{save_name}'

    params_gen, state_gen, params_disc, state_disc, history = train_loop(
        dataset, key, config,
        n_epochs=n_epochs,
        display_step=display_step,
        num_images=num_images,
        plot=plot,
        model_path=model_path,
        save_step=save_step,
    )

    if save_name is not None and save_name != '':
        print()
        model_path = f'pre_trained/{save_name}'
        save_gan(params_gen, state_gen, params_disc, state_disc, config,
                 model_path, verbose=True)
        jnp.save(f'{model_path}/gen_history.npy', history['loss_gen'])
        jnp.save(f'{model_path}/disc_history.npy', history['loss_disc'])
