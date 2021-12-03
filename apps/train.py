from time import time

from jax import numpy as jnp, random
import matplotlib.pyplot as plt

from gan.dcgan import DCGAN as GAN
from utils.data import load_images_celeba_64 as load_images
from utils.means import Mean
from utils.plot_img import plot_curves, plot_tensor_images
from utils.save_and_load import save_gan

gan = GAN()
trainer = gan.Trainer() # The loss function can be change here

# Global training function

def main(dataset, key, config, n_epochs=60, max_time=None,
         display_step=500, num_images=(10, 10), plot=True,
         save_step=None, model_path=None):
    max_time = max_time or jnp.inf # max_time = inf iif max_time in {0, None}
    kwargs_gen, kwargs_disc = config.get_models_kwargs()
    X_real = jnp.array(dataset.take(1).as_numpy_iterator().next())
    batch_size = X_real.shape[0]

    # Initialize generator and discriminator
    z = trainer.input_func(key, batch_size, config.zdim)
    state_gen, opt_gen, opt_state_gen, params_gen = trainer.init_gen(
        key, config, z, kwargs_gen
        )
    state_disc, opt_disc, opt_state_disc, params_disc = trainer.init_disc(
        key, config, X_real, kwargs_disc
        )
    print('Initialization succeeded.')

    mean_loss_gen, mean_loss_disc = Mean(), Mean()
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
            key, subkey = random.split(key, 2)

            # Training both generator and discriminator
            (params_gen, params_disc, state_gen, state_disc,
             opt_state_gen, opt_state_disc, mean_loss_gen,
             mean_loss_disc) = trainer.cycle_train(
                X_real, key, params_gen, params_disc, state_gen, state_disc,
                opt_gen, opt_state_gen, opt_disc, opt_state_disc, kwargs_gen,
                kwargs_disc, mean_loss_gen, mean_loss_disc, config
                )

            itr += 1
            # Plot images
            if plot and itr % display_step == 0 and itr > 0:
                z = trainer.input_func(subkey, num_images[0]*num_images[1],
                                       config.zdim)
                X_fake, state_gen = trainer.gen_fwd_apply(
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
                save_gan(params_gen, state_gen, params_disc, state_disc,
                         config, model_path + f'/itr{itr}', verbose=False)

            if time() - start >= max_time:
                print('\n\nTraining time limit reached.')
                break

        mean_loss_gen.reset(), mean_loss_disc.reset()
        print()

        if time() - start >= max_time:
            break

    history = {'loss_gen': mean_loss_gen.history,
               'loss_disc': mean_loss_disc.history}
    if plot:
        z = trainer.input_func(key, num_images[0]*num_images[1], config.zdim)
        X_fake, state_gen = trainer.gen_fwd_apply(
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
    save_name = "MyGann" # None or empty to not save
    batch_size = 128
    n_epochs = 2
    max_time = 10  # in seconds
    display_step = 100
    save_step = 10000
    num_images = (6, 6)
    plot = True

    # DCGAN configs
    config = gan.Config(
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

    params_gen, state_gen, params_disc, state_disc, history = main(
        dataset=dataset,
        key=key,
        config=config,
        n_epochs=n_epochs,
        max_time=max_time,
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
