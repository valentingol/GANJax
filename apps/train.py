import os

from jax import numpy as jnp, random

from gan.dcgan import DCGAN as GAN
from utils.data import load_images_cifar10 as load_images
from utils.save_and_load import save_gan

gan = GAN()
trainer = gan.Trainer()  # The loss function can be change here

if __name__ == '__main__':
    # Global configs
    seed = 0
    save_name = "MyGAN"  # None or empty to not save
    batch_size = 128
    n_epochs = 20
    max_time = None  # in seconds
    display_step = 50
    save_step = 10000
    num_images = (6, 6)
    plot = True
    with_states = True  # save states or not
    config_model_path = "pre_trained/CIFAR-10/dcgan"

    # DCGAN configs
    config = gan.Config().load(os.path.join(config_model_path,
                                            'config.pickle'))

    key = random.PRNGKey(seed)
    dataset = load_images(batch_size=batch_size, seed=seed)
    model_path = f'pre_trained/{save_name}'

    params_gen, state_gen, params_disc, state_disc, history = trainer.main(
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
                 model_path, with_states=with_states)
        jnp.save(f'{model_path}/gen_history.npy', history['loss_gen'])
        jnp.save(f'{model_path}/disc_history.npy', history['loss_disc'])
