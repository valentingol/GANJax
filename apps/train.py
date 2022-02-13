from jax import numpy as jnp, random

from gan.progan import ProGAN as GAN
from utils.data import load_images_cifar10 as load_images
from utils.save_and_load import save_gan

gan = GAN()
trainer = gan.Trainer()  # The loss function can be change here

if __name__ == '__main__':
    # Global configs
    seed = 0
    save_name = "MyGAN"  # None or empty to not save
    batch_size = 128
    max_time = None  # in seconds
    display_step = 200
    save_step = 10000
    num_images = (6, 6)
    plot = True
    with_states = False  # save states or not

    # DCGAN configs
    config = gan.Config(
        zdim=64,
        channels_gen=64,
        channels_disc=64,
        first_resolution=8,
        n_steps=1,
        growing_epochs = (100,),
        fixed_epochs = (100,),
        cylce_train_disc=2,
        lr_gen=1e-5,
        lr_disc=1e-6,
        lambda_gp=10,
    )

    key = random.PRNGKey(seed)
    dataset = load_images(batch_size=batch_size, seed=seed)
    print('Dataset loaded.')
    model_path = f'pre_trained/{save_name}'

    params_gen, state_gen, params_disc, state_disc, history = trainer.main(
        dataset=dataset,
        key=key,
        config=config,
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
