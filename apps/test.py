import os
from typing import Generator

import haiku as hk
from jax import random

from gan.dcgan import DCGAN as GAN
from utils.save_and_load import load_jax_model
from utils.plot_img import plot_tensor_images

Generator = GAN().Generator
Config = GAN().Config
input_func = GAN().input_func

@hk.transform_with_state
def generate(zseed, config, n_samples=1, is_training=False):
    zkey = random.PRNGKey(zseed)
    kwargs_gen = config.get_models_kwargs()[0]
    generator = Generator(**kwargs_gen)
    z = input_func(zkey, n_samples, config.zdim)
    X_fake = generator(z, is_training=is_training)
    return X_fake


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    model_path = 'pre_trained/CIFAR-10/dcgan'
    seed = 0
    zseed = 0
    num_images = (10, 10)
    cmap = None

    key = random.PRNGKey(seed)
    config = Config().load(os.path.join(model_path, 'config.pickle'))
    params, state = load_jax_model(os.path.join(model_path, 'generator'))

    generate.init(key, zseed, config, 1, is_training=True)

    images, _ = generate.apply(params, state, key,
                               zseed=zseed,
                               config=config,
                               n_samples=num_images[0]*num_images[1],
                               is_training=False)

    plot_tensor_images(images, num_images=num_images, cmap=cmap)
    plt.show()
