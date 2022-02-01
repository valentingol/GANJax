import os
import argparse
import pickle

from jax import random
import haiku as hk

from gan.dcgan import DCGAN as GAN
from utils.save_and_load import load_jax_model
from utils.plot_img import plot_tensor_images

gan = GAN()
trainer = gan.Trainer()

@hk.transform_with_state
def generate(zseed, config, n_samples=1, is_training=False):
    zkey = random.PRNGKey(zseed)
    kwargs_gen = config.get_models_kwargs()[0]
    generator = gan.Generator(**kwargs_gen)
    z = trainer.input_func(zkey, n_samples, config.zdim)
    X_fake = generator(z, is_training=is_training)
    return X_fake


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_images_path', type=str, help='Path to save generated images')
    args = parser.parse_args()
    model_path = 'pre_trained/CelebA-64/dcgan'
    seed = 0
    zseed = 0
    num_images = (10, 10)
    cmap = 'gray'

    key = random.PRNGKey(seed)
    config = gan.Config().load(os.path.join(model_path, 'config.pickle'))
    params, state = load_jax_model(os.path.join(model_path, 'generator'))

    generate.init(key, zseed, config, 1, is_training=True)

    images, _ = generate.apply(params, state, key,
                               zseed=zseed,
                               config=config,
                               n_samples=num_images[0]*num_images[1],
                               is_training=False)
    
    if args.save_images_path is not None:
        with open(args.save_images_path, 'wb') as save_images:
            pickle.dump(images, save_images)


    plot_tensor_images(images, num_images=num_images, cmap=cmap)
    plt.show()

