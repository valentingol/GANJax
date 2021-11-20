import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

def plot_tensor_images(images, num_images=(10, 10), cmap='gray'):
    # Normalize to [0, 1]
    if images.min() < 0:
        images = (images + 1.0) / 2.0
        images = jnp.clip(images, 0.0, 1.0)
    h, w = images.shape[1:3]
    nh, nw = num_images
    if len(images) < nh * nw:
        raise ValueError("Not enough images to show (number of images "
                         f"received: {len(images)}, number of image "
                         f"needed : {nh}x{nw}.")
    image_grid = images[: nh * nw].reshape(nh, nw, h, w, -1)
    image_grid = jnp.transpose(image_grid, (0, 2, 1, 3, 4))
    image_grid = image_grid.reshape(nh * h, nw * w, -1)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(image_grid, cmap=cmap)


def plot_curves(history, n_epochs):
    loss_gen, loss_disc = history['loss_gen'], history['loss_disc']
    loss_gen, loss_disc = jnp.array(loss_gen), jnp.array(loss_disc)
    # Downsample the points to reduce the length of the plots
    len_gen, len_disc = min(1000, len(loss_gen)), min(1000, len(loss_disc))
    time_gen = jnp.linspace(0, n_epochs, len_gen)
    time_disc = jnp.linspace(0, n_epochs, len_disc)
    loss_gen = jnp.interp(time_gen,
                          jnp.linspace(0, n_epochs, len(loss_gen)),
                          loss_gen)
    loss_disc = jnp.interp(time_disc,
                          jnp.linspace(0, n_epochs, len(loss_disc)),
                          loss_disc)

    plt.plot(time_gen, loss_gen, color='#ff9100', label='generator loss')
    plt.plot(time_disc, loss_disc, color='#00aaff', label='discriminator loss')
    plt.ylim([0, 5.0])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')


if __name__ == '__main__':
    h, w = 128, 128
    num_images = (4, 4)
    for seed in range(5):
        images = []
        key = jax.random.PRNGKey(seed)
        for _ in range(num_images[0]*num_images[1]):
            key, subkey = jax.random.split(key, 2)
            row = jax.random.uniform(key, shape=(w, 1))
            rows = [row]
            for i in range(h - 1):
                queue, tail = row[:-1], row[-1:]
                row = jnp.concatenate([tail, queue], axis=0)
                rows.append(row)
            images.append(jnp.stack(rows, axis=0))
        images = jnp.stack(images, axis=0)
        plot_tensor_images(images, num_images)
        plt.show(block=False)
        plt.pause(1)

    # loss_gen = jnp.linspace(1, 20, 50000)
    # loss_disc = jnp.linspace(1, 50, 100000)
    # history = {'loss_gen': loss_gen, 'loss_disc': loss_disc}
    # plot_curves(history, 30)
    # plt.show()
