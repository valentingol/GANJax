import haiku as hk
from haiku.initializers import VarianceScaling as VScaling
import jax
from jax import numpy as jnp

from gan.progan.auxiliary import upsample_2d, add_batch_std, pixel_norm

class ProGeneratorBlock(hk.Module):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.conv1 = hk.Conv2D(self.channels, 3, padding='SAME',
                              w_init=VScaling(2.0), with_bias=False,
                              name='conv1')
        self.conv2 = hk.Conv2D(self.channels, 3, padding='SAME',
                              w_init=VScaling(2.0), with_bias=False,
                              name='conv2')

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = pixel_norm(x)

        x = self.conv2(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = pixel_norm(x)
        return x


class ProGenerator(hk.Module):
    def __init__(self, n_blocks, channels, first_resolution, name=None):
        super().__init__(name=name)
        if isinstance(channels, int):
            channels = [channels] * (n_blocks + 1)
        self.resolution = first_resolution
        self.n_blocks = n_blocks
        self.channels = channels
        self.base_lin = hk.Linear(first_resolution**2 * channels[0],
                                    w_init=VScaling(2.0), with_bias=False,
                                    name='base_lin')
        self.base_conv1 = hk.Conv2D(channels[0], 4, padding='SAME',
                                    w_init=VScaling(2.0), with_bias=False,
                                    name='base_conv1')
        self.base_conv2 = hk.Conv2D(channels[0], 3, padding='SAME',
                                    w_init=VScaling(2.0), with_bias=False,
                                    name='base_conv2')
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(ProGeneratorBlock(channels[i + 1],
                                                 name=f'block_{i}'))

        # 1*1 convolutions ('toRGB' layers) from the previous model
        # (lower resolution) and the current model
        self.prev_conv11 = hk.Conv2D(3, 1, padding='SAME',w_init=VScaling(2.0),
                                with_bias=False, name='prev_conv11')
        self.new_conv11 = hk.Conv2D(3, 1, padding='SAME',w_init=VScaling(2.0),
                                with_bias=False, name='new_conv11')

    def forward(self, z, alpha):
        # Base generator
        x = pixel_norm(z)
        x = self.base_lin(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = pixel_norm(x)
        x = jnp.reshape(x, (x.shape[0], self.resolution, self.resolution, -1))
        x = self.base_conv1(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = pixel_norm(x)
        x = self.base_conv2(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = pixel_norm(x)

        # Blocks
        for i in range(0, self.n_blocks - 1):
            x = upsample_2d(x, 2)
            x = self.blocks[i](x)

        # Weight sum with last block
        x1 = upsample_2d(x, 2)
        x1 = self.blocks[-1](x1)
        x1 = self.new_conv11(x1)
        x2 = self.prev_conv11(x)
        x2 = upsample_2d(x2, 2)
        x = alpha * x1 + (1 - alpha) * x2
        return x

    def __call__(self, z, alpha):
        return self.forward(z, alpha)


class ProDiscriminatorBlock(hk.Module):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.conv1 = hk.Conv2D(self.channels, 3, padding='SAME',
                              w_init=VScaling(2.0), with_bias=True,
                              name='conv1')
        self.conv2 = hk.Conv2D(self.channels, 3, padding='SAME',
                              w_init=VScaling(2.0), with_bias=True,
                              name='conv2')

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = jax.nn.leaky_relu(x, 0.2)
        return x


class ProDiscriminator(hk.Module):
    def __init__(self, n_blocks, channels, name=None):
        super().__init__(name=name)
        if isinstance(channels, int):
            channels = [channels] * (n_blocks + 1)
        self.n_blocks = n_blocks
        self.channels = channels
        self.base_conv1 = hk.Conv2D(channels[0], 3, padding='SAME',
                                    w_init=VScaling(2.0), with_bias=True,
                                    name='base_conv1')
        self.base_conv2 = hk.Conv2D(channels[0], 4, padding='SAME',
                                    w_init=VScaling(2.0), with_bias=True,
                                    name='base_conv2')
        self.base_lin = hk.Linear(1, w_init=VScaling(2.0), with_bias=True,
                                  name='base_lin')
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(ProGeneratorBlock(channels[i + 1],
                                                 name=f'block_{i}'))

        # 1*1 convolutions ('fromRGB' layers) from the previous model
        # (lower resolution) and the current model
        self.prev_conv11 = hk.Conv2D(channels[0], 1, padding='SAME',
                                     w_init=VScaling(2.0), with_bias=True,
                                     name='prev_conv11')
        self.new_conv11 = hk.Conv2D(channels[0], 1, padding='SAME',
                                    w_init=VScaling(2.0), with_bias=True,
                                    name='new_conv11')

    def forward(self, x, alpha):
        # Note: the blocks are apply in reverse order to keep the same
        # name for the layers in models with other resolution

        # Weight sum with last block
        x1 = self.new_conv11(x)
        x1 = jax.nn.leaky_relu(x1, 0.2)
        x1 = self.blocks[-1](x1)
        x1 = hk.avg_pool(x1, 2, 2, 'SAME')
        x2 = hk.avg_pool(x, 2, 2, 'SAME')
        x2 = self.prev_conv11(x2)
        x2 = jax.nn.leaky_relu(x2, 0.2)
        x = alpha * x1 + (1 - alpha) * x2

        # Blocks
        for i in range(self.n_blocks - 2, -1, -1):
            x = self.blocks[i](x)
            x = hk.avg_pool(x, 2, 2, 'SAME')

        # Base discriminator
        x = add_batch_std(x)
        x = self.base_conv1(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = self.base_conv2(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = hk.Flatten()(x)
        x = self.base_lin(x)
        return x

    def __call__(self, x, alpha):
        return self.forward(x, alpha)
