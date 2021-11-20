import haiku as hk
from haiku.initializers import Constant, RandomNormal
import jax
from jax import numpy as jnp, random

from utils.config import GANConfig


class DCGenerator(hk.Module):
    def __init__(self, channels, ker_shapes, strides, padding, name=None):
        super().__init__(name=name)
        self.name = name
        self.channels = channels
        self.ker_shapes = ker_shapes
        self.strides = strides
        self.padding = padding
        self.n_layers = len(channels)

        if isinstance(ker_shapes, int):
            ker_shapes = [ker_shapes] * self.n_layers
        if isinstance(strides, int):
            strides = [strides] * self.n_layers
        if isinstance(padding, int):
            padding = [padding] * self.n_layers

        self.layers = [
            hk.Conv2DTranspose(
                        channels[i],
                        kernel_shape=ker_shapes[i],
                        stride=strides[i],
                        padding='VALID' if padding[i]==0 else 'SAME',
                        with_bias=False,
                        w_init=RandomNormal(stddev=0.02, mean=0.0)
                        )
            for i in range(self.n_layers)
            ]

        self.batch_norms = [
            hk.BatchNorm(True, True, 0.99) for _ in range(self.n_layers - 1)
            ]

    def forward(self, z, is_training):
        x = jnp.reshape(z, (-1, 1, 1, z.shape[-1]))
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = self.batch_norms[i](x, is_training)
            x = jax.nn.relu(x)
        x = self.layers[-1](x)
        x = jnp.tanh(x)

        return x

    def __call__(self, z, is_training=True):
        return self.forward(z, is_training)


class DCDiscriminator(hk.Module):
    def __init__(self, channels, ker_shapes, strides, padding, name=None):
        super().__init__(name=name)
        self.name = name
        self.channels = channels
        self.ker_shapes = ker_shapes
        self.strides = strides
        self.padding = padding
        self.n_layers = len(channels)

        if isinstance(ker_shapes, int):
            ker_shapes = [ker_shapes] * self.n_layers
        if isinstance(strides, int):
            strides = [strides] * self.n_layers
        if isinstance(padding, int):
            padding = [padding] * self.n_layers

        self.layers = [
            hk.Conv2D(channels[i],
                      kernel_shape=ker_shapes[i],
                      stride=strides[i],
                      padding='VALID' if padding[i]==0 else 'SAME',
                      w_init=RandomNormal(stddev=0.02, mean=0.0),
                      b_init=Constant(0.0)
                      )
            for i in range(self.n_layers)
            ]
        self.batch_norms = [
            hk.BatchNorm(True, True, 0.99) for _ in range(self.n_layers - 1)
            ]

    def forward(self, x, is_training):
        if x.ndim == 3:
            x = jnp.expand_dims(x, axis=-1)
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = self.batch_norms[i](x, is_training)
            x = jax.nn.leaky_relu(x, 0.2)

        x = self.layers[-1](x)
        x = jnp.squeeze(x)
        return x

    def __call__(self, z, is_training=True):
        return self.forward(z, is_training)


class DCGANConfig(GANConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default_init(self):
        # Gneral
        self.cylce_train_disc = 5

        # Latent input vector
        self.zdim = 64

        # Generator
        self.channels_gen = (256, 128, 64, 1)
        self.ker_shapes_gen = (3, 4, 3, 4)
        self.strides_gen = (2, 1, 2, 2)
        self.padding_gen = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.name_gen = 'DCGAN_gen'
        self.lr_gen = 1e-4
        self.beta1_gen = 0.5
        self.beta2_gen = 0.999

        # Discriminator
        self.channels_disc = (16, 32, 1)
        self.ker_shapes_disc = 4
        self.strides_disc = 2
        self.padding_disc = ((0, 0), (0, 0), (0, 0))
        self.name_disc = 'DCGAN_disc'
        self.lr_disc = 1e-4
        self.beta1_disc = 0.5
        self.beta2_disc = 0.999

    def get_models_kwargs(self):
        kwargs_gen = {
            "channels": self.channels_gen,
            "ker_shapes": self.ker_shapes_gen,
            "strides": self.strides_gen,
            "padding": self.padding_gen,
            "name": self.name_gen
            }
        kwargs_disc = {
            "channels": self.channels_disc,
            "ker_shapes": self.ker_shapes_disc,
            "strides": self.strides_disc,
            "padding": self.padding_disc,
            "name": self.name_disc
            }
        kwargs_gen = hk.data_structures.to_haiku_dict(kwargs_gen)
        kwargs_disc = hk.data_structures.to_haiku_dict(kwargs_disc)
        return kwargs_gen, kwargs_disc
