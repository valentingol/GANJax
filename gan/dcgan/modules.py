import haiku as hk
from haiku.initializers import Constant, RandomNormal
import jax
from jax import numpy as jnp

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
            hk.BatchNorm(False, False, 0.99) for _ in range(self.n_layers - 1)
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
