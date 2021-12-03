import haiku as hk

from utils.GANConfig import GANConfig

class DCGANConfig(GANConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default_init(self):
        """ Overwrite the default_init to define the
        configs for the DCGAN. """
        # Gneral
        self.cylce_train_disc = 5

        # Latent input vector
        self.zdim = 64

        # Generator
        self.channels_gen = (256, 128, 64, 1)
        self.ker_shapes_gen = (3, 4, 3, 4)
        self.strides_gen = (2, 1, 2, 2)
        self.padding_gen = (0, 0, 0, 0)
        self.name_gen = 'DCGAN_gen'
        self.lr_gen = 1e-4
        self.beta1_gen = 0.5
        self.beta2_gen = 0.999

        # Discriminator
        self.channels_disc = (16, 32, 1)
        self.ker_shapes_disc = 4
        self.strides_disc = 2
        self.padding_disc = (0, 0, 0, 0)
        self.name_disc = 'DCGAN_disc'
        self.lr_disc = 1e-4
        self.beta1_disc = 0.5
        self.beta2_disc = 0.999

    def get_models_kwargs(self):
        """ Overwrite the get_models_kwargs to get the
        configs required to init the modules of DCGAN. """
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
        kwargs_gen = hk.data_structures.to_immutable_dict(kwargs_gen)
        kwargs_disc = hk.data_structures.to_immutable_dict(kwargs_disc)
        return kwargs_gen, kwargs_disc
