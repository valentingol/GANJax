import haiku as hk

from utils.GANConfig import GANConfig

class ProGANConfig(GANConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default_init(self):
        """ Overwrite the default_init to define the
        configs for the ProGAN. """
        # General
        # NOTE: output resolution is first_resolution * 2**n_steps
        self.first_resolution = 4
        self.n_steps = 3
        self.growing_epochs = (3, 4, 5)  # of length n_steps
        self.fixed_epochs = (5, 8, 10)  # of length n_steps
        self.lambda_gp = 10.0
        self.cylce_train_disc = 1

        # Latent input vector
        self.zdim = 64

        # Generator
        # NOTE: len(channels_gen) = n_steps + 1
        # (or integer for constant channels)
        self.channels_gen = 128
        self.name_gen = 'ProGAN_gen'
        self.lr_gen = 1e-3
        self.beta1_gen = 0.0
        self.beta2_gen = 0.99

        # Discriminator
        # NOTE: len(channels_disc) = n_blocks + 1
        # (or integer for constant channels)
        self.channels_disc = 128
        self.name_disc = 'ProGAN_disc'
        self.lr_disc = 1e-3
        self.beta1_disc = 0.0
        self.beta2_disc = 0.99

    def get_models_kwargs(self):
        """ Overwrite the get_models_kwargs to get the
        configs required to init the modules of ProGAN. """
        kwargs_gen = {
            "channels": self.channels_gen,
            "first_resolution": self.first_resolution,
            "name": self.name_gen
            }
        kwargs_disc = {
            "channels": self.channels_disc,
            "name": self.name_disc
            }
        kwargs_gen = hk.data_structures.to_immutable_dict(kwargs_gen)
        kwargs_disc = hk.data_structures.to_immutable_dict(kwargs_disc)
        return kwargs_gen, kwargs_disc
