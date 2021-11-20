from typing import NamedTuple
from gan.dcgan.modules import DCGenerator, DCDiscriminator, DCGANConfig
from gan.dcgan.training_func import init_gen, init_disc, train_gen, train_disc, input_func, gen_fwd, disc_fwd

class DCGAN(object):
    def __init__(self):
        self.Generator = DCGenerator
        self.Discriminator = DCDiscriminator
        self.Config = DCGANConfig
        self.init_gen = init_gen
        self.init_disc = init_disc
        self.train_gen = train_gen
        self.train_disc = train_disc
        self.input_func = input_func
        self.gen_fwd_apply = gen_fwd.apply
        self.disc_fwd_apply = disc_fwd.apply

