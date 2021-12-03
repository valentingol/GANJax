from typing import NamedTuple
from gan.dcgan.modules import DCGenerator, DCDiscriminator
from gan.dcgan.config import DCGANConfig
from gan.dcgan.trainer import DCGANTrainer

class DCGAN(object):
    def __init__(self):
        self.Generator = DCGenerator
        self.Discriminator = DCDiscriminator
        self.Config = DCGANConfig
        self.Trainer = DCGANTrainer
