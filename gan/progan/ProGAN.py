from gan.progan.modules import ProGenerator, ProDiscriminator
from gan.progan.config import ProGANConfig
from gan.progan.trainer import ProGANTrainer

class ProGAN(object):
    def __init__(self):
        self.Generator = ProGenerator
        self.Discriminator = ProDiscriminator
        self.Config = ProGANConfig
        self.Trainer = ProGANTrainer
