import haiku as hk

from gan.dcgan.modules import DCGenerator, DCDiscriminator
from utils.BaseTrainer import BaseTrainer
from utils.losses import cross_entropy

class DCGANTrainer(BaseTrainer):
    def __init__(self, loss_func=cross_entropy):
        super(DCGANTrainer, self).__init__(loss_fn=loss_func)

    @hk.transform_with_state
    def gen_fwd(self, z, kwargs_gen, is_training):
        """ (transformed) Forward pass of generator. """
        generator = DCGenerator(**kwargs_gen)
        X_fake = generator(z, is_training=is_training)
        return X_fake

    @hk.transform_with_state
    def disc_fwd(self, X, kwargs_disc, is_training):
        """ (transformed) Discriminator pass of generator. """
        discriminator = DCDiscriminator(**kwargs_disc)
        y_pred = discriminator(X, is_training=is_training)
        return y_pred

    # The rest of the functions are similar to BaseTrainer
    # and inherit to them.
