from utils.BaseTrainer import BaseTrainer
from utils.losses import cross_entropy

class DCGANTrainer(BaseTrainer):
    def __init__(self, loss_func=cross_entropy):
        super(DCGANTrainer, self).__init__(loss_fn=loss_func)

    # The rest of the functions are similar to BaseTrainer
    # and inherit to them.
