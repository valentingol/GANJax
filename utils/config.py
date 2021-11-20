import os
import pickle

class GANConfig(object):
    def __init__(self, **kwargs):
        self.default_init()
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f'Error in configs initialization: {k} '
                                 'is not a valid argument.')
            setattr(self, k, v)

    def default_init(self):
        """ Initialize default values for all attributes.

        self.att1 = default_val1
        self.att2 = default val2
        ...

        Warning: you should define a default value for ALL valid attributes.
        """
        raise NotImplementedError("You must overload default_init.")

    def get_models_kwargs(self):
        """ Return intitialization kwargs for generator
        and discriminator (parameters of __init__).
        See hk.data_structures.to_haiku_dict to transform
        a dict to a haiku dict.

        Returns:
            kwargs_gen: haiku dict
            kwargs_disc: haiku dict
        """
        raise NotImplementedError("You must overload get_models_kwargs.")

    def get_configs(self):
        configs = {}
        for attr in self.__dict__:
            if not attr.startswith('_'):
                configs[attr] = getattr(self, attr)
        return configs

    def save(self, path, verbose=True):
        configs = self.get_configs()
        head = os.path.split(path)[0]
        os.makedirs(head, exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(configs, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"Config saved at '{path}'.")

    def load(self, path):
        with open(path, 'rb') as handle:
            configs = pickle.load(handle)
        for attr, value in configs.items():
            setattr(self, attr, value)
        return self
