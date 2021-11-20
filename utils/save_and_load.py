import os
import pickle

def save_jax_model(params, state, model_path, verbose=True):
    os.makedirs(model_path+'/params', exist_ok=True)
    os.makedirs(model_path+'/state', exist_ok=True)
    param_path = os.path.join(model_path, 'params', 'params.pickle')
    state_path = os.path.join(model_path, 'state', 'state.pickle')
    with open(param_path, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(state_path, 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"Model saved at '{model_path}'.")


def load_jax_model(model_path):
    param_path = os.path.join(model_path, 'params', 'params.pickle')
    state_path = os.path.join(model_path, 'state', 'state.pickle')
    with open(param_path, 'rb') as handle:
        params = pickle.load(handle)
    with open(state_path, 'rb') as handle:
        state = pickle.load(handle)
    return params, state


def save_gan(params_gen, state_gen, params_disc, state_disc, config,
             model_path, verbose=True):
    save_jax_model(params_gen, state_gen,
                   os.path.join(model_path, 'generator'), verbose)
    save_jax_model(params_disc, state_disc,
                   os.path.join(model_path, 'discriminator'), verbose)
    config.save(os.path.join(model_path, 'config.pickle'), verbose)