import os
import pickle

def save_jax_model(params, state, model_path, with_states=True, verbose=True):
    os.makedirs(model_path+'/params', exist_ok=True)
    param_path = os.path.join(model_path, 'params', 'params.pickle')
    with open(param_path, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if with_states:
        os.makedirs(model_path+'/state', exist_ok=True)
        state_path = os.path.join(model_path, 'state', 'state.pickle')
        with open(state_path, 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"Model saved at '{model_path}'.")


def load_jax_model(model_path):
    param_path = os.path.join(model_path, 'params', 'params.pickle')
    state_path = os.path.join(model_path, 'state', 'state.pickle')
    with open(param_path, 'rb') as handle:
        params = pickle.load(handle)
    if os.path.exists(state_path):
        with open(state_path, 'rb') as handle:
            state = pickle.load(handle)
        return params, state
    else:
        return params


def save_gan(params_gen, state_gen, params_disc, state_disc, config,
             model_path, with_states=True, verbose=True):
    save_jax_model(params_gen, state_gen,
                   os.path.join(model_path, 'generator'),
                   with_states=with_states, verbose=verbose)
    save_jax_model(params_disc, state_disc,
                   os.path.join(model_path, 'discriminator'),
                   with_states=with_states, verbose=verbose)
    config.save(os.path.join(model_path, 'config.pickle'), verbose)