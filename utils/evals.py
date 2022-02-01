import jax
from jax import jit, numpy as jnp
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

import functools
from jax_fid import fid, inception

def compute_statistics_from_image(images, params, apply_fn, batch_size=1, img_size=None):
    if type(images) == str:
        imgs = []
        for f in os.listdir(images):
            img = Image.open(os.path.join(images, f))
            img = jnp.array(img) / 255.0
            if img_size is not None:
                img = jax.image.resize(img, shape=(img_size[0], img_size[1], img.shape[2]), method='bilinear', antialias=False)
            imgs.append(img)
        images = jnp.array(imgs)

    else:
        imgs = []
        for img in images:
            #img = jnp.array(img) / 255.0
            if img_size is not None:
                img = jax.image.resize(img, shape=(img_size[0], img_size[1], img.shape[2]), method='bilinear', antialias=False)
            imgs.append(img)
        images = jnp.array(imgs)
    
    num_batches = int(np.ceil(images.shape[0] / batch_size))
    act = []
    for i in tqdm(range(num_batches)):
        x = images[i * batch_size:i * batch_size + batch_size]
        x = 2 * x - 1

        pred = apply_fn(params, jax.lax.stop_gradient(x))
        act.append(pred.squeeze(axis=1).squeeze(axis=1))
    act = jnp.concatenate(act, axis=0)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def frechet_inception_distance(images1, images2, batch_size=50, img_size=None, key=0):

    rng = jax.random.PRNGKey(key)
    model = inception.InceptionV3(pretrained=True)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    print('Computing statistics for dataset 1...')
    mu1,sigma1 = compute_statistics_from_image(images1, params, apply_fn, batch_size, img_size)
    print('Computing statistics for dataset 2...')
    mu2,sigma2 = compute_statistics_from_image(images2, params, apply_fn, batch_size, img_size)

    fid_score = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
    
    return fid_score

