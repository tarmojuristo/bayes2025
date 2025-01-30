print('Loading libraries...')
import numpy as np
import pymc as pm

def run():
    size = 100
    rng = np.random.default_rng(1234)
    x = rng.normal(size=size)
    g = rng.choice(list("ABC"), size=size)
    y = rng.integers(0, 1, size=size, endpoint=True)

    g_levels, g_idxs = np.unique(g, return_inverse=True)
    coords = {"group": g_levels}

    with pm.Model(coords=coords) as model:
        coef_x = pm.Normal("x")
        coef_g = pm.Normal("g", dims="group")
        p = pm.math.softmax(coef_x + coef_g[g_idxs])
        pm.Bernoulli("y", p=p, observed=y)
        idata = pm.sample()

    return idata, model


if __name__ == "__main__":
    idata, model = run()
