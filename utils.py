import numpy as np
import scipy
import pymc as pm
import pandas as pd

# Multivariate verion of Binomial distribution
def MvBinomial(name, p, N, n, **kwargs):
    cn = np.arange(N+1)

    probs = scipy.special.binom(N,cn) * \
            (p[:,None]**cn[None,:]) * \
            (1-p[:,None])**(N-cn[None,:])
    
    return pm.Multinomial(name, p=probs, n=n, **kwargs)

# Abstracted from mrp.ipynb
def poststratify(model, cdata, new_data, obs_name='y', n_samples=1000000):
    with model:
        print("Sampling model")
        idata = pm.sample(nuts_sampler='nutpie', idata_kwargs={"log_likelihood": True})

        pm.set_data(
            coords={
                "obs_idx": cdata.index,
            },
            new_data=new_data
        )
        
        print("Sampling posterior predictive on census data")
        # Reduce the sample to just 100 draws from 2 chains each
        idata_s = idata.sel(draw=slice(0,500),chain=[0,1])
        idata_ps = pm.sample_posterior_predictive(
            idata_s,
            predictions=True
        )

    print("Sampling synthetic population")
    pred = idata_ps.predictions[obs_name]

    # This needed some technical work to not explode memory usage
    df = pd.DataFrame(
        pred.values.reshape(-1), dtype=pd.Int16Dtype(),
        columns=['N']
    ).assign(
        chain=np.repeat(pred.chain.values, len(pred.draw) * len(pred.obs_idx)*len(pred.response)).astype('int8'),
        draw=np.tile(np.repeat(pred.draw.values, len(pred.obs_idx)*len(pred.response)), len(pred.chain)).astype('int16'),
        obs_idx=np.tile(pred.obs_idx.values, len(pred.chain) * len(pred.draw)*len(pred.response)).astype('int16'),
        response=np.tile(np.arange(len(pred.response)), len(pred.chain) * len(pred.draw)*len(pred.obs_idx)).astype('int8')
    )
    df['adraw'] = df['draw'] + df['chain'] * len(pred.draw)
    #df.memory_usage(deep=True)

    # Sample a "synthetic population" of 1M rows by sampling with replacement with weights based on N
    df = df.sample(weights='N',n=1000000,replace=True).drop(columns=['N'])
    df['response'] = pd.Categorical(pred.response.values[df['response']],categories=pred.response.values,ordered=True)

    # Merge with the census data on obs_idx
    df = df.merge(cdata,left_on='obs_idx',right_index=True)
    df = df.drop(columns=['obs_idx','N','chain','draw'])

    return df, idata