import numpy as np
import scipy
import pymc as pm
import pandas as pd
import altair as alt

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
        print("Sampling model...")
        idata = pm.sample(nuts_sampler='nutpie') #, idata_kwargs={"log_likelihood": True})
        print('Computing log-likelihood...') 
        pm.compute_log_likelihood(idata, extend_inferencedata=True)

        pm.set_data(
            coords={
                "obs_idx": cdata.index,
            },
            new_data=new_data
        )
        
        print("Sampling posterior predictive on census data...")
        # Reduce the sample to just 100 draws from 2 chains each
        idata_s = idata.sel(draw=slice(0,500),chain=[0,1])
        idata_ps = pm.sample_posterior_predictive(
            idata_s,
            predictions=True
        )

    print("Sampling synthetic population...")
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
    

# Plotting function for synthetic control
def plot_treatment_effect(df, trace, treatment_country, control_countries, treatment_year):
    # Prepare data for visualization
    years = df.index.values
    treatment_idx = years.searchsorted(treatment_year)
    
    # Calculate synthetic control prediction
    beta = trace.posterior.beta.mean(('chain', 'draw')).values
    synthetic = df[control_countries].values @ beta
    
    # Create DataFrame for visualization
    viz_data = pd.DataFrame({
        'year': years,
        'actual': df[treatment_country].values,
        'synthetic': synthetic,
        'treatment_effect': np.zeros_like(years)
    })
    
    # Calculate treatment effect
    viz_data.loc[treatment_idx:, 'treatment_effect'] = (
        viz_data.loc[treatment_idx:, 'actual'] - 
        viz_data.loc[treatment_idx:, 'synthetic']
    )

     # Normalize synthetic control to match treatment at intervention time
    normalization_factor = viz_data.loc[treatment_idx, 'actual'] / viz_data.loc[treatment_idx, 'synthetic']
    viz_data['synthetic'] = viz_data['synthetic'] * normalization_factor
    
    # Create base chart
    base = alt.Chart(viz_data).encode(
        x=alt.X('year:Q').axis(format='.0f').title('Year')
    )
    
    upper = np.ceil(viz_data[['actual', 'synthetic']].max().max()*2)/2
    
    # Create actual vs synthetic lines
    lines = base.mark_line().encode(
        y=alt.Y('value:Q', title='TFR').scale(domain=[1.0, upper]),
        color=alt.Color('type:N').title(None)
    ).transform_fold(
        ['actual', 'synthetic'],
        as_=['type', 'value']
    )
    
    # Add treatment year line
    treatment_line = alt.Chart(pd.DataFrame({'year': [treatment_year]})).mark_rule(
        color='red', strokeDash=[4, 4]
    ).encode(x='year:Q')
    
    # Create treatment effect area
    effect = alt.Chart(viz_data).mark_area(
        opacity=0.3,
        color='green'
    ).encode(
        x=alt.X('year:Q').axis(format='.0f').title('Year'),
        y=alt.Y('treatment_effect:Q', title='Treatment Effect').scale(domain=[-0.3, 0.3])
    )

    # Add reference line for treatment effect
    effect_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule().encode(y='y')
    
    # Combine charts
    chart = (lines + treatment_line).properties(
        title=f'TFR and Treatment Effect for {treatment_country}',
        width=500,
        height=300
    ) | (effect + treatment_line + effect_line).properties(
        width=400,
        height=300
    )
    
    return chart.configure(font='SF Compact Display')
    
# Prepare data for TFR example
def prepare_data(df, treatment_country, treatment_year):
    """
    Prepare data for the synthetic control model.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with TFR values
    treatment_country : str
        Name of the treated country
    treatment_year : int
        Year when the treatment happened
        
    Returns:
    --------
    X_control : numpy.ndarray
        Control countries TFR values
    y_treated_pre : numpy.ndarray
        Treated country pre-treatment TFR values
    y_treated_post : numpy.ndarray
        Treated country post-treatment TFR values
    control_countries : list
        Names of control countries
    pre_years : array
        Pre-treatment years
    post_years : array
        Post-treatment years
    """
    # Get treatment year index
    treatment_idx = df.index.get_loc(treatment_year)
    
    # Split into pre and post treatment periods
    df_pre = df.iloc[:treatment_idx]
    df_post = df.iloc[treatment_idx:]
    
    # Get years
    pre_years = df_pre.index.values
    post_years = df_post.index.values
    
    # Extract treated country data
    y_treated_pre = df_pre[treatment_country].values
    y_treated_post = df_post[treatment_country].values
    
    # Extract control countries data
    control_countries = [c for c in df.columns if c != treatment_country]
    X_control_pre = df_pre[control_countries].values
    X_control_post = df_post[control_countries].values
    
    # Combine pre and post control data
    X_control = np.vstack([X_control_pre, X_control_post])
    
    return X_control, y_treated_pre, y_treated_post, control_countries, pre_years, post_years