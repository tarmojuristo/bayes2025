import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import arviz as az

@st.cache_data
def load_data():
	return az.from_netcdf("bb.nc")
	
trace = load_data()

teams = list(trace.posterior.team.to_numpy())
team_to_idx = {team: idx for idx, team in enumerate(teams)}

def simulate_matchup(trace, team_to_idx, home_team, away_team, num_samples=3000):
	
	np.random.seed(seed=42) # To stabilise score plots between reruns
	
	pp = trace.posterior_predictive
		
	# Look up team indices
	home_idx = team_to_idx[home_team]
	away_idx = team_to_idx[away_team]
	
	#home_expected = posterior['home_expected'].values[:, :, home_idx].flatten()[:num_samples]
	#away_expected = posterior['away_expected'].values[:, :, away_idx].flatten()[:num_samples]
	
	# Apply Poisson likelihood by sampling scores
	home_score_samples = pp['ap_home_score'].values[:, :, home_idx,away_idx].flatten()[:num_samples] 
		#np.random.poisson(np.maximum(1, home_expected))
	away_score_samples = pp['ap_away_score'].values[:, :, home_idx,away_idx].flatten()[:num_samples]
		#np.random.poisson(np.maximum(1, away_expected))
	score_diff_samples = home_score_samples - away_score_samples
	
	# Return results in a DataFrame
	result_df = pd.DataFrame({
		"home_team": home_team,
		"away_team": away_team,
		"home_score": home_score_samples,
		"away_score": away_score_samples,
		"score_diff": score_diff_samples
	})
	
	win_prob = (home_score_samples > away_score_samples).mean()
	
	return result_df, win_prob
	
def create_score_heatmap(results, home_team, away_team):
	# Define score ranges
	home_range = range(40, 110)
	away_range = range(40, 110)
	
	# Process data for heatmap
	heatmap_df = results[results.score_diff != 0].groupby(
		['home_score', 'away_score'], observed=True
	).size().reset_index(name='count')
	
	# Create full grid of combinations
	full_index = pd.MultiIndex.from_product(
		[home_range, away_range], 
		names=['home_score', 'away_score']
	)
	
	# Reindex to include all combinations with zeros for missing values
	# and remove ties
	df_full = (heatmap_df.set_index(['home_score', 'away_score'])
			   .reindex(full_index, fill_value=0)
			   .reset_index())
	df_full = df_full[df_full['home_score'] != df_full['away_score']] # Remove draws
	
	mode = df_full[df_full['count']==df_full['count'].max()].values[0]
	
	# Create heatmap
	axis_values = list(np.arange(8, 25) * 5)
	heatmap = alt.Chart(df_full).mark_rect(strokeWidth=0).encode(
		x=alt.X('home_score:O').axis(
			bandPosition=0.5, 
			grid=False, 
			values=axis_values
		).title([f'Home team: {home_team}', f'Most likely score: {mode[0]}:{mode[1]}']),
		y=alt.Y('away_score:O').axis(
			bandPosition=0.5, 
			domain=False, 
			grid=False,
			values=axis_values
		).title(
			f'Away team: {away_team}'
			),
		color=alt.Color('count:Q').scale(scheme='blues').legend(None),
		tooltip=[
			alt.Tooltip('home_score', title=f'{home_team}:'), 
			alt.Tooltip('away_score', title=f'{away_team}:'), 
			'count'
		]
	).properties(
		width=600, 
		height=650, 
		title='Expected Score Dispersion'
	)
	
	return heatmap

def plot_result(results, win_prob, team_A, team_B, cutoff=0):
	
	diff = (results.score_diff > cutoff).mean() if cutoff >= 0  else (results.score_diff.astype(int) < cutoff).mean()
	
	chart = alt.Chart(results[results.score_diff!=0]['score_diff'].value_counts(normalize=True).reset_index()).mark_bar(
		cornerRadiusTopLeft=2,
		cornerRadiusTopRight=2
	).encode(
		x=alt.X('score_diff:Q').title([
			#f"Probability of {team_A} winning: {win_prob:.1%}", 
			f"Probability of score difference of more than {cutoff} points: {diff:.2%}",
			f"Most likely result: {results.score_diff.mode().iloc[0]:+.0f}"
		]).axis(format='+.0f'),
		y=alt.Y('proportion:Q').axis(format='.1%').title(None),
		tooltip=[alt.Tooltip('score_diff:Q', title='Score difference:'), alt.Tooltip("proportion:Q", format='.1%', title='Probability:')]
	)
	
	tie_line = alt.Chart(pd.DataFrame({"score_diff": [cutoff]})).mark_rule(
			color="red", strokeDash=[4, 4], size=1.5
	).encode(x=alt.X("score_diff:Q")) 
	
	return (chart + tie_line).configure(font='IBM Plex Sans').properties(
		width=700, height=400, title=f"Simulated Score Difference: {team_A} vs {team_B}")


probs = st.container(border=True)

left, middle, right = probs.columns(3)

team_A = left.selectbox('Home team:', np.sort(teams))
team_B = middle.selectbox('Away team:', np.sort([t for t in teams if t!=team_A]))
cutoff = right.number_input('Points cutoff:', step=1, value=0)

results, win_prob = simulate_matchup(trace, team_to_idx, team_A, team_B, num_samples=3000)

io1 = 1/(win_prob) if win_prob!=0 else 1.0 #np.inf
io2 = 1/(1-win_prob) if win_prob!=1 else 1000.0 #np.inf

probs.write(' ')
probs.write(f"Probability of **{team_A}** beating **{team_B}** at home: **{win_prob:.2%}** (odds: {io1:.2f}:{io2:.2f})")

plot = plot_result(results, win_prob, team_A, team_B, cutoff)

sim = st.container(border=True)

tab1, tab2= sim.tabs(["Score Difference", "Score Dispersion Heatmap"])

with tab1:
	tab1.altair_chart(plot)
	
with tab2:
	heatmap = create_score_heatmap(results, team_A, team_B)
	tab2.altair_chart(heatmap, use_container_width=False)

odds = st.container(border=True)

with odds.expander('How does this work?'):
	st.write('Simulator is initially seeded by implied decimal odds, derived from the winning probability ' +
		'determined by the model. It is then possible to replace them with new values -- such as odds of a ' +
		'particular bookmaker -- based on which the simulator calculates the expected value assuming a €100 bet ' + 
		'placed on the outcome with respective odds.'
)

c1, c2 = odds.columns(2)

#odds.markdown(f"##### Implied odds at {win_prob:.2%} win probability: :red[{io1:.2f}] : :red[{io2:.2f}]")

if io1 == np.inf or io2 == np.inf:
	odds.write('**Error:** Cannot compute, infinite odds')
	
else:
	short_odds = c1.number_input(team_A+' :', value=io1, step=.01, min_value=1.0)
	long_odds = c2.number_input(team_B+' :', value=io2, step=.01, min_value=1.0)
	c1.metric('Expected value on a €100 bet:', f"{(short_odds-1) * win_prob * 100 - (1-win_prob)*100:,.2f}€")
	c2.metric('Expected value on a €100 bet:', f"{(long_odds-1) * (1-win_prob) * 100 - (win_prob)*100:,.2f}€")





