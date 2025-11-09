import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from model import PeatlandABM, run_simulation, monte_carlo_runs


st.set_page_config(page_title='Peatland ABM', layout='wide')

st.title('Peatland Adoption Model with Intention-Adoption Split')
st.write(
    "Simulate peatland farmers' adoption decisions with an explicit intention stage, "
    "prospect-theory economics (Rommel et al., 2022), and emissions factors from van Leeuwen et al. (2024)."
)

if "simulation_complete" not in st.session_state:
    st.session_state["simulation_complete"] = False

with st.sidebar:
    st.markdown('---')
    st.subheader('Simulation Setup')
    n_agents = st.slider('Number of farmers', 50, 1000, 500, 50)
    steps = st.slider('Simulation steps', 10, 200, 50, 10, help='Number of periods to simulate.')
    n_runs = st.number_input(
        'Monte Carlo runs', min_value=1, max_value=500, value=1, step=1,
        help='How many seeds to evaluate for the Monte Carlo ensemble.'
    )
    seed_base = st.number_input(
        'Monte Carlo seed base', value=42, step=1,
        help='Run i uses seed_base + i for reproducibility.'
    )
    seed = st.number_input(
        'Primary random seed', value=42, step=1,
        help='Used when running a single simulation.'
    )

    st.markdown('---')
    st.subheader('Policy & Initial Conditions')
    subsidy_eur_per_ha = st.slider(
        'Subsidy S (EUR/ha/year)', 0.0, 500.0, 100.0, 10.0,
        help='Annual subsidy paid to adopters.'
    )
    hetero_persistence = st.checkbox(
        'Heterogeneous adopter persistence', value=True,
        help='If selected, each adopter draws their own stay probability in [0.7, 0.99].'
    )
    stay_adopter_prob = st.slider(
        'Stay-adopter probability (if homogeneous)', 0.5, 0.99, 0.90, 0.01,
        help='Used only when heterogeneity is disabled.'
    )

    st.markdown('---')
    st.subheader('Stage A - Intention Formation')
    st.latex(r"I_i = w_{econ} \tilde{U}_{econ,i} + w_{soc} \tilde{U}_{soc,i} + w_{pers} V_i + w_{self} B_i + b")
    st.latex(r"p^{intent}_i = \sigma\!\left(T \cdot I_i\right)")
    risk_aversion_factor = st.slider(
        'Loss aversion lambda', 0.5, 3.0, 1.2, 0.1,
        help='Prospect theory loss aversion parameter (Rommel et al., 2022).'
    )
    weight_econ = st.slider('Weight: economic (w_econ)', 0.0, 3.0, 1.0, 0.1)
    weight_social = st.slider('Weight: social (w_soc)', 0.0, 3.0, 1.0, 0.1)
    weight_personal = st.slider('Weight: personal values (w_pers)', 0.0, 3.0, 0.6, 0.1)
    weight_self = st.slider('Weight: self-belief (w_self)', 0.0, 3.0, 0.6, 0.1)
    intention_intercept = st.slider('Intercept b', -2.0, 2.0, 0.0, 0.1)
    intention_temperature = st.slider('Sigmoid temperature T', 0.1, 5.0, 1.0, 0.1)

    st.markdown('---')
    st.subheader('Learning Dynamics')
    social_learning_rate = st.slider('Social learning rate', 0.0, 1.0, 0.1, 0.01)
    econ_learning_rate = st.slider('Economic learning rate', 0.0, 1.0, 0.1, 0.01)

    st.markdown('---')
    st.subheader('Stage B - Adoption Friction')
    st.latex(r"p^{adopt}_i = p^{intent}_i \cdot C_i \cdot O_i")
    st.write("Capability and opportunity are now loaded from the CSV file.")

st.markdown('---')
run_clicked = st.button('Run simulation')

if run_clicked:
    st.session_state['simulation_complete'] = True

    weights = {
        'econ': weight_econ,
        'social': weight_social,
        'personal': weight_personal,
        'self': weight_self,
    }

    model_params = dict(
        n_agents=int(n_agents),
        subsidy_eur_per_ha=float(subsidy_eur_per_ha),
        risk_aversion_factor=float(risk_aversion_factor),
        hetero_persistence=bool(hetero_persistence),
        stay_adopter_prob=float(stay_adopter_prob),
        intention_weights=weights,
        intention_intercept=float(intention_intercept),
        intention_steepness=float(intention_temperature),
        social_learning_rate=float(social_learning_rate),
        econ_learning_rate=float(econ_learning_rate),
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/monte_carlo_run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    with st.spinner('Running simulations...'):
        if int(n_runs) == 1:
            model = PeatlandABM(seed=int(seed), **model_params)
            ds = run_simulation(model, int(steps))
            results_list = [ds]
        else:
            results_list = monte_carlo_runs(
                int(n_runs), steps=int(steps), seed_base=int(seed_base), **model_params
            )

    metrics = [
        'adoption_rate',
        'intention_rate',
        'avg_emissions_tCO2_ha',
        'policy_cost_eur_per_ha',
        'cost_per_tonne_eur_per_tCO2',
    ]
    stacked = {m: np.stack([ds[m].values for ds in results_list]) for m in metrics}
    steps_arr = results_list[0].coords['step'].values

    agg = {}
    for m, arr in stacked.items():
        agg[f'{m}_mean'] = np.nanmean(arr, axis=0)
        agg[f'{m}_std'] = np.nanstd(arr, axis=0)
        agg[f'{m}_q25'] = np.nanpercentile(arr, 25, axis=0)
        agg[f'{m}_q75'] = np.nanpercentile(arr, 75, axis=0)

    df_mc = pd.DataFrame({'step': steps_arr})
    for key, values in agg.items():
        df_mc[key] = values
    df_mc.to_csv(output_dir / 'monte_carlo_stats.csv', index=False)

    mean_utility_arr = np.stack([ds['mean_utility'].values for ds in results_list])
    mean_econ_arr = np.stack([ds['mean_econ_utility'].values for ds in results_list])
    mean_social_arr = np.stack([ds['mean_social_utility'].values for ds in results_list])
    utility_summary = {
        'overall': (np.mean(mean_utility_arr, axis=0), np.std(mean_utility_arr, axis=0)),
        'economic': (np.mean(mean_econ_arr, axis=0), np.std(mean_econ_arr, axis=0)),
        'social': (np.mean(mean_social_arr, axis=0), np.std(mean_social_arr, axis=0)),
    }

    final_adoption_mean = agg['adoption_rate_mean'][-1]
    final_intention_mean = agg['intention_rate_mean'][-1]
    st.success(f'Mean adoption rate after {int(steps)} steps: {final_adoption_mean:.1%}')
    st.caption(
        f'Mean intention rate ends at {final_intention_mean:.1%}, giving an intention-adoption gap of '
        f'{(final_intention_mean - final_adoption_mean):+.1%}.'
    )

    first_ds = results_list[0]
    adoption_series = first_ds['adoption_rate'].values
    intention_series = first_ds['intention_rate'].values
    col1, col2 = st.columns(2)
    col1.metric(
        'Adoption rate (first run, final step)',
        f'{adoption_series[-1]:.1%}',
        delta=f'{(adoption_series[-1] - adoption_series[0]):+.1%}',
    )
    col2.metric(
        'Intention rate (first run, final step)',
        f'{intention_series[-1]:.1%}',
        delta=f'{(intention_series[-1] - intention_series[0]):+.1%}',
    )

    st.subheader('Monte Carlo summary (first five rows)')
    st.dataframe(df_mc.head())
    st.download_button(
        'Download Monte Carlo summary CSV',
        data=df_mc.to_csv(index=False),
        file_name='monte_carlo_stats.csv',
    )

    adoption_low = np.clip(agg['adoption_rate_mean'] - agg['adoption_rate_std'], 0.0, 1.0)
    adoption_high = np.clip(agg['adoption_rate_mean'] + agg['adoption_rate_std'], 0.0, 1.0)
    intention_low = np.clip(agg['intention_rate_mean'] - agg['intention_rate_std'], 0.0, 1.0)
    intention_high = np.clip(agg['intention_rate_mean'] + agg['intention_rate_std'], 0.0, 1.0)

    st.subheader('Intention vs adoption over time')
    fig_beh, ax_beh = plt.subplots()
    ax_beh.plot(steps_arr, agg['intention_rate_mean'], label='Intention rate', color='tab:orange')
    ax_beh.fill_between(steps_arr, intention_low, intention_high, alpha=0.2, color='tab:orange')
    ax_beh.plot(steps_arr, agg['adoption_rate_mean'], label='Adoption rate', color='tab:blue')
    ax_beh.fill_between(steps_arr, adoption_low, adoption_high, alpha=0.2, color='tab:blue')
    ax_beh.set_xlabel('Step')
    ax_beh.set_ylabel('Share of agents')
    ax_beh.set_ylim(0, 1)
    ax_beh.grid(alpha=0.3)
    ax_beh.legend()
    st.pyplot(fig_beh)
    fig_beh.savefig(output_dir / 'intention_adoption_plot.png')

    st.subheader('Average emissions trajectory')
    emis_low = agg['avg_emissions_tCO2_ha_mean'] - agg['avg_emissions_tCO2_ha_std']
    emis_high = agg['avg_emissions_tCO2_ha_mean'] + agg['avg_emissions_tCO2_ha_std']
    fig_emis, ax_emis = plt.subplots()
    ax_emis.plot(
        steps_arr, agg['avg_emissions_tCO2_ha_mean'], label='Mean emissions', color='tab:red'
    )
    ax_emis.fill_between(steps_arr, emis_low, emis_high, alpha=0.2, color='tab:red')
    ax_emis.set_xlabel('Step')
    ax_emis.set_ylabel('t CO$_2$-eq/ha/year')
    ax_emis.grid(alpha=0.3)
    ax_emis.legend()
    st.pyplot(fig_emis)
    fig_emis.savefig(output_dir / 'emissions_plot.png')

    st.subheader('Utility decomposition')
    util_mean, util_std = utility_summary['overall']
    econ_mean, econ_std = utility_summary['economic']
    social_mean, social_std = utility_summary['social']
    fig_util, ax_util = plt.subplots()
    ax_util.plot(steps_arr, util_mean, label='Overall utility', color='tab:blue')
    ax_util.fill_between(steps_arr, util_mean - util_std, util_mean + util_std, color='tab:blue', alpha=0.2)
    ax_util.plot(steps_arr, econ_mean, label='Economic utility', color='tab:green')
    ax_util.fill_between(steps_arr, econ_mean - econ_std, econ_mean + econ_std, color='tab:green', alpha=0.2)
    ax_util.plot(steps_arr, social_mean, label='Social utility', color='tab:orange')
    ax_util.fill_between(steps_arr, social_mean - social_std, social_mean + social_std, color='tab:orange', alpha=0.2)
    ax_util.set_xlabel('Step')
    ax_util.set_ylabel('Utility (EUR/ha)')
    ax_util.grid(alpha=0.3)
    ax_util.legend()
    st.pyplot(fig_util)
    fig_util.savefig(output_dir / 'utility_plot.png')

    st.subheader('Agent-level diagnostics (final step, first run)')
    final_diag = pd.DataFrame(
        {
            'Intention probability': first_ds['intention_prob_per_agent'].values[-1],
            'Adoption probability': first_ds['adoption_prob_per_agent'].values[-1],
            'Capability': first_ds['capability_per_agent'].values[-1],
            'Opportunity': first_ds['opportunity_per_agent'].values[-1],
            'Profit change (EUR/ha)': first_ds['expected_profit_change_eur_per_ha'].values[-1],
        }
    )
    with st.expander('Show agent distributions', expanded=False):
        st.dataframe(final_diag.describe(percentiles=[0.1, 0.5, 0.9]).T)

        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(
            final_diag['Intention probability'],
            final_diag['Adoption probability'],
            alpha=0.4,
            s=18,
            label='Agent',
        )
        ax_scatter.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='p_intent = p_adopt')
        ax_scatter.set_xlabel('Intention probability')
        ax_scatter.set_ylabel('Adoption probability')
        ax_scatter.set_xlim(0, 1)
        ax_scatter.set_ylim(0, 1)
        ax_scatter.grid(alpha=0.2)
        ax_scatter.legend()
        st.pyplot(fig_scatter)
        fig_scatter.savefig(output_dir / 'intention_vs_adoption_scatter.png')

        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(final_diag['Intention probability'], bins=30, alpha=0.6, label='Intention', color='tab:orange')
        ax_hist.hist(final_diag['Adoption probability'], bins=30, alpha=0.6, label='Adoption', color='tab:blue')
        ax_hist.set_xlabel('Probability')
        ax_hist.set_ylabel('Number of agents')
        ax_hist.legend()
        ax_hist.grid(alpha=0.2)
        st.pyplot(fig_hist)
        fig_hist.savefig(output_dir / 'intention_adoption_hist.png')

        fig_cap, ax_cap = plt.subplots()
        ax_cap.hist(final_diag['Capability'], bins=30, alpha=0.6, label='Capability', color='tab:green')
        ax_cap.hist(final_diag['Opportunity'], bins=30, alpha=0.6, label='Opportunity', color='tab:purple')
        ax_cap.set_xlabel('Scaled factor')
        ax_cap.set_ylabel('Number of agents')
        ax_cap.legend()
        ax_cap.grid(alpha=0.2)
        st.pyplot(fig_cap)
        fig_cap.savefig(output_dir / 'capability_opportunity_hist.png')

    st.caption(f'Outputs saved to `{output_dir}`.')

elif not st.session_state['simulation_complete']:
    st.info('Adjust parameters in the sidebar and click Run simulation.')
