# ---------------------------------------------
# app.py — Streamlit UI for Peatland ABM demo
# ---------------------------------------------

# Import Streamlit for web UI, NumPy for math, Pandas for tables, and Matplotlib for plots
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports for file handling and date/time
import os
from pathlib import Path
from datetime import datetime
import json

# Import ABM model logic
from model import PeatlandABM, run_simulation

# Set web page title and layout
st.set_page_config(page_title="Peatland ABM", layout="wide")

# Title and description
st.title("Conceptual Agent-Based Model of Sustainable Land-Use Farming Adoption")
st.write("This interactive demo explores sustainable land-use adoption using a utility function that combines social and economic components at the farmer level (peer and economic weights) and a regional context parameter (alpha).")

# Sidebar: model parameters
with st.sidebar:
    st.markdown("---")
    st.subheader("Monte Carlo Simulation")
    n_runs = st.number_input(
        "Number of Monte Carlo Runs", min_value=1, max_value=500, value=1, step=1,
        help="How many times to repeat the simulation with different random seeds.")
    seed_base = st.number_input(
        "Monte Carlo Seed Base", value=42, step=1,
        help="Base seed for reproducibility. Each run uses seed_base + run_index.")
    st.header("Model Equations & Parameters")

    # Utility Calculation
    st.latex(r"U_i = \alpha \cdot U_{econ,i} + (1 - \alpha) \cdot U_{social,i}")
    st.latex(r"U_{econ,i} = S - w^{(p)}_i \cdot P^{(conv)}_i + w^{(p)}_i \cdot P^{(nat)}_i")
    st.latex(r"U_{social,i} = SCF \cdot w^{(s)}_i \cdot \phi")
    st.markdown("**Parameters for Utility Calculation:**")
    subsidy_eur_per_ha = st.slider(
        "Subsidy for adoption $S$ (EUR/ha/year)", 0.0, 500.0, 100.0, 10.0,
        help="Annual subsidy paid to adopters. Higher values increase economic utility.")
    alpha = st.slider(
        "Alpha $\alpha$ (Economic vs Social Weight)", 0.0, 1.0, 0.7, 0.05,
        help="Relative weight of economic utility vs social utility. 1 = only economic, 0 = only social.")
    social_capital_factor = st.slider(
        "Social Capital Factor $SCF$ (EUR/ha)", 0, 1000, 500, 50,
        help="Maximum value of social pressure. Higher values make peer influence stronger.")
    initial_share_adopters = st.slider(
        "Initial Share of Adopters $\phi$ (%)", 0, 100, 5, 1,
        help="Percentage of agents starting as adopters.")

    st.markdown("---")
    # Adoption Probability
    st.latex(r"p_i = \frac{1}{1 + \exp\left(-\frac{U_i}{k}\right)}")
    scaling_factor = st.slider(
        "Adoption Sensitivity $k$ (Logistic Scaling Factor)", 50, 500, 100, 10,
        help="Controls how sensitive adoption probability is to utility. Higher = less sensitive.")

    st.markdown("---")
    # Agent Heterogeneity
    st.latex(r"w^{(p)}_i, w^{(s)}_i \in [0.5, 2.0]")
    profit_weight_min, profit_weight_max = st.slider(
        "Profit Weight Range $w^{(p)}_i$", 0.5, 2.0, (0.5, 2.0), 0.05,
        help="Range for agent profit weights. Higher = more profit-driven.")
    peer_weight_min, peer_weight_max = st.slider(
        "Peer Weight Range $w^{(s)}_i$", 0.5, 2.0, (0.5, 2.0), 0.05,
        help="Range for agent peer weights. Higher = more peer-driven.")
    stay_adopter_prob_min, stay_adopter_prob_max = st.slider(
        "Stay Adopter Probability Range", 0.7, 0.99, (0.7, 0.99), 0.05,
        help="Probability that an adopter remains an adopter at each step.")

    st.markdown("---")
    # Learning Rates
    st.latex(r"w^{(s)}_i(t+1) = w^{(s)}_i(t) + \lambda_{social} (\phi(t) - w^{(s)}_i(t))")
    st.latex(r"w^{(p)}_i(t+1) = w^{(p)}_i(t) + \lambda_{econ} (U_{econ,i}(t) - w^{(p)}_i(t))")
    social_learning_rate = st.slider(
        "Social Learning Rate $\lambda_{social}$", 0.0, 1.0, 0.1, 0.05,
        help="How quickly agents update peer weights based on observed adoption.")
    econ_learning_rate = st.slider(
        "Economic Learning Rate $\lambda_{econ}$", 0.0, 1.0, 0.1, 0.05,
        help="How quickly agents update profit weights based on economic experience.")

    st.markdown("---")
    # Emissions and Policy Cost
    st.latex(r"E_i = 5.0 \cdot (1 - 0.5 \cdot A_i)")
    st.latex(r"\text{PolicyCost}_{ha} = S \cdot \phi")
    st.latex(r"\text{EmissionsSaved}_{ha} = \max(5.0 - \overline{E}, 0)")
    st.latex(r"\text{CostPerTonne} = \frac{\text{PolicyCost}_{ha}}{\text{EmissionsSaved}_{ha}}")
    steps = st.slider(
        "Simulation Steps", 10, 200, 50, 10,
        help="Number of time steps to run the simulation.")
    seed = st.number_input(
        "Random Seed", value=42, step=1,
        help="Seed for random number generation (reproducibility).")

# Run button

if st.button("Run Simulation"):
    # Prepare model parameters
    model_params = dict(
        subsidy_eur_per_ha=subsidy_eur_per_ha,
        alpha=alpha,
        social_capital_factor=social_capital_factor,
        scaling_factor=scaling_factor,
        initial_share_adopters=initial_share_adopters / 100.0
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/monte_carlo_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Monte Carlo or single simulation
    if n_runs == 1:
        from model import run_simulation
        model = PeatlandABM(seed=seed, **model_params)
        ds = run_simulation(model, steps)
        results_list = [ds]
    else:
        from model import monte_carlo_runs
        results_list = monte_carlo_runs(n_runs, steps=steps, seed_base=seed_base, **model_params)

    # Utility: initial vs final step (Monte Carlo)
    mean_utility = np.stack([ds['mean_utility'].values for ds in results_list])
    # Initial step stats
    initial_mean = mean_utility[:, 0]
    initial_mean_mu = np.mean(initial_mean)
    initial_mean_std = np.std(initial_mean)
    initial_mean_q25 = np.percentile(initial_mean, 25)
    initial_mean_q75 = np.percentile(initial_mean, 75)
    # Final step stats
    final_mean = mean_utility[:, -1]
    final_mean_mu = np.mean(final_mean)
    final_mean_std = np.std(final_mean)
    final_mean_q25 = np.percentile(final_mean, 25)
    final_mean_q75 = np.percentile(final_mean, 75)
    # Prepare model parameters
    model_params = dict(
        subsidy_eur_per_ha=subsidy_eur_per_ha,
        alpha=alpha,
        social_capital_factor=social_capital_factor,
        scaling_factor=scaling_factor,
        initial_share_adopters=initial_share_adopters / 100.0
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/monte_carlo_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Monte Carlo or single simulation
    if n_runs == 1:
        from model import run_simulation
        model = PeatlandABM(seed=seed, **model_params)
        ds = run_simulation(model, steps)
        results_list = [ds]
    else:
        from model import monte_carlo_runs
        results_list = monte_carlo_runs(n_runs, steps=steps, seed_base=seed_base, **model_params)

    # Aggregate results
    # Stack metrics for each run
    metrics = ["adoption_rate", "avg_emissions_tCO2_ha", "policy_cost_eur_per_ha", "cost_per_tonne_eur_per_tCO2"]
    stacked = {m: np.stack([ds[m].values for ds in results_list]) for m in metrics}
    steps_arr = results_list[0].coords["step"].values

    # Compute mean, std, 25th, 75th percentiles
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = np.mean(stacked[m], axis=0)
        agg[f"{m}_std"] = np.std(stacked[m], axis=0)
        agg[f"{m}_q25"] = np.percentile(stacked[m], 25, axis=0)
        agg[f"{m}_q75"] = np.percentile(stacked[m], 75, axis=0)

    # Save aggregated results to CSV
    df_mc = pd.DataFrame({"step": steps_arr})
    for k, v in agg.items():
        df_mc[k] = v
    df_mc.to_csv(output_dir / "monte_carlo_stats.csv", index=False)

    # Display results table
    st.subheader("Monte Carlo Results (first 5 rows)")
    st.dataframe(df_mc.head(5))

    # Plot mean ± std for adoption rate
    st.subheader("Adoption Rate Over Time (Monte Carlo)")
    fig, ax = plt.subplots()
    ax.plot(steps_arr, agg['adoption_rate_mean'], label="Mean Adoption Rate")
    ax.fill_between(steps_arr, agg['adoption_rate_mean'] - agg['adoption_rate_std'], agg['adoption_rate_mean'] + agg['adoption_rate_std'], alpha=0.3, label="±1 Std Dev")
    ax.set_xlabel("Step")
    ax.set_ylabel("Share of Adopters")
    ax.legend()
    st.pyplot(fig)
    fig.savefig(output_dir / "adoption_plot.png")

    # Plot mean ± std for emissions
    st.subheader("Average Emissions Over Time (Monte Carlo)")
    fig2, ax2 = plt.subplots()
    ax2.plot(steps_arr, agg['avg_emissions_tCO2_ha_mean'], label="Mean Emissions", color='red')
    ax2.fill_between(steps_arr, agg['avg_emissions_tCO2_ha_mean'] - agg['avg_emissions_tCO2_ha_std'], agg['avg_emissions_tCO2_ha_mean'] + agg['avg_emissions_tCO2_ha_std'], alpha=0.3, label="±1 Std Dev", color='red')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Avg Emissions (t CO₂-eq/ha/year)")
    ax2.legend()
    st.pyplot(fig2)
    fig2.savefig(output_dir / "emissions_plot.png")

    # Utility figure (now last)
    st.subheader("Utility Over Time (Monte Carlo)")
    steps_arr = results_list[0].coords["step"].values
    # Stack all runs for each utility type
    mean_utility_arr = np.stack([ds['mean_utility'].values for ds in results_list])
    mean_econ_utility_arr = np.stack([ds['mean_econ_utility'].values for ds in results_list])
    mean_social_utility_arr = np.stack([ds['mean_social_utility'].values for ds in results_list])

    # Compute mean and std for each time step
    mean_utility_mu = np.mean(mean_utility_arr, axis=0)
    mean_utility_std = np.std(mean_utility_arr, axis=0)
    mean_econ_utility_mu = np.mean(mean_econ_utility_arr, axis=0)
    mean_econ_utility_std = np.std(mean_econ_utility_arr, axis=0)
    mean_social_utility_mu = np.mean(mean_social_utility_arr, axis=0)
    mean_social_utility_std = np.std(mean_social_utility_arr, axis=0)

    fig_util, ax_util = plt.subplots()
    # Plot mean lines
    ax_util.plot(steps_arr, mean_utility_mu, label="Overall Utility", color="blue")
    ax_util.plot(steps_arr, mean_econ_utility_mu, label="Economic Utility", color="green")
    ax_util.plot(steps_arr, mean_social_utility_mu, label="Social Utility", color="orange")
    # Plot std shaded area
    ax_util.fill_between(steps_arr, mean_utility_mu - mean_utility_std, mean_utility_mu + mean_utility_std, color="blue", alpha=0.2)
    ax_util.fill_between(steps_arr, mean_econ_utility_mu - mean_econ_utility_std, mean_econ_utility_mu + mean_econ_utility_std, color="green", alpha=0.2)
    ax_util.fill_between(steps_arr, mean_social_utility_mu - mean_social_utility_std, mean_social_utility_mu + mean_social_utility_std, color="orange", alpha=0.2)
    ax_util.set_xlabel("Step")
    ax_util.set_ylabel("Utility (EUR/ha)")
    ax_util.set_title("Utility Over Time (Mean ± Std)")
    ax_util.legend()
    st.pyplot(fig_util)
    fig_util.savefig(output_dir / "utility_plot.png")

    # CSV download button
    st.download_button("Download Monte Carlo Results CSV", data=df_mc.to_csv(index=False), file_name="monte_carlo_stats.csv")

else:
    st.info("Set parameters and click Run Simulation.")
