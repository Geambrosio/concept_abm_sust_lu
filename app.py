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
st.title("Peatland ABM — Agent-Based Modeling Demo")
st.write("This minimal agent-based model simulates how farmers adopt sutainable practices based on simplified policy, economic and social aspects.")

# Sidebar: model parameters
with st.sidebar:
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
    # Initialize model with real units
    model = PeatlandABM(
        subsidy_eur_per_ha=subsidy_eur_per_ha,
        seed=seed,
        alpha=alpha,
        social_capital_factor=social_capital_factor,
        scaling_factor=scaling_factor,
        initial_share_adopters=initial_share_adopters / 100.0
    )



    # Run simulation for the number of steps
    ds = run_simulation(model, steps)

    # Use xarray for all metrics and plotting
    steps_arr = ds.coords["step"].values

    # Cumulative metrics
    cum_policy_cost_eur_per_ha = ds["policy_cost_eur_per_ha"].cumsum(dim="step")
    cum_emissions_saved_tCO2_ha = ds["emissions_saved_tCO2_ha"].cumsum(dim="step")
    cum_cost_per_tonne_eur_per_tCO2 = cum_policy_cost_eur_per_ha / cum_emissions_saved_tCO2_ha.where(cum_emissions_saved_tCO2_ha != 0)

    # Moving averages
    adoption_rate_ma10 = ds["adoption_rate"].rolling(step=10, center=False).mean()
    avg_emissions_ma10 = ds["avg_emissions_tCO2_ha"].rolling(step=10, center=False).mean()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/run_{timestamp}")    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata dictionary
    metadata = {
        "timestamp": timestamp,
        "parameters": {
            "steps": steps,
            "seed": seed,
            "subsidy_eur_per_ha": subsidy_eur_per_ha,
        },
        "results_summary": {
            "final_adoption_rate": float(ds["adoption_rate"].isel(step=-1).values),
            "final_emissions_tCO2_ha": float(ds["avg_emissions_tCO2_ha"].isel(step=-1).values),
            "mean_adoption_rate": float(ds["adoption_rate"].mean().values),
            "mean_emissions_tCO2_ha": float(ds["avg_emissions_tCO2_ha"].mean().values),
            "final_cost_per_tonne_eur_per_tCO2": float(ds["cost_per_tonne_eur_per_tCO2"].isel(step=-1).values),
            "cumulative_cost_per_tonne_eur_per_tCO2": float(cum_cost_per_tonne_eur_per_tCO2.isel(step=-1).values)
        },
        "metrics": list(ds.data_vars.keys()),
        "files_generated": [
            "abm_results.csv",
            "metadata.json",
            "metadata.txt",
            "adoption_plot.png",
            "emissions_plot.png",
            "cost_per_tonne_plot.png"
        ]
    }

    # Save metadata as JSON
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Save metadata as TXT
    with open(output_dir / "metadata.txt", "w") as f:
        f.write("Peatland ABM Simulation Metadata\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Parameters (real units):\n")
        for param, value in metadata["parameters"].items():
            f.write(f"- {param}: {value}\n")
        f.write("\nResults Summary (real units):\n")
        for metric, value in metadata["results_summary"].items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\nMetrics Saved:\n")
        for metric in metadata["metrics"]:
            f.write(f"- {metric}\n")
        f.write("\nFiles Generated:\n")
        for file in metadata["files_generated"]:
            f.write(f"- {file}\n")

    # Save results to CSV (convert only for download)
    df = ds.to_dataframe().reset_index()
    df.to_csv(output_dir / "abm_results.csv", index=False)

    # Display results table
    st.subheader("Simulation Output (first 2 rows)")
    st.dataframe(df.head(2))

    st.subheader("Key policy metric")
    col1, col2 = st.columns(2)
    col1.metric("Final cost per tCO₂ saved",
                f"{ds['cost_per_tonne_eur_per_tCO2'].isel(step=-1).values:.0f} EUR/tCO₂")
    col2.metric("Cumulative cost per tCO₂ saved",
                f"{cum_cost_per_tonne_eur_per_tCO2.isel(step=-1).values:.0f} EUR/tCO₂")

    # Plot adoption rate
    st.subheader("Adoption Rate Over Time")
    fig, ax = plt.subplots()
    slope, intercept = np.polyfit(steps_arr, ds['adoption_rate'].values, 1)
    ax.plot(steps_arr, slope * steps_arr + intercept, linestyle="--", color="black", label="Trend")
    ax.plot(steps_arr, ds['adoption_rate'].values, label="Adoption Rate", color='blue')
    ax.set_xlabel("Step")
    ax.set_ylabel("Share of Adopters")
    ax.legend()
    st.pyplot(fig)
    fig.savefig(output_dir / "adoption_plot.png")

    # Plot emissions (t CO2-eq/ha/year)
    st.subheader("Average Emissions Over Time (t CO₂-eq/ha/year)")
    fig2, ax2 = plt.subplots()
    slope, intercept = np.polyfit(steps_arr, ds['avg_emissions_tCO2_ha'].values, 1)
    ax2.plot(steps_arr, slope * steps_arr + intercept, linestyle="--", color="black", label="Trend")
    ax2.plot(steps_arr, ds['avg_emissions_tCO2_ha'].values, label="Emissions", color='red')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Avg Emissions (t CO₂-eq/ha/year)")
    ax2.legend()
    st.pyplot(fig2)
    fig2.savefig(output_dir / "emissions_plot.png")

    # Plot change in agent utility from first to last timestep
    st.subheader("Change in Agent Utility: First vs Last Step")
    first_utilities = ds["utility_per_agent"].isel(step=0).values
    last_utilities = ds["utility_per_agent"].isel(step=-1).values
    fig4, ax4 = plt.subplots()
    ax4.hist(first_utilities, bins=20, alpha=0.5, label="First Step", color="orange", edgecolor="black")
    ax4.hist(last_utilities, bins=20, alpha=0.5, label="Last Step", color="blue", edgecolor="black")
    ax4.set_xlabel("Agent Utility (EUR/ha)")
    ax4.set_ylabel("Number of Agents")
    ax4.set_title("Agent Utility: First vs Last Step")
    ax4.legend()
    st.pyplot(fig4)
    fig4.savefig(output_dir / "utility_change_histogram.png")
    st.write("This plot compares the distribution of agent utility at the start and end of the simulation.")

    # CSV download button
    st.download_button("Download Results CSV", data=df.to_csv(index=False), file_name="abm_results.csv")


else:
    st.info("Set parameters and click Run Simulation.")
