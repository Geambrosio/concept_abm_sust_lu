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
    st.header("Model Parameters")

    # Simulation settings
    n_agents = st.slider("Number of farmers", min_value=20, max_value=500, value=100, step=10)
    steps = st.slider("Simulation steps", min_value=10, max_value=200, value=50, step=10)
    seed = st.number_input("Random seed", value=42, step=1)

    # Economic and social parameters
    subsidy_eur_per_ha = st.slider("Subsidy for adoption (EUR/ha/year)", min_value=0.0, max_value=500.0, value=100.0, step=10.0)
    profit_diff_eur_per_ha = st.slider("Profit difference (conventional - nature-inclusive, EUR/ha/year)", -200.0, 200.0, 50.0, 10.0)
    peer_weight = st.slider("Peer influence weight (0-1)", 0.0, 1.0, 0.3, 0.05)

# Run button

if st.button("Run Simulation"):
    # Initialize model with real units
    model = PeatlandABM(
        n_agents=n_agents,
        subsidy_eur_per_ha=subsidy_eur_per_ha,
        profit_diff_eur_per_ha=profit_diff_eur_per_ha,
        peer_weight=peer_weight,
        seed=seed
    )

    # Run simulation for the number of steps
    results = run_simulation(model, steps)

    # Cumulative (assumes each step ~ 1 year)
    results["cum_policy_cost_eur_per_ha"] = results["policy_cost_eur_per_ha"].cumsum()
    results["cum_emissions_saved_tCO2_ha"] = results["emissions_saved_tCO2_ha"].cumsum()
    results["cum_cost_per_tonne_eur_per_tCO2"] = (
        results["cum_policy_cost_eur_per_ha"] / results["cum_emissions_saved_tCO2_ha"].replace(0, np.nan)
    )


    # Calculate moving averages for smoother plots
    results["adoption_rate_ma10"] = results["adoption_rate"].rolling(window=10).mean()
    results["avg_emissions_ma10"] = results["avg_emissions_tCO2_ha"].rolling(window=10).mean()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/run_{timestamp}")    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata dictionary
    metadata = {
        "timestamp": timestamp,
        "parameters": {
            "n_agents": n_agents,
            "steps": steps,
            "seed": seed,
            "subsidy_eur_per_ha": subsidy_eur_per_ha,
            "profit_diff_eur_per_ha": profit_diff_eur_per_ha,
            "peer_weight": peer_weight
        },
        "results_summary": {
            "final_adoption_rate": float(results["adoption_rate"].iloc[-1]),
            "final_emissions_tCO2_ha": float(results["avg_emissions_tCO2_ha"].iloc[-1]),
            "mean_adoption_rate": float(results["adoption_rate"].mean()),
            "mean_emissions_tCO2_ha": float(results["avg_emissions_tCO2_ha"].mean()),
            "final_cost_per_tonne_eur_per_tCO2": float(results["cost_per_tonne_eur_per_tCO2"].iloc[-1]),
            "cumulative_cost_per_tonne_eur_per_tCO2": float(results["cum_cost_per_tonne_eur_per_tCO2"].iloc[-1])
        },
        "metrics": list(results.columns),
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

    # Save results to CSV
    results.to_csv(output_dir / "abm_results.csv", index=False)

    # Display results table
    st.subheader("Simulation Output (first 2 rows)")
    st.dataframe(results.head(2))

    st.subheader("Key policy metric")
    col1, col2 = st.columns(2)
    col1.metric("Final cost per tCO₂ saved",
                f"{results['cost_per_tonne_eur_per_tCO2'].iloc[-1]:.0f} EUR/tCO₂")
    col2.metric("Cumulative cost per tCO₂ saved",
                f"{results['cum_cost_per_tonne_eur_per_tCO2'].iloc[-1]:.0f} EUR/tCO₂")

    # Plot adoption rate
    st.subheader("Adoption Rate Over Time")
    fig, ax = plt.subplots()
    # Calculate trend line (slope & intercept)
    slope, intercept = np.polyfit(results['step'], results['adoption_rate'], 1)
    ax.plot(results['step'], slope * results["step"] + intercept, linestyle="--", color="black", label="Trend")
    ax.plot(results['step'], results['adoption_rate'], label="Adoption Rate", color='blue')
    ax.set_xlabel("Step")
    ax.set_ylabel("Share of Adopters")
    ax.legend()
    st.pyplot(fig)
    fig.savefig(output_dir / "adoption_plot.png")  # Save to local file

    # Plot emissions (t CO2-eq/ha/year)
    st.subheader("Average Emissions Over Time (t CO₂-eq/ha/year)")
    fig2, ax2 = plt.subplots()
    # Calculate trend line (slope & intercept)
    slope, intercept = np.polyfit(results['step'], results['avg_emissions_tCO2_ha'], 1)
    ax2.plot(results['step'], slope * results["step"] + intercept, linestyle="--", color="black", label="Trend")
    ax2.plot(results['step'], results['avg_emissions_tCO2_ha'], label="Emissions", color='red')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Avg Emissions (t CO₂-eq/ha/year)")
    ax2.legend()
    st.pyplot(fig2)
    fig2.savefig(output_dir / "emissions_plot.png")  # Save to local file

    st.subheader("Cost per tCO₂ saved over time")
    fig3, ax3 = plt.subplots()
    ax3.plot(results["step"], results["cost_per_tonne_eur_per_tCO2"], label="Per-step EUR/tCO₂", color="orange")
    ax3.plot(results["step"], results["cum_cost_per_tonne_eur_per_tCO2"], linestyle="--", label="Cumulative EUR/tCO₂", color="black")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("EUR per tCO₂ saved")
    ax3.legend()
    st.pyplot(fig3)
    fig3.savefig(output_dir / "cost_per_tonne_plot.png")
    st.write("Note: Cumulative cost per tCO₂ can be very high initially when few emissions are saved.")

    # CSV download button
    st.download_button("Download Results CSV", data=results.to_csv(index=False), file_name="abm_results.csv")


else:
    st.info("Set parameters and click Run Simulation.")
