# ---------------------------------------------
# app.py — Streamlit UI for Peatland ABM demo
# ---------------------------------------------

# Import Streamlit for web UI, NumPy for math, Pandas for tables, and Matplotlib for plots
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our ABM model logic
from model import PeatlandABM, run_simulation

# Set web page title and layout
st.set_page_config(page_title="Peatland ABM", layout="wide")

# Title and description
st.title("Peatland ABM — Agent-Based Modeling Demo")
st.write("This minimal agent-based model simulates how farmers on Dutch peatlands may adopt nature-inclusive practices under different policy scenarios.")

# Sidebar: model parameters
with st.sidebar:
    st.header("Model Parameters")

    # Simulation settings
    n_agents = st.slider("Number of farmers", min_value=20, max_value=500, value=100, step=10)
    steps = st.slider("Simulation steps", min_value=10, max_value=200, value=50, step=10)
    seed = st.number_input("Random seed", value=42, step=1)

    # Economic and social parameters
    subsidy = st.slider("Subsidy for adoption", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    profit_diff = st.slider("Profit difference (conventional - nature-inclusive)", -2.0, 2.0, 0.5, 0.1)
    peer_weight = st.slider("Peer influence weight", 0.0, 1.0, 0.3, 0.05)

# Run button
if st.button("Run Simulation"):
    # Initialize model
    model = PeatlandABM(
        n_agents=n_agents,
        subsidy=subsidy,
        profit_diff=profit_diff,
        peer_weight=peer_weight,
        seed=seed
    )

    # Run simulation for the number of steps
    results = run_simulation(model, steps)

    # Display results table
    st.subheader("Simulation Output (first 5 rows)")
    st.dataframe(results.head())

    # Plot adoption rate
    st.subheader("Adoption Rate Over Time")
    fig, ax = plt.subplots()
    ax.plot(results['step'], results['adoption_rate'], label="Adoption Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("Share of Adopters")
    ax.legend()
    st.pyplot(fig)

    # Plot emissions
    st.subheader("Average Emissions Over Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(results['step'], results['avg_emissions'], label="Emissions", color='red')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Avg Emissions")
    ax2.legend()
    st.pyplot(fig2)

    # CSV download button
    st.download_button("Download Results CSV", data=results.to_csv(index=False), file_name="abm_results.csv")
else:
    st.info("Set parameters and click Run Simulation.")
