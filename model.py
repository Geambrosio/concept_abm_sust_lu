# ---------------------------------------------
# model.py — Core agent-based model logic
# ---------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr

# Define ABM class

class PeatlandABM:
    """
    Agent-based model where each agent (farmer) decides whether to adopt
    a nature-inclusive practice based on economic and peer influence.
    Uses real units for emissions (t CO2-eq/ha/year) and monetary values (EUR/ha/year).
    
    Emission values based on empirical data from:
    https://doi.org/10.5194/bg-21-4099-2024
    - Conventional peatland agriculture: 3.77 tCO2-eq/ha/year
    - Water infiltration systems: 2.66 tCO2-eq/ha/year
    
    Risk aversion factor based on empirical data from:
    https://doi.org/10.1002/aepp.13330
    Rommel et al. 2022 - Loss aversion factor of 1.2 for Dutch agricultural sector
    """

    def __init__(self, n_agents=500, subsidy_eur_per_ha=100.0, seed=42, stay_adopter_prob=0.9, hetero_persistence=True, alpha=0.7, social_capital_factor=100, scaling_factor=100, initial_share_adopters=0.05, profits_csv='profits_agents.csv', risk_aversion_factor=1.2):
        self.n = n_agents  # Number of agents (farmers)
        self.subsidy_eur_per_ha = subsidy_eur_per_ha  # Subsidy paid to adopters (EUR/ha/year)

        # Data-driven assignment from CSV
        df = pd.read_csv(profits_csv)
        if len(df) < self.n:
            raise ValueError(f"CSV file must have at least {self.n} rows for agent profits.")

        # Store individual profit values as xarray DataArrays
        self.profit_conventional = xr.DataArray(
            df['profit_conventional_eur_per_ha'].values[:self.n], dims=["agent"])
        self.profit_nature_based = xr.DataArray(
            df['profit_nature_based_eur_per_ha'].values[:self.n], dims=["agent"])

        # Internal calculation of profit difference
        self.profit_diff_eur_per_ha = self.profit_conventional - self.profit_nature_based

        self.rng = np.random.default_rng(seed)  # Random generator for reproducibility
        self.stay_adopter_prob = stay_adopter_prob  # Probability to remain adopter if already adopted
        self.alpha = alpha  # Weight for economic vs social utility
        self.social_capital_factor = social_capital_factor  # Social pressure factor
        self.scaling_factor = scaling_factor  # Logistic scaling factor
        self.risk_aversion_factor = risk_aversion_factor  # Loss aversion factor from prospect theory

        # Always randomize stay_adopter_probs per agent
        self.stay_adopter_probs = xr.DataArray(
            self.rng.uniform(0.7, 0.99, size=self.n), dims=["agent"])

        # Randomize profit and peer weights for each agent for more heterogeneity
        # Profit weights: between 0.5 and 2.0 (higher = more profit-driven)
        self.profit_weights = xr.DataArray(
            self.rng.uniform(0.5, 2.0, size=self.n), dims=["agent"])
        # Peer weights: between 0.5 and 2.0 (higher = more peer-driven)
        self.peer_weights = xr.DataArray(
            self.rng.uniform(0.5, 2.0, size=self.n), dims=["agent"])

        # Initialize given share of farmers as adopters
        self.adopt = xr.DataArray(
            self.rng.binomial(1, initial_share_adopters, size=self.n), dims=["agent"])

    def step(self):
        """
        One time step: compute adoption decisions and return summary stats.
        """
        # Calculate average adoption in the population (as a peer proxy)
        peer_share = float(self.adopt.mean())

        # Calculate economic utility of adoption of nature-based practices using prospect theory
        # Profit change from adopting nature-based practices (subsidy + profit difference)
        profit_change = self.subsidy_eur_per_ha - self.profit_diff_eur_per_ha

        # Calculate economic utility using prospect theory
        # Step 1: Apply prospect theory to each agent's profit change
        prospect_values = []  # Will store the prospect theory value for each agent
        for i in range(self.n):
            agent_profit_change = float(profit_change.values[i])  # Get this agent's profit change
            agent_prospect_value = self.prospect_value(agent_profit_change)  # Apply prospect theory
            prospect_values.append(agent_prospect_value)

        # Step 2: Convert to xarray and multiply by profit weights
        prospect_values_array = xr.DataArray(prospect_values, dims=["agent"])
        econ_utility = self.profit_weights * prospect_values_array

        # Monetize social utility to make it comparable
        social_utility = self.social_capital_factor * self.peer_weights * peer_share

        # Combine with alpha (now weighting two monetary values)
        utility = self.alpha * econ_utility + (1 - self.alpha) * social_utility

        # Store agent-level utility and mean utility
        mean_utility = float(utility.mean())
        utility_per_agent = utility.values.copy()
        mean_econ_utility = float(econ_utility.mean())
        utility_econ_per_agent = econ_utility.values.copy()
        mean_social_utility = float(social_utility.mean())
        utility_social_per_agent = social_utility.values.copy()

        # Logistic transformation: maps utility (in EUR) to [0,1] probability
        # The scaling factor determines sensitivity to utility changes
        prob = 1 / (1 + np.exp(-utility.values / self.scaling_factor))

        # Each farmer adopts with probability 'prob'
        new_adopt = self.rng.binomial(1, prob)
        # Use xarray for agent state
        new_adopt_arr = self.adopt.copy()
        for i in range(self.n):
            if self.adopt.values[i] == 1 and self.rng.random() < self.stay_adopter_probs.values[i]:
                new_adopt_arr.values[i] = 1
            else:
                new_adopt_arr.values[i] = new_adopt[i]
        self.adopt = new_adopt_arr
        # Agent learning: update peer_weights based on observed peer adoption
        social_learning_rate = 0.1  # You can tune this value
        for i in range(self.n):
            # Move peer_weight slightly toward current peer_share (bounded between 0.5 and 2.0)
            self.peer_weights.values[i] += social_learning_rate * (peer_share - self.peer_weights.values[i])
            self.peer_weights.values[i] = np.clip(self.peer_weights.values[i], 0.5, 2.0)

        # Agent learning: update profit_weights based on economic experience
        econ_learning_rate = 0.1  # Tune as needed
        for i in range(self.n):
            # If agent adopted, update profit_weight toward economic utility (bounded between 0.5 and 2.0)
            if self.adopt.values[i] == 1:
                self.profit_weights.values[i] += econ_learning_rate * (econ_utility.values[i] - self.profit_weights.values[i])
                self.profit_weights.values[i] = np.clip(self.profit_weights.values[i], 0.5, 2.0)

        # Emissions: adopters emit less (t CO2-eq/ha/year)
        # Based on empirical data from https://doi.org/10.5194/bg-21-4099-2024
        # Conventional: 3.77 tCO2-eq/ha/year, Water infiltration: 2.66 tCO2-eq/ha/year
        conventional_emissions = 3.77
        nature_based_emissions = 2.66
        emis = conventional_emissions * (1 - self.adopt.values) + nature_based_emissions * self.adopt.values

        # New: compute policy cost & cost-effectiveness (per ha)
        baseline = conventional_emissions  # t CO2-eq/ha/year under conventional
        adoption_rate = float(np.mean(self.adopt))
        avg_emiss = float(np.mean(emis))
        emisssions_reduced = max(baseline - avg_emiss, 0.0) # t CO2-eq/ha/year reduced never negative

        policy_cost_per_ha = self.subsidy_eur_per_ha * adoption_rate  # EUR/ha/year

        # Set a threshold to avoid division by very small numbers
        threshold = 0.01
        if emisssions_reduced > threshold:
            cost_per_tonne  = policy_cost_per_ha / emisssions_reduced  # EUR per t CO2-eq reduced
        else:
            cost_per_tonne  = float('nan')  # Avoid division by zero or near-zero

        return {
            "adoption_rate": float(np.mean(self.adopt)),
            "avg_emissions_tCO2_ha": float(np.mean(emis)),
            "subsidy_eur_per_ha": self.subsidy_eur_per_ha,
            "profit_diff_eur_per_ha": self.profit_diff_eur_per_ha,
            "emissions_saved_tCO2_ha": emisssions_reduced,
            "policy_cost_eur_per_ha": policy_cost_per_ha,
            "cost_per_tonne_eur_per_tCO2": cost_per_tonne,
            "mean_utility": mean_utility,
            "utility_per_agent": utility_per_agent,  # numpy array
            "mean_econ_utility": mean_econ_utility,
            "utility_econ_per_agent": utility_econ_per_agent,
            "mean_social_utility": mean_social_utility,
            "utility_social_per_agent": utility_social_per_agent
        }

    def prospect_value(self, profit_change):
        """
        Apply prospect theory value function with loss aversion.
        
        Args:
            profit_change: Change in profit (EUR/ha/year) - can be positive or negative
            
        Returns:
            Value under prospect theory (dimensionless)
        """
        # Reference point is 0 (no change from current profit)
        if profit_change >= 0:
            # Gains: value = profit_change^0.88 (concave for gains)
            return profit_change ** 0.88
        else:
            # Losses: value = -self.risk_aversion_factor * |profit_change|^0.88 (convex for losses)
            return -self.risk_aversion_factor * (abs(profit_change) ** 0.88)

# Run the model over multiple time steps
def run_simulation(model, steps=50):
    """
    Run the ABM for a number of time steps and collect metrics.
    Returns a pandas DataFrame.
    """
    # Prepare storage for each variable
    adoption_rate = []
    avg_emissions_tCO2_ha = []
    subsidy_eur_per_ha = []
    emissions_saved_tCO2_ha = []
    policy_cost_eur_per_ha = []
    cost_per_tonne_eur_per_tCO2 = []
    mean_utility = []
    utility_per_agent = []
    mean_econ_utility = []
    utility_econ_per_agent = []
    mean_social_utility = []
    utility_social_per_agent = []

    for t in range(steps):
        result = model.step()
        adoption_rate.append(result["adoption_rate"])
        avg_emissions_tCO2_ha.append(result["avg_emissions_tCO2_ha"])
        subsidy_eur_per_ha.append(result["subsidy_eur_per_ha"])
        emissions_saved_tCO2_ha.append(result["emissions_saved_tCO2_ha"])
        policy_cost_eur_per_ha.append(result["policy_cost_eur_per_ha"])
        cost_per_tonne_eur_per_tCO2.append(result["cost_per_tonne_eur_per_tCO2"])
        mean_utility.append(result["mean_utility"])
        utility_per_agent.append(result["utility_per_agent"])
        mean_econ_utility.append(result["mean_econ_utility"])
        utility_econ_per_agent.append(result["utility_econ_per_agent"])
        mean_social_utility.append(result["mean_social_utility"])
        utility_social_per_agent.append(result["utility_social_per_agent"])

    # Convert lists to arrays
    steps_arr = np.arange(1, steps + 1)
    agent_dim = np.arange(model.n)

    ds = xr.Dataset({
        "adoption_rate": ("step", np.array(adoption_rate)),
        "avg_emissions_tCO2_ha": ("step", np.array(avg_emissions_tCO2_ha)),
        "subsidy_eur_per_ha": ("step", np.array(subsidy_eur_per_ha)),
        "profit_diff_eur_per_ha": ("agent", model.profit_diff_eur_per_ha.data),
        "emissions_saved_tCO2_ha": ("step", np.array(emissions_saved_tCO2_ha)),
        "policy_cost_eur_per_ha": ("step", np.array(policy_cost_eur_per_ha)),
        "cost_per_tonne_eur_per_tCO2": ("step", np.array(cost_per_tonne_eur_per_tCO2)),
        "mean_utility": ("step", np.array(mean_utility)),
        "utility_per_agent": (["step", "agent"], np.stack(utility_per_agent)),
        "mean_econ_utility": ("step", np.array(mean_econ_utility)),
        "utility_econ_per_agent": (["step", "agent"], np.stack(utility_econ_per_agent)),
        "mean_social_utility": ("step", np.array(mean_social_utility)),
        "utility_social_per_agent": (["step", "agent"], np.stack(utility_social_per_agent)),
    }, coords={"step": steps_arr, "agent": agent_dim})
    return ds

# Monte Carlo simulation: run ABM multiple times with different seeds
def monte_carlo_runs(n_runs, steps=50, seed_base=42, **model_params):
    """
    Run the ABM n_runs times, each with a different random seed.
    Returns a list of xarray Datasets (one per run).
    """
    results_list = []
    for i in range(n_runs):
        model = PeatlandABM(seed=seed_base + i, **model_params)
        ds = run_simulation(model, steps)
        results_list.append(ds)
    return results_list
