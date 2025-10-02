# ---------------------------------------------
# model.py — Core agent-based model logic
# ---------------------------------------------

import numpy as np
import pandas as pd

# Define ABM class

class PeatlandABM:
    """
    Agent-based model where each agent (farmer) decides whether to adopt
    a nature-inclusive practice based on economic and peer influence.
    Uses real units for emissions (t CO2-eq/ha/year) and monetary values (EUR/ha/year).
    """

    def __init__(self, n_agents=100, subsidy_eur_per_ha=100.0, profit_diff_eur_per_ha=50.0, peer_weight=0.3, seed=42, stay_adopter_prob=0.9, hetero_persistence=True, alpha=0.7):
        self.n = n_agents  # Number of agents (farmers)
        self.subsidy_eur_per_ha = subsidy_eur_per_ha  # Subsidy paid to adopters (EUR/ha/year)
        self.profit_diff_eur_per_ha = profit_diff_eur_per_ha  # Profit advantage of conventional (EUR/ha/year)
        self.peer_weight = peer_weight  # Importance of neighbors' choices
        self.rng = np.random.default_rng(seed)  # Random generator for reproducibility
        self.stay_adopter_prob = stay_adopter_prob  # Probability to remain adopter if already adopted
        self.alpha = alpha  # Weight for economic vs social utility

        if hetero_persistence:
            self.stay_adopter_probs = self.rng.uniform(0.7, 0.99, size=self.n)
        else:
            self.stay_adopter_probs = np.full(self.n, stay_adopter_prob)

        # Randomize profit and peer weights for each agent for more heterogeneity
        # Profit weights: between 0.5 and 2.0 (higher = more profit-driven)
        self.profit_weights = self.rng.uniform(0.5, 2.0, size=self.n)
        # Peer weights: between 0.5 and 2.0 (higher = more peer-driven)
        self.peer_weights = self.rng.uniform(0.5, 2.0, size=self.n)

        # Initialize 5% of farmers as adopters
        self.adopt = self.rng.binomial(1, 0.05, size=self.n)

    def step(self):
        """
        One time step: compute adoption decisions and return summary stats.
        """
        # Calculate average adoption in the population (as a peer proxy)
        peer_share = np.mean(self.adopt)

        # Calculate raw economic utility (EUR/ha/year)
        econ_utility_raw = self.subsidy_eur_per_ha - self.profit_weights * self.profit_diff_eur_per_ha
        # Normalize economic utility to [0, 1]
        econ_min = np.min(econ_utility_raw)
        econ_max = np.max(econ_utility_raw)
        econ_utility = (econ_utility_raw - econ_min) / (econ_max - econ_min + 1e-8)

        # Calculate raw social utility (unitless)
        social_utility_raw = self.peer_weights * peer_share
        # Normalize social utility to [0, 1]
        social_min = np.min(social_utility_raw)
        social_max = np.max(social_utility_raw)
        social_utility = (social_utility_raw - social_min) / (social_max - social_min + 1e-8)

        # Combine with alpha
        utility = self.alpha * econ_utility + (1 - self.alpha) * social_utility

        # Logistic transformation: maps utility to [0,1] probability
        prob = 1 / (1 + np.exp(-utility / 100.0))  # scale utility for probability

        # Each farmer adopts with probability 'prob'
        new_adopt = self.rng.binomial(1, prob)
        for i in range(self.n):
            if self.adopt[i] == 1 and self.rng.random() < self.stay_adopter_probs[i]:
                new_adopt[i] = 1
        self.adopt = new_adopt

        # Emissions: adopters emit less (t CO2-eq/ha/year)
        emis = 5.0 * (1 - 0.5 * self.adopt)  # non-adopter: 5, adopter: 2.5

        # New: compute policy cost & cost-effectiveness (per ha)
        baseline = 5.0  # t CO2-eq/ha/year under conventional
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
        }

# Run the model over multiple time steps
def run_simulation(model, steps=50):
    """
    Run the ABM for a number of time steps and collect metrics.
    Returns a pandas DataFrame.
    """
    rows = []
    for t in range(steps):
        result = model.step()
        result["step"] = t + 1
        rows.append(result)
    return pd.DataFrame(rows)
