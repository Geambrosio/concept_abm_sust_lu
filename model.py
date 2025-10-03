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

    def __init__(self, n_agents=100, subsidy_eur_per_ha=100.0, seed=42, stay_adopter_prob=0.9, hetero_persistence=True, alpha=0.7, profits_csv='profits_agents.csv'):
        self.n = n_agents  # Number of agents (farmers)
        self.subsidy_eur_per_ha = subsidy_eur_per_ha  # Subsidy paid to adopters (EUR/ha/year)
        
        # Data-driven assignment from CSV
        df = pd.read_csv(profits_csv)
        if len(df) < self.n:
            raise ValueError(f"CSV file must have at least {self.n} rows for agent profits.")
        
        # Store individual profit values
        self.profit_conventional = df['profit_conventional_eur_per_ha'].values[:self.n]
        self.profit_nature_based = df['profit_nature_based_eur_per_ha'].values[:self.n]

        # Internal calculation of profit difference
        self.profit_diff_eur_per_ha = self.profit_conventional - self.profit_nature_based
        
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

        # Calculate economic utility (now in real EUR/ha/year, no normalization)
        econ_utility = self.subsidy_eur_per_ha - self.profit_weights * self.profit_diff_eur_per_ha

        # Monetize social utility to make it comparable
        social_capital_factor = 500  # EUR/ha, represents the max value of social pressure
        social_utility = social_capital_factor * self.peer_weights * peer_share

        # Combine with alpha (now weighting two monetary values)
        utility = self.alpha * econ_utility + (1 - self.alpha) * social_utility

        # Store agent-level utility and mean utility
        mean_utility = float(np.mean(utility))
        utility_per_agent = utility.copy()
        mean_econ_utility = float(np.mean(econ_utility))
        utility_econ_per_agent = econ_utility.copy()
        mean_social_utility = float(np.mean(social_utility))
        utility_social_per_agent = social_utility.copy()

        # Logistic transformation: maps utility (in EUR) to [0,1] probability
        # The scaling factor (e.g., 100) determines sensitivity to utility changes
        prob = 1 / (1 + np.exp(-utility / 100.0))

        # Each farmer adopts with probability 'prob'
        new_adopt = self.rng.binomial(1, prob)
        for i in range(self.n):
            if self.adopt[i] == 1 and self.rng.random() < self.stay_adopter_probs[i]:
                new_adopt[i] = 1
        self.adopt = new_adopt
        # Agent learning: update peer_weights based on observed peer adoption
        social_learning_rate = 0.1  # You can tune this value
        for i in range(self.n):
            # Move peer_weight slightly toward current peer_share (bounded between 0.5 and 2.0)
            self.peer_weights[i] += social_learning_rate * (peer_share - self.peer_weights[i])
            self.peer_weights[i] = np.clip(self.peer_weights[i], 0.5, 2.0)

        # Agent learning: update profit_weights based on economic experience
        econ_learning_rate = 0.1  # Tune as needed
        for i in range(self.n):
            # If agent adopted, update profit_weight toward economic utility (bounded between 0.5 and 2.0)
            if self.adopt[i] == 1:
                self.profit_weights[i] += econ_learning_rate * (econ_utility[i] - self.profit_weights[i])
                self.profit_weights[i] = np.clip(self.profit_weights[i], 0.5, 2.0)

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
            "mean_utility": mean_utility,
            "utility_per_agent": utility_per_agent,  # numpy array
            "mean_econ_utility": mean_econ_utility,
            "utility_econ_per_agent": utility_econ_per_agent,
            "mean_social_utility": mean_social_utility,
            "utility_social_per_agent": utility_social_per_agent
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
