# ---------------------------------------------
# model.py — Core agent-based model logic
# ---------------------------------------------

import numpy as np
import pandas as pd

# Define ABM class
class PeatlandABM:
    """
    A minimal agent-based model where each agent (farmer) decides whether to adopt
    a nature-inclusive practice based on economic and peer influence.
    """

    def __init__(self, n_agents=100, subsidy=1.0, profit_diff=0.5, peer_weight=0.3, seed=42):
        self.n = n_agents                     # Number of agents (farmers)
        self.subsidy = subsidy               # Subsidy paid to adopters
        self.profit_diff = profit_diff       # Profit advantage of conventional farming
        self.peer_weight = peer_weight       # Importance of neighbors' choices
        self.rng = np.random.default_rng(seed)  # Random generator for reproducibility

        # Initialize 10% of farmers as adopters
        self.adopt = self.rng.binomial(1, 0.1, size=self.n)

    def step(self):
        """
        One time step: compute adoption decisions and return summary stats.
        """
        # Calculate average adoption in the population (as a peer proxy)
        peer_share = np.mean(self.adopt)

        # Calculate adoption utility:
        # If utility > 0, more likely to adopt
        utility = self.subsidy - self.profit_diff + self.peer_weight * peer_share

        # Logistic transformation: maps utility to [0,1] probability
        prob = 1 / (1 + np.exp(-utility))

        # Each farmer adopts with probability 'prob'
        self.adopt = self.rng.binomial(1, prob, size=self.n)

        # Emissions: adopters emit less
        emis = 5.0 * (1 - 0.5 * self.adopt)  # baseline 5.0, reduced by 50% for adopters

        # Return stats
        return {
            "adoption_rate": float(np.mean(self.adopt)),
            "avg_emissions": float(np.mean(emis))
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
