import numpy as np
import pandas as pd

# SETTINGS: Change these to generate new sets of agent profit differences
N_AGENTS = 100  # Number of agents/farmers
PROFIT_DIFF_MEAN = 50.0  # Mean profit difference = conventional - nature-inclusive (EUR/ha/year)
PROFIT_DIFF_STD = 30.0   # Standard deviation (EUR/ha/year)
PROFIT_DIFF_MIN = -50.0  # Minimum allowed value
PROFIT_DIFF_MAX = 150.0  # Maximum allowed value
SEED = 42                # Random seed for reproducibility

# Generate profit differences
rng = np.random.default_rng(SEED)
profit_diff = rng.normal(PROFIT_DIFF_MEAN, PROFIT_DIFF_STD, N_AGENTS)
profit_diff = np.clip(profit_diff, PROFIT_DIFF_MIN, PROFIT_DIFF_MAX)

# Create DataFrame
df = pd.DataFrame({'profit_diff_eur_per_ha': profit_diff})

# Save to CSV in main project folder
csv_path = 'profit_diff_agents.csv'
df.to_csv(csv_path, index=False)
print(f"Saved agent profit differences to {csv_path}")
print(df.head())
