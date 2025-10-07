# -----------------------------------------------------------------
# make_profits_csv.py - Generates agent-level profit data
# -----------------------------------------------------------------

import pandas as pd
import numpy as np

def generate_profit_data(n_agents=500, mean_conv=1500, std_conv=200, mean_nat=1450, std_nat=250, seed=42, filename="profits_agents.csv"):
    """
    Generates a CSV file with randomized profit values for conventional and
    nature-based agriculture for each agent.

    Args:
        n_agents (int): Number of agents.
        mean_conv (float): Mean profit for conventional ag (EUR/ha).
        std_conv (float): Standard deviation for conventional ag profit.
        mean_nat (float): Mean profit for nature-based ag (EUR/ha).
        std_nat (float): Standard deviation for nature-based ag profit.
        seed (int): Random seed for reproducibility.
        filename (str): Name of the output CSV file.
    """
    rng = np.random.default_rng(seed)

    # Generate profit for conventional agriculture
    profit_conventional = rng.normal(loc=mean_conv, scale=std_conv, size=n_agents)
    profit_conventional = np.clip(profit_conventional, 1000, 2000)  # Clip to a realistic range

    # Generate profit for nature-based agriculture
    profit_nature_based = rng.normal(loc=mean_nat, scale=std_nat, size=n_agents)
    profit_nature_based = np.clip(profit_nature_based, 900, 1900)  # Clip to a realistic range

    # Create DataFrame
    df = pd.DataFrame({
        'agent_id': range(n_agents),
        'profit_conventional_eur_per_ha': profit_conventional,
        'profit_nature_based_eur_per_ha': profit_nature_based
    })

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Successfully generated '{filename}' with {n_agents} agents.")
    print("\nData preview:")
    print(df.head())
    print(f"\nMean Conventional Profit: {df['profit_conventional_eur_per_ha'].mean():.2f} EUR/ha")
    print(f"Mean Nature-Based Profit: {df['profit_nature_based_eur_per_ha'].mean():.2f} EUR/ha")
    mean_diff = (df['profit_conventional_eur_per_ha'] - df['profit_nature_based_eur_per_ha']).mean()
    print(f"Mean Profit Difference (Conventional - Nature-Based): {mean_diff:.2f} EUR/ha")


if __name__ == "__main__":
    generate_profit_data(n_agents=500)
