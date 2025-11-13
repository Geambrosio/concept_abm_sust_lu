# -----------------------------------------------------------------
# make_profits_csv.py - Generates agent-level profit data
# -----------------------------------------------------------------

import pandas as pd
import numpy as np

def generate_profit_data(
    n_agents=100,
    mean_conv=1500,
    std_conv=200,
    mean_nat=1450,
    std_nat=250,
    seed=42,
    filename="profits_agents.csv",
):
    """Generate a CSV with profit and behavioural inputs for each agent.

    The output aligns with the optional arrays accepted by ``PeatlandABM`` so
    the model can be driven entirely from the CSV without relying on internal
    random draws.

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

    # Generate behavioural and capability inputs to mirror model defaults
    personal_values = np.clip(rng.normal(0.0, 0.4, size=n_agents), -1.0, 1.0)
    social_weights = np.clip(rng.uniform(0.5, 2.0, size=n_agents), 0.5, 2.0)
    profit_weights = np.clip(rng.uniform(0.5, 2.0, size=n_agents), 0.5, 2.0)
    stay_probs = np.clip(rng.uniform(0.7, 0.99, size=n_agents), 0.7, 0.99)
    initial_adopt = rng.binomial(1, 0.05, size=n_agents)
    self_belief = np.clip(rng.beta(2.0, 2.0, size=n_agents), 0.0, 1.0)
    capability = np.clip(rng.uniform(0.6, 0.95, size=n_agents), 0.0, 1.0)
    opportunity = np.clip(rng.uniform(0.5, 0.95, size=n_agents), 0.0, 1.0)

    # Generate land-related attributes
    area_ha = rng.uniform(1.0, 10.0, size=n_agents)  # Area in hectares
    carbon_stock = rng.normal(50.0, 10.0, size=n_agents)  # Carbon stock in t/ha
    carbon_stock = np.clip(carbon_stock, 0.0, 100.0)  # Clip to realistic range
    fauna_abundance = rng.uniform(0.0, 1.0, size=n_agents)  # Fauna abundance index

    # Create DataFrame
    df = pd.DataFrame(
        {
            "farmer_id": range(n_agents),
            "profit_conventional_eur_per_ha": profit_conventional,
            "profit_nature_based_eur_per_ha": profit_nature_based,
            "environmental_value": personal_values,
            "social_weights": social_weights,
            "profit_weights": profit_weights,
            "stay_probs": stay_probs,
            "initial_adopt": initial_adopt,
            "self_belief": self_belief,
            "capability": capability,
            "opportunity": opportunity,
            "area_ha": area_ha,
            "carbon_stock": carbon_stock,
            "fauna_abundance": fauna_abundance,
        }
    )

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Successfully generated '{filename}' with {n_agents} agents.")
    print("\nData preview:")
    print(df.head())
    print(f"\nMean Conventional Profit: {df['profit_conventional_eur_per_ha'].mean():.2f} EUR/ha")
    print(f"Mean Nature-Based Profit: {df['profit_nature_based_eur_per_ha'].mean():.2f} EUR/ha")
    mean_diff = (df['profit_conventional_eur_per_ha'] - df['profit_nature_based_eur_per_ha']).mean()
    print(f"Mean Profit Difference (Conventional - Nature-Based): {mean_diff:.2f} EUR/ha")
    print(f"Mean Area: {df['area_ha'].mean():.2f} ha")
    print(f"Mean Carbon Stock: {df['carbon_stock'].mean():.2f} t/ha")
    print(f"Mean Fauna Abundance: {df['fauna_abundance'].mean():.2f}")


def generate_consumer_data(
    n_consumers=1000,
    health_mean=0.2,
    health_std=0.35,
    price_mean=0.4,
    price_std=0.25,
    eco_mean=0.3,
    eco_std=0.35,
    eco_weight_mean=1.0,
    eco_weight_std=0.3,
    seed=84,
    filename="consumers.csv",
):
    """Generate a CSV with consumer preferences aligned with the Consumer dataclass."""

    rng = np.random.default_rng(seed)
    health_pref = np.clip(rng.normal(health_mean, health_std, size=n_consumers), -1.0, 1.0)
    price_sensitivity = np.clip(rng.normal(price_mean, price_std, size=n_consumers), 0.0, 2.0)
    eco_pref = np.clip(rng.normal(eco_mean, eco_std, size=n_consumers), -1.0, 1.0)
    eco_weight = np.clip(rng.normal(eco_weight_mean, eco_weight_std, size=n_consumers), 0.2, 2.0)
    current_choice = np.full(n_consumers, "conventional", dtype=object)
    rewetted_count = max(1, int(round(0.05 * n_consumers)))
    rewetted_indices = rng.choice(n_consumers, size=rewetted_count, replace=False)
    current_choice[rewetted_indices] = "rewetted"

    df = pd.DataFrame(
        {
            "household_id": range(n_consumers),
            "health_pref": health_pref,
            "price_sensitivity": price_sensitivity,
            "eco_pref": eco_pref,
            "eco_weight": eco_weight,
            "current_choice": current_choice,
        }
    )

    df.to_csv(filename, index=False)
    print(f"Successfully generated '{filename}' with {n_consumers} consumers.")
    print("\nConsumer data preview:")
    print(df.head())
    print(f"\nMean Health Preference: {df['health_pref'].mean():.2f}")
    print(f"Mean Price Sensitivity: {df['price_sensitivity'].mean():.2f}")
    print(f"Mean Eco Preference: {df['eco_pref'].mean():.2f}")
    print(f"Mean Eco Weight: {df['eco_weight'].mean():.2f}")
    rewetted_share = (df['current_choice'] == 'rewetted').mean()
    print(f"Initial Rewetted Share: {rewetted_share:.2%}")


if __name__ == "__main__":
    generate_profit_data(n_agents=100)
    generate_consumer_data(n_consumers=1000)
