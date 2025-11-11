"""Peatland ABM with intention/adoption architecture.

This module separates the behavioural logic of the peatland adoption model into
coherent components so that its clearly defined how decisions are formed.
Economic parameters follow Rommel et al. (2022)
(https://doi.org/10.1002/aepp.13330) and emission factors follow van Leeuwen
et al. (2024) (https://doi.org/10.5194/bg-21-4099-2024).
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr


def _norm01(arr) -> xr.DataArray:
    """Return a 0-1 normalized copy of ``arr`` while preserving coordinates.
    Used to normalize the values of economic utility (econ), social utility (social),
    capability, and opportunity so that these components are scaled to the [0, 1]"""
    data = np.asarray(arr, dtype=float)
    span = data.max() - data.min()
    if span < 1e-8:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - data.min()) / span
    if isinstance(arr, xr.DataArray):
        return xr.DataArray(normalized, dims=arr.dims, coords=arr.coords)
    else:
        return xr.DataArray(normalized, dims=["agent"])


def _sigmoid_scalar(value: float, steepness: float) -> float:
    """Scaled sigmoid used to map latent intention scores to probabilities.
    If steepness  = 1.0, the sigmoid is standard.
    If steepness  = 0.5, agents with slightly positive intention are much more likely to adopt.
    If steepness  = 2.0, even agents with high intention only slowly approach probability 1."""
    return 1.0 / (1.0 + np.exp(-steepness * value))


@dataclass
class AgentState:
    """Container for agent-level attributes kept as :class:`xarray.DataArray`.

    Keeping these grouped improves readability and makes it clear which inputs
    feed the behavioural model (values, beliefs, capability) versus economic
    parameters (profits, peer weights).
    """

    expected_profit_conv: xr.DataArray
    expected_profit_nat: xr.DataArray
    expected_profit_diff: xr.DataArray
    profit_weights: xr.DataArray
    social_weights: xr.DataArray
    environmental_value: xr.DataArray
    self_belief: xr.DataArray
    capability: xr.DataArray
    opportunity: xr.DataArray
    adopt: xr.DataArray
    stay_probs: xr.DataArray


@dataclass
class IntentionConfig:
    """Configuration for the intention formation engine.

    These parameters determine how behavioural components are combined into the
    latent intention score derived from the choice experiment evidence."""

    weights: Dict[str, float]
    intercept: float
    steepness: float


@dataclass
class IntentionBundle:
    """Outputs of intention formation used for diagnostics and adoption."""

    econ: xr.DataArray
    social: xr.DataArray
    personal: xr.DataArray
    selfeff: xr.DataArray
    econ_norm: xr.DataArray
    social_norm: xr.DataArray
    latent: xr.DataArray
    intention_prob: xr.DataArray
    expected_profit_change: xr.DataArray

    def as_dict(self) -> Dict[str, xr.DataArray]:
        """Return a plain dictionary for backwards-compatible access."""

        return {
            "econ": self.econ,
            "social": self.social,
            "personal": self.personal,
            "selfeff": self.selfeff,
            "econ_n": self.econ_norm,
            "social_n": self.social_norm,
            "latent": self.latent,
            "intention_prob": self.intention_prob,
            "expected_profit_change": self.expected_profit_change,
        }


class IntentionEngine:
    """Stage A: compute intention probabilities from behavioural components.

    The engine blends economic advantage (Δπᵢ), social norms, personal values,
    and self-belief into a latent intention score. The score
    is mapped to [0, 1] with a sigmoid so that it can be read as a probability.
    """

    def __init__(self, prospect_fn, config: IntentionConfig):
        self._prospect_fn = prospect_fn
        self._config = config

    def compute(self, state: AgentState, peer_share: float, subsidy: float) -> IntentionBundle:
        """Return intention components for all agents given the current context."""

        expected_profit_change = subsidy - state.expected_profit_diff
        prospect_values = xr.apply_ufunc(
            lambda val: self._prospect_fn(float(val)),
            expected_profit_change,
            vectorize=True,
        )
        econ = state.profit_weights * prospect_values
        social = state.social_weights * peer_share

        environmental = (state.environmental_value + 1.0) * 0.5
        selfeff = state.self_belief

        econ_norm = _norm01(econ)
        social_norm = _norm01(social)

        weights = self._config.weights
        latent = (
            weights["econ"] * econ_norm
            + weights["social"] * social_norm
            + weights["personal"] * environmental
            + weights["self"] * selfeff
        ) + self._config.intercept

        intention_prob = xr.apply_ufunc(
            lambda val: _sigmoid_scalar(val, self._config.steepness),
            latent,
            vectorize=True,
        )

        return IntentionBundle(
            econ=econ,
            social=social,
            personal=environmental,
            selfeff=selfeff,
            econ_norm=econ_norm,
            social_norm=social_norm,
            latent=latent,
            intention_prob=intention_prob,
            expected_profit_change=expected_profit_change,
        )


def apply_adoption(
    intention_prob: xr.DataArray,
    capability: xr.DataArray,
    opportunity: xr.DataArray,
    current_adopt: xr.DataArray,
    stay_probs: xr.DataArray,
    rng: np.random.Generator,
) -> Dict[str, xr.DataArray]:
    """Stage B: translate intention into adoption with friction."""

    capability_norm = _norm01(capability)
    opportunity_norm = _norm01(opportunity)
    adoption_prob = intention_prob * capability_norm * opportunity_norm
    adoption_prob = xr.DataArray(
        np.clip(adoption_prob.values, 0.0, 1.0),
        dims=adoption_prob.dims,
        coords=adoption_prob.coords,
    )

    draws = rng.binomial(1, adoption_prob.values)
    updated = current_adopt.copy()
    for idx in range(updated.sizes["agent"]):
        if current_adopt.values[idx] == 1 and rng.random() < stay_probs.values[idx]:
            updated.values[idx] = 1
        else:
            updated.values[idx] = draws[idx]

    return {"adoption_prob": adoption_prob, "updated_adopt": updated}


class PeatlandABM:
    """Peatland adoption model with explicit intention and realization stages.

    - Economic signals use observed profits per hectare and a prospect theory
      value function with loss aversion set to 1.2 (Rommel et al., 2022).
    - Emissions for conventional vs. water infiltration practices follow van
      Leeuwen et al. (2024) so that mitigation impacts stay in physical units.
    - Intention and adoption follow a COM-B inspired split: intention is shaped
      by motivation (economic, social, personal) and beliefs, while capability,
      opportunity, and inertia govern actual behaviour.
    """

    def __init__(
        self,
        n_agents: int = 500,
        subsidy_eur_per_ha: float = 100.0,
        seed: int = 42,
        stay_adopter_prob: float = 0.9,
        hetero_persistence: bool = True,
        profits_csv: str = "profits_agents.csv",
        risk_aversion_factor: float = 1.2,
        environmental_value=None,
        social_weights=None,
        intention_weights: Dict[str, float] | None = None,
        intention_intercept: float = 0.0,
        intention_steepness : float = 1.0,
        social_learning_rate: float = 0.1,
        econ_learning_rate: float = 0.1,
    ) -> None:
        self.n = n_agents
        self.subsidy_eur_per_ha = subsidy_eur_per_ha
        self.risk_aversion_factor = risk_aversion_factor
        self.rng = np.random.default_rng(seed)

        df = pd.read_csv(profits_csv)
        if len(df) < self.n:
            raise ValueError(f"CSV file must have at least {self.n} rows for agent profits.")

        expected_profit_conv = xr.DataArray(
            df["profit_conventional_eur_per_ha"].values[: self.n], dims=["agent"]
        )
        expected_profit_nat = xr.DataArray(
            df["profit_nature_based_eur_per_ha"].values[: self.n], dims=["agent"]
        )
        expected_profit_diff = expected_profit_conv - expected_profit_nat

        if hetero_persistence:
            stay_probs = df["stay_probs"].values[: self.n]
        else:
            stay_probs = np.full(self.n, stay_adopter_prob)
        stay_probs_da = xr.DataArray(np.clip(np.asarray(stay_probs), 0.7, 0.99), dims=["agent"])

        profit_weights = df["profit_weights"].values[: self.n]
        profit_weights_da = xr.DataArray(np.clip(np.asarray(profit_weights), 0.5, 2.0), dims=["agent"])

        if social_weights is None:
            social_weights = df["social_weights"].values[: self.n]
        social_weights_da = xr.DataArray(np.clip(np.asarray(social_weights), 0.5, 2.0), dims=["agent"])

        if environmental_value is None:
            environmental_value = df["environmental_value"].values[: self.n]
        environmental_da = xr.DataArray(np.clip(np.asarray(environmental_value), -1.0, 1.0), dims=["agent"])

        self_belief = df["self_belief"].values[: self.n]
        self_belief_da = xr.DataArray(np.clip(np.asarray(self_belief), 0.0, 1.0), dims=["agent"])

        capability = df["capability"].values[: self.n]
        capability_da = xr.DataArray(np.clip(np.asarray(capability), 0.0, 1.0), dims=["agent"])

        opportunity = df["opportunity"].values[: self.n]
        opportunity_da = xr.DataArray(np.clip(np.asarray(opportunity), 0.0, 1.0), dims=["agent"])

        adopt = xr.DataArray(df["initial_adopt"].values[: self.n], dims=["agent"])

        self.state = AgentState(
            expected_profit_conv=expected_profit_conv,
            expected_profit_nat=expected_profit_nat,
            expected_profit_diff=expected_profit_diff,
            profit_weights=profit_weights_da,
            social_weights=social_weights_da,
            environmental_value=environmental_da,
            self_belief=self_belief_da,
            capability=capability_da,
            opportunity=opportunity_da,
            adopt=adopt,
            stay_probs=stay_probs_da,
        )

        default_weights = {"econ": 1.0, "social": 1.0, "personal": 0.6, "self": 0.6}
        if intention_weights is None:
            weights = default_weights
        else:
            overrides = {k: v for k, v in dict(intention_weights).items() if k in default_weights}
            weights = {**default_weights, **overrides}

        self.intention_config = IntentionConfig(
            weights=weights,
            intercept=float(intention_intercept),
            steepness=float(intention_steepness),
        )
        self.engine = IntentionEngine(self.prospect_value, self.intention_config)

        self.social_learning_rate = social_learning_rate
        self.econ_learning_rate = econ_learning_rate

        # Preserve legacy attributes used elsewhere in the code base
        self.expected_profit_diff_eur_per_ha = self.state.expected_profit_diff

    def step(self) -> Dict[str, object]:
        """Advance the simulation by one period and return diagnostics."""

        peer_share = float(self.state.adopt.mean())
        bundle = self.engine.compute(self.state, peer_share, self.subsidy_eur_per_ha)
        adoption_out = apply_adoption(
            bundle.intention_prob,
            self.state.capability,
            self.state.opportunity,
            self.state.adopt,
            self.state.stay_probs,
            self.rng,
        )
        self.state.adopt = adoption_out["updated_adopt"]

        mean_utility = float(bundle.latent.mean())
        mean_econ_utility = float(bundle.econ_norm.mean())
        mean_social_utility = float(bundle.social_norm.mean())

        # Update social weights based on new peer share
        new_peer_share = float(self.state.adopt.mean())
        for idx in range(self.state.social_weights.sizes["agent"]):
            self.state.social_weights.values[idx] += self.social_learning_rate * (
                new_peer_share - self.state.social_weights.values[idx]
            )
            self.state.social_weights.values[idx] = np.clip(self.state.social_weights.values[idx], 0.5, 2.0)

        econ_learning_rate = self.econ_learning_rate
        for idx in range(self.state.profit_weights.sizes["agent"]):
            if self.state.adopt.values[idx] == 1:
                self.state.profit_weights.values[idx] += econ_learning_rate * (
                    bundle.econ.values[idx] - self.state.profit_weights.values[idx]
                )
                self.state.profit_weights.values[idx] = np.clip(self.state.profit_weights.values[idx], 0.5, 2.0)

        conventional_emissions = 3.77
        nature_based_emissions = 2.66
        emis = conventional_emissions * (1 - self.state.adopt.values) + nature_based_emissions * self.state.adopt.values

        baseline = conventional_emissions
        adoption_rate = float(np.mean(self.state.adopt.values))
        avg_emiss = float(np.mean(emis))
        emissions_reduced = max(baseline - avg_emiss, 0.0)
        policy_cost_per_ha = self.subsidy_eur_per_ha * adoption_rate
        threshold = 0.01
        if emissions_reduced > threshold:
            cost_per_tonne = policy_cost_per_ha / emissions_reduced
        else:
            cost_per_tonne = float("nan")

        intention_prob = bundle.intention_prob
        intention_rate = float(intention_prob.mean())
        adoption_prob = adoption_out["adoption_prob"]

        result = {
            "adoption_rate": adoption_rate,
            "avg_emissions_tCO2_ha": avg_emiss,
            "subsidy_eur_per_ha": self.subsidy_eur_per_ha,
            "expected_profit_diff_eur_per_ha": self.state.expected_profit_diff.copy(),
            "emissions_saved_tCO2_ha": emissions_reduced,
            "policy_cost_eur_per_ha": policy_cost_per_ha,
            "cost_per_tonne_eur_per_tCO2": cost_per_tonne,
            "mean_utility": mean_utility,
            "utility_per_agent": bundle.latent.copy(),
            "mean_econ_utility": mean_econ_utility,
            "utility_econ_per_agent": bundle.econ_norm.copy(),
            "mean_social_utility": mean_social_utility,
            "utility_social_per_agent": bundle.social_norm.copy(),
            "intention_prob": intention_prob.copy(),
            "intention_rate": intention_rate,
            "adoption_prob_per_agent": adoption_prob.copy(),
            "intention_prob_per_agent": intention_prob.copy(),
            "capability_per_agent": self.state.capability.copy(),
            "opportunity_per_agent": self.state.opportunity.copy(),
            "environmental_value_per_agent": self.state.environmental_value.copy(),
            "social_weights_per_agent": self.state.social_weights.copy(),
            "self_belief_per_agent": self.state.self_belief.copy(),
            "expected_profit_change_eur_per_ha": bundle.expected_profit_change.copy(),
            "intention_components": bundle.as_dict(),
        }

        return result

    def prospect_value(self, profit_change: float) -> float:
        """Prospect theory value function with calibrated loss aversion."""

        if profit_change >= 0:
            return profit_change ** 0.88
        return -self.risk_aversion_factor * (abs(profit_change) ** 0.88)


def run_simulation(model: PeatlandABM, steps: int = 50) -> xr.Dataset:
    """Run the ABM for ``steps`` periods and assemble diagnostics.

    The dataset includes adoption dynamics, emissions, policy cost indicators,
    and the behaviourally rich intention/adoption signals for each agent.
    """

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
    intention_rate = []
    intention_prob_per_agent = []
    adoption_prob_per_agent = []
    capability_per_agent = []
    opportunity_per_agent = []
    environmental_value_per_agent = []
    social_weights_per_agent = []
    self_belief_per_agent = []
    expected_profit_change_eur_per_ha = []

    for _ in range(steps):
        result = model.step()
        adoption_rate.append(result["adoption_rate"])
        avg_emissions_tCO2_ha.append(result["avg_emissions_tCO2_ha"])
        subsidy_eur_per_ha.append(result["subsidy_eur_per_ha"])
        emissions_saved_tCO2_ha.append(result["emissions_saved_tCO2_ha"])
        policy_cost_eur_per_ha.append(result["policy_cost_eur_per_ha"])
        cost_per_tonne_eur_per_tCO2.append(result["cost_per_tonne_eur_per_tCO2"])
        mean_utility.append(result["mean_utility"])
        utility_per_agent.append(result["utility_per_agent"].values)
        mean_econ_utility.append(result["mean_econ_utility"])
        utility_econ_per_agent.append(result["utility_econ_per_agent"].values)
        mean_social_utility.append(result["mean_social_utility"])
        utility_social_per_agent.append(result["utility_social_per_agent"].values)
        intention_rate.append(result["intention_rate"])
        intention_prob_per_agent.append(result["intention_prob_per_agent"].values)
        adoption_prob_per_agent.append(result["adoption_prob_per_agent"].values)
        capability_per_agent.append(result["capability_per_agent"].values)
        opportunity_per_agent.append(result["opportunity_per_agent"].values)
        environmental_value_per_agent.append(result["environmental_value_per_agent"].values)
        social_weights_per_agent.append(result["social_weights_per_agent"].values)
        self_belief_per_agent.append(result["self_belief_per_agent"].values)
        expected_profit_change_eur_per_ha.append(result["expected_profit_change_eur_per_ha"].values)

    steps_arr = np.arange(1, steps + 1)
    agent_dim = np.arange(model.n)

    ds = xr.Dataset(
        {
            "adoption_rate": ("step", np.array(adoption_rate)),
            "avg_emissions_tCO2_ha": ("step", np.array(avg_emissions_tCO2_ha)),
            "subsidy_eur_per_ha": ("step", np.array(subsidy_eur_per_ha)),
            "expected_profit_diff_eur_per_ha": ("agent", model.expected_profit_diff_eur_per_ha.values),
            "emissions_saved_tCO2_ha": ("step", np.array(emissions_saved_tCO2_ha)),
            "policy_cost_eur_per_ha": ("step", np.array(policy_cost_eur_per_ha)),
            "cost_per_tonne_eur_per_tCO2": ("step", np.array(cost_per_tonne_eur_per_tCO2)),
            "mean_utility": ("step", np.array(mean_utility)),
            "utility_per_agent": (["step", "agent"], np.stack(utility_per_agent)),
            "mean_econ_utility": ("step", np.array(mean_econ_utility)),
            "utility_econ_per_agent": (["step", "agent"], np.stack(utility_econ_per_agent)),
            "mean_social_utility": ("step", np.array(mean_social_utility)),
            "utility_social_per_agent": (["step", "agent"], np.stack(utility_social_per_agent)),
            "intention_rate": ("step", np.array(intention_rate)),
            "intention_prob_per_agent": (["step", "agent"], np.stack(intention_prob_per_agent)),
            "adoption_prob_per_agent": (["step", "agent"], np.stack(adoption_prob_per_agent)),
            "capability_per_agent": (["step", "agent"], np.stack(capability_per_agent)),
            "opportunity_per_agent": (["step", "agent"], np.stack(opportunity_per_agent)),
            "environmental_value_per_agent": (["step", "agent"], np.stack(environmental_value_per_agent)),
            "social_weights_per_agent": (["step", "agent"], np.stack(social_weights_per_agent)),
            "self_belief_per_agent": (["step", "agent"], np.stack(self_belief_per_agent)),
            "expected_profit_change_eur_per_ha": (["step", "agent"], np.stack(expected_profit_change_eur_per_ha)),
        },
        coords={"step": steps_arr, "agent": agent_dim},
    )
    return ds


def monte_carlo_runs(n_runs: int, steps: int = 50, seed_base: int = 42, **model_params) -> list[xr.Dataset]:
    """Run the ABM across multiple seeds to create a Monte Carlo ensemble."""

    results_list = []
    for i in range(n_runs):
        model = PeatlandABM(seed=seed_base + i, **model_params)
        ds = run_simulation(model, steps)
        results_list.append(ds)
    return results_list
