"""Peatland ABM with intention/adoption architecture.

This module separates the behavioural logic of the peatland adoption model into
coherent components so that its clearly defined how decisions are formed.
Economic parameters follow Rommel et al. (2022)
(https://doi.org/10.1002/aepp.13330) and emission factors follow van Leeuwen
et al. (2024) (https://doi.org/10.5194/bg-21-4099-2024).
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class LandPatch:
    """Minimal land representation linked to a single farmer."""

    id: int
    area_ha: float
    carbon_stock: float
    fauna_abundance: float


@dataclass
class Farmer:
    """Container for farmer-level attributes and their associated land patch."""

    id: int
    land_patch: LandPatch
    expected_profit_conv: float
    expected_profit_nat: float
    expected_profit_diff: float
    profit_weights: float
    social_weights: float
    environmental_value: float
    self_belief: float
    capability: float
    opportunity: float
    adopt: int
    stay_prob: float


@dataclass
class Consumer:
    """Minimal household representation for downstream market interactions."""

    household_id: int
    health_pref: float
    price_sensitivity: float
    eco_pref: float
    eco_weight: float
    current_choice: str

    def decide(self, eco_weight_scale: float = 1.0) -> str:
        """Choose product by weighing health and eco gains against price pressure."""

        effective_eco_weight = self.eco_weight * eco_weight_scale
        score = self.health_pref + effective_eco_weight * self.eco_pref - self.price_sensitivity
        self.current_choice = "rewetted" if score >= 0.0 else "conventional"
        return self.current_choice


def _norm01(arr) -> xr.DataArray:
    """Return a 0-1 normalized copy of ``arr`` while preserving coordinates."""

    data = np.asarray(arr, dtype=float)
    span = data.max() - data.min()
    if span < 1e-8:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - data.min()) / span
    if isinstance(arr, xr.DataArray):
        return xr.DataArray(normalized, dims=arr.dims, coords=arr.coords)
    return xr.DataArray(normalized, dims=["agent"])


def _sigmoid_scalar(value: float, steepness: float) -> float:
    """Scaled sigmoid used to map latent intention scores to probabilities."""

    return 1.0 / (1.0 + np.exp(-steepness * value))


@dataclass
class AgentState:
    """Container for agent-level attributes kept as :class:`xarray.DataArray`."""

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
    adoption_lock: xr.DataArray
    stay_probs: xr.DataArray


@dataclass
class IntentionConfig:
    """Configuration for the intention formation engine."""

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


class IntentionEngine:
    """Stage A: compute intention probabilities from behavioural components.

    The engine blends economic advantage (Δπᵢ), social norms, personal values,
    and self-belief into a latent intention score. The score
    is mapped to [0, 1] with a sigmoid so that it can be read as a probability.
    """

    def __init__(self, prospect_fn, config: IntentionConfig):
        self._prospect_fn = prospect_fn
        self._config = config

    def compute(self, state: AgentState, peer_share: float) -> IntentionBundle:
        """Return intention components for all agents given the current context."""

        expected_profit_change = -state.expected_profit_diff
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
    opportunity_bonus: float = 0.0,
    adoption_lock: xr.DataArray | None = None,
    adoption_lock_years: int = 10,
    capability_scale: float = 1.0,
    opportunity_scale: float = 1.0,
) -> Dict[str, xr.DataArray]:
    """Stage B: translate intention into adoption with friction."""

    capability_norm = _norm01(capability)
    capability_norm = xr.DataArray(
        np.clip(capability_norm.values * capability_scale, 0.0, 1.0),
        dims=capability_norm.dims,
        coords=capability_norm.coords,
    )
    opportunity_norm = _norm01(opportunity)
    opportunity_norm = xr.DataArray(
        np.clip(opportunity_norm.values * opportunity_scale, 0.0, 1.0),
        dims=opportunity_norm.dims,
        coords=opportunity_norm.coords,
    )
    opportunity_adjusted = xr.DataArray(
        np.clip(opportunity_norm.values + opportunity_bonus, 0.0, 1.0),
        dims=opportunity_norm.dims,
        coords=opportunity_norm.coords,
    )
    adoption_prob = intention_prob * capability_norm * opportunity_adjusted
    adoption_prob = xr.DataArray(
        np.clip(adoption_prob.values, 0.0, 1.0),
        dims=adoption_prob.dims,
        coords=adoption_prob.coords,
    )

    draws = rng.binomial(1, adoption_prob.values)
    updated = current_adopt.copy()
    if adoption_lock is None:
        adoption_lock = xr.zeros_like(current_adopt)
    updated_lock = adoption_lock.copy()
    lock_years = max(1, int(adoption_lock_years))
    for idx in range(updated.sizes["agent"]):
        lock_remaining = float(adoption_lock.values[idx])
        if current_adopt.values[idx] == 1:
            if lock_remaining > 0:
                updated.values[idx] = 1
                updated_lock.values[idx] = max(lock_remaining - 1, 0)
                continue

            if rng.random() < stay_probs.values[idx]:
                updated.values[idx] = 1
                updated_lock.values[idx] = 0
                continue

            updated.values[idx] = draws[idx]
            updated_lock.values[idx] = 0
            continue

        if draws[idx] == 1:
            updated.values[idx] = 1
            updated_lock.values[idx] = max(lock_years - 1, 0)
        else:
            updated.values[idx] = 0
            updated_lock.values[idx] = 0

    return {
        "adoption_prob": adoption_prob,
        "updated_adopt": updated,
        "opportunity_norm": opportunity_adjusted,
        "adoption_lock": updated_lock,
    }


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

    conventional_carbon_stock = 50.0  # tC/ha for conventional practices
    nature_based_carbon_stock = 60.0  # tC/ha for nature-based practices

    def __init__(
        self,
        n_agents: int = 100,
        subsidy_eur_per_ha: float = 100.0,
        seed: int = 42,
        stay_adopter_prob: float = 0.9,
        hetero_persistence: bool = True,
        profits_csv: str = "profits_agents.csv",
        consumers_csv: str = "consumers.csv",
        risk_aversion_factor: float = 1.2,
        environmental_value=None,
        social_weights=None,
        intention_weights: Dict[str, float] | None = None,
        intention_intercept: float = 0.0,
        intention_steepness : float = 1.0,
        social_learning_rate: float = 0.1,
        econ_learning_rate: float = 0.1,
        consumer_demand_strength: float = 12.0,
        consumer_eco_weight_scale: float = 1.0,
        consumer_demand_curvature: float = 2.0,
        subsidy_alpha: float = 0.2,
        subsidy_feasibility_beta: float = 0.1,
        subsidy_ref_eur_per_ha: float = 200.0,
        consumer_share_update_rate: float = 0.05,
        adoption_lock_years: int = 10,
        capability_scale: float = 1.0,
        opportunity_scale: float = 1.0,
    ) -> None:
        self.n = n_agents
        self.subsidy_eur_per_ha = subsidy_eur_per_ha
        self.risk_aversion_factor = risk_aversion_factor
        self.rng = np.random.default_rng(seed)
        self.consumer_demand_strength = float(consumer_demand_strength)
        self.consumer_eco_weight_scale = float(consumer_eco_weight_scale)
        self.consumer_demand_curvature = max(0.0, float(consumer_demand_curvature))
        self.subsidy_alpha = float(np.clip(subsidy_alpha, 0.0, 1.0))
        self.subsidy_feasibility_beta = max(0.0, float(subsidy_feasibility_beta))
        self.subsidy_ref_eur_per_ha = max(1e-6, float(subsidy_ref_eur_per_ha))
        self.consumer_share_update_rate = float(np.clip(consumer_share_update_rate, 0.0, 1.0))
        self.capability_scale = float(max(0.0, capability_scale))
        self.opportunity_scale = float(max(0.0, opportunity_scale))
        self.adoption_lock_years = max(1, int(adoption_lock_years))

        df = pd.read_csv(profits_csv)
        if len(df) < self.n:
            raise ValueError(
                f"CSV file '{profits_csv}' must have at least {self.n} rows for farmer profits."
            )

        agent_ids = (
            df["farmer_id"].values[: self.n]
            if "farmer_id" in df
            else (df["agent_id"].values[: self.n] if "agent_id" in df else np.arange(self.n))
        )
        area_values = df["area_ha"].values[: self.n] if "area_ha" in df else np.ones(self.n)
        carbon_values = df["carbon_stock"].values[: self.n] if "carbon_stock" in df else np.zeros(self.n)
        fauna_values = df["fauna_abundance"].values[: self.n] if "fauna_abundance" in df else np.zeros(self.n)

        self.land_patches: List[LandPatch] = [
            LandPatch(
                id=int(agent_ids[idx]),
                area_ha=float(area_values[idx]),
                carbon_stock=float(carbon_values[idx]),
                fauna_abundance=float(fauna_values[idx]),
            )
            for idx in range(self.n)
        ]

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
        adoption_lock = xr.DataArray(
            np.where(adopt.values == 1, max(self.adoption_lock_years - 1, 0), 0),
            dims=["agent"],
        )

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
            adoption_lock=adoption_lock,
            stay_probs=stay_probs_da,
        )

        self._base_expected_profit_conv = self.state.expected_profit_conv.copy()
        self._base_expected_profit_nat = self.state.expected_profit_nat.copy()

        self.consumer_rewetted_share = 0.0
        self.consumer_rewetted_share_raw = 0.0
        self.consumer_profit_bonus_nat = 0.0
        self.consumer_profit_bonus_conv = 0.0
        self.consumer_demand_bias = 0.0
        self._consumer_rewetted_share_next = 0.0
        self._consumer_rewetted_share_raw_next = 0.0
        self._consumer_bonus_nat_current = 0.0
        self._consumer_bonus_conv_current = 0.0
        self._consumer_bonus_nat_next = 0.0
        self._consumer_bonus_conv_next = 0.0

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

        self.farmers: List[Farmer] = [
            Farmer(
                id=int(agent_ids[idx]),
                land_patch=self.land_patches[idx],
                expected_profit_conv=float(self.state.expected_profit_conv.values[idx]),
                expected_profit_nat=float(self.state.expected_profit_nat.values[idx]),
                expected_profit_diff=float(self.state.expected_profit_diff.values[idx]),
                profit_weights=float(self.state.profit_weights.values[idx]),
                social_weights=float(self.state.social_weights.values[idx]),
                environmental_value=float(self.state.environmental_value.values[idx]),
                self_belief=float(self.state.self_belief.values[idx]),
                capability=float(self.state.capability.values[idx]),
                opportunity=float(self.state.opportunity.values[idx]),
                adopt=int(self.state.adopt.values[idx]),
                stay_prob=float(self.state.stay_probs.values[idx]),
            )
            for idx in range(self.n)
        ]

        self.consumers = self._load_consumers(consumers_csv)

    def _load_consumers(self, consumers_csv: str | None) -> List[Consumer]:
        """Load consumer preferences from CSV and return instantiated consumers."""

        if consumers_csv is None:
            raise ValueError("Consumer data must be provided via CSV; got None.")

        try:
            consumer_df = pd.read_csv(consumers_csv)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Consumer CSV '{consumers_csv}' could not be found."
            ) from exc

        required_cols = {"household_id", "health_pref", "price_sensitivity", "eco_pref", "current_choice"}
        missing = required_cols - set(consumer_df.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(
                f"Consumer CSV '{consumers_csv}' is missing required columns: {missing_cols}"
            )

        has_eco_weight = "eco_weight" in consumer_df.columns

        consumers: List[Consumer] = []
        for _, row in consumer_df.iterrows():
            choice = str(row["current_choice"]).strip().lower()
            if choice not in {"conventional", "rewetted"}:
                choice = "conventional"
            eco_weight_value = float(row["eco_weight"]) if has_eco_weight else 1.0
            if not np.isfinite(eco_weight_value):
                eco_weight_value = 1.0
            eco_weight_value = float(np.clip(eco_weight_value, 0.0, 5.0))
            consumers.append(
                Consumer(
                    household_id=int(row["household_id"]),
                    health_pref=float(row["health_pref"]),
                    price_sensitivity=float(row["price_sensitivity"]),
                    eco_pref=float(row["eco_pref"]),
                    eco_weight=eco_weight_value,
                    current_choice=choice,
                )
            )
        return consumers

    def apply_consumer_feedback(self, demand_strength: float) -> float:
        """Adjust expected profits according to the aggregate consumer choice."""

        self.state.expected_profit_conv = self._base_expected_profit_conv.copy()
        self.state.expected_profit_nat = self._base_expected_profit_nat.copy()

        if demand_strength <= 0.0 or not self.consumers:
            self._consumer_bonus_nat_current = 0.0
            self._consumer_bonus_conv_current = 0.0
            self._consumer_bonus_nat_next = 0.0
            self._consumer_bonus_conv_next = 0.0
            self.consumer_rewetted_share = 0.0
            self._consumer_rewetted_share_next = 0.0
            self.consumer_rewetted_share_raw = 0.0
            self._consumer_rewetted_share_raw_next = 0.0
            self.consumer_profit_bonus_nat = 0.0
            self.consumer_profit_bonus_conv = 0.0
            self.consumer_demand_bias = 0.0
            self.state.expected_profit_diff = self.state.expected_profit_conv - self.state.expected_profit_nat
            self.expected_profit_diff_eur_per_ha = self.state.expected_profit_diff
            return 0.0

        if self._consumer_bonus_nat_current > 0.0:
            self.state.expected_profit_nat = self.state.expected_profit_nat + self._consumer_bonus_nat_current
        if self._consumer_bonus_conv_current > 0.0:
            self.state.expected_profit_conv = self.state.expected_profit_conv + self._consumer_bonus_conv_current

        self.state.expected_profit_diff = self.state.expected_profit_conv - self.state.expected_profit_nat
        self.expected_profit_diff_eur_per_ha = self.state.expected_profit_diff
        self.consumer_profit_bonus_nat = self._consumer_bonus_nat_current
        self.consumer_profit_bonus_conv = self._consumer_bonus_conv_current

        applied_share = self.consumer_rewetted_share

        rewetted_votes = 0
        for consumer in self.consumers:
            choice = consumer.decide(self.consumer_eco_weight_scale)
            if choice == "rewetted":
                rewetted_votes += 1

        rewetted_share_raw = rewetted_votes / len(self.consumers)
        self.consumer_rewetted_share_raw = rewetted_share_raw

        update_rate = self.consumer_share_update_rate
        if update_rate <= 0.0:
            rewetted_share_smoothed = rewetted_share_raw
        else:
            rewetted_share_smoothed = (
                (1.0 - update_rate) * self.consumer_rewetted_share + update_rate * rewetted_share_raw
            )
        rewetted_share_smoothed = float(np.clip(rewetted_share_smoothed, 0.0, 1.0))

        neutral_share = 0.5
        delta_share = rewetted_share_smoothed - neutral_share
        curvature = self.consumer_demand_curvature
        if curvature <= 0.0:
            rewetted_benefit = float(np.clip(rewetted_share_smoothed, 0.0, 1.0))
        else:
            rewetted_benefit = 1.0 / (1.0 + np.exp(-curvature * delta_share))

        conv_benefit = 1.0 - rewetted_benefit
        self.consumer_demand_bias = float(2.0 * rewetted_benefit - 1.0)
        self._consumer_bonus_nat_next = demand_strength * rewetted_benefit
        self._consumer_bonus_conv_next = demand_strength * conv_benefit
        self._consumer_rewetted_share_next = rewetted_share_smoothed
        self._consumer_rewetted_share_raw_next = rewetted_share_raw

        return applied_share

    def step(self) -> Dict[str, object]:
        """Advance the simulation by one period and return diagnostics."""

        consumer_share = self.apply_consumer_feedback(self.consumer_demand_strength)
        peer_share = float(self.state.adopt.mean())
        bundle = self.engine.compute(self.state, peer_share)

        subsidy_stage_a = self.subsidy_alpha * self.subsidy_eur_per_ha
        subsidy_stage_b = (1.0 - self.subsidy_alpha) * self.subsidy_eur_per_ha
        subsidy_stage_a_norm = float(min(subsidy_stage_a / self.subsidy_ref_eur_per_ha, 1.0))
        subsidy_stage_b_norm = float(min(subsidy_stage_b / self.subsidy_ref_eur_per_ha, 1.0))

        latent_with_subsidy = bundle.latent + subsidy_stage_a_norm
        intention_prob = xr.apply_ufunc(
            lambda val: _sigmoid_scalar(val, self.intention_config.steepness),
            latent_with_subsidy,
            vectorize=True,
        )
        bundle.latent = latent_with_subsidy
        bundle.intention_prob = intention_prob

        opportunity_bonus = self.subsidy_feasibility_beta * subsidy_stage_b_norm
        adoption_out = apply_adoption(
            bundle.intention_prob,
            self.state.capability,
            self.state.opportunity,
            self.state.adopt,
            self.state.stay_probs,
            self.rng,
            opportunity_bonus=opportunity_bonus,
            adoption_lock=self.state.adoption_lock,
            adoption_lock_years=self.adoption_lock_years,
            capability_scale=self.capability_scale,
            opportunity_scale=self.opportunity_scale,
        )
        self.state.adopt = adoption_out["updated_adopt"]
        self.state.adoption_lock = adoption_out["adoption_lock"]

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

        for idx, farmer in enumerate(self.farmers):
            farmer.adopt = int(self.state.adopt.values[idx])
            farmer.profit_weights = float(self.state.profit_weights.values[idx])
            farmer.social_weights = float(self.state.social_weights.values[idx])
            farmer.expected_profit_conv = float(self.state.expected_profit_conv.values[idx])
            farmer.expected_profit_nat = float(self.state.expected_profit_nat.values[idx])
            farmer.expected_profit_diff = float(self.state.expected_profit_diff.values[idx])
            # Update carbon stock based on adoption
            farmer.land_patch.carbon_stock = (
                self.nature_based_carbon_stock if farmer.adopt else self.conventional_carbon_stock
            )

        conventional_emissions = 3.77
        nature_based_emissions = 2.66
        emis_per_ha = np.array([
            nature_based_emissions if farmer.adopt else conventional_emissions
            for farmer in self.farmers
        ])
        total_emis = emis_per_ha * np.array([farmer.land_patch.area_ha for farmer in self.farmers])
        total_area = sum(farmer.land_patch.area_ha for farmer in self.farmers)
        avg_emiss = total_emis.sum() / total_area if total_area > 0 else 0.0

        baseline_per_ha = conventional_emissions
        baseline_total = baseline_per_ha * total_area
        adoption_rate = float(np.mean(self.state.adopt.values))
        emissions_reduced = max(baseline_total - total_emis.sum(), 0.0)
        policy_cost_per_ha = self.subsidy_eur_per_ha * adoption_rate
        policy_cost_total = self.subsidy_eur_per_ha * sum(farmer.land_patch.area_ha * farmer.adopt for farmer in self.farmers)
        threshold = 0.01
        if emissions_reduced > threshold:
            cost_per_tonne = policy_cost_total / emissions_reduced
        else:
            cost_per_tonne = float("nan")

        intention_prob = bundle.intention_prob
        intention_rate = float(intention_prob.mean())
        adoption_prob = adoption_out["adoption_prob"]
        opportunity_adjusted = adoption_out["opportunity_norm"]
        opportunity_adjusted_mean = float(opportunity_adjusted.mean())

        result = {
            "adoption_rate": adoption_rate,
            "avg_emissions_tCO2_ha": avg_emiss,
            "subsidy_eur_per_ha": self.subsidy_eur_per_ha,
            "expected_profit_diff_eur_per_ha": self.state.expected_profit_diff.copy(),
            "emissions_saved_tCO2_ha": emissions_reduced / total_area if total_area > 0 else 0.0,
            "policy_cost_eur_per_ha": policy_cost_per_ha,
            "cost_per_tonne_eur_per_tCO2": cost_per_tonne,
            "mean_econ_utility": mean_econ_utility,
            "utility_econ_per_agent": bundle.econ_norm.copy(),
            "mean_social_utility": mean_social_utility,
            "utility_social_per_agent": bundle.social_norm.copy(),
            "intention_rate": intention_rate,
            "adoption_prob_per_agent": adoption_prob.copy(),
            "intention_prob_per_agent": intention_prob.copy(),
            "capability_per_agent": self.state.capability.copy(),
            "opportunity_per_agent": self.state.opportunity.copy(),
            "opportunity_adjusted_per_agent": opportunity_adjusted.copy(),
            "environmental_value_per_agent": self.state.environmental_value.copy(),
            "social_weights_per_agent": self.state.social_weights.copy(),
            "self_belief_per_agent": self.state.self_belief.copy(),
            "adoption_lock_remaining_per_agent": self.state.adoption_lock.copy(),
            "expected_profit_change_eur_per_ha": bundle.expected_profit_change.copy(),
            "consumer_rewetted_share": consumer_share,
            "consumer_rewetted_share_raw": self.consumer_rewetted_share_raw,
            "consumer_profit_bonus_nat": self.consumer_profit_bonus_nat,
            "consumer_profit_bonus_conv": self.consumer_profit_bonus_conv,
            "consumer_demand_bias": self.consumer_demand_bias,
            "consumer_eco_weight_scale": self.consumer_eco_weight_scale,
            "subsidy_stageA_eur_per_ha": subsidy_stage_a,
            "subsidy_stageB_eur_per_ha": subsidy_stage_b,
            "subsidy_stageA_norm": subsidy_stage_a_norm,
            "subsidy_stageB_norm": subsidy_stage_b_norm,
            "subsidy_alpha": self.subsidy_alpha,
            "subsidy_feasibility_beta": self.subsidy_feasibility_beta,
            "opportunity_bonus": opportunity_bonus,
            "opportunity_adjusted_mean": opportunity_adjusted_mean,
            "capability_scale": float(self.capability_scale),
            "opportunity_scale": float(self.opportunity_scale),
        }

        self._consumer_bonus_nat_current = self._consumer_bonus_nat_next
        self._consumer_bonus_conv_current = self._consumer_bonus_conv_next
        self.consumer_rewetted_share = self._consumer_rewetted_share_next
        self.consumer_rewetted_share_raw = self._consumer_rewetted_share_raw_next

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
    adoption_lock_remaining_per_agent = []
    expected_profit_change_eur_per_ha = []
    consumer_rewetted_share = []
    consumer_rewetted_share_raw = []
    consumer_demand_bias = []
    consumer_profit_bonus_nat = []
    consumer_profit_bonus_conv = []
    opportunity_adjusted_per_agent = []
    opportunity_adjusted_mean = []
    subsidy_stageA_eur_per_ha = []
    subsidy_stageB_eur_per_ha = []
    subsidy_stageA_norm = []
    subsidy_stageB_norm = []
    opportunity_bonus = []
    subsidy_alpha_series = []
    subsidy_feasibility_beta_series = []
    capability_scale_series = []
    opportunity_scale_series = []

    for _ in range(steps):
        result = model.step()
        adoption_rate.append(result["adoption_rate"])
        avg_emissions_tCO2_ha.append(result["avg_emissions_tCO2_ha"])
        subsidy_eur_per_ha.append(result["subsidy_eur_per_ha"])
        emissions_saved_tCO2_ha.append(result["emissions_saved_tCO2_ha"])
        policy_cost_eur_per_ha.append(result["policy_cost_eur_per_ha"])
        cost_per_tonne_eur_per_tCO2.append(result["cost_per_tonne_eur_per_tCO2"])
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
        adoption_lock_remaining_per_agent.append(result["adoption_lock_remaining_per_agent"].values)
        expected_profit_change_eur_per_ha.append(result["expected_profit_change_eur_per_ha"].values)
        consumer_rewetted_share.append(result["consumer_rewetted_share"])
        consumer_rewetted_share_raw.append(result["consumer_rewetted_share_raw"])
        consumer_demand_bias.append(result["consumer_demand_bias"])
        consumer_profit_bonus_nat.append(result["consumer_profit_bonus_nat"])
        consumer_profit_bonus_conv.append(result["consumer_profit_bonus_conv"])
        opportunity_adjusted_per_agent.append(result["opportunity_adjusted_per_agent"].values)
        opportunity_adjusted_mean.append(result["opportunity_adjusted_mean"])
        subsidy_stageA_eur_per_ha.append(result["subsidy_stageA_eur_per_ha"])
        subsidy_stageB_eur_per_ha.append(result["subsidy_stageB_eur_per_ha"])
        subsidy_stageA_norm.append(result["subsidy_stageA_norm"])
        subsidy_stageB_norm.append(result["subsidy_stageB_norm"])
        opportunity_bonus.append(result["opportunity_bonus"])
        subsidy_alpha_series.append(result["subsidy_alpha"])
        subsidy_feasibility_beta_series.append(result["subsidy_feasibility_beta"])
        capability_scale_series.append(result["capability_scale"])
        opportunity_scale_series.append(result["opportunity_scale"])

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
            "mean_econ_utility": ("step", np.array(mean_econ_utility)),
            "utility_econ_per_agent": (["step", "agent"], np.stack(utility_econ_per_agent)),
            "mean_social_utility": ("step", np.array(mean_social_utility)),
            "utility_social_per_agent": (["step", "agent"], np.stack(utility_social_per_agent)),
            "intention_rate": ("step", np.array(intention_rate)),
            "intention_prob_per_agent": (["step", "agent"], np.stack(intention_prob_per_agent)),
            "adoption_prob_per_agent": (["step", "agent"], np.stack(adoption_prob_per_agent)),
            "capability_per_agent": (["step", "agent"], np.stack(capability_per_agent)),
            "opportunity_per_agent": (["step", "agent"], np.stack(opportunity_per_agent)),
            "opportunity_adjusted_per_agent": (["step", "agent"], np.stack(opportunity_adjusted_per_agent)),
            "environmental_value_per_agent": (["step", "agent"], np.stack(environmental_value_per_agent)),
            "social_weights_per_agent": (["step", "agent"], np.stack(social_weights_per_agent)),
            "self_belief_per_agent": (["step", "agent"], np.stack(self_belief_per_agent)),
            "adoption_lock_remaining_per_agent": (["step", "agent"], np.stack(adoption_lock_remaining_per_agent)),
            "expected_profit_change_eur_per_ha": (["step", "agent"], np.stack(expected_profit_change_eur_per_ha)),
            "consumer_rewetted_share": ("step", np.array(consumer_rewetted_share)),
            "consumer_rewetted_share_raw": ("step", np.array(consumer_rewetted_share_raw)),
            "consumer_demand_bias": ("step", np.array(consumer_demand_bias)),
            "consumer_profit_bonus_nat": ("step", np.array(consumer_profit_bonus_nat)),
            "consumer_profit_bonus_conv": ("step", np.array(consumer_profit_bonus_conv)),
            "opportunity_adjusted_mean": ("step", np.array(opportunity_adjusted_mean)),
            "subsidy_stageA_eur_per_ha": ("step", np.array(subsidy_stageA_eur_per_ha)),
            "subsidy_stageB_eur_per_ha": ("step", np.array(subsidy_stageB_eur_per_ha)),
            "subsidy_stageA_norm": ("step", np.array(subsidy_stageA_norm)),
            "subsidy_stageB_norm": ("step", np.array(subsidy_stageB_norm)),
            "opportunity_bonus": ("step", np.array(opportunity_bonus)),
            "subsidy_alpha": ("step", np.array(subsidy_alpha_series)),
            "subsidy_feasibility_beta": ("step", np.array(subsidy_feasibility_beta_series)),
            "capability_scale": ("step", np.array(capability_scale_series)),
            "opportunity_scale": ("step", np.array(opportunity_scale_series)),
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
