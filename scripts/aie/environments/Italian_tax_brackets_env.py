import numpy as np
from ai_economist.foundation.base.base_component import BaseComponent, component_registry
from ai_economist.foundation.components.utils import annealed_tax_limit, annealed_tax_mask

@component_registry.add
class ItalianPeriodicBracketTax(BaseComponent):
    name = "ItalianPeriodicBracketTax"
    component_type = "PeriodicTax"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        disable_taxes=False,
        tax_model="italian_brackets_2020",
        period=100,
        rate_min=0.0,
        rate_max=1.0,
        rate_disc=0.05,
        n_brackets=5,
        top_bracket_cutoff=100,
        usd_scaling= 863.93,                        # exchange rate at 1/10/21  ddmmyy 1157.5
        bracket_spacing="it-brackets",
        tax_annealing_schedule=None,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Whether to turn off taxes. Disabling taxes will prevent any taxes from
        # being collected but the observation space will be the same as if taxes were
        # enabled, which can be useful for controlled tax/no-tax comparisons.
        self.disable_taxes = bool(disable_taxes)

        # How to set taxes
        self.tax_model = tax_model
        assert self.tax_model in [
            "italian_brackets_2020",
        ]

        # How many timesteps a tax period lasts
        self.period = int(period)
        assert self.period > 0

        # Minimum marginal bracket rate
        self.rate_min = 0.0 if self.disable_taxes else float(rate_min)
        # Maximum marginal bracket rate
        self.rate_max = 0.0 if self.disable_taxes else float(rate_max)
        assert 0 <= self.rate_min <= self.rate_max <= 1.0

        # Interval for discretizing tax rate options
        # (only applies if tax_model == "model_wrapper")
        self.rate_disc = float(rate_disc)

        self.use_discretized_rates = self.tax_model == "model_wrapper"

        if self.use_discretized_rates:
            self.disc_rates = np.arange(
                self.rate_min, self.rate_max + self.rate_disc, self.rate_disc
            )
            self.disc_rates = self.disc_rates[self.disc_rates <= self.rate_max]
            assert len(self.disc_rates) > 1 or self.disable_taxes
            self.n_disc_rates = len(self.disc_rates)
        else:
            self.disc_rates = None
            self.n_disc_rates = 0

        # === income bracket definitions ===
        self.n_brackets = int(n_brackets)
        assert self.n_brackets >= 2

        self.top_bracket_cutoff = float(top_bracket_cutoff)
        assert self.top_bracket_cutoff >= 10

        self.usd_scale = float(usd_scaling)
        assert self.usd_scale > 0

        self.bracket_spacing = bracket_spacing.lower()
        assert self.bracket_spacing in ["it-brackets"]

        if self.bracket_spacing == "it-brackets":
            self.bracket_cutoffs = (
                np.array([0, 15000, 28000, 55000, 75000])
                / self.usd_scale
            )
            self.n_brackets = len(self.bracket_cutoffs)
            self.top_bracket_cutoff = float(self.bracket_cutoffs[-1])
        else:
            raise NotImplementedError

        self.bracket_edges = np.concatenate([self.bracket_cutoffs, [np.inf]])
        self.bracket_sizes = self.bracket_edges[1:] - self.bracket_edges[:-1]

        assert self.bracket_cutoffs[0] == 0

        if self.tax_model == "italian_brackets_2020":
            assert self.bracket_spacing == "it-brackets"

        else:
            self._fixed_bracket_rates = None

        # === bracket tax rates ===
        self.curr_bracket_tax_rates = np.zeros_like(self.bracket_cutoffs)
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        # === tax cycle definitions ===
        self.tax_cycle_pos = 1
        self.last_coin = [0 for _ in range(self.n_agents)]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        # === trackers ===
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [0] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self.taxes = []

        # === tax annealing ===
        # for annealing of non-planner max taxes
        self._annealed_rate_max = float(self.rate_max)
        self._last_completions = 0

        # for annealing of planner actions
        self.tax_annealing_schedule = tax_annealing_schedule
        if tax_annealing_schedule is not None:
            assert isinstance(self.tax_annealing_schedule, (tuple, list))
            self._annealing_warmup = self.tax_annealing_schedule[0]
            self._annealing_slope = self.tax_annealing_schedule[1]
            self._annealed_rate_max = annealed_tax_limit(
                self._last_completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )
        else:
            self._annealing_warmup = None
            self._annealing_slope = None

        if self.tax_model == "model_wrapper" and not self.disable_taxes:
            planner_action_tuples = self.get_n_actions("BasicPlanner")
            self._planner_tax_val_dict = {
                k: self.disc_rates for k, v in planner_action_tuples
            }
        else:
            self._planner_tax_val_dict = {}
        self._planner_masks = None

        # === placeholders ===
        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

    @property
    def Italian_brackets_2020(self):
            return [0.23, 0.27, 0.38, 0.41, 0.43]

    @property
    def fixed_bracket_rates(self):
        """Return whatever fixed bracket rates were set during initialization."""
        return self._fixed_bracket_rates

    @property
    def curr_rate_max(self):
        """Maximum allowable tax rate, given current progress of any tax annealing."""
        if self.tax_annealing_schedule is None:
            return self.rate_max
        return self._annealed_rate_max
        
    @property
    def curr_marginal_rates(self):
        """The current set of marginal tax bracket rates."""
        if self.use_discretized_rates:
            return self.disc_rates[self.curr_rate_indices]

        if self.tax_model == "italian_brackets_2020":
            return np.minimum(
                np.array(self.Italian_brackets_2020), self.curr_rate_max
            )

        raise NotImplementedError


    def set_new_period_rates_model(self):
        """Update taxes using actions from the tax model."""
        if self.disable_taxes:
            return

        # AI version
        for i, bracket in enumerate(self.bracket_cutoffs):
            planner_action = self.world.planner.get_component_action(
                self.name, "TaxIndexBracket_{:03d}".format(int(bracket))
            )
            if planner_action == 0:
                pass
            elif planner_action <= self.n_disc_rates:
                self.curr_rate_indices[i] = int(planner_action - 1)
            else:
                raise ValueError

    # Methods for collecting and redistributing taxes
    # -----------------------------------------------

    def income_bin(self, income):
        """Return index of tax bin in which income falls."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.bracket_cutoffs[np.argmax(bracket_bool)]

    def marginal_rate(self, income):
        """Return the marginal tax rate applied at this income level."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.curr_marginal_rates[np.argmax(bracket_bool)]

    def taxes_due(self, income):
        """Return the total amount of taxes due at this income level."""
        past_cutoff = np.maximum(0, income - self.bracket_cutoffs)
        bin_income = np.minimum(self.bracket_sizes, past_cutoff)
        bin_taxes = self.curr_marginal_rates * bin_income
        return np.sum(bin_taxes)

    def enact_taxes(self):
        """Calculate period income & tax burden. Collect taxes and redistribute."""
        net_tax_revenue = 0
        tax_dict = dict(
            schedule=np.array(self.curr_marginal_rates),
            cutoffs=np.array(self.bracket_cutoffs),
        )

        for curr_rate, bracket_cutoff in zip(
            self.curr_marginal_rates, self.bracket_cutoffs
        ):
            self._schedules["{:03d}".format(int(bracket_cutoff))].append(
                float(curr_rate)
            )

        self.last_income = []
        self.last_effective_tax_rate = []
        self.last_marginal_rate = []
        for agent, last_coin in zip(self.world.agents, self.last_coin):
            income = agent.total_endowment("Coin") - last_coin
            tax_due = self.taxes_due(income)
            effective_taxes = np.minimum(
                agent.state["inventory"]["Coin"], tax_due
            )  # Don't take from escrow.
            marginal_rate = self.marginal_rate(income)
            effective_tax_rate = float(effective_taxes / np.maximum(0.000001, income))
            tax_dict[str(agent.idx)] = dict(
                income=float(income),
                tax_paid=float(effective_taxes),
                marginal_rate=marginal_rate,
                effective_rate=effective_tax_rate,
            )

            # Actually collect the taxes
            agent.state["inventory"]["Coin"] -= effective_taxes
            net_tax_revenue += effective_taxes

            self.last_income.append(float(income))
            self.last_marginal_rate.append(float(marginal_rate))
            self.last_effective_tax_rate.append(effective_tax_rate)

            self.all_effective_tax_rates.append(effective_tax_rate)
            self._occupancy["{:03d}".format(int(self.income_bin(income)))] += 1

        self.total_collected_taxes += float(net_tax_revenue)

        lump_sum = net_tax_revenue / self.n_agents
        for agent in self.world.agents:
            agent.state["inventory"]["Coin"] += lump_sum
            tax_dict[str(agent.idx)]["lump_sum"] = float(lump_sum)
            self.last_coin[agent.idx] = float(agent.total_endowment("Coin"))

        self.taxes.append(tax_dict)

        # Pre-compute some things that will be useful for generating observations
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        If using the "model_wrapper" tax model and taxes are enabled, the planner's
        action space includes an action subspace for each of the tax brackets. Each
        such action space has as many actions as there are discretized tax rates.
        """
        # Only the planner takes actions through this component
        if agent_cls_name == "BasicPlanner":
            if self.tax_model == "model_wrapper" and not self.disable_taxes:
                # For every bracket, the planner can select one of the discretized
                # tax rates.
                return [
                    ("TaxIndexBracket_{:03d}".format(int(r)), self.n_disc_rates)
                    for r in self.bracket_cutoffs
                ]

        # Return 0 (no added actions) if the other conditions aren't met
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any agent state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        On the first day of each tax period, update taxes. On the last day, enact them.
        """

        # 1. On the first day of a new tax period: Set up the taxes for this period.
        if self.tax_cycle_pos == 1:
            if self.tax_model == "model_wrapper":
                self.set_new_period_rates_model()

            self._curr_rates_obs = np.array(self.curr_marginal_rates)

        # 2. On the last day of the tax period: Get $-taxes AND update agent endowments
        if self.tax_cycle_pos >= self.period:
            self.enact_taxes()
            self.tax_cycle_pos = 0

        else:
            self.taxes.append([])

        # increment timestep
        self.tax_cycle_pos += 1

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Agents observe where in the tax period cycle they are, information about the
        last period's incomes, and the current marginal tax rates, including the
        marginal rate that will apply to their next unit of income.

        The planner observes the same type of information, but for all of the agents. It
        also sees, for each agent, their marginal tax rate and reported income from
        the previous tax period.
        """
        is_tax_day = float(self.tax_cycle_pos >= self.period)
        is_first_day = float(self.tax_cycle_pos == 1)
        tax_phase = self.tax_cycle_pos / self.period

        obs = dict()

        obs[self.world.planner.idx] = dict(
            is_tax_day=is_tax_day,
            is_first_day=is_first_day,
            tax_phase=tax_phase,
            last_incomes=self._last_income_obs_sorted,
            curr_rates=self._curr_rates_obs,
        )

        for agent in self.world.agents:
            i = agent.idx
            k = str(i)

            curr_marginal_rate = self.marginal_rate(
                agent.total_endowment("Coin") - self.last_coin[i]
            )

            obs[k] = dict(
                is_tax_day=is_tax_day,
                is_first_day=is_first_day,
                tax_phase=tax_phase,
                last_incomes=self._last_income_obs_sorted,
                curr_rates=self._curr_rates_obs,
                marginal_rate=curr_marginal_rate,
            )

            obs["p" + k] = dict(
                last_income=self._last_income_obs[i],
                last_marginal_rate=self.last_marginal_rate[i],
                curr_marginal_rate=curr_marginal_rate,
            )

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Masks only apply to the planner and if tax_model == "model_wrapper" and taxes
        are enabled.
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps
        except when self.tax_cycle_pos==1 (meaning a new tax period is starting).
        When self.tax_cycle_pos==1, tax actions are masked in order to enforce any
        tax annealing.
        """
        if (
            completions != self._last_completions
            and self.tax_annealing_schedule is not None
        ):
            self._last_completions = int(completions)
            self._annealed_rate_max = annealed_tax_limit(
                completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )

        if self.disable_taxes:
            return {}

        if self.tax_model == "model_wrapper":
            # No annealing. Generate masks using default method.
            if self.tax_annealing_schedule is None:
                if self._planner_masks is None:
                    masks = super().generate_masks(completions=completions)
                    self._planner_masks = dict(
                        new_taxes=deepcopy(masks[self.world.planner.idx]),
                        zeros={
                            k: np.zeros_like(v)
                            for k, v in masks[self.world.planner.idx].items()
                        },
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

            # Doing annealing.
            else:
                # Figure out what the masks should be this episode.
                if self._planner_masks is None:
                    planner_masks = {
                        k: annealed_tax_mask(
                            completions,
                            self._annealing_warmup,
                            self._annealing_slope,
                            tax_values,
                        )
                        for k, tax_values in self._planner_tax_val_dict.items()
                    }
                    self._planner_masks = dict(
                        new_taxes=deepcopy(planner_masks),
                        zeros={k: np.zeros_like(v) for k, v in planner_masks.items()},
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

        # We are not using a learned planner. Generate masks by the default method.
        else:
            masks = super().generate_masks(completions=completions)

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset trackers.
        """
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        self.tax_cycle_pos = 1
        self.last_coin = [
            float(agent.total_endowment("Coin")) for agent in self.world.agents
        ]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

        self.taxes = []
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self._planner_masks = None

    def get_metrics(self):
        """
        See base_component.py for detailed description.

        Return metrics related to bracket rates, bracket occupancy, and tax collection.
        """
        out = dict()

        n_observed_incomes = np.maximum(1, np.sum(list(self._occupancy.values())))
        for c in self.bracket_cutoffs:
            k = "{:03d}".format(int(c))
            out["avg_bracket_rate/{}".format(k)] = np.mean(self._schedules[k])
            out["bracket_occupancy/{}".format(k)] = (
                self._occupancy[k] / n_observed_incomes
            )

        if not self.disable_taxes:
            out["avg_effective_tax_rate"] = np.mean(self.all_effective_tax_rates)
            out["total_collected_taxes"] = float(self.total_collected_taxes)

            # Indices of richest and poorest agents
            agent_coin_endows = np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            )
            idx_poor = np.argmin(agent_coin_endows)
            idx_rich = np.argmax(agent_coin_endows)

            tax_days = self.taxes[(self.period - 1) :: self.period]
            for i, tag in zip([idx_poor, idx_rich], ["poorest", "richest"]):
                total_income = np.maximum(
                    0, [tax_day[str(i)]["income"] for tax_day in tax_days]
                ).sum()
                total_tax_paid = np.sum(
                    [tax_day[str(i)]["tax_paid"] for tax_day in tax_days]
                )
                # Report the overall tax rate over the episode
                # for the richest and poorest agents
                out["avg_tax_rate/{}".format(tag)] = total_tax_paid / np.maximum(
                    0.001, total_income
                )

        return out

    def get_dense_log(self):
        """
        Log taxes.

        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single
                timestep. Entries are empty except for timesteps where a tax period
                ended and taxes were collected. For those timesteps, each entry
                contains the tax schedule, each agent's reported income, tax paid,
                and redistribution received.
                Returns None if taxes are disabled.
        """
        if self.disable_taxes:
            return None
        return self.taxes
