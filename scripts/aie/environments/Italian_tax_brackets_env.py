import numpy as np
from ai_economist.foundation.base.base_env import scenario_registry
from ai_economist.foundation.base.base_component import BaseComponent, component_registry
from ai_economist.foundation.scenarios.simple_wood_and_stone.layout_from_file import LayoutFromFile
from ai_economist.foundation.components.redistribution import PeriodicBracketTax
from ai_economist.foundation.components.utils import annealed_tax_limit



@scenario_registry.add
class ItalianBracketsEnv(LayoutFromFile):
    name = "ItalianBracketsEnv"


@component_registry.add
class ItalianPeriodicBracketTax(PeriodicBracketTax):
    name = "ItalianPeriodicBracketTax"

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
    def curr_marginal_rates(self):
        """The current set of marginal tax bracket rates."""
        if self.use_discretized_rates:
            return self.disc_rates[self.curr_rate_indices]

        if self.tax_model == "italian_brackets_2020":
            return np.minimum(
                np.array(self.Italian_brackets_2020), self.curr_rate_max
            )

        raise NotImplementedError
