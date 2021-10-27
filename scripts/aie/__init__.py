from ai_economist.foundation.base.base_env import component_registry
from ai_economist.foundation.base.base_env import scenario_registry

from aie.environments.Italian_tax_brackets_env import ItalianPeriodicBracketTax
from aie.environments.thesis_env import ThesisEnv

scenario_registry.add(ThesisEnv)
component_registry.add(ItalianPeriodicBracketTax)
