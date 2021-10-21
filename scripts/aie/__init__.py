from ai_economist.foundation.base.base_env import scenario_registry

from aie.environments.Italian_tax_brackets_env import ItalianBracketsEnv

scenario_registry.add(ItalianBracketsEnv)
