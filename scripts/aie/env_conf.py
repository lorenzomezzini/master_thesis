import numpy as np

ENV_CONF_DEFAULT = {
    'scenario_name': 'layout_from_file/simple_wood_and_stone',
    'components': [
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        ('Gather', {}),
        ('PeriodicBracketTax', {
            'bracket_spacing': "us-federal",
            'period': 100,
            'tax_model': 'fixed-bracket-rates',
            'fixed_bracket_rates': np.zeros(7).tolist(),
        }),
    ],

    'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 0,
    'fixed_four_skill_and_loc': True,

    'n_agents': 4,  
    'world_size': [25, 25], 
    'episode_length': 1000,  

    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,

    'flatten_observations': False,
    'flatten_masks': True,
}

ENV_PHASE_ONE ={
    'energy_warmup_method': 'auto',
    'energy_warmup_constant': 10000
}

ENV_NO_LABOUR ={
    'energy_cost' : 0,
}

ENV_ITALY = {
    'components': [
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        ('Gather', {}),
        ('ItalianPeriodicBracketTax', {
            'bracket_spacing': "it-brackets",
            'period': 100,
            'tax_model': 'italian_brackets_2020',
        }),
    ],
}

ENV_US = {
    'components': [
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        ('Gather', {}),
        ('PeriodicBracketTax', {
            'bracket_spacing': "us-federal",
            'period': 100,
            'tax_model': 'us-federal-single-filer-2018-scaled',
        }),
    ],
}

ENV_FLAT_TAX = {
    'components': [
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        ('Gather', {}),
        ('PeriodicBracketTax', {
            'bracket_spacing': "us-federal",
            'period': 100,
            'tax_model': 'fixed-bracket-rates',
            'fixed_bracket_rates': [21,21,21,21,21,21,21],
        }),
    ],
}