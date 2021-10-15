from pathlib import Path

from ray.rllib.agents import ppo

from aie.aie_env import AIEEnv, OBS_SPACE_AGENT, ACT_SPACE_AGENT, AIEEnvFv, OBS_SPACE_AGENT_FV
from aie.callbacks import MyCallbacks

BASE_CONF = {
    "env": AIEEnv,
    "callbacks": MyCallbacks,
    "multiagent": {
        "policies_to_train": ["learned"],
        "policies": {
            "learned": (None, OBS_SPACE_AGENT, ACT_SPACE_AGENT, {
                "model": {
                    "custom_model": "my_model",
                },
            }),
        },
        "policy_mapping_fn": lambda x: 'learned',
    },
    "no_done_at_end": False,
}

BASE_CONF_FV = {
    "env": AIEEnvFv,
    "callbacks": MyCallbacks,
    "multiagent": {
        "policies_to_train": ["learned"],
        "policies": {
            "learned": (None, OBS_SPACE_AGENT_FV, ACT_SPACE_AGENT, {
                "model": {
                    "custom_model": "my_model",
                },
            }),
        },
        "policy_mapping_fn": lambda x: 'learned',
    },
    "no_done_at_end": False,
}

FV_PPO_CONF = {
    **ppo.DEFAULT_CONFIG,
    **BASE_CONF_FV,
}

BASE_PPO_CONF = {
    **ppo.DEFAULT_CONFIG,
    **BASE_CONF,
}

#OUT_DIR = Path('/media/lorenzo/SAMSUNG/Tesi/NEW/ray_results')

def get_base_ppo_conf(num_workers: int):
    return {
        **BASE_PPO_CONF,

        #"num_gpus": .4,
        "num_workers": num_workers,
        #"num_gpus_per_worker": .4 / num_workers,
        'num_envs_per_worker': 60 / num_workers,

        "rollout_fragment_length": 200,
        "train_batch_size": 3000,
        "sgd_minibatch_size": 3000,
        "num_sgd_iter": 10,

        "vf_loss_coeff": 0.05,
        "clip_param": 0.25,
        "lambda": 0.98,
        "gamma": 0.998,
        "entropy_coeff": 0.025,
        "lr": 3e-4,
    }


def get_fv_ppo_conf(num_workers: int):
    return {
        **FV_PPO_CONF,

        #"num_gpus": .4,
        "num_workers": num_workers,
        #"num_gpus_per_worker": .4 / num_workers,
        'num_envs_per_worker': 60 / num_workers,

        "rollout_fragment_length": 200,
        "train_batch_size": 3000,
        "sgd_minibatch_size": 3000,
        "num_sgd_iter": 10,

        "vf_loss_coeff": 0.05,
        "clip_param": 0.25,
        "lambda": 0.98,
        "gamma": 0.998,
        "entropy_coeff": 0.025,
        "lr": 3e-4,
    }
