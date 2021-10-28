import time

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from aie.aie_env import OBS_SPACE_AGENT, ACT_SPACE_AGENT
from aie.env_conf import  ENV_COMMUNISM
from rl.conf import get_base_ppo_conf_LSTM
from rl.models.tf.fcnet_lstm import RNNModel

def get_conf():
    return {
        **get_base_ppo_conf_LSTM(num_workers=4),
        "multiagent": {
            "policies_to_train": ["learned"],
            "policies": {
                "learned": (None, OBS_SPACE_AGENT, ACT_SPACE_AGENT, {
                    "model": {
                        "custom_model": "my_model",
                        'max_seq_len': 50,
                    },
                }),
            },
            "policy_mapping_fn": lambda x: 'learned',
        },
        "env_config" : ENV_COMMUNISM,

    }


def run(load_dir=None):
    ModelCatalog.register_custom_model("my_model", RNNModel)
    trainer = ppo.PPOTrainer(config=get_conf())

    if load_dir != None:
        # should add an assert
        trainer.restore(load_dir)

    t = time.monotonic()
    while True:
        trainer.train()
        checkpoint = trainer.save()
        print(time.monotonic() - t, "checkpoint saved at", checkpoint)
        t = time.monotonic()


if __name__ == '__main__':
    ray.init()
    run('/home/lorenzo/ray_results/LSTM_80M/checkpoint_6315/checkpoint-6315')
