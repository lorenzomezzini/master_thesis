import time

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from rl.conf import get_base_ppo_conf_FCNet
from rl.models.tf.fcnet import FCNet


def get_conf():
    lr = 1e-4

    return {
        **get_base_ppo_conf_FCNet(num_workers=2),
        "lr_schedule": [
            [0, lr],
            [50_000_000, lr],
            [100_000_000, 0],
        ],
    }


def run(load_dir=None):
    ModelCatalog.register_custom_model("my_model", FCNet)
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
    ray.init() #need to add dir to the phase 1 results
    run()
