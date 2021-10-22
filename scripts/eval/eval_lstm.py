import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import ModelCatalog
from tqdm import tqdm

import sys
sys.path.insert(1, '/home/lorenzo/Desktop/master_thesis/scripts')

from aie import plotting
from aie.aie_env import AIEEnv
from rl.conf import BASE_PPO_CONF
from rl.models.tf.fcnet_lstm import RNNModel

ray.init()
ModelCatalog.register_custom_model("my_model", RNNModel)

trainer = ppo.PPOTrainer(config={
    **BASE_PPO_CONF,
    "num_workers": 0,
})

ckpt_path = '/home/lorenzo/ray_results/LSTM_80M/checkpoint_6310/checkpoint-6310'

trainer.restore(str(ckpt_path))

env = AIEEnv({}, force_dense_logging=True)
obs = env.reset()
hidden_states = {
    k: [
        np.zeros(128, np.float32),
        np.zeros(128, np.float32),
    ]
    for k in obs.keys()
}

for t in tqdm(range(1000)):
    results = {
        k: trainer.compute_action(
            v,
            state=hidden_states[k],
            policy_id='learned',
            explore=False,
        )
        for k, v in obs.items()
    }
    actions = {
        k: v[0]
        for k, v in results.items()
    }
    hidden_states = {
        k: v[1]
        for k, v in results.items()
    }
    obs, reward, done, info = env.step(actions)
    #plotting.plot_env_state(env.env)

plotting.breakdown(env.env.previous_episode_dense_log)
plt.show()

env.env.scenario_metrics()
