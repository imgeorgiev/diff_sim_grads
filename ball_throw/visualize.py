import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
import sys

sys.path.append(PARENT_DIR)
from utils.customized_integrator_euler import CustomizedSymplecticEulerIntegrator

# from ball_throw._ball_throw import BallThrow
from _ball_throw import BallThrow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import seaborn as sns
from tqdm import tqdm
import warp

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, "config.yaml"))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps
cfg.name = os.path.basename(__file__)[:-3]
cfg.name += f"_mu_{cfg.customized_mu}"
cfg.THIS_DIR = THIS_DIR

std = 0.2
runs = 100
np.random.seed(0)
thetas = np.linspace(0.0, np.pi / 2, 50)
# thetas = [0.68]
special_thetas = np.linspace(-np.pi, np.pi, 10)
grads = []
trajectories = {}
cost = []
for theta in tqdm(thetas):
    cfg["init_vel"] = [math.cos(theta) * cfg.init_v, math.sin(theta) * cfg.init_v]

    system = BallThrow(
        cfg,
        integrator_class=CustomizedSymplecticEulerIntegrator,
        adapter="cpu",
        render=False,
    )
    loss = system.compute_loss()
    x = []
    for i in range(len(system.states)):
        x.append(system.states[i].particle_q.numpy())
    x = np.array(x).squeeze()
    trajectories.update({theta: x})
    cost.append(loss.numpy()[0])


# create plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

for k, v in trajectories.items():
    ax1.plot(v[:, 0], v[:, 1], label="th={:.2f}".format(k))

ax1.vlines(x=cfg.wall_x, ymin=0.0, ymax=cfg.wall_y, linewidth=2, color="black")

# ax1.legend()
ax1.axis("equal")
ax2.plot(thetas, cost)
ax2.set_ylabel("theta")
ax2.set_ylabel("cost")
plt.savefig(
    "trajectories_kn_{:.2f}_kd_{:.2f}_mu_{:.2f}.pdf".format(
        cfg.customized_kn, cfg.customized_kd, cfg.customized_mu
    )
)
