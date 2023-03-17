import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
import sys

sys.path.append(PARENT_DIR)
from utils.customized_integrator_euler import CustomizedSymplecticEulerIntegrator
from _ball_throw import BallThrow
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, "config.yaml"))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps  # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.name += f"_mu_{cfg.customized_mu}"
cfg.THIS_DIR = THIS_DIR

eps = np.finfo(float).eps
np.random.seed(0)
thetas = np.linspace(0.0, np.pi / 2, 50)
baselines = {}
grads = []
for theta in tqdm(thetas):
    # first compute baseline
    cfg["init_vel"] = [math.cos(theta) * cfg.init_v, math.sin(theta) * cfg.init_v]
    system = BallThrow(
        cfg,
        integrator_class=CustomizedSymplecticEulerIntegrator,
        adapter="cpu",
        render=False,
    )
    baseline = system.compute_loss().numpy()[0]
    baselines.update({theta: baseline})

    # Now do stochastic runs
    for i in range(cfg.samples):
        # add noise to theta
        noise_sample = np.random.normal(scale=cfg.std)
        theta_noisy = theta + noise_sample
        # use theta to set angle
        cfg["init_vel"] = [
            math.cos(theta_noisy) * cfg.init_v,
            math.sin(theta_noisy) * cfg.init_v,
        ]

        system = BallThrow(
            cfg,
            integrator_class=CustomizedSymplecticEulerIntegrator,
            adapter="cpu",
            render=False,
        )
        loss = system.compute_loss().numpy()[0]

        # Get FoG
        v_grad = system.check_grad(system.states[0].particle_qd)
        print(v_grad)
        theta_grad = np.tan(v_grad.numpy()[0, 1] / (v_grad.numpy()[0, 0] + eps))

        # Get old style of linear ZoG
        zog_old = (
            1
            / cfg.std**2
            * cfg.init_v
            * loss
            * noise_sample
            * np.array([-np.sin(theta), np.cos(theta)])
        )
        zog_theta_old = np.tan(zog_old[1] / (zog_old[0] + eps))

        # Get non-linear stochastic ZoG
        zog = cfg.init_v * loss * np.array([-np.sin(theta_noisy), np.cos(theta_noisy)])
        zog_theta = np.tan(zog[1] / (zog[0] + eps))

        # Get non-linear stochastic ZoG with baseline
        zog_bl = (
            cfg.init_v
            * (loss - baselines[theta])
            * np.array([-np.sin(theta_noisy), np.cos(theta_noisy)])
        )
        zog_bl_theta = np.tan(zog_bl[1] / (zog_bl[0] + eps))

        # Store
        grads.append(
            {
                "run": i,
                "th": theta,
                "th_s": theta_noisy,
                "FoG": theta_grad,
                "ZoG": zog_theta,
                "ZoG_old": zog_theta_old,
                "ZoG_bl": zog_bl_theta,
                "cost": loss,
                "baseline": baseline,
            }
        )

grads = pd.DataFrame(grads)
grads.to_csv("grads_samples_{:}_std_{:.2f}.npy".format(cfg.samples, cfg.std))
