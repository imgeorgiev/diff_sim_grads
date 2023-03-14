import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
import sys
sys.path.append(PARENT_DIR)
from utils.customized_integrator_euler import CustomizedSymplecticEulerIntegrator
from utils.customized_integrator_xpbd import CustomizedXPBDIntegratorForGroundWall
from _ground_wall_warp import GroundWall
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from omegaconf import OmegaConf

def to_numpy(array):
    arr = []
    for i in range(len(array)):
        arr.append(array[i].numpy())
    return np.array(arr).squeeze()


yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'ground_wall.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.name += f"_mu_{cfg.customized_mu}"
cfg.THIS_DIR = THIS_DIR

print("------------Task 2: Compliant Model (Warp)-----------")
np.random.seed(0)


iters = 100
noise = 0.2
xs = []
x_grads = {}
for it in range(iters):

    system = GroundWall(
        cfg,
        integrator_class=CustomizedXPBDIntegratorForGroundWall,
        adapter='cpu',
        render=False,
        noise=noise
    )
    loss = system.compute_loss()
    print(f"iter {it} loss: {loss}")

    x = []
    for i in range(len(system.states)):
        x.append(system.states[i].particle_q.numpy())
    x = np.array(x).squeeze()
    x_grad = system.check_grad_pos(system.states)
    x_grad = to_numpy(x_grad)
    ts = np.arange(x_grad.shape[0])
    for t in ts:
        xs.append({"run": it, "t": t,"x": x[t, 0], "y": x[t, 1], "dx": x_grad[t, 0], "dy": x_grad[t, 1]})

xs = pd.DataFrame(xs)
print(xs.shape)

# make plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
sns.lineplot(xs, x="x", y="y", sort=False, ax=ax1)
sns.lineplot(xs, x="t", y="dx", ax=ax2, label="dx")
sns.lineplot(xs, x="t", y="dy", ax=ax2, label="dy")
plt.tight_layout()

ax1.scatter(cfg.init_pos[0], cfg.init_pos[1], label="init_pos")
ax1.scatter(cfg.target[0], cfg.target[1], label="target")
ax1.legend()
ax1.axis("equal")
ax1.set_title("trajectories")

ax2.set_xlabel("timestep")
ax2.legend()
ax2.set_title("gradients")
plt.tight_layout()
plt.savefig("ground_wall_grads_pdb.pdf")
exit(1)
# this should be velocity?
v_grad = system.check_grad(system.states[0].particle_qd)

# this should be velocity control or force?
ctrl0_grad = system.check_grad(system.states[0].external_particle_f)
print(f"gradient of loss w.r.t. initial position dl/dx0: {x_grad.numpy()[0, 0:2]}")
print(f"gradient of loss w.r.t. initial velocity dl/dv0: {v_grad.numpy()[0, 0:2]}")
print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrl0_grad.numpy()[0, 0:2]}")

if cfg.is_train:
    print("---------start training------------")
    loss_np, init_vel_np, last_traj_np = system.train()
    print("---------finish training------------")

    np.savez(
        os.path.join(system.save_dir, cfg.name), 
        loss=loss_np, 
        init_vel=init_vel_np,
        last_traj=last_traj_np,
    )
    print(f"optimized_velocity: {system.states[0].particle_qd}")