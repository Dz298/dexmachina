import genesis as gs
import torch
import numpy as np
gs.init(logging_level="warning")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    rigid_options=gs.options.RigidOptions(dt=0.01),
    show_viewer=False
)
mat = gs.materials.Rigid(gravity_compensation=0.0) # we will compensate manually
sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0,0,1)), surface=gs.surfaces.Default(), material=mat)
scene.build(n_envs=1)


rigid_solver = scene.rigid_solver

mass = sphere.get_mass()
print("Mass:", mass)
# apply upward force equal to gravity
force = torch.zeros((1, sphere.n_links, 3), device=gs.device)
force[..., 2] = mass * 9.81

pos_before = sphere.get_pos()[0, 2].item()
for _ in range(100):
    # apply_links_external_force(force, links_idx) where force is (num_envs, num_links, 3)
    rigid_solver.apply_links_external_force(force, [sphere.links[0].idx])
    scene.step()
pos_after = sphere.get_pos()[0, 2].item()
print("pos before:", pos_before, "pos after:", pos_after)
