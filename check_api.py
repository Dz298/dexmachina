import genesis as gs
gs.init(logging_level="warning")
print("Writing test")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    rigid_options=gs.options.RigidOptions(dt=0.01)
)
sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0,0,1)))
scene.build(n_envs=2)

print(type(sphere.gravity_compensation))
print("val:", sphere.gravity_compensation)
try:
    sphere.gravity_compensation = 0.5
    print("Set to 0.5 success. val:", sphere.gravity_compensation)
except Exception as e:
    print("set failed:", e)
    
try:
    import torch
    sphere.gravity_compensation = torch.tensor([0.5, 0.2]).cuda()
    print("Set to tensor success. val:", sphere.gravity_compensation)
except Exception as e:
    print("set tensor failed:", e)
