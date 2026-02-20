import genesis as gs
gs.init(logging_level="warning")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    rigid_options=gs.options.RigidOptions(dt=0.01)
)
mat = gs.materials.Rigid(gravity_compensation=0.5)
sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0,0,1)), surface=gs.surfaces.Default(), material=mat)
scene.build(n_envs=1)

print("material:", sphere.material)
try:
    sphere.material.gravity_compensation = 0.2
    print("set material grav comp success!")
except Exception as e:
    print("Failed to set via material:", e)

# test applying control force to base
try:
    sphere.control_dofs_force(torch.tensor([[0,0,1,0,0,0]]))
    print("control_dofs_force works")
except Exception as e:
    print("Failed control_dofs_force:", e)
