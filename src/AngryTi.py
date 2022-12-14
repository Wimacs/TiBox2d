import TiBox2d
import taichi as ti
screen_res = (800, 800)
screen_to_world_ratio = 12
tibox_collector = TiBox2d.Collector()
tibox_collector.add_water_box(0.0, 0.0, 15.0, 15.0)
tibox_collector.add_rigid_box(35.0, 0.0, 37.0, 45.0)
tibox_collector.add_rigid_box(45.0, 0.0, 47.0, 45.0)
tibox_solver = TiBox2d.Solver((screen_res[0] / screen_to_world_ratio,screen_res[1] / screen_to_world_ratio), tibox_collector)
tibox_solver.Compile(tibox_collector)
gui = ti.GUI('PBD',screen_res)
while True:
    tibox_solver.step(tibox_collector)
    gui.circles(tibox_solver.positions.to_numpy() * screen_to_world_ratio / screen_res[0],radius=2,palette=[0x23144, 0x068587, 0xED553B, 0xEEEEF0],palette_indices= tibox_solver.material)
    gui.show()