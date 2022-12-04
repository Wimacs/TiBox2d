import TiBox2d
import taichi as ti

screen_res = (800, 800)
screen_to_world_ratio = 12
tibox_collector = TiBox2d.Collector()
tibox_collector.add_water_box(5.0, 5.0, 15.0, 15.0)
tibox_collector.add_plastic_box(16.0, 20.0, 26.0, 30.0)
tibox_collector.add_rigid_box(27.0, 35.0, 37.0, 45.0)
tibox_collector.add_elastic_box(38.0, 50.0, 48.0, 60.0)
tibox_collector.add_rope(50.0, 0.0, 30.0)
tibox_solver = TiBox2d.Solver((screen_res[0] / screen_to_world_ratio,screen_res[1] / screen_to_world_ratio), tibox_collector)
tibox_solver.Compile(tibox_collector)
gui = ti.GUI('PBD',screen_res)

for i in range(1000):
    tibox_solver.step(tibox_collector)
    gui.circles(tibox_solver.positions.to_numpy() * screen_to_world_ratio / screen_res[0],radius=2,palette=[ 0x068587,0xADFF2F, 0xED553B, 0xEEEEF0, 0xEE82EE],palette_indices= tibox_solver.material)
    filename = f'frame_{i:05d}.png'   # create filename with suffix png
    print(f'Frame {i} is recorded in {filename}')
    gui.show(filename)  # export and show in GUI