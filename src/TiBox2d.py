import taichi as ti
import math
import numpy as np

ti.init(arch=ti.cpu)

cell_size = 2.51
cell_recpr = 1.0 / cell_size
def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


class Collector:
    def __init__(self) -> None:
        self.rigid_box_list = []  
        self.water_box_list = []  
        self.plastic_box_list = []  
        self.num_rigid = 0
        self.num_plastic = 0
        self.num_water = 0
        self.num_particle = 0
        self.h_ = 1.2

    def add_rigid_box(self, position_0_x, position_0_y, position_1_x, position_1_y):
        self.rigid_box_list.append(position_0_x)
        self.rigid_box_list.append(position_0_y)
        self.rigid_box_list.append(position_1_x)
        self.rigid_box_list.append(position_1_y)
        self.num_rigid += 1
        AABB_length_x = position_1_x - position_0_x
        AABB_length_y = position_1_y - position_0_y
        delta = self.h_ * 0.5
        num_particle_x = AABB_length_x // delta
        num_particle_y = AABB_length_y // delta
        self.num_particle += num_particle_x * num_particle_y
    
    def add_plastic_box(self, position_0_x, position_0_y, position_1_x, position_1_y):
        self.plastic_box_list.append(position_0_x)
        self.plastic_box_list.append(position_0_y)
        self.plastic_box_list.append(position_1_x)
        self.plastic_box_list.append(position_1_y)
        self.num_plastic += 1
        AABB_length_x = position_1_x - position_0_x
        AABB_length_y = position_1_y - position_0_y
        delta = self.h_ * 0.5
        num_particle_x = AABB_length_x // delta
        num_particle_y = AABB_length_y // delta
        self.num_particle += num_particle_x * num_particle_y

    def add_water_box(self, position_0_x, position_0_y, position_1_x, position_1_y):
        self.water_box_list.append(position_0_x)
        self.water_box_list.append(position_0_y)
        self.water_box_list.append(position_1_x)
        self.water_box_list.append(position_1_y)
        self.num_water += 1
        AABB_length_x = position_1_x - position_0_x
        AABB_length_y = position_1_y - position_0_y
        delta = self.h_ * 0.5
        num_particle_x = AABB_length_x // delta
        num_particle_y = AABB_length_y // delta
        self.num_particle += num_particle_x * num_particle_y

@ti.data_oriented
class Solver:
    def __init__(self, boundary, Collector : Collector):
        self.grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

        self.dim = 2
        self.num_particles_x = 15
        self.screen_to_world_ratio = 15
        self.num_particles = self.num_particles_x * 45
        self.max_num_particles_per_cell = 100
        self.max_num_neighbors = 100
        self.time_delta = 1.0 / 20.0
        self.epsilon = 1e-2
        self.particle_radius = 2
        self.particle_radius_in_world = self.particle_radius / self.screen_to_world_ratio

        # PBF params
        self.h_ = 1.2
        self.mass = 1.0
        self.rho0 = 1.0
        self.lambda_epsilon = 100.0
        self.pbf_num_iters = 5
        self.corr_deltaQ_coeff = 0.2
        self.corrK = 0.001
        self.neighbor_radius = self.h_ * 1.0
        self.poly6_factor = 315.0 / 64.0 / math.pi
        self.spiky_grad_factor = -45.0 / math.pi


        self.num_particles = int(Collector.num_particle)
        self.num_rigid = int(Collector.num_rigid)
        self.num_plastic = int(Collector.num_plastic)
        self.num_warter = int(Collector.num_water)


        self.old_positions = ti.Vector.field(self.dim, float)
        self.positions = ti.Vector.field(self.dim, float)
        self.positions_rest = ti.Vector.field(self.dim, float)
        self.velocities = ti.Vector.field(self.dim, float)
        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        self.lambdas = ti.field(float)
        self.position_deltas = ti.Vector.field(self.dim, float)
        # 0: x-pos, 1: timestep in sin()
        self.board_states = ti.Vector.field(2, float)
        self.material = ti.field(dtype=int, shape=self.num_particles)

        ti.root.dense(ti.i, self.num_particles).place(self.old_positions, self.positions, self.velocities, self.positions_rest)
        grid_snode = ti.root.dense(ti.ij, self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.k, self.max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.num_particles).place(self.lambdas, self.position_deltas)
        ti.root.place(self.board_states)

        if (self.num_rigid > 0):
            self.rigid_box_start = ti.field(int)
            self.rigid_box_end = ti.field(int)
            self.rigid_box_cm = ti.Vector.field(self.dim,float)
            ti.root.dense(ti.i, self.num_rigid).place(self.rigid_box_start, self.rigid_box_end, self.rigid_box_cm)

        if (self.num_plastic > 0):
            self.plastic_box_start = ti.field(int)
            self.plastic_box_end = ti.field(int)
            self.plastic_box_cm = ti.Vector.field(self.dim,float)
            self.plastic_deform = ti.Matrix.field(self.dim, self.dim, float)
            ti.root.dense(ti.i, self.num_plastic).place(self.plastic_box_start, self.plastic_box_end, self.plastic_box_cm, self.plastic_deform)

        self.board_states[None] = ti.Vector([boundary[0] - self.epsilon, -0.0])

    # compute center of mass
    @ti.func
    def compute_cos(self, x: int, y: int):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0])
        for i in range(x, y):
            cm += self.mass * self.positions[i]
            sum_m += self.mass
        cm /= sum_m
        return cm
        
    @ti.kernel
    def init_rigid_partical(self,rigid_index : int, start_index : int, AABB_bl_x : float, AABB_bl_y : float, AABB_tr_x : float, AABB_tr_y : float) -> int:
        delta = self.h_ * 0.5
        cur_index = start_index
        AABB_bl = ti.Vector([AABB_bl_x, AABB_bl_y])
        AABB_tr = ti.Vector([AABB_tr_x, AABB_tr_y]) 
        AABB_length_x = AABB_tr[0] - AABB_bl[0]
        AABB_length_y = AABB_tr[1] - AABB_bl[1]
        num_particle_x = int(AABB_length_x // delta)
        num_particle_y = int(AABB_length_y // delta)

        self.rigid_box_start[rigid_index] = cur_index

        for i, j in ti.ndrange(num_particle_x, num_particle_y):
            self.positions[cur_index] = ti.Vector([AABB_bl[0] + i * delta, AABB_bl[1] + j * delta])
            self.positions_rest[cur_index] = ti.Vector([AABB_bl[0] + i* delta, AABB_bl[1] + j * delta])
            cur_index += 1
        
        self.rigid_box_end[rigid_index] = cur_index
        self.rigid_box_cm[rigid_index] =  self.compute_cos(self.rigid_box_start[rigid_index], self.rigid_box_end[rigid_index])
        return cur_index

    @ti.kernel
    def init_plastic_partical(self,plastic_index : int, start_index : int, AABB_bl_x : float, AABB_bl_y : float, AABB_tr_x : float, AABB_tr_y : float) -> int:
        delta = self.h_ * 0.5
        cur_index = start_index
        AABB_bl = ti.Vector([AABB_bl_x, AABB_bl_y])
        AABB_tr = ti.Vector([AABB_tr_x, AABB_tr_y]) 
        AABB_length_x = AABB_tr[0] - AABB_bl[0]
        AABB_length_y = AABB_tr[1] - AABB_bl[1]
        num_particle_x = int(AABB_length_x // delta)
        num_particle_y = int(AABB_length_y // delta)

        self.plastic_box_start[plastic_index] = cur_index

        for i, j in ti.ndrange(num_particle_x, num_particle_y):
            self.positions[cur_index] = ti.Vector([AABB_bl[0] + i* delta, AABB_bl[1] + j * delta])
            self.positions_rest[cur_index] = ti.Vector([AABB_bl[0] + i* delta, AABB_bl[1] + j * delta])
            cur_index += 1
        
        self.plastic_box_end[plastic_index] = cur_index
        self.plastic_box_cm[plastic_index] =  self.compute_cos(self.plastic_box_start[plastic_index], self.plastic_box_end[plastic_index])
        self.plastic_deform[plastic_index] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        return cur_index

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = self.poly6_factor * x * x * x
        return result


    @ti.func
    def spiky_gradient(self, r, h):
        result = ti.Vector([0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result


    @ti.func
    def compute_scorr(self, pos_ji):
        # Eq (13)
        x = self.poly6_value(pos_ji.norm(), self.h_) / self.poly6_value(self.corr_deltaQ_coeff * self.h_,
                                                        self.h_)
        # pow(x, 4)
        x = x * x
        x = x * x
        return (-self.corrK) * x


    @ti.func
    def get_cell(self,pos):
        return int(pos * cell_recpr)


    @ti.func
    def is_in_grid(self,c):
        # @c: Vector(i32)
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[
            1] < self.grid_size[1]


    @ti.func
    def confine_position_to_boundary(self, p):
        bmin = self.particle_radius_in_world
        bmax = ti.Vector([self.board_states[None][0], boundary[1]
                        ]) - self.particle_radius_in_world
        for i in ti.static(range(self.dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin 
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - self.epsilon * ti.random()
        return p

    @ti.func
    def confine_velocity_to_boundary(self, p, v):
        bmin = self.particle_radius_in_world
        bmax = ti.Vector([self.board_states[None][0], boundary[1]
                        ]) - self.particle_radius_in_world
        for i in ti.static(range(self.dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                v[i] = -v[i] 
            elif bmax[i] <= p[i]:
                v[i] = -v[i]
        return v


    @ti.kernel
    def nearest_neigbour(self):
        # save old positions
        for i in self.positions:
            self.old_positions[i] = self.positions[i]
        # apply gravity within boundary
        for i in self.positions:
            g = ti.Vector([0.0, -9.8])
            pos, vel = self.positions[i], self.velocities[i]
            vel += g * self.time_delta
            pos += vel * self.time_delta
            self.positions[i]  = self.confine_position_to_boundary(pos)

        # clear neighbor lookup table
        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        # update grid
        for p_i in self.positions:
            cell = self.get_cell(self.positions[p_i])
            # ti.Vector doesn't seem to support unpacking yet
            # but we can directly use int Vectors as indices
            offs = ti.atomic_add(self.grid_num_particles[cell], 1)
            self.grid2particles[cell, offs] = p_i
        # find particle neighbors
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            cell = self.get_cell(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        p_j = self.grid2particles[cell_to_check, j]
                        if nb_i < self.max_num_neighbors and p_j != p_i and (
                                pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i


    @ti.kernel
    def substep(self):
        # compute lambdas
        # Eq (8) ~ (11)
        for p_i in self.positions:
            pos_i = self.positions[p_i]

            grad_i = ti.Vector([0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                grad_j = self.spiky_gradient(pos_ji, self.h_)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += self.poly6_value(pos_ji.norm(), self.h_)

            # Eq(1)
            density_constraint = (self.mass * density_constraint / self.rho0) - 1.0

            sum_gradient_sqr += grad_i.dot(grad_i)
            self.lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                    self.lambda_epsilon)
        # compute position deltas
        # Eq(12), (14)
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            lambda_i = self.lambdas[p_i]

            pos_delta_i = ti.Vector([0.0, 0.0])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                lambda_j = self.lambdas[p_j]
                pos_ji = pos_i - self.positions[p_j]
                scorr_ij = self.compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                    self.spiky_gradient(pos_ji, self.h_)

            pos_delta_i /= self.rho0
            self.position_deltas[p_i] = pos_delta_i
        # apply position deltas
        for i in self.positions:
            self.positions[i] += self.position_deltas[i]


    @ti.kernel
    def apply_boundary(self):
        # update velocities
        for i in self.positions:
            self.velocities[i] = (self.positions[i] - self.old_positions[i]) / self.time_delta
            pos = self.positions[i]
            vel = self.velocities[i]
            #self.velocities[i] = self.confine_velocity_to_boundary(pos, vel)
        # confine to boundary
        for i in self.positions:
            pos = self.positions[i]
            self.positions[i] = self.confine_position_to_boundary(pos)

    @ti.kernel
    def shape_matching_rigid(self, num_rigid_body : int):
        for rigid_index in range(2):
            start_index = self.rigid_box_start[rigid_index]
            end_index = self.rigid_box_end[rigid_index]
            rest_cm = self.rigid_box_cm[rigid_index]
            cm = self.compute_cos(start_index, end_index)
            # A
            A = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            for idx in range(start_index, end_index):
                q = self.positions_rest[idx] - rest_cm
                p = self.positions[idx] - cm
                A += p @ q.transpose()
            R, S = ti.polar_decompose(A)
            for idx in range(start_index, end_index):
                goal = cm + R @ (self.positions_rest[idx] - rest_cm)
                corr = (goal - self.positions[idx])
                self.positions[idx] += corr
    
    @ti.kernel
    def shape_matching_plastic(self, num_plastic_body : int):
        for plastic_index in range(2):
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            start_index = self.plastic_box_start[plastic_index]
            end_index = self.plastic_box_end[plastic_index]
            rest_cm = self.plastic_box_cm[plastic_index]
            cm = self.compute_cos(start_index, end_index)
            SP = self.plastic_deform[plastic_index]
            # A
            A = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            for idx in range(start_index, end_index):
                q = SP @ (self.positions_rest[idx] - rest_cm)
                p = self.positions[idx] - cm
                A += p @ q.transpose()
            R, S = ti.polar_decompose(A)
            SP = (I + 0.001 * (S - I)) @ SP
            self.plastic_deform[plastic_index] = SP
            for idx in range(start_index, end_index):
                goal = cm + R @ (self.positions_rest[idx] - rest_cm)
                corr = (goal - self.positions[idx])
                self.positions[idx] += corr

screen_res = (800, 800)
screen_to_world_ratio = 20
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)
fuck = Collector()
fuck.add_rigid_box(5.0, 10.0, 8.0, 13.0)
fuck.add_rigid_box(5.0, 5.0, 8.0, 8.0)
fuck.add_plastic_box(20.0, 20.0, 24.0, 24.0)
shit = Solver(boundary, fuck)

particle_index = 0
for rigid_index in range(fuck.num_rigid):
    particle_index = shit.init_rigid_partical(rigid_index, particle_index, fuck.rigid_box_list[rigid_index * 4 + 0], fuck.rigid_box_list[rigid_index * 4 + 1]
        , fuck.rigid_box_list[rigid_index * 4 + 2], fuck.rigid_box_list[rigid_index * 4 + 3])
for plastic_index in range(fuck.num_plastic):
    particle_index = shit.init_plastic_partical(plastic_index, particle_index, fuck.plastic_box_list[plastic_index * 4 + 0], fuck.plastic_box_list[plastic_index * 4 + 1]
        , fuck.plastic_box_list[plastic_index * 4 + 2], fuck.plastic_box_list[plastic_index * 4 + 3])
    
gui = ti.GUI('PBD',screen_res)

while True:

    shit.nearest_neigbour()
    #for _ in range(shit.pbf_num_iters):
    if (fuck.num_rigid > 0):
        shit.shape_matching_rigid(fuck.num_rigid)
    if (fuck.num_plastic > 0):
        shit.shape_matching_plastic(fuck.num_plastic)
    for _ in range(shit.pbf_num_iters):
        shit.substep()
    shit.apply_boundary()

    pos_np = shit.positions.to_numpy()
    for j in range(2):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    #print(pos_np)
    gui.circles(pos_np,
                radius=2,
                color=0x168587)
    # Change to gui.show(f'{frame:06d}.png') to write images to disk
    gui.show()