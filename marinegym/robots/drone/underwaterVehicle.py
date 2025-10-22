import logging
from typing import Type, Dict

import torch
import torch.distributions as D
import yaml
from functorch import vmap
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict

from marinegym.views import RigidPrimView
from marinegym.actuators.rotor_group import RotorGroup
from marinegym.actuators.t200 import T200
from marinegym.controllers import LeePositionController

from marinegym.robots import RobotBase, RobotCfg
from marinegym.utils.torch import (
    normalize, off_diag, quat_rotate, quat_rotate_inverse, quat_axis, symlog, quaternion_to_euler
)

from dataclasses import dataclass
from collections import defaultdict

import pprint
import torch.distributions as D

@dataclass
class UnderwaterVehicleCfg(RobotCfg):
    force_sensor: bool = False
    
class UnderwaterVehicle(RobotBase):
    
    param_path: str
    DEFAULT_CONTROLLER: Type = LeePositionController
    cfg_cls = UnderwaterVehicleCfg

    def __init__(
        self, 
        name: str = None, 
        cfg: UnderwaterVehicleCfg=None, 
        is_articulation: bool = True
    ) -> None:
        super().__init__(name, cfg, is_articulation)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {self.name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            "com": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
        }).to(self.device)
        
        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 6
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)
        self.randomization = defaultdict(dict)

    def initialize(
        self, 
        prim_paths_expr: str = None,
        track_contact_forces: bool = False
    ):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/base_link",
                name="base_link",
                track_contact_forces=track_contact_forces,
                shape=self.shape,
            )
            self.base_link.initialize()
            print(self._view.dof_names)
            print(self._view._dof_indices)
            rotor_joint_indices = [
                i for i, dof_name in enumerate(self._view._dof_names) 
                if dof_name.startswith("rotor")
            ]
            if len(rotor_joint_indices):
                self.rotor_joint_indices = torch.tensor(
                    rotor_joint_indices,
                    device=self.device
                )
            else:
                self.rotor_joint_indices = None
        else:
            super().initialize(prim_paths_expr=f"{prim_paths_expr}/base_link")
            self.base_link = self._view
            self.prim_paths_expr = prim_paths_expr

        self.rotors_view = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_*",
            name="rotors",
            shape=(*self.shape, self.num_rotors)
        )
        self.rotors_view.initialize()

        rotor_config = self.params["rotor_configuration"]
        self.rotors = T200(rotor_config, dt=self.dt).to(self.device)

        rotor_params = make_functional(self.rotors)
        self.rotor_params = rotor_params.expand(self.shape).clone()
        self.TIME_CONSTANTS_0 = torch.tensor(self.params["rotor_configuration"]['time_constants'], device=self.device)
        self.FORCE_CONSTANTS_0 = torch.tensor(self.params["rotor_configuration"]['force_constants'], device=self.device)

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
        self.throttle = self.rotor_params["throttle"]
        self.directions = self.rotor_params["directions"]

        self.thrusts = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)
        self.forces = torch.zeros(*self.shape, 3, device=self.device)

        self.pos, self.rot = self.get_world_poses(True)
        self.throttle_difference = torch.zeros(self.throttle.shape[:-1], device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = self.vel_w = torch.zeros(*self.shape, 6, device=self.device)
        self.vel_b = torch.zeros_like(self.vel_w)
        self.acc = self.acc_w = torch.zeros(*self.shape, 6, device=self.device)
        self.acc_b = torch.zeros_like(self.acc_w)

        self.alpha = 0.9

        self.masses = self.base_link.get_masses().clone()
        self.gravity = self.masses * 9.81
        self.inertias = self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        self.volumes = torch.full_like(self.masses, self.params["volume"])
        self.coBMs = torch.full_like(self.masses, self.params["coBM"])
        self.MASS_0 = self.masses[0].clone()
        self.INERTIA_0 = (
            self.base_link
            .get_inertias()
            .reshape(*self.shape, 3, 3)[0]
            .diagonal(0, -2, -1)
            .clone()
        )
        self.VOLUME_0 = torch.tensor([[self.params["volume"]]]).to(self.MASS_0)
        self.CoBM_0 = torch.tensor([[self.params["coBM"]]]).to(self.MASS_0)
        self.ADDED_MASS_0 = torch.tensor(self.params["hydro_coef"]["added_mass"]).to(self.MASS_0)
        self.LINEAR_DAMPING_0 = torch.tensor(self.params["hydro_coef"]["linear_damping"]).to(self.MASS_0)
        self.QUADRATIC_DAMPING_0 = torch.tensor(self.params["hydro_coef"]["quadratic_damping"]).to(self.MASS_0)

        logging.info(str(self))

        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()
        
        self.prev_body_vels = torch.zeros(*self.shape, 6, device=self.device)     
        self.prev_body_acc= torch.zeros(*self.shape, 6, device=self.device)   
        hydro_coef=self.params['hydro_coef']
        self.added_mass_matrix = torch.diag(torch.tensor(hydro_coef["added_mass"])).repeat(*self.shape, 1, 1).to(self.device)
        self.linear_damping_matrix = torch.diag(torch.tensor(hydro_coef["linear_damping"])).repeat(*self.shape, 1, 1).to(self.device)
        self.quadratic_damping_matrix = torch.diag(torch.tensor(hydro_coef["quadratic_damping"])).repeat(*self.shape, 1, 1).to(self.device)
        self.flow_vels = torch.zeros(*self.shape, 6, device=self.device)
        self.max_flow_vel = torch.zeros(*self.shape, 6, device=self.device)
        self.flow_noise_scale = torch.zeros(*self.shape, 6, device=self.device)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)        
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(0, (*self.shape, self.num_rotors))

        self.thrusts[..., 0] = thrusts
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
        self.forces.zero_()
        flow_vels = self.flow_vels  + torch.rand_like(self.flow_vels) * self.flow_noise_scale
        hydro_forces, hydro_torques = self.apply_hydrodynamic_forces(flow_vels)
        self.forces += hydro_forces
        self.torques += hydro_torques

        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3), 
            is_global=False
        )

        self.base_link.apply_forces_and_torques_at_pos(
            self.forces.reshape(-1, 3), 
            self.torques.reshape(-1, 3),
            is_global=False
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)

    def apply_hydrodynamic_forces(self, flow_vels_w) -> TensorDict:

        body_vels = self.vel_b.clone()
        body_rpy = quaternion_to_euler(self.rot)
        flow_vels_b = torch.cat([
            quat_rotate_inverse(self.rot, flow_vels_w[..., :3]),
            quat_rotate_inverse(self.rot, flow_vels_w[..., 3:])
        ], dim=-1)
        body_vels -=  flow_vels_b
        body_vels[..., [1,2,4,5]] *= -1
        body_rpy[...,[1,2]] *= -1

        body_acc = self.calculate_acc(body_vels)        
        damping = self.calculate_damping(body_vels.squeeze(1))
        added_mass = self.calculate_added_mass(body_acc.squeeze(1))
        coriolis = self.calculate_corilis(body_vels.squeeze(1))
        buoyancy = self.calculate_buoyancy(body_rpy.squeeze(1))
        
        hydro = - (added_mass + coriolis + damping)
        hydro[:, [1, 2, 4, 5]] *= -1
        buoyancy[:, [1, 2, 4, 5]] *= -1
        hydro = hydro.unsqueeze(1)
        buoyancy = buoyancy.unsqueeze(1)

        return hydro[..., 0:3] + buoyancy[..., 0:3], hydro[..., 3:6] + buoyancy[..., 3:6]
    
    def calculate_acc(self, body_vels):
        alpha = 0.3
        acc = (body_vels - self.prev_body_vels) / self.dt
        filteredAcc = (1.0-alpha)* self.prev_body_acc + alpha * acc
        self.prev_body_vels = body_vels.clone()
        self.prev_body_acc = filteredAcc.clone() 

        return filteredAcc
    
    def calculate_damping(self, body_vels):
        maintained_body_vels = torch.diag_embed(body_vels)
        maintained_body_vels[:, 1, 5] = body_vels[:, 5]
        maintained_body_vels[:, 2, 4] = body_vels[:, 4]
        maintained_body_vels[:, 4, 2] = body_vels[:, 2]
        maintained_body_vels[:, 5, 1] = body_vels[:, 1] 
        damping_matrix = self.linear_damping_matrix[:,0,:,:] + self.quadratic_damping_matrix[:,0,:,:] * torch.abs(maintained_body_vels)
        damping = damping_matrix @ body_vels.unsqueeze(2)
        damping = damping.squeeze(2)   
        
        return damping
    
    def calculate_added_mass(self, body_acc):
        added_mass = self.added_mass_matrix[:,0,:,:] @ body_acc.unsqueeze(2)
        added_mass = added_mass.squeeze(2)
        
        return added_mass
    
    def calculate_corilis(self, body_vels):
        ab = self.added_mass_matrix[:,0,:,:] @ body_vels.unsqueeze(2)
        ab =ab.squeeze(2)
        coriolis = torch.zeros(*self.shape, 6, device=self.device)
        coriolis.squeeze_(dim=1)
        coriolis[:, 0:3] = - torch.cross(ab[:, 0:3], body_vels[:, 3:6], dim=1)
        coriolis[:, 3:6] = - (torch.cross(ab[:, 0:3], body_vels[:, 0:3], dim=1) + torch.cross(ab[:, 3:6], body_vels[:, 3:6], dim=1))
        
        return coriolis
    
    def calculate_buoyancy(self, rpy):
        buoyancy = torch.zeros(*self.shape, 6, device=self.device)
        buoyancy.squeeze_(dim=1)
        buoyancyForce = 997 * 9.8 * self.volumes[:,0,0]
        dis = self.coBMs[:,0,0]
        buoyancy[:, 0] = buoyancyForce * torch.sin(rpy[:,1])
        buoyancy[:, 1] = -buoyancyForce * torch.sin(rpy[:,0]) * torch.cos(rpy[:,1])
        buoyancy[:, 2] = -buoyancyForce * torch.cos(rpy[:,0]) * torch.cos(rpy[:,1])
        buoyancy[:, 3] = - dis * buoyancyForce * torch.cos(rpy[:,1]) * torch.sin(rpy[:,0])
        buoyancy[:, 4] = - dis * buoyancyForce * torch.sin(rpy[:,1])
        
        return buoyancy

    def get_state(self, check_nan: bool=False, env_frame: bool=True):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if env_frame and hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions)
        
        vel_w = self.get_velocities(True)
        vel_b = torch.cat([
            quat_rotate_inverse(self.rot, vel_w[..., :3]),
            quat_rotate_inverse(self.rot, vel_w[..., 3:])
        ], dim=-1)
        self.vel_w[:] = vel_w
        self.vel_b[:] = vel_b
        
        self.heading[:] = quat_axis(self.rot, axis=0)
        self.up[:] = quat_axis(self.rot, axis=2)
        state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle * 2 - 1]
        if self.use_force_sensor:
            self.force_readings, self.torque_readings = self.get_force_sensor_forces().chunk(2, -1)
            # normalize by mass and inertia
            force_reading_norms = self.force_readings.norm(dim=-1, keepdim=True)
            force_readings = (
                self.force_readings
                / force_reading_norms
                * symlog(force_reading_norms)
                / self.gravity.unsqueeze(-2)
            )
            torque_readings = self.torque_readings / self.INERTIA_0.unsqueeze(-2)
            state.append(force_readings.flatten(-2))
            state.append(torque_readings.flatten(-2))
        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        self.thrusts[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
        self.throttle[env_ids] = 0.0
        self.throttle_difference[env_ids] = 0.0
        self.prev_body_acc[env_ids] = 0.0
        self.prev_body_vels[env_ids] = 0.0
        self.flow_vels[env_ids] = torch.rand_like(self.flow_vels[env_ids]) * self.max_flow_vel[env_ids]
        return env_ids
    
    def set_flow_velocities(self, env_ids, max_flow_velocity, flow_velocity_gaussian_noise):
        self.max_flow_vel[env_ids,0,:] = torch.tensor(max_flow_velocity,dtype=torch.float32,device=self.device)
        self.flow_noise_scale[env_ids,0,:] = torch.tensor(flow_velocity_gaussian_noise,dtype=torch.float32,device=self.device)
    
    def get_thrust_to_weight_ratio(self):
        return self.KF.sum(-1, keepdim=True) / (self.masses * 9.81)

    def get_linear_smoothness(self):
        return - (
            torch.norm(self.acc[..., :3], dim=-1) 
            + torch.norm(self.jerk[..., :3], dim=-1)
        )
    
    def get_angular_smoothness(self):
        return - (
            torch.sum(self.acc[..., 3:].abs(), dim=-1)
            + torch.sum(self.jerk[..., 3:].abs(), dim=-1)
        )
    
    def __str__(self):
        default_params = "\n".join([
            "Default parameters:",
            f"Mass: {self.MASS_0.tolist()}",
            f"Inertia: {self.INERTIA_0.tolist()}",
        ])
        return default_params

    @staticmethod
    def make(drone_model: str, controller: str=None, device: str="cpu"):
        drone_cls = UnderwaterVehicle.REGISTRY[drone_model]
        drone = drone_cls()
        from marinegym.controllers import ControllerBase
        controller = None
        return drone, controller

