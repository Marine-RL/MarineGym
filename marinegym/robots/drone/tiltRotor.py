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
class TiltRotorCfg(RobotCfg):
    force_sensor: bool = False
    
class TiltRotor(RobotBase):
    
    param_path: str
    DEFAULT_CONTROLLER: Type = LeePositionController
    cfg_cls = TiltRotorCfg

    def __init__(
        self, 
        name: str = None, 
        cfg: TiltRotorCfg=None, 
        is_articulation: bool = True
    ) -> None:
        super().__init__(name, cfg, is_articulation)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {self.name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]
        self.num_arms = self.params["rotor_configuration"]["num_arms"]

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_arms+self.num_rotors, device=self.device)
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
         
        self.arm_view = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/arm_*",
            name="arm",
            shape=(*self.shape, self.num_arms)
        )
        self.arm_view.initialize()
        arm_joint_indices = [
                i for i, dof_name in enumerate(self._view._dof_names) 
                if dof_name.startswith("arm")
            ]
        self.arm_joint_indices = torch.tensor(arm_joint_indices,device=self.device)
        self.arm_pos = torch.zeros(*self.shape,self.num_arms,device=self.device).squeeze(1)

        self.rotors_view = RigidPrimView(
            # prim_paths_expr=f"{self.prim_paths_expr}/rotor_[0-{self.num_rotors-1}]",
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_*",
            name="rotors",
            shape=(*self.shape, self.num_rotors)
        )
        self.rotors_view.initialize()

        rotor_config = self.params["rotor_configuration"]
        self.rotors = T200(rotor_config, dt=self.dt).to(self.device)

        rotor_params = make_functional(self.rotors)
        # self.KF_0 = rotor_params["KF"].clone()
        # self.KM_0 = rotor_params["KM"].clone()
        self.MAX_ROT_VEL = (
            torch.as_tensor(rotor_config["max_rotation_velocities"])
            .float()
            .to(self.device)
        )
        self.rotor_params = rotor_params.expand(self.shape).clone()
        self.TIME_CONSTANTS_0 = torch.tensor(self.params["rotor_configuration"]['time_constants'], device=self.device)
        self.FORCE_CONSTANTS_0 = torch.tensor(self.params["rotor_configuration"]['force_constants'], device=self.device)

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
        # self.KF = self.rotor_params["KF"]
        # self.KM = self.rotor_params["KM"]
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

        # self.jerk = torch.zeros(*self.shape, 6, device=self.device)
        self.alpha = 0.9

        self.masses = self.base_link.get_masses().clone()
        self.gravity = self.masses * 9.81
        self.inertias = self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        self.volumes = torch.full_like(self.masses, self.params["volume"])
        self.coBMs = torch.full_like(self.masses, self.params["coBM"])
        # default/initial parameters
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
        # self.THRUST2WEIGHT_0 = self.KF_0 / (self.MASS_0 * 9.81) # TODO: get the real g
        # self.FORCE2MOMENT_0 = torch.broadcast_to(self.KF_0 / self.KM_0, self.THRUST2WEIGHT_0.shape)
        
        logging.info(str(self))

        # self.drag_coef = torch.zeros(*self.shape, 1, device=self.device) * self.params["drag_coef"]
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

    

    def setup_randomization(self, cfg):
        if not self.initialized:
            raise RuntimeError

        if not cfg['enable_randomization']:
            return

        self.randomization['enable'] = True
        # 质量随机化 done
        mass_scale = cfg['body'].get("mass_scale", None)
        if mass_scale is not None:
            low = self.MASS_0 * mass_scale[0]
            high = self.MASS_0 * mass_scale[1]
            self.randomization["mass"] = D.Uniform(low, high)

        # 体积随机化 done
        volume_scale = cfg['body'].get("volume_scale", None)
        if volume_scale is not None:
            low = self.VOLUME_0 * volume_scale[0]
            high = self.VOLUME_0 * volume_scale[1]
            self.randomization["volume"] = D.Uniform(low, high)

        # 质心偏移随机化
        coBM_scale = cfg['body'].get("coBM_scale", None)
        if coBM_scale is not None:
            low = self.CoBM_0 * coBM_scale[0]
            high = self.CoBM_0 * coBM_scale[1]
            self.randomization["coBM"] = D.Uniform(low, high)

        # 惯量随机化
        inertia_scale = cfg['body'].get("inertia_scale", None)
        if inertia_scale is not None:
            low = self.INERTIA_0 * torch.as_tensor(inertia_scale[0], device=self.device)
            high = self.INERTIA_0 * torch.as_tensor(inertia_scale[1], device=self.device)
            self.randomization["inertia"] = D.Uniform(low, high)

        # 附加质量随机化
        added_mass_scale = cfg['body'].get("added_mass_scale", None)
        if added_mass_scale is not None:
            low = self.ADDED_MASS_0 * added_mass_scale[0]
            high = self.ADDED_MASS_0 * added_mass_scale[1]
            self.randomization["added_mass"] = D.Uniform(low, high)

        # 线性阻尼随机化
        linear_damping_scale = cfg['body'].get("linear_damping_scale", None)
        if linear_damping_scale is not None:
            low = self.LINEAR_DAMPING_0 * linear_damping_scale[0]
            high = self.LINEAR_DAMPING_0 * linear_damping_scale[1]
            self.randomization["linear_damping"] = D.Uniform(low, high)

        # 二次阻尼随机化
        quadratic_damping_scale = cfg['body'].get("quadratic_damping_scale", None)
        if quadratic_damping_scale is not None:
            low = self.QUADRATIC_DAMPING_0 * quadratic_damping_scale[0]
            high = self.QUADRATIC_DAMPING_0 * quadratic_damping_scale[1]
            self.randomization["quadratic_damping"] = D.Uniform(low, high)

        # 推进器时间常数随机化
        time_constants_scale = cfg['rotor'].get("time_constants_scale", None)
        if time_constants_scale is not None:
            low = self.TIME_CONSTANTS_0 * time_constants_scale[0]
            high = self.TIME_CONSTANTS_0 * time_constants_scale[1]
            self.randomization["time_constants"] = D.Uniform(low, high)

        # 推进器推力常数随机化
        force_constants_scale = cfg['rotor'].get("force_constants_scale", None)
        if force_constants_scale is not None:
            low = self.FORCE_CONSTANTS_0 * force_constants_scale[0]
            high = self.FORCE_CONSTANTS_0 * force_constants_scale[1]
            self.randomization["force_constants"] = D.Uniform(low, high)
        
        logging.info(f"Setup randomization:\n" + pprint.pformat(dict(self.randomization)))

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        # actions[:,:,0:4] = torch.tensor([0.0,0.0,0.0,0.0], device=self.device) # test
        
        cmds = actions.expand(*self.shape, self.num_rotors+self.num_arms)        
        last_throttle = self.throttle.clone()
        arm_cmds, rotor_cmds = cmds.split([self.num_rotors, self.num_arms], dim=-1)
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(0, (*self.shape, self.num_rotors))

        self.thrusts[..., 2] = thrusts
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
        # TODO@btx0424: general rotating rotor
        if self.is_articulation and self.rotor_joint_indices is not None:
            rot_vel = (self.throttle * self.directions * self.MAX_ROT_VEL)/50
            self._view.set_joint_velocities(
                rot_vel.reshape(-1, self.num_rotors),
                joint_indices=self.rotor_joint_indices
            )
            # from omni.isaac.core.utils.types import ArticulationActions
            # view_action = ArticulationActions(joint_velocities=rot_vel,joint_indices=self.rotor_joint_indices)
            # self._view.apply_action(view_action)
        self.arm_pos[:] = arm_cmds.squeeze(1) * 3.14
        self.set_tiltrotor_poses(self.arm_pos)

        self.forces.zero_()
        # # TODO: global downwash
        # if self.n > 1:
        #     self.forces[:] += vmap(self.downwash)(
        #         self.pos,
        #         self.pos,
        #         quat_rotate(self.rot, self.thrusts.sum(-2)),
        #         kz=0.3
        #     ).sum(-2)
        # self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]
        flow_vels = self.flow_vels  + torch.rand_like(self.flow_vels) * self.flow_noise_scale # add guassian noise
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
        body_vels -=  flow_vels_b # relative velocity to the current
        # Rotate the body velocities to the NED frame
        body_vels[..., [1,2,4,5]] *= -1
        body_rpy[...,[1,2]] *= -1

        
        # Calculate accelerations
        body_acc = self.calculate_acc(body_vels)        
        # Calculate damping forces
        damping = self.calculate_damping(body_vels.squeeze(1))
        # Calculate added mass forces
        added_mass = self.calculate_added_mass(body_acc.squeeze(1))
        # Calculate Coriolis forces
        coriolis = self.calculate_corilis(body_vels.squeeze(1))
        # Calculate Buoyancy forces
        buoyancy = self.calculate_buoyancy(body_rpy.squeeze(1))
        
        hydro = - (added_mass + coriolis + damping)
        
        # Rotate the hydrodynamic forces to the ENU frame
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
        # damping = self.linear_damping_matrix[:,0,:,:] @ body_vels.unsqueeze(2)
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
        
        # acc = self.acc.lerp((vel - self.vel) / self.dt, self.alpha)
        # self.acc[:] = acc
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
        # self.jerk[env_ids] = 0.
        if train and 'enable' in self.randomization:
            self._randomize(env_ids, self.randomization)
        # init_throttle = self.gravity[env_ids] / self.KF[env_ids].sum(-1, keepdim=True)
        self.throttle[env_ids] = 0.0
        self.throttle_difference[env_ids] = 0.0
        self.prev_body_acc[env_ids] = 0.0
        self.prev_body_vels[env_ids] = 0.0
        self.flow_vels[env_ids] = torch.rand_like(self.flow_vels[env_ids]) * self.max_flow_vel[env_ids]
        self.arm_pos[env_ids] = 0.0
        return env_ids
    
    def set_tiltrotor_poses(self, arm_pos):
        from omni.isaac.core.utils.types import ArticulationActions
        view_action = ArticulationActions(joint_positions=arm_pos,joint_indices=self.arm_joint_indices)
        self._view.apply_action(view_action)

    def set_flow_velocities(self, env_ids, max_flow_velocity, flow_velocity_gaussian_noise):
        self.max_flow_vel[env_ids,0,:] = torch.tensor(max_flow_velocity,dtype=torch.float32,device=self.device)
        self.flow_noise_scale[env_ids,0,:] = torch.tensor(flow_velocity_gaussian_noise,dtype=torch.float32,device=self.device)
    
    def _randomize(self, env_ids: torch.Tensor, distributions: Dict[str, D.Distribution]):
        shape = env_ids.shape
        if "mass" in distributions:
            masses = distributions["mass"].sample(shape)
            self.base_link.set_masses(masses, env_indices=env_ids)
            self.masses[env_ids] = masses
            self.gravity[env_ids] = masses * 9.81
            self.intrinsics["mass"][env_ids] = (masses / self.MASS_0)
        if 'volume' in distributions:
            self.volumes[env_ids] = distributions["volume"].sample(shape)
            # self.intrinsics["volume"][env_ids] = volumes / self.VOLUME_0
        if 'coBM' in distributions:
            self.coBMs[env_ids] = distributions["coBM"].sample(shape)
        if "inertia" in distributions:
            inertias = distributions["inertia"].sample(shape)
            self.inertias[env_ids] = inertias
            self.base_link.set_inertias(
                torch.diag_embed(inertias).flatten(-2), env_indices=env_ids
            )
            self.intrinsics["inertia"][env_ids] = inertias / self.INERTIA_0
            
        if 'added_mass' in distributions:
            added_mass = distributions["added_mass"].sample(shape)
            self.added_mass_matrix[env_ids,0] = torch.diag_embed(added_mass)
        if 'linear_damping' in distributions:
            linear_damping = distributions["linear_damping"].sample(shape)
            self.linear_damping_matrix[env_ids,0] = torch.diag_embed(linear_damping)
        if 'quadratic_damping' in distributions:
            quadratic_damping = distributions["quadratic_damping"].sample(shape)
            self.quadratic_damping_matrix[env_ids,0] = torch.diag_embed(quadratic_damping)
        if 'time_constants' in distributions:
            time_constants = distributions["time_constants"].sample(shape)
            self.rotor_params['time_constants'][env_ids,0] = time_constants
        if 'force_constants' in distributions:
            force_constants = distributions["force_constants"].sample(shape)
            self.rotor_params['force_constants'][env_ids,0] = force_constants
        # if "com" in distributions:
        #     coms = distributions["com"].sample((*shape, 3))
        #     self.base_link.set_coms(coms, env_indices=env_ids)
        #     self.intrinsics["com"][env_ids] = coms.reshape(*shape, 1, 3)
        # if "thrust2weight" in distributions:
        #     thrust2weight = distributions["thrust2weight"].sample(shape)
        #     KF = thrust2weight * self.masses[env_ids] * 9.81 
        #     self.KF[env_ids] = KF
        #     self.intrinsics["KF"][env_ids] = KF / self.KF_0
        # if "force2moment" in distributions:
        #     force2moment = distributions["force2moment"].sample(shape)
        #     KM = self.KF[env_ids] / force2moment
        #     self.KM[env_ids] = KM
        #     self.intrinsics["KM"][env_ids] = KM / self.KM_0
        # if "drag_coef" in distributions:
        #     drag_coef = distributions["drag_coef"].sample(shape).reshape(-1, 1, 1)
        #     self.drag_coef[env_ids] = drag_coef
        #     self.intrinsics["drag_coef"][env_ids] = drag_coef
        # if "tau_up" in distributions:
        #     tau_up = distributions["tau_up"].sample(shape+self.rotors_view.shape[1:])
        #     self.tau_up[env_ids] = tau_up
        #     self.intrinsics["tau_up"][env_ids] = tau_up
        # if "tau_down" in distributions:
        #     tau_down = distributions["tau_down"].sample(shape+self.rotors_view.shape[1:])
        #     self.tau_down[env_ids] = tau_down
        #     self.intrinsics["tau_down"][env_ids] = tau_down
    
    # def set_world_flow_vels(self, flow_vels: torch.Tensor):
    #     self.flow
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
            # f"Thrust2Weight: {self.THRUST2WEIGHT_0.tolist()}",
            # f"Force2Moment: {self.FORCE2MOMENT_0.tolist()}",
        ])
        return default_params

    @staticmethod
    def downwash(
        p0: torch.Tensor, 
        p1: torch.Tensor,
        p1_t: torch.Tensor,
        kr: float=2,
        kz: float=1,
    ):
        """
        A highly simplified downwash effect model. 
        
        References:
        https://arxiv.org/pdf/2207.09645.pdf
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798116

        """
        z, r = separation(p0, p1, normalize(p1_t))
        z = torch.clip(z, 0)
        v = torch.exp(-0.5 * torch.square(kr * r / z)) / (1 + kz * z)**2
        f = off_diag(v * - p1_t)
        return f
    @staticmethod
    def make(drone_model: str, controller: str=None, device: str="cpu"):
        drone_cls = TiltRotor.REGISTRY[drone_model]
        drone = drone_cls()
        from marinegym.controllers import ControllerBase
        # if controller is not None:
        #     controller_cls = ControllerBase.REGISTRY[controller]
        #     controller = controller_cls(drone.gravity[1], drone.params).to(device)
        controller = None
        return drone, controller

def separation(p0, p1, p1_d):
    rel_pos = rel_pos =  p1.unsqueeze(0) - p0.unsqueeze(1)
    z_distance = (rel_pos * p1_d).sum(-1, keepdim=True)
    z_displacement = z_distance * p1_d

    r_displacement = rel_pos - z_displacement
    r_distance = torch.norm(r_displacement, dim=-1, keepdim=True)
    return z_distance, r_distance
