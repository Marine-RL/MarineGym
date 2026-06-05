import torch
import torch.nn as nn
from marinegym.utils.torch import (
    normalize, off_diag, quat_rotate, quat_rotate_inverse, quat_axis, symlog, quaternion_to_euler, euler_to_quaternion, quat_mul
)

class fin200(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        self.liftConstant = nn.Parameter(torch.as_tensor(rotor_config["lift_constants"]))
        self.dragConstant = nn.Parameter(torch.as_tensor(rotor_config["drag_constants"]))
        self.num_fins = len(self.liftConstant)
        self.max_joint_limit = nn.Parameter(torch.as_tensor(rotor_config["angle_threshold"]))
        self.time_constants = nn.Parameter(torch.as_tensor(rotor_config["time_constants"]))

        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        self.noise_scale = 0.002

        self.throttle = nn.Parameter(torch.zeros(self.num_fins))

        self.angle = nn.Parameter(torch.zeros(self.num_fins))

        self.f = torch.square
        self.f_inv = torch.sqrt

        self.requires_grad_(False)

    def forward(self, cmds: torch.Tensor, flow_vels: torch.Tensor, fin_pos, fin_rot, fin_vel):
        target_angle = torch.clamp(cmds, -1, 1)
        alpha = torch.exp(-self.dt / self.time_constants)
        noise = torch.randn_like(self.angle) * self.noise_scale * 0.

        angle = alpha * self.angle + (1 - alpha) * target_angle
        self.angle = torch.clamp(angle + noise, -self.max_joint_limit, self.max_joint_limit)
        


        unitz = torch.zeros(fin_rot.shape[0], 3, device='cuda')
        unitz[:, 2] = 1
        ldNormalI = quat_rotate(fin_rot,unitz)
        velI = fin_vel - flow_vels
        velInLDPlaneI = torch.cross(ldNormalI, torch.cross(velI[...,0:3], ldNormalI,dim=1), dim=1)
        _velInLDPlaneL = quat_rotate_inverse(fin_rot, velInLDPlaneI)

        attack_angle = torch.atan(_velInLDPlaneL[..., 1] / _velInLDPlaneL[..., 0])
        update_angle = torch.where(attack_angle > 1.57, attack_angle - 3.14, attack_angle)
        update_angle = torch.where(attack_angle < -1.57, attack_angle + 3.14, attack_angle)
        update_velInLDPlaneL = torch.where(attack_angle.unsqueeze(1).repeat(1, 3) > 1.57, -_velInLDPlaneL, _velInLDPlaneL)
        update_velInLDPlaneL = torch.where(attack_angle.unsqueeze(1).repeat(1, 3) < -1.57, -_velInLDPlaneL, _velInLDPlaneL)

        u = torch.norm(update_velInLDPlaneL, dim=1)
        u2 = u ** 2
        du2 = update_angle * u2

        drag = update_angle * du2 * self.dragConstant
        lift = du2 * self.liftConstant
        liftDirectionL = - torch.cross(unitz, _velInLDPlaneL, dim=1)
        dragDirectionL = -_velInLDPlaneL

        liftDirectionL_unit = liftDirectionL / liftDirectionL.norm(dim=1, keepdim=True)
        dragDirectionL_unit = dragDirectionL / dragDirectionL.norm(dim=1, keepdim=True)

        lift_vector = lift.unsqueeze(1) * liftDirectionL_unit
        drag_vector = drag.unsqueeze(1) * dragDirectionL_unit

        total_force = lift_vector + drag_vector
        return total_force, self.angle