import torch
import torch.nn as nn


class T200(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        self.force_constants = nn.Parameter(torch.as_tensor(rotor_config["force_constants"]))
        self.moment_constants = nn.Parameter(torch.as_tensor(rotor_config["moment_constants"]))
        self.max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        self.num_rotors = len(self.force_constants)

        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        self.noise_scale = 0.002

        # self.KF = nn.Parameter(max_rot_vels.square() * force_constants)
        # self.KM = nn.Parameter(max_rot_vels.square() * moment_constants)
        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]).float())

        self.tau_up = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        
        self.rpm = nn.Parameter(torch.zeros(self.num_rotors))
        self.time_constants = nn.Parameter(torch.as_tensor(rotor_config["time_constants"]))

        self.f = torch.square
        self.f_inv = torch.sqrt

        self.requires_grad_(False)

    def forward(self, cmds: torch.Tensor):
        target_throttle = torch.clamp(cmds, -1, 1)

        tau = torch.where(target_throttle > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0, 1)
        self.throttle.add_(tau * (target_throttle - self.throttle))
        
        # 修改目标转速的计算逻辑
        target_rpm = torch.where(self.throttle > 0.075, 3.6599e+03 * self.throttle + 3.4521e+02,
        torch.where(self.throttle < -0.075, 3.4944e+03 * self.throttle - 4.3350e+02, torch.zeros_like(self.throttle)
        ))
        alpha = torch.exp(-self.dt / self.time_constants)
        
        noise = torch.randn_like(self.rpm) * self.noise_scale * 0.
        rpm = alpha * self.rpm + (1 - alpha) * target_rpm
        self.rpm = torch.clamp(rpm + noise, -3900, 3900)
        
        thrusts = self.force_constants /4.4e-7 * 9.81 * torch.where(self.rpm>0, 4.7368e-07 * self.f(self.rpm) - 1.9275e-04 * self.rpm + 8.4452e-02, -3.8442e-07 * self.f(self.rpm) - 1.6186e-04 * self.rpm - 3.9139e-02)# rpm2force
        
        moments = thrusts * -self.directions * 0        

        return thrusts, moments
