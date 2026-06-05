import torch
import torch.nn as nn


class M600(nn.Module):
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
        
        target_rpm = torch.where(self.throttle > 0.075, 5.6599e+02 * self.throttle + 3.4521e+01,
        torch.where(self.throttle < -0.075, 5.4944e+02 * self.throttle - 4.3350e+01, torch.zeros_like(self.throttle)
        ))
        alpha = torch.exp(-self.dt / self.time_constants)
        
        noise = torch.randn_like(self.rpm) * self.noise_scale * 0.
        rpm = alpha * self.rpm + (1 - alpha) * target_rpm
        self.rpm = torch.clamp(rpm + noise, -600, 600)
        
        thrusts = self.force_constants * torch.abs(self.rpm) * self.rpm
        moments = thrusts * -self.directions * 0        

        return thrusts, moments