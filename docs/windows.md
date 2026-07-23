# Windows installation

MarineGym provides experimental native Windows support through the Python environment bundled with
Isaac Sim 4.1.0. A separate Conda environment is not used because Isaac Sim
does not ship a Windows equivalent of `setup_conda_env.sh`.

## Requirements

- Windows 11
- NVIDIA RTX GPU and a driver compatible with Isaac Sim 4.1.0
- Isaac Sim 4.1.0 installed as a workstation package
- PowerShell 5.1 or newer
- Git

Enable Windows long paths before installation. Run the following command in an
elevated PowerShell terminal, then restart the terminal:

```powershell
New-ItemProperty `
  -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled `
  -Value 1 `
  -PropertyType DWORD `
  -Force
```

## Install

Set the Isaac Sim installation path for the current terminal. The directory
must contain `python.bat` and `apps\omni.isaac.sim.python.kit`.

```powershell
$env:ISAACSIM_PATH = "C:\isaacsim"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

The setup script installs MarineGym and its Python dependencies into the
Python environment bundled with Isaac Sim.

## Run

Run training from the repository root:

```powershell
.\scripts\marinegym.ps1 train `
  task=Hover `
  algo=ppo `
  headless=false `
  enable_livestream=false
```

For a short smoke test, reduce the environment count and total frames using
Hydra overrides appropriate for the selected task.

## Limitations

- The Docker workflow remains Linux-only.
- The tested runtime is Isaac Sim 4.1.0 with its bundled Python 3.10.
- WSL is not part of this native Windows workflow.
- GPU simulation behavior must still meet NVIDIA's Isaac Sim requirements.
