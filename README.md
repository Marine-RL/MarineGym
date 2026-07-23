![Visualization of MarineGym](docs/overview.png)

---

# MarineGym

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-4.1.0-orange.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/archived_release_notes.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://marinegym.netlify.app/)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmarine-gym.com&label=website&up_message=online&down_message=offline)](https://marine-gym.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*MarineGym* is a large-scale parallel framework designed for reinforcement learning research on unmanned underwater vehicles (UUVs). It is built upon [OmniDrones](https://github.com/btx0424/OmniDrones) and [Isaac Sim](https://developer.nvidia.com/isaac/sim), offering the following features:

- Efficiency: Achieve a simulation speed of up to 10<sup>7</sup> steps per second.
- Fidelity: Accurately replicate the physical environment, including physical laws, kinematics, and dynamics.
- Flexibility:  Ensure compatibility with existing RL frameworks and offer user-friendly APIs to facilitate seamless integration and usage.
- Evaluation: Assesses and contrasts various RL strategies through multiple tasks.

> [!TIP]
>
> 🚀 **Collaborate with us on Underwater Embodied AI!**
>
> We are actively seeking research partners in the field of Underwater Embodied Intelligence and Reinforcement Learning. If you are interested in leveraging MarineGym for your project, please contact us at:
>
> 📮 **Email**: zjuoyh@163.com

## Installation

To install MarineGym, we recommend reading one of the following guides:
- [Installation from Source](https://marinegym.netlify.app/installation_from_source) (recommended for development)
- [Native Windows Installation](docs/windows.md) (Isaac Sim 4.1.0 workstation package)
- [Docker Environment](https://marinegym.netlify.app/docker_environment) (recommended for training purposes; no visualization interface)

If you encounter any issues, you can find solutions to common problems in the [FAQ](https://marinegym.netlify.app/faq) or feel free to open an issue.

For training and evaluation commands, please take a look at the [Quick Start](https://marinegym.netlify.app/quick_start).

## Usage
For installation details, please refer to our [Setup Guide](https://marinegym.netlify.app/installation_from_source/).

Currently, five gym environments are verified: Hover, Circle Tracking, Helical Tracking, Lemniscate Tracking, and Landing. Additional environments, including vision-based and sonar-based tasks, are under development.

The training script is located in the `scripts` folder, named `train.py`.


To start the training process, run:

```bash
python train.py task=Hover algo=ppo headless=false enable_livestream=false
```
where `task` specifies the training scenario, which can be `Hover`, `Track`, or `Landing`.


## Citation

If you build on this work, please cite our paper:

```bibtex
@inproceedings{chu2025marinegym,
  title={MarineGym: A high-performance reinforcement learning platform for underwater robotics},
  author={Chu, Shuguang and Huang, Zebin and Li, Yutong and Lin, Mingwei and Li, Dejun and Carlucho, Ignacio and Petillot, Yvan R and Yang, Canjun},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={17146--17153},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement

The architecture and certain implementation ideas build upon concepts introduced in [OmniDrones](https://github.com/btx0424/OmniDrones).
