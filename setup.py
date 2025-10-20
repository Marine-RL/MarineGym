from setuptools import find_packages, setup

setup(
    name="marinegym",
    version="1.0",
    author="chusg@zju.edu.cn",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "imageio",
        "plotly",
        "einops",
        "pandas",
        "moviepy",
        "av",
        "torchrl==0.4.0", # for torch==2.2.2
        "tensordict==0.4.0",
    ],
)
