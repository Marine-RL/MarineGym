import torch

from marinegym.robots.robot import ASSET_PATH
from marinegym.robots.drone.underwaterVehicle import UnderwaterVehicle

class BlueROV(UnderwaterVehicle):

    usd_path: str = ASSET_PATH + "/usd/BlueROV/BlueROV.usd"
    param_path: str = ASSET_PATH + "/usd/BlueROV/BlueROV.yaml"
    fixed_usd_path: str = ASSET_PATH + "/usd/BlueROV/fixed_usd/BlueROV/BlueROV.usd"