import torch

from marinegym.robots.robot import ASSET_PATH
from marinegym.robots.drone.underwaterVehicle import UnderwaterVehicle

class BlueROVHeavy(UnderwaterVehicle):

    usd_path: str = ASSET_PATH + "/usd/BlueROVHeavy/BlueROVHeavy.usd"
    param_path: str = ASSET_PATH + "/usd/BlueROVHeavy/BlueROVHeavy.yaml"
    fixed_usd_path: str = ASSET_PATH + "/usd/BlueROVHeavy/fixed_usd/BlueROVHeavy/BlueROVHeavy.usd"