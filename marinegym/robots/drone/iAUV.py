import torch

from marinegym.robots.robot import ASSET_PATH
from marinegym.robots.drone.underwaterVehicle import UnderwaterVehicle
from marinegym.robots.drone.underwaterVehicleFin import UnderwaterVehicleFin

class iAUV(UnderwaterVehicleFin):

    usd_path: str = ASSET_PATH + "/usd/iAUV/iAUV.usd"
    param_path: str = ASSET_PATH + "/usd/iAUV/iAUV.yaml"
    fixed_usd_path: str = ASSET_PATH + "/usd/iAUV/fixed_usd/iAUV/iAUV.usd"