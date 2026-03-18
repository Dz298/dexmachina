"""Environment modules for DexMachina."""

__all__ = ["BaseEnv", "BaseRobot", "get_default_robot_cfg"]


def __getattr__(name):
    if name == "BaseEnv":
        from .base_env import BaseEnv

        return BaseEnv
    if name == "BaseRobot":
        from .robot import BaseRobot

        return BaseRobot
    if name == "get_default_robot_cfg":
        from .robot import get_default_robot_cfg

        return get_default_robot_cfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
