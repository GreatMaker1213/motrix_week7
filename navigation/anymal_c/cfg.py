import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.join(os.path.dirname(__file__), "xmls", "scene.xml")


@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    # ANYmal C 电机的缩放比例 (将神经网络的 [-1, 1] 映射到实际的关节弧度变化)
    action_scale = 0.5  


@dataclass
class InitState:
    # 机器人在世界坐标系下的初始位置 (Z轴高度设置为0.6m，防止初始穿模)
    pos = [0.0, 0.0, 0.6]  

    # 训练时的位置随机化范围 [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  

    # ANYmal C 站立时的标称关节角度
    default_joint_angles = {
        "LF_HAA": 0.03,  # [rad] Left Front
        "RF_HAA": -0.03, # [rad] Right Front
        "LH_HAA": 0.03,  # [rad] Left Hind
        "RH_HAA": -0.03, # [rad] Right Hind
        "LF_HFE": 0.4,   # [rad]
        "RF_HFE": 0.4,   # [rad]
        "LH_HFE": -0.4,  # [rad]
        "RH_HFE": -0.4,  # [rad]
        "LF_KFE": -0.8,  # [rad]
        "RF_KFE": -0.8,  # [rad]
        "LH_KFE": 0.8,   # [rad]
        "RH_KFE": 0.8,   # [rad]
    }


@dataclass
class Commands:
    # Flat Navigation 任务中的目标指令范围 [vx_min, vy_min, yaw_vel_min, vx_max, vy_max, yaw_vel_max]
    velocity_command_range = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]


@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05


@dataclass
class Asset:
    # 对应 anymal_c.xml 中的躯干名称
    body_name = "base"
    # 对应 anymal_c.xml 中的足底接触点或几何体名称
    foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    # 允许发生接触的惩罚部位
    terminate_after_contacts_on = ["base"]
    # 对应我们在 scene.xml 中定义的地面 geom 名称
    ground_name = "floor" 


@dataclass
class Sensor:
    base_linvel = "base_linvel"
    base_gyro = "base_gyro"


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -200.0,
            "tracking_lin_vel": 1.0,     # 线速度跟踪奖励
            "tracking_ang_vel": 0.5,     # 角速度跟踪奖励
            "lin_vel_z": -2.0,           # 惩罚弹跳
            "ang_vel_xy": -0.05,         # 惩罚异常姿态
            "action_rate": -0.01,        # 平滑动作
            "dof_pos_limits": -10.0,     # 关节限位惩罚
        }
    )

# 环境注册，注意名称统一
@registry.envcfg("anymal_c_navigation_flat-v0")
@dataclass
class AnymalCEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.05
    max_episode_seconds: float = 20.0  # 预留较长时间以供导航
    sim_dt: float = 0.005              # 200Hz 物理频率 (与 scene.xml 中的 timestep 一致)
    ctrl_dt: float = 0.02              # 50Hz 策略频率
    reset_yaw_scale: float = 3.14
    max_dof_vel: float = 120.0         # 允许的最大关节速度

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)