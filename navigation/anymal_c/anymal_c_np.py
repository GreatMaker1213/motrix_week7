

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCEnvCfg

# 注意保持注册名称一致
@registry.env("anymal_c_navigation_flat-v0", "np")
class AnymalCEnv(NpEnv):
    _cfg: AnymalCEnvCfg

    def __init__(self, cfg: AnymalCEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        
        # get body handle
        self._body = self._model.get_body(cfg.asset.body_name)
        
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # obs space
        # 指令(3) + 重力投影(3) + 线速度(3) + 角速度(3) + 关节位置(12) + 关节速度(12) + 上次动作(12)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        
        self._num_dof = self._model.num_actuators
        self._init_buffer()

    def _init_buffer(self):
        """初始化默认关节角度和状态缓冲"""
        self.default_angles = np.zeros(self._num_dof, dtype=np.float32)
        for i in range(self._num_dof):
            for name, angle in self._cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
                    
        # 缓存初始的自由度姿态
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_pos[-self._num_dof :] = self.default_angles

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        """处理神经网络动作并下发到物理引擎"""
        # 记录动作供观测使用
        if "current_actions" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # 缩放动作并加上默认站立姿态
        actions_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + actions_scaled
        
        # 直接下发给 PD 控制器
        state.data.actuator_ctrls = target_pos
        return state

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def update_state(self, state: NpEnvState):
        data = state.data
        
        # 获取物理状态与属性
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        
        # get velocity
        base_lin_vel = self._model.get_sensor_value("base_linvel", data)
        base_ang_vel = self._model.get_sensor_value("base_gyro", data)
        
        joint_pos = self._body.get_joint_dof_pos(data)
        joint_vel = self._body.get_joint_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # get instruction
        commands = state.info.get("commands", np.zeros((self._num_envs, 3), dtype=np.float32))

        # simple obs
        projected_gravity = quaternion.rotate_vector(root_quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.concatenate([
            commands,                           # 3
            projected_gravity,                  # 3
            base_lin_vel,                       # 3
            base_ang_vel,                       # 3
            joint_pos_rel,                      # 12
            joint_vel,                          # 12
            state.info["last_actions"]          # 12
        ], axis=-1)

        # simple reward
        lin_vel_error = np.sum(np.square(commands[:, :2] - base_lin_vel[:, :2]), axis=1)
        ang_vel_error = np.square(commands[:, 2] - base_ang_vel[:, 2])
        
        reward = (
            1.0 * np.exp(-lin_vel_error / 0.25) +          # XY linear velocity
            0.5 * np.exp(-ang_vel_error / 0.25) -          # Yawn angle velocity
            2.0 * np.square(base_lin_vel[:, 2]) -          # Z 
            0.01 * np.sum(np.square(state.info["current_actions"] - state.info["last_actions"]), axis=1) # smooth
        )

        # truncated
        base_height = pose[:, 2]
        terminated = base_height < 0.35  # 如果基座高度低于 0.35m，认为摔倒
        
        # 数值保护
        vel_max = np.abs(joint_vel).max(axis=1)
        terminated = np.logical_or(terminated, vel_max > 100.0)

        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        return state

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        """环境重置"""
        num_envs = data.shape[0]

        # 随机生成速度指令 [vx, vy, yaw_vel]
        cmd_range = self._cfg.commands.velocity_command_range
        commands = np.random.uniform(
            low=cmd_range[:3], 
            high=cmd_range[3:], 
            size=(num_envs, 3)
        ).astype(np.float32)

        # 状态重置 有噪声
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        noise = np.random.uniform(-0.05, 0.05, size=(num_envs, self._num_dof))
        dof_pos[:, -self._num_dof:] += noise  # 仅给电机加噪声
        
        dof_vel = np.zeros((num_envs, self._model.num_dof_vel), dtype=np.float32)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 构造初始观测
        info = {
            "commands": commands,
            "current_actions": np.zeros((num_envs, self._num_dof), dtype=np.float32),
            "last_actions": np.zeros((num_envs, self._num_dof), dtype=np.float32)
        }
        
        # 复用 update_state 中的提取逻辑来生成初始 obs
        state = NpEnvState(data=data, obs=None, reward=None, terminated=None, truncated=None, info=info)
        state = self.update_state(state)
        
        return state.obs, state.info