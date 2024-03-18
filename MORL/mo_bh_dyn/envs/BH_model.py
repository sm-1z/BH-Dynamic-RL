import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import numpy as np
import os

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0, 0.8, 1.017)),
    "azimuth": 180.0,
    "elevation": -10.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class BhModelEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        forward_reward_weight=10.0,
        ctrl_cost_weight=0.1,
        healthy_reward=4.0,
        robustness_reward=20.0,
        effic_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 1.5),  # 健康质心高度
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self.count = 1
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward

        self._robustness_reward = robustness_reward
        self._effic_reward = effic_reward
        self.COM_pos_x = []
        self.COM_pos_y = []
        # self.COM_pos_z = 1.0
        self.left_foot_pos_x = []
        self.left_foot_pos_y = []
        self.left_foot_pos_z = []
        self.right_foot_pos_x = []
        self.right_foot_pos_y = []
        self.right_foot_pos_z = []
        self.left_jiont_pos = np.zeros((3500, 7))
        self.right_jiont_pos = np.zeros((3500, 7))
        self._get_target()  # 初始化获得质心与双足轨迹

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(455,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(457,), dtype=np.float64
            )

        current_file_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_file_dir, "bhmodel.xml")
        MujocoEnv.__init__(
            self,
            # "./mo_bh_dyn/envs/bhmodel.xml",
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property  # 健康奖励
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(
            np.square(self.data.ctrl)
        )
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (
            (not self.is_healthy)
            if self._terminate_when_unhealthy or self.data.time > 500
            else False
        )
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    # YKB Add
    def _get_target(self):
        # total_time = 6

        # self.COM_pos_x, self.COM_pos_y, \
        # self.left_foot_pos_x, self.left_foot_pos_y, self.left_foot_pos_z, \
        # self.right_foot_pos_x, self.right_foot_pos_y, self.right_foot_pos_z, \
        # self.left_jiont_pos, self.right_jiont_pos = Calculate(total_time)
        current_file_dir = os.path.dirname(__file__)
        trace_path = os.path.join(current_file_dir, "bipedal_robot_data.npz")
        # loaded_data = np.load("./mo_bh_dyn/envs/bipedal_robot_data.npz")
        loaded_data = np.load(trace_path)
        # loaded_data = np.load("./mo_bh_dyn/envs/bipedal_robot_data.npz")
        self.COM_pos_x = loaded_data["COM_pos_x"]
        self.COM_pos_y = loaded_data["COM_pos_y"]
        self.left_foot_pos_x = loaded_data["left_foot_pos_x"]
        self.left_foot_pos_y = loaded_data["left_foot_pos_y"]
        self.left_foot_pos_z = loaded_data["left_foot_pos_z"]
        self.right_foot_pos_x = loaded_data["right_foot_pos_x"]
        self.right_foot_pos_y = loaded_data["right_foot_pos_y"]
        self.right_foot_pos_z = loaded_data["right_foot_pos_z"]
        self.left_joint_pos = loaded_data["left_joint_pos"]
        self.right_joint_pos = loaded_data["right_joint_pos"]

    # YKB Add
    @property  # 鲁棒性奖励
    def robustness_reward(self):
        weight = [0.02, 0.02, 0.1]
        step = int(self.data.time / 0.01)

        # 关节[7:]位置
        q_pos = self.data.qpos.flat.copy()
        q_left_pos = self.left_jiont_pos[step]
        q_right_pos = self.right_jiont_pos[step]

        target_q_pos = np.hstack((q_right_pos, q_left_pos))
        q_loss = -0.5 * np.sum((q_pos[7:] - target_q_pos) ** 2)

        body_pos = self.data.xipos.flat.copy()
        # 计算质心位置
        mass = np.expand_dims(self.model.body_mass, axis=1)
        body_center = (np.sum(mass * body_pos, axis=0) / np.sum(mass))[
            0:3
        ].copy()

        # 一个时间步是0.01=0.02*5=timestep*frame_skip

        target_center = np.array(
            [self.COM_pos_x[step], self.COM_pos_y[step], 1.064]
        )
        center_loss = -0.5 * np.sum((body_center - target_center) ** 2)

        # 双足当前位置 右[27:30] 左[51:]
        r_foot = body_pos[27:30]
        l_foot = body_pos[51:]
        target_r_foot = np.array(
            [
                self.right_foot_pos_x[step],
                self.right_foot_pos_y[step],
                self.right_foot_pos_z[step],
            ]
        )
        target_l_foot = np.array(
            [
                self.left_foot_pos_x[step],
                self.left_foot_pos_y[step],
                self.left_foot_pos_z[step],
            ]
        )
        foot_loss = -0.5 * np.sum(
            (r_foot - target_r_foot) ** 2 + (l_foot - target_l_foot) ** 2
        )

        robustness_reward = (
            weight[0] * center_loss
            + weight[1] * foot_loss
            + weight[2] * q_loss
        )

        return robustness_reward

    # YKB Add
    @property  # 能效性奖励
    def effic_reward(self):
        q_force = self.data.qfrc_actuator.flat.copy()
        q_vel = self.data.qvel.flat.copy()
        effic_reward = 50.0 / (
            np.sum(q_force[6:] * q_vel[6:]) / np.sum(self.xy_dis**2)
        )

        return effic_reward

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        self.xy_dis = xy_position_after - xy_position_before  # YKB Add
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        forward_reward = self._forward_reward_weight * y_velocity
        healthy_reward = self.healthy_reward
        robustness_reward = self._robustness_reward * self.robustness_reward
        effic_reward = self._effic_reward * self.effic_reward
        rewards = (
            forward_reward + healthy_reward + robustness_reward + effic_reward
        )  # YKB Change

        observation = self._get_obs()  # 观察空间
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward + healthy_reward,
            "healthy_reward": healthy_reward,
            "robustness_reward": robustness_reward + healthy_reward,
            "effic_reward": effic_reward + healthy_reward,
        }
        if self.count % 100 == 0:
            # print(
            #     f"forward_reward:{forward_reward}\nhealthy_reward:{healthy_reward}\nrobustness_reward:{robustness_reward}\neffic_reward:{effic_reward}\n"
            # )
            self.count = 1
        else:
            self.count += 1
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
