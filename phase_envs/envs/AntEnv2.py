import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math

class AntEnv2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.phase = 0
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        zposbefore = self.get_body_com("torso")[2]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        zposafter = self.get_body_com("torso")[2]
        forward_reward = (xposafter - xposbefore)/self.dt
        backward_reward = (xposbefore - xposafter)/self.dt
        leftward_reward = (zposafter - zposbefore)/self.dt
        rightward_reward = (zposbefore - zposafter)/self.dt

        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        if self.phase == 0:
                reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        elif self.phase == math.pi/2:
                reward = leftward_reward - ctrl_cost - contact_cost + survive_reward
        elif self.phase == math.pi:
                reward = backward_reward - ctrl_cost - contact_cost + survive_reward
        elif self.phase == 3*math.pi/2:
                reward = rightward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_backward=backward_reward,
            reward_rightward=rightward_reward,
            reward_leftward=leftward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
