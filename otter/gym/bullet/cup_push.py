import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import kinova
import random
import pybullet_data
from pkg_resources import parse_version
import os
from deepx import *
import cv2

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


from ..gym_wrapper import GymWrapper

__all__ = ['CupPush']

class GymCupPush(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self._isDiscrete = self.isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = self.urdfRoot
        self._actionRepeat = self.actionRepeat
        self._isEnableSelfCollision = self.isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = self.render
        self._maxSteps = self.maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._hard_reset = True

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.5, 180, -41, [0.32, 0.1, -0.00])
        else:
            p.connect(p.DIRECT)

        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        self._seed()
        self.reset()
        observationDim = len(self._get_obs())

        observation_high = np.array([largeValObservation] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 3
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

        self._hard_reset = self.hard_reset  # This assignment need to be after reset()

    def reset(self):
        # print("KukaGymEnv _reset")
        if self._hard_reset:

            p.resetSimulation()
            p.setPhysicsEngineParameter(numSolverIterations=150)
            p.setTimeStep(self._timeStep)

            # build gym env
            p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
            self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), [0.5000000, 0.00000, -.620000],
                                       [0.000000, 0.000000, 0.0, 1.0])

            xpos = 0.55 + 0.12 * random.random()
            ypos = 0
            ang = math.pi / 2
            zpos = 0.25
            orn = p.getQuaternionFromEuler([ang, 0, 0])
            self.cubeUid = p.loadURDF(os.path.join(self._robot_urdfRoot, "Cube.urdf"), [0.55, 0, 0.15],
                                      [0, 0, 0, 1],
                                      useFixedBase=True)

            self.teacupUid = p.loadURDF(os.path.join(self._robot_urdfRoot, "TeaCup.urdf"),
                                        [xpos, ypos, zpos],
                                        [orn[0], orn[1], orn[2], orn[3]],
                                        useFixedBase=False)
            p.setGravity(0, 0, -9.81)

            # camera configuration
            self._CameraViewMatrix = p.computeViewMatrixFromYawPitchRoll([xpos, ypos, zpos], 1, -90, -60, 0, 2)
            # self._CameraProjMatrix = p.computeProjectionMatrix(-0.5000, 0.5000, -0.5000, 1.5000, 1.0000, 6.0000)
            self._CameraProjMatrix = p.computeProjectionMatrixFOV(80, 0.5, 0, 8)

            self.kinova = kinova.Kinova(p,
                                        robot_type='j2s7s300',
                                        urdfRootPath=self._robot_urdfRoot,
                                        timeStep=self._timeStep,
                                        building_env=False,  # use gym env
                                        useInverseKinematics=True,  # IMPORTANCE! It determines the mode of the motion.
                                        torque_control_enabled=False,
                                        is_fixed=True)
        else:
            self.kinova.reset(reload_urdf=False)

        self.terminated = 0
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self._get_obs()

        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        image_raw = p.getCameraImage(self.image_dim, self.image_dim, self._CameraViewMatrix, self._CameraProjMatrix)
        image_obs = image_raw[2].flatten()
        cupPos, cupOrn = p.getBasePositionAndOrientation(self.teacupUid)
        kinovaState = self.kinova.GetObservation()

        # relative position
        EndEffector_pos, EndEffector_orn = self.kinova.EndEffectorObersavations()
        invEndEffector_pos, invEndEffector_orn = p.invertTransform(EndEffector_pos, EndEffector_orn)
        cupPosInEE, cupOrnInEE = p.multiplyTransforms(invEndEffector_pos, invEndEffector_orn, cupPos, cupOrn)

        self._observation = \
            np.concatenate([np.array(cupPos), np.array(cupOrn), np.array(cupPosInEE), np.array(cupOrnInEE), np.array(kinovaState), image_obs]).flatten()
        return self._observation

    def step(self, action):
        if (self._isDiscrete):
            dv = 0.005
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        else:
            dv = 0.005
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            # da = action[2] * 0.05

            # Compute EndEffector Oritation: Oriented to the cup
            # cupPos, cupOrn = p.getBasePositionAndOrientation(self.teacupUid)
            # EndEffector_pos, EndEffector_orn = self.kinova.EndEffectorObersavations()
            # invEndEffector_pos, invEndEffector_orn = p.invertTransform(EndEffector_pos, EndEffector_orn)
            # cupPosInEE, cupOrnInEE = p.multiplyTransforms(invEndEffector_pos, invEndEffector_orn, cupPos, cupOrn)
            # Orn = [item * 0.05 for item in list(cupOrnInEE)]

            # Compute EndEffector Oritation: Oriented fixed direction
            f = 0.3

            realAction = [dx, dy, dz, 0, f]
        return self.step2(realAction)

    def step2(self, action):
        for i in range(self._actionRepeat):
            self.kinova.ApplyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self._get_obs()


        done = self._termination()
        npaction = np.array(
            [action[3]])  # only penalize rotation until learning works well [action[0],action[1],action[3]])
        actionCost = np.linalg.norm(npaction) * 10.
        reward = self.reward() - actionCost

        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        state = p.getLinkState(self.kinova.kinovaUid, self.kinova.EndEffectorIndex)
        actualEndEffectorPos = state[0]

        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self._get_obs()
            return True
        maxDist = 0.005
        closestPoints_1 = p.getClosestPoints(self.teacupUid, self.kinova.kinovaUid, maxDist)
        closestPoints_2 = p.getClosestPoints(self.tableUid, self.kinova.kinovaUid, maxDist)

        if (len(closestPoints_1)):  # (actualEndEffectorPos[2] <= -0.43):
            self.terminated = 1

            # print("terminating, closing gripper, attempting grasp")
            # start grasp and terminate
            fingerAngle = 0.3
            for i in range(100):
                graspAction = [0, 0, 0.0001, 0, fingerAngle]
                self.kinova.ApplyAction(graspAction)
                p.stepSimulation()
                fingerAngle = fingerAngle - (0.3 / 100.)
                if (fingerAngle < 0):
                    fingerAngle = 0

            for i in range(1000):
                graspAction = [0, 0, 0.001, 0, fingerAngle]
                self.kinova.ApplyAction(graspAction)
                p.stepSimulation()
                cupPos, cupOrn = p.getBasePositionAndOrientation(self.teacupUid)
                if (cupPos[2] > 0.23):
                    # print("BLOCKPOS!")
                    # print(blockPos[2])
                    break
                state = p.getLinkState(self.kinova.kinovaUid, self.kinova.EndEffectorIndex)
                actualEndEffectorPos = state[0]  ##TODO COM need to modified
                if (actualEndEffectorPos[2] > 0.5):
                    break

            self._observation = self._get_obs()
            return True
        return False

    def reward(self):
        maxDist = 0.003

        # rewards is height of target object
        cupPos, cupOrn = p.getBasePositionAndOrientation(self.teacupUid)
        closestPoints = p.getClosestPoints(self.teacupUid, self.kinova.kinovaUid, maxDist, -1,
                                           self.kinova.EndEffectorIndex)

        reward = -1000

        numPt = len(closestPoints)
        if (numPt > 0):
            reward = -closestPoints[0][8] * 10
        if (closestPoints):
            reward = reward + 10000
            print("successfully grasped a cup!!!")
        return reward


class CupPush(GymWrapper):
    environment_name = 'CupPush'
    entry_point = "otter.gym.bullet.cup_push:GymCupPush"
    max_episode_steps = 50
    reward_threshold = -3.75

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'random_target': kwargs.pop('random_target', False),
            'random_start': kwargs.pop('random_start', False),
            'default_goal': kwargs.pop('default_goal', [0.5, 0, 0.5]),
            'image_dim': kwargs.pop('image_dim', 128),
            'urdfRoot': kwargs.pop('urdfRoot', pybullet_data.getDataPath()),
            'actionRepeat': kwargs.pop('actionRepeat', 1),
            'isEnableSelfCollision': kwargs.pop('isEnableSelfCollision', True),
            'render': kwargs.pop('render', False),
            'isDiscrete': kwargs.pop('isDiscrete', False),
            'maxSteps': kwargs.pop('maxSteps', 1000),
            'hard_reset': kwargs.pop('hard_reset', False),
            '_robot_urdfRoot': kwargs.pop('_robot_urdfRoot', os.path.abspath('..')+'/otter/gym/bullet/assets/urdf/')
        }
        super(CupPush, self).__init__(config)

    def torque_matrix(self):
        return 2 * np.eye(self.get_action_dim())

    def make_summary(self, observations, name):
        if self.image:
            pass
            # observations = T.reshape(observations, [-1] + self.image_size())
            # T.core.summary.image(name, observations)

    def is_image(self):
        return self.image

    def image_size(self):
        if self.image:
            return [self.image_dim, self.image_dim, 3]
        return None

    def cost_fn(self, s, a):
        return np.linalg.norm(s[:,-3:], axis=-1) + np.sum(np.square(a), axis=-1)
