# Based on https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c
import numpy as np


class PIDController:

    def __init__(self, goal, mask, P, I, D):
        self.goal, self.mask = goal, mask
        self.P, self.I, self.D = P, I, D
        self.integral, self.derivative, self.prev_error = 0, 0, 0

    def reset(self):
        self.integral, self.derivative, self.prev_error = 0, 0, 0

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def get_distribution(self, state, discrete=True, n_actions=None):
        #error = np.sqrt(state**2 - np.array(self.goal)**2)
        error = state - self.goal
        #error = np.abs(np.array(error))
        integral = self.integral + error
        derivative = error - self.prev_error
        prev_error = error

        pid = np.dot(self.P * error + self.I * self.integral + self.D * self.derivative, self.mask)
        action = self.sigmoid(pid)

        if discrete:
            action = np.round(action).astype(np.int32)
            distribution = [0] * n_actions
            distribution[action] = 1
            return distribution

    def step(self, state, discrete=True, sigmoid=False):
        #error = np.sqrt(state**2 - np.array(self.goal)**2)
        error = state - self.goal
        #error = np.abs(np.array(error))
        self.integral += error
        self.derivative = error - self.prev_error
        self.prev_error = error

        pid = np.dot(self.P * error + self.I * self.integral + self.D * self.derivative, self.mask)

        if sigmoid:
            action = self.sigmoid(pid)
        else:
            action = pid

        if discrete:
            action = np.round(action).astype(np.int32)
        return action