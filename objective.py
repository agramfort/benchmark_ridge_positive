from benchopt import BaseObjective
import numpy as np


class Objective(BaseObjective):
    name = "Ridge regression with postivity constraints"

    parameters = {"reg": [1]}

    def __init__(self, reg=1):
        self.reg = reg

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        if (beta >= 0).all():
            diff = self.y - self.X.dot(beta)
            return 0.5 * diff.dot(diff) + 0.5 * self.reg * beta.dot(beta)
        else:
            return np.inf

    def to_dict(self):
        return dict(X=self.X, y=self.y, reg=self.reg)
