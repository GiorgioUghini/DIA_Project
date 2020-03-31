import numpy as np
import utils


def fun(t, userType, phase):
    return utils.getClickCurve(phase, userType, t)  # phase and userType are inverted, it's correct


class NSCMABEnvironment():
    def __init__(self, budgets_list, sigma, horizon):
        self.N_PHASES = 4
        # We hardcode N_PHASES and all the functions here because it would compromise the main code
        # to pass all these as parameters. However, it can be easily done, making the code more
        # reusable but less readable
        self.t = 0
        self.budgets = budgets_list
        self.horizon = horizon
        self.means = [fun(budgets_list[userType], userType, phase) for userType in range(0, len(budgets_list))
                      for phase in range(0, self.N_PHASES)]
        # [ FUN(0,0), FUN(0,1), FUN(0,2), ..., FUN(1,0), FUN(1,1), FUN(1,2), ...]

        self.sigma = sigma

    def round(self, pulled_arm, userType):
        phase_size = self.horizon / self.N_PHASES
        current_phase = int(self.t / phase_size)
        mean = self.means[userType * self.N_PHASES + current_phase][pulled_arm]
        # We are supposing same variance among all userTypes and phases.
        # If there are some phases with more uncertainty than others, just keep track
        # of it by instantiating self.sigma as a list based upon the current phase
        return np.random.normal(mean, self.sigma)

    def ahead(self):
        self.t += 1
        return
