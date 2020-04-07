from context_generator.Context import *
from context_generator.SplitFailedEx import *
import math


class TSContextGenerator:

    def __init__(self, class_probabilities, arm_demand_means, arms, demands_names=None):
        self.realizations_per_arm_per_demand = [[
            [] for _ in range(len(class_probabilities))
        ] for _ in range(len(arms))]
        self.realizations_per_arm_per_demand_timestamps = [[
            [] for _ in range(len(class_probabilities))
        ] for _ in range(len(arms))]
        self.beta_params = np.ones([len(arms), len(class_probabilities), 2], dtype=np.int64)
        self.time_steps = 0
        self.regrets = []
        self.clairvoyants = []
        self.rewards = []
        self.arms = arms
        self.current_scale_factor = 1
        self.class_probabilities = class_probabilities
        self.arm_demand_means = arm_demand_means
        self.demands_names = demands_names
        self.nodes = [Context([i for i in range(len(class_probabilities))])]

    def n(self, arm, demand):
        val = len(self.realizations_per_arm_per_demand[arm][demand])
        # print(arm, demand, val)
        return val

    def emp_mean(self, arm, dem):
        return np.mean(self.realizations_per_arm_per_demand[arm][dem])

    def calc_clair_rew_regret(self):
        clair = 0
        rew = 0
        self.time_steps += 1
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            clair += node.get_clairvoyant(self.class_probabilities, self.arm_demand_means, self.arms)
            bestarm = node.get_best_arm(self.class_probabilities, self, self.arms)
            for dem in node.dem_ind_list:
                sort = np.random.binomial(n=1, size=1, p=self.arm_demand_means[bestarm][dem] * self.current_scale_factor)[0]
                self.realizations_per_arm_per_demand[bestarm][dem].append(sort)
                self.realizations_per_arm_per_demand_timestamps[bestarm][dem].append(self.time_steps)
                self.beta_params[bestarm][dem] += np.array([sort, 1 - sort])

            # print(self.realizations_per_arm_per_demand)

            rew += node.get_arm_reward(self.class_probabilities, self, bestarm, self.arms)

        rew = min(rew, clair)
        self.clairvoyants.append(clair)
        self.rewards.append(rew)
        self.regrets.append(clair - rew)

    def split(self):
        for i in range(len(self.nodes)):
            try:
                n1, n2 = self.nodes[i].split(self.class_probabilities, self, self.arms)
                self.nodes.pop(i)
                self.nodes.append(n1)
                self.nodes.append(n2)
                break
            except SplitFailedEx:
                continue

    def print_tree(self):
        for node in self.nodes:
            linetopr = "IF "
            for dem in node.dem_ind_list:
                linetopr += self.demands_names[dem]
                if node.dem_ind_list[-1] != dem:
                    linetopr += " OR "
                else:
                    linetopr += " ---> "
            ba = node.get_best_arm(self.class_probabilities, self, self.arms)
            linetopr += "ARM " + str(ba)
            print(linetopr)
