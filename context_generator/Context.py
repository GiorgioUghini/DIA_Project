import numpy as np
import copy
from context_generator.SplitFailedEx import *


class Context:
    def __init__(self, demands_index_list):
        self.demands_index_list = demands_index_list

    def get_clairvoyant(self, probabilities_of_users, arms_demand_for_each_user, arms):
        scores = np.zeros(len(arms))
        for i in range(arms_demand_for_each_user.shape[0]):
            for j in self.demands_index_list:
                scores[i] += arms_demand_for_each_user[i][j] * arms[i] * probabilities_of_users[j]

        return np.max(scores)

    def get_best_arm(self, probabilities_of_users, context_generator, arms):
        scores = np.zeros(len(arms))
        for d in self.demands_index_list:
            for i in range(len(arms)):
                scores[i] += np.random.beta(context_generator.beta_params[i][d][0], context_generator.beta_params[i][d][1], size=1)[0] * probabilities_of_users[d]

        scores = scores * arms
        return np.argmax(scores)

    def get_arm_reward(self, probabilities_of_users, context_generator, arm, arms):
        score = 0
        for j in self.demands_index_list:
            score += context_generator.emp_mean(arm, j) * arms[arm] * probabilities_of_users[j]

        return score

    def get_arm_hoeffding_lowbound(self, probabilities_of_users, context_generator, arm, arms):
        score = 0
        cum_prob = 0
        n_cum = 0
        for j in self.demands_index_list:
            score += (context_generator.emp_mean(arm, j)) * arms[arm] * probabilities_of_users[j]
            cum_prob += probabilities_of_users[j]
            n_cum += context_generator.n(arm, j)
        if n_cum == 0:
            raise SplitFailedEx

        return score - np.sqrt(-np.log10(0.1) * 2 / (cum_prob * n_cum))

    def split(self, probabilities_of_users, context_generator, arms):
        if len(self.demands_index_list) <= 1:
            raise SplitFailedEx()
        for i in range(len(self.demands_index_list)):
            one_feature = [self.demands_index_list[i]]
            other_features = copy.deepcopy(self.demands_index_list)
            other_features.pop(i)
            c1 = Context(one_feature)
            c2 = Context(other_features)
            best_arm_1 = c1.get_best_arm(probabilities_of_users, context_generator, arms)
            best_arm_2 = c2.get_best_arm(probabilities_of_users, context_generator, arms)
            best_arm_this = self.get_best_arm(probabilities_of_users, context_generator, arms)
            lower_bound_1 = c1.get_arm_hoeffding_lowbound(probabilities_of_users, context_generator, best_arm_1, arms)
            lower_bound_2 = c2.get_arm_hoeffding_lowbound(probabilities_of_users, context_generator, best_arm_2, arms)
            lower_bound_this = self.get_arm_hoeffding_lowbound(probabilities_of_users, context_generator, best_arm_this, arms)
            if lower_bound_1 + lower_bound_2 > lower_bound_this:
                return c1, c2
        raise SplitFailedEx()
