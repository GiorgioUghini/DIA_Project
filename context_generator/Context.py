import numpy as np
import copy
from context_generator.SplitFailedEx import *


class Context:
    def __init__(self, demands_index_list):
        self.dem_ind_list = demands_index_list

    def get_clairvoyant(self, class_probs, arm_demand_means, arms):
        scores = np.zeros(len(arms))
        for i in range(arm_demand_means.shape[0]):
            for j in self.dem_ind_list:
                scores[i] += arm_demand_means[i][j] * arms[i] * class_probs[j]
        return np.max(scores)

    def get_best_arm(self, class_probs, cont_gen, arms):
        scores = np.zeros(len(arms))
        for d in self.dem_ind_list:
            for i in range(len(arms)):
                scores[i] += np.random.beta(cont_gen.beta_params[i][d][0], cont_gen.beta_params[i][d][1], size=1)[0] * class_probs[d]
        scores = scores * arms
        return np.argmax(scores)

    def get_arm_reward(self, class_probs, cont_gen, arm, arms):
        score = 0
        for j in self.dem_ind_list:
            score += cont_gen.emp_mean(arm, j) * arms[arm] * class_probs[j]
        return score

    def get_arm_hoeffding_lowbound(self, class_probs, cont_gen, arm, arms):
        score = 0
        cum_prob = 0
        n_cum = 0
        for j in self.dem_ind_list:
            score += (cont_gen.emp_mean(arm, j)) \
                     * arms[arm] * class_probs[j]
            cum_prob += class_probs[j]
            n_cum += cont_gen.n(arm, j)
        return score - np.sqrt(-np.log10(0.1) * 2 / (cum_prob * n_cum))

    def split(self, class_probs, cont_gen, arms):
        if len(self.dem_ind_list) <= 1:
            raise SplitFailedEx
        for i in range(len(arms)):
            for dem in self.dem_ind_list:
                if cont_gen.n(i, dem) == 0:
                    raise SplitFailedEx
        for i in range(len(self.dem_ind_list)):
            spl1 = [self.dem_ind_list[i]]
            spl2 = copy.deepcopy(self.dem_ind_list)
            spl2.pop(i)
            n1 = Context(spl1)
            n2 = Context(spl2)
            b1 = n1.get_best_arm(class_probs, cont_gen, arms)
            b2 = n2.get_best_arm(class_probs, cont_gen, arms)
            b = self.get_best_arm(class_probs, cont_gen, arms)
            l1 = n1.get_arm_hoeffding_lowbound(class_probs, cont_gen, b1, arms)
            l2 = n2.get_arm_hoeffding_lowbound(class_probs, cont_gen, b2, arms)
            l = self.get_arm_hoeffding_lowbound(class_probs, cont_gen, b, arms)
            if l1 + l2 > l:
                return n1, n2
        raise SplitFailedEx
