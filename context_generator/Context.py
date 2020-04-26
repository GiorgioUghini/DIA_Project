import numpy as np
import copy
from context_generator.SplitFailedEx import *


class Context:
    def __init__(self, user_type_list):
        self.user_type_list = user_type_list

    def get_clairvoyant(self, probabilities_of_users, arms_samples_for_each_user, arms):
        values = np.zeros(len(arms))
        for arm in range(len(arms)):
            for user_type in self.user_type_list:
                values[arm] += arms_samples_for_each_user[arm][user_type] * arms[arm] * probabilities_of_users[user_type]

        return np.max(values)

    def get_best_arm(self, probabilities_of_users, context_generator, arms):
        values = np.zeros(len(arms))
        for user_type in self.user_type_list:
            for arm in range(len(arms)):
                values[arm] += np.random.beta(context_generator.get_successes(arm, user_type) + 1,
                                              context_generator.get_failures(arm, user_type) + 1, size=1)[0] * probabilities_of_users[user_type]
        values = values * arms
        return np.argmax(values)

    def get_arm_reward(self, probabilities_of_users, context_generator, arm, arms):
        value = 0
        for user_type in self.user_type_list:
            value += context_generator.mean_success(arm, user_type) * arms[arm] * probabilities_of_users[user_type]

        return value

    def get_arm_hoeffding_lowbound(self, probabilities_of_users, context_generator, arm, arms):
        value = 0
        context_probability = 0
        number_of_extractions = 0
        for user_type in self.user_type_list:
            value += (context_generator.mean_success(arm, user_type)) * arms[arm] * probabilities_of_users[user_type]
            context_probability += probabilities_of_users[user_type]
            number_of_extractions += context_generator.number_of_extractions(arm, user_type)

        if number_of_extractions == 0:
            raise SplitFailedEx

        return value - np.sqrt(-np.log10(0.1) * 2 / (context_probability * number_of_extractions))

    def split(self, probabilities_of_users, context_generator, arms):
        if len(self.user_type_list) <= 1:
            raise SplitFailedEx()
        for user_type in range(len(self.user_type_list)):
            one_user_type = [self.user_type_list[user_type]]
            other_user_types = copy.deepcopy(self.user_type_list)
            other_user_types.pop(user_type)
            c1 = Context(one_user_type)
            c2 = Context(other_user_types)
            best_arm_1 = c1.get_best_arm(probabilities_of_users, context_generator, arms)
            best_arm_2 = c2.get_best_arm(probabilities_of_users, context_generator, arms)
            best_arm_this = self.get_best_arm(probabilities_of_users, context_generator, arms)
            lower_bound_1 = c1.get_arm_hoeffding_lowbound(probabilities_of_users, context_generator, best_arm_1, arms)
            lower_bound_2 = c2.get_arm_hoeffding_lowbound(probabilities_of_users, context_generator, best_arm_2, arms)
            lower_bound_this = self.get_arm_hoeffding_lowbound(probabilities_of_users, context_generator, best_arm_this, arms)
            if lower_bound_1 + lower_bound_2 > lower_bound_this:
                return c1, c2

        raise SplitFailedEx()
