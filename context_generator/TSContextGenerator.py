from context_generator.Context import *
from context_generator.SplitFailedEx import *


class TSContextGenerator:

    def __init__(self, probabilities_of_users, arms_samples_for_each_user, arms):
        self.probabilities_of_users = probabilities_of_users
        self.arms_samples_for_each_user = arms_samples_for_each_user
        self.arms = arms
        self.success_fail_per_arm_per_user = [[[0, 0] for _ in range(len(probabilities_of_users))] for _ in range(len(arms))]
        self.regrets = []
        self.contexts = [Context([i for i in range(len(probabilities_of_users))])]

    def get_successes(self, arm, user_type):
        return self.success_fail_per_arm_per_user[arm][user_type][0]

    def get_failures(self, arm, user_type):
        return self.success_fail_per_arm_per_user[arm][user_type][1]

    def number_of_extractions(self, arm, user_type):
        return self.get_successes(arm, user_type) + self.get_failures(arm, user_type)

    def mean_success(self, arm, user_type):
        if self.get_successes(arm, user_type) == 0:
            return 0
        else:
            return self.get_successes(arm, user_type) / self.number_of_extractions(arm, user_type)

    def update_regret_after_day_passed(self, clicks_per_day):
        for _ in range(0, clicks_per_day):
            clairvoyant = 0
            reward = 0
            for i in range(len(self.contexts)):
                context = self.contexts[i]
                clairvoyant += context.get_clairvoyant(self.probabilities_of_users, self.arms_samples_for_each_user, self.arms)
                best_arm = context.get_best_arm(self.probabilities_of_users, self, self.arms)

                for user_type in context.user_type_list:
                    sort = np.random.binomial(n=1, size=1, p=self.arms_samples_for_each_user[best_arm][user_type])[0]
                    self.success_fail_per_arm_per_user[best_arm][user_type][0] += sort
                    self.success_fail_per_arm_per_user[best_arm][user_type][1] += 1 - sort

                reward += context.get_arm_reward(self.probabilities_of_users, self, best_arm, self.arms)

            reward = min(reward, clairvoyant)
            self.regrets.append(clairvoyant - reward)

    def split(self):
        for i in range(len(self.contexts)):
            try:
                context_split_1, context_split_2 = self.contexts[i].split(self.probabilities_of_users, self, self.arms)
                self.contexts.pop(i)
                self.contexts.append(context_split_1)
                self.contexts.append(context_split_2)
                break
            except SplitFailedEx:
                continue
