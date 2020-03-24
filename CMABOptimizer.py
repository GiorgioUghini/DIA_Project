import numpy as np


class CMABOptimizer():
    def __init__(self, max_budget, campaign_number, step):
        if (max_budget % step == 0):
            self.budgets = [i for i in range(0, int(max_budget) + 1, step)]  # +1 to max_budget because range does not include the right extreme of the interval by default
        else:
            self.budgets = [i for i in range(0, int(max_budget), step)]
        self.campaign_number = campaign_number
        self.step = step

    def optimize(self, matrix):
        array_old = np.zeros(matrix.shape[1])
        array_next = np.zeros(matrix.shape[1])
        budget_assigned_old = [[ 0 for i in range(0, self.campaign_number)]] * matrix.shape[1]
        budget_assigned_new = [[ 0 for i in range(0, self.campaign_number)]] * matrix.shape[1]
        for r in range(0, matrix.shape[0]):
            for c in range(0, matrix.shape[1]):
                values = np.zeros(c + 1)
                for i in range(0, len(values)):
                    np.put(values, i, matrix[r, c - i] + array_old[i])

                idx = int(np.argmax(values))
                budget_assigned_new[c] = budget_assigned_old[idx].copy()
                budget_assigned_new[c][r] = self.budgets[c - idx]

                np.put(array_next, [c], max(values))
            array_old = array_next
            array_next = np.zeros(int(matrix.shape[1]))
            budget_assigned_old = budget_assigned_new.copy()
        chosen_budget = budget_assigned_new[int(np.argmax(array_old))]
        return chosen_budget      # If using integer step this can be done
