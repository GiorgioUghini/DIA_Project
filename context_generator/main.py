import seaborn as sb
from tqdm import tqdm
import matplotlib.pyplot as mplt
import utils as u
from context_generator.TSContextGenerator import *


def smoothen_curve(curve, smoothing_window_size=800):
    smooth_curve = np.zeros(len(curve))
    for i in range(len(curve)):
        smooth_curve[i] = np.mean(curve[max(0, int(i - smoothing_window_size / 2)):
                                                    min(len(curve), int(i + smoothing_window_size / 2))])
    return smooth_curve


prod_cost = 10
arms = np.array([30 + (320 /(5)) * i for i in range(6)])
class_probs = np.array([108/329, 48/329, 173/329])
arm_dem_means = np.zeros([len(arms), len(class_probs)])
for i in range(len(arms)):
    for j in range(len(class_probs)):
        arm_dem_means[i][j] = u.getDemandCurve(j, arms[i])

clairs = []
rews = []
regrs = []
T_HOR = 1400
N_EXPS = 1
arms = arms - prod_cost
for _ in tqdm(range(N_EXPS)):
    contgen = TSContextGenerator(class_probs,
                                     arm_dem_means,
                                     arms)
    for i in range(T_HOR):
        contgen.calc_clair_rew_regret()
        if i % 7 == 0:
            contgen.split()
    clairs.append(contgen.clairvoyants)
    rews.append(contgen.rewards)
    regrs.append(contgen.regrets)
# sb.lineplot(range(T_HOR), u.smoothen_curve(np.mean(clairs, axis=0)))

smooth_rews = smoothen_curve(np.mean(rews, axis=0))
#gr = sb.lineplot(range(T_HOR), smooth_rews)
#gr.set_title(str(6) + " ARMS - REVENUE")
#mplt.show()
smooth_regrs = smoothen_curve(np.mean(regrs, axis=0))
#gr = sb.lineplot(range(T_HOR), smooth_regrs)
#gr.set_title(str(6) + " ARMS - REGRET")
#mplt.show()
clairmean = np.mean(clairs, axis=0)
#gr = sb.lineplot(range(T_HOR), clairmean)
#gr.set_title(str(6) + " ARMS - CLAIRVOYANT")
#mplt.show()

mplt.figure(0)
mplt.plot(smooth_rews, 'r')
mplt.plot(smooth_regrs, 'b')
mplt.plot(clairmean, 'g')
mplt.legend(["Reward", "Regret", "Clairvoiant"])
mplt.show()
