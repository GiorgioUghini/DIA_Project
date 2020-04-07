import seaborn as sb
from tqdm import tqdm
import matplotlib.pyplot as mplt
import utils as u
from context_generator.TSContextGenerator import *

n_arms = 6
arms = np.array([100 + 50 * i for i in range(n_arms)])
class_probs = np.array([0.3, 0.5, 0.2])
arm_dem_means = np.zeros([len(arms), len(class_probs)])
for i in range(len(arms)):
    for j in range(len(class_probs)):
        arm_dem_means[i][j] = u.getDemandCurve(j, arms[i])
clairs = []
rews = []
regrs = []
T_HOR = 18250
N_EXPS = 1
for _ in tqdm(range(N_EXPS)):
    contgen = TSContextGenerator(class_probs,
                                     arm_dem_means,
                                     arms,
                                     ["A", "B", "C"])
    for i in range(T_HOR):
        contgen.calc_clair_rew_regret()
        if i % 350 == 0:
            contgen.split()
    clairs.append(contgen.clairvoyants)
    rews.append(contgen.rewards)
    regrs.append(contgen.regrets)
    contgen.print_tree()
# sb.lineplot(range(T_HOR), u.smoothen_curve(np.mean(clairs, axis=0)))
smooth_rews = u.smoothen_curve(np.mean(rews, axis=0))
gr = sb.lineplot(range(T_HOR), smooth_rews)
gr.set_title(str(n_arms) + " ARMS - REVENUE")
mplt.show()
smooth_regrs = u.smoothen_curve(np.mean(regrs, axis=0))
gr = sb.lineplot(range(T_HOR), smooth_regrs)
gr.set_title(str(n_arms) + " ARMS - REGRET")
mplt.show()
clairmean = np.mean(clairs, axis=0)
gr = sb.lineplot(range(T_HOR), clairmean)
gr.set_title(str(n_arms) + " ARMS - CLAIRVOYANT")
mplt.show()
