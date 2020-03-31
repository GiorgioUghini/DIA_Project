import matplotlib.pyplot as plt
import numpy as np
import utils


# Data for plotting
t = np.arange(90., 450., 0.1)

s1 = utils.getDemandCurve(0, t)
s2 = utils.getDemandCurve(1, t)
s3 = utils.getDemandCurve(2, t)

fig, ax = plt.subplots()
ax.plot(t, s1, "r")
ax.plot(t, s2, "g")
ax.plot(t, s3, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax.set(xlabel='price of the item [€]', ylabel='conversion rate [%]',
       title='Conversion rate curve')
ax.grid()
plt.savefig('curves/conversion_rate_curve.png')
plt.show()