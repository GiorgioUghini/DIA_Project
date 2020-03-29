import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(90., 450., 0.1)

s1 = 0.75*np.exp(-np.power(t - 200, 2.) / (2 * np.power(90, 2.))) + 0.3*np.exp(-np.power(t - 60, 2.) / (2 * np.power(60, 2.)))
s2 = 0.75*np.exp(-np.power(t - 250, 2.) / (2 * np.power(90, 2.))) + 0.55*np.exp(-np.power(t - 65, 2.) / (2 * np.power(90, 2.)))
s3 = 0.42*np.exp(-np.power(t - 180, 2.) / (2 * np.power(90, 2.))) + 0.35*np.exp(-np.power(t - 85, 2.) / (2 * np.power(120, 2.)))

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