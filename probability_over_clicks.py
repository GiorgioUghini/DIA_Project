import matplotlib.pyplot as plt
import numpy as np
import utils

# Data for plotting
t = np.arange(0.0, 100.0, 0.1)

# High interest / No competitors
p00 = utils.getClickCurve(0, 0, t)
p01 = utils.getClickCurve(0, 1, t)
p02 = utils.getClickCurve(0, 2, t)
# Low interest / No competitors
p10 = utils.getClickCurve(1, 0, t)
p11 = utils.getClickCurve(1, 1, t)
p12 = utils.getClickCurve(1, 2, t)
# Low interest / With competitors
p20 = utils.getClickCurve(2, 0, t)
p21 = utils.getClickCurve(2, 1, t)
p22 = utils.getClickCurve(2, 2, t)
# High interest / With competitors
p30 = utils.getClickCurve(3, 0, t)
p31 = utils.getClickCurve(3, 1, t)
p32 = utils.getClickCurve(3, 2, t)


fig, ax = plt.subplots()
ax.plot(t, p10, "r")
ax.plot(t, p11, "g")
ax.plot(t, p12, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Number of clicks over daily budget - Low interest, No competitors')
ax.grid()
plt.savefig('curves/real_curves/daily_clicks_00.png')
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t, p20, "r")
ax2.plot(t, p21, "g")
ax2.plot(t, p22, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax2.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Number of clicks over daily budget - Low interest, With competitors')
ax2.grid()
plt.savefig('curves/real_curves/daily_clicks_01.png')
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(t, p30, "r")
ax3.plot(t, p31, "g")
ax3.plot(t, p32, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax3.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Number of clicks over daily budget - High interest, With competitors')
ax3.grid()
plt.savefig('curves/real_curves/daily_clicks_11.png')
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(t, p00, "r")
ax4.plot(t, p01, "g")
ax4.plot(t, p02, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax4.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Number of clicks over daily budget - High interest, No competitors')
ax4.grid()
plt.savefig('curves/real_curves/daily_clicks_10.png')
plt.show()
