import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 100.0, 0.1)
#Phase 0
s1 = -16500 * np.exp(-np.power(t - 0, 2.) / (2 * np.power(50, 2.))) + 16500
s2 = 14500 * (1 - np.exp((-1*t)/70)) + 1000 * np.log(t+1) + 1200*np.exp(-np.power(t - 10, 2.) / (2 * np.power(5, 2.)))
s3 = 5500 * (1 - np.exp((-1*t)/10))
#Phase 1
w1 = 8200 * (1 - np.exp((-1*t)/85))
w2 = -7500 * np.exp(-np.power(t - 0, 2.) / (2 * np.power(50, 2.))) + 7500
w3 = 4500 * (1 - np.exp((-1*t)/20)) - 1500*np.exp(-np.power(t - 80, 2.) / (2 * np.power(40, 2.))) + 200
#Phase 2
pr1 = 5500 * np.log(t/27+1)
pr2 = 6500 * np.log(t/20+1)
pr3 = 4500 * (1 - np.exp((-1*t)/40)) + 1500 * np.log(t/35+1)

fig, ax = plt.subplots()
ax.plot(t, s1, "r")
ax.plot(t, s2, "g")
ax.plot(t, s3, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Estimation of the number of clicks over daily budget (normal)')
ax.grid()
plt.savefig('curves/daily_clicks_normal.png')
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t, w1, "r")
ax2.plot(t, w2, "g")
ax2.plot(t, w3, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax2.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Estimation of the number of clicks over daily budget (new product)')
ax2.grid()
plt.savefig('curves/daily_clicks_competitor.png')
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(t, pr1, "r")
ax3.plot(t, pr2, "g")
ax3.plot(t, pr3, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax3.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Estimation of the number of clicks over daily budget (covid-19)')
ax3.grid()
plt.savefig('curves/daily_clicks_covid.png')
plt.show()

