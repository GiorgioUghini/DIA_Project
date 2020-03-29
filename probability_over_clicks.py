import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 100.0, 0.1)
#Fall
s1 = -15000 * np.exp(-np.power(t - 0, 2.) / (2 * np.power(50, 2.))) + 15000
s2 = 12000 * (1 - np.exp((-1*t)/70)) + 1000 * np.log(t+1) + 1000*np.exp(-np.power(t - 10, 2.) / (2 * np.power(5, 2.)))
s3 = 3500 * (1 - np.exp((-1*t)/10))
#Christmas
w1 = 23000 * (1 - np.exp((-1*t)/85))
w2 = -20000 * np.exp(-np.power(t - 0, 2.) / (2 * np.power(50, 2.))) + 20000
w3 = 11500 * (1 - np.exp((-1*t)/20)) + 13*t * np.sin(t)
#Spring
pr1 = 900 * np.log(t/27+1)
pr2 = 1100 * np.log(t/20+1)
pr3 = 700 * (1 - np.exp((-1*t)/40)) + 200 * np.log(t/35+1)

fig, ax = plt.subplots()
ax.plot(t, s1, "r")
ax.plot(t, s2, "g")
ax.plot(t, s3, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Probability over daily clicks - Autumn')
ax.grid()
plt.savefig('curves/daily_clicks_fall.png')
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t, w1, "r")
ax2.plot(t, w2, "g")
ax2.plot(t, w3, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax2.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Probability over daily clicks - Christmas time')
ax2.grid()
plt.savefig('curves/daily_clicks_xmas.png')
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(t, pr1, "r")
ax3.plot(t, pr2, "g")
ax3.plot(t, pr3, "b")
plt.legend(["Northern Italy with childrens", "Northern Italy without childrens", "Southern Italy with childrens"])
ax3.set(xlabel='Budget allocated to subcampaign in k€', ylabel='number of clicks',
       title='Probability over daily clicks - Spring')
ax3.grid()
plt.savefig('curves/daily_clicks_spring.png')
plt.show()

