import matplotlib.pyplot as plt
import numpy as np

data_a = [[5,3,31,26,125,24,47,63,25,81,122,28,80,22,56,47,3,18,33,40,6],
          [67,114,104,246,66,106,69,68,193,100,187,95,130,253,372,298,132,252,231,211,122],
          [106,123,146,452,93,121,183,151,238,131,284,99,1000,1000,388,100,147,220,258,454,218]]

data_b = [[14.39130435,10.18315789,4.005681818,6.931034483,12.18894009,67.38655462,17.20348837,8.169201521,12.36406619,4.945823928,21.88395904,5.95221843,12.20327103,5.670212766,19.50154799,17.99013158,15.1618799,7.939393939],
          [140.9727891,96.99759615,10.18417047,6.223285486,91.37272727,218.0721649,142.3972603,12.43695441,31.48083942,36.59657321,72.46125461,70.46750903,169.4516129,3.004531722,149.9748201,158.3656388,49.17236842,11.92095588,8.495590829],
          [190.4492754,132.5793103,7.486203616,5.059139785,89.73863636,132,138.1284404,57.13263158,18.7413148,16.37885463,240.8421053,176.3540146,72.91749175,3,151.4848485,15.36490615,47.43520309,4.113221884,4.003795066]]


ticks = ['20k', '100k', '200k']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, 'black') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='black', label='CGP')
plt.plot([], c='#2C7BB6', label='CM-GP')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.title('Inverted Pendulum training')
plt.ylabel('Episodic return')
plt.xlabel('Interactions')
plt.ylim(0, 500)
plt.tight_layout()
#plt.show()
plt.savefig('comp.pdf')