import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as mpatches


def f(obs):
    a = 4*obs[1] + 2*obs[2] + 2*obs[3]
    return a




import numpy as np
import matplotlib.pyplot as plt

obs, acts = None, None

with open('obs.npy', 'rb') as f:
    obs = np.load(f)
with open('acts.npy', 'rb') as f:
    acts = np.load(f)


X_RANGE = [-0.03, 0.03]
Y_RANGE = [-0.04, 0.04]


def plot(o, a):
    plt.clf()
    fig = plt.figure(figsize=(19, 6))
    x = np.linspace(X_RANGE[0], X_RANGE[1], 100)
    f = lambda a: 2*a
    y = f(x)

    ax = fig.add_subplot()
    #ax.set_title('Angle pole (rad)')
    ax.set_xlim(X_RANGE[0], X_RANGE[1])
    plt.rcParams.update({'font.size': 20})
    ax.set_ylim(Y_RANGE[0], Y_RANGE[1])
    ax.set_xlabel('Velocity pole (rad/s)', size=30)
    ax.xaxis.label.set_position((X_RANGE[0] + 0.01, Y_RANGE[1]))
    ax.set_title('Contribution', size=30)
    #ax.yaxis.label.set_position((1, 1))
    ax.plot(x, y)

    ax.scatter(o, f(o), s=1000, color='r')
    print(o)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    # remove the ticks from the top and right edges
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks(np.arange(X_RANGE[0], X_RANGE[1], 0.02))
    ax.yaxis.set_ticks(np.arange(Y_RANGE[0], Y_RANGE[1], 0.02))
    ax.yaxis.set_ticks_position('left')
    #plt.show()

i = 0
for o, a in zip(obs, acts):
    fig = plot(o[3], a)
    plt.savefig(f'./pole_angle/{i}.pdf')
    i += 1

