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


X_RANGE = [-0.01, 0.011]
Y_RANGE = []


def plot(o, a):
    plt.rcParams.update({'font.size': 40})
    plt.close()
    #plt.clf()


    fig = plt.figure(figsize=(16, 5))
    x = np.linspace(X_RANGE[0], X_RANGE[1], 100)
    f = lambda a: 2*a
    y = f(x)

    ax = fig.add_subplot()
    ax.set_xlim(X_RANGE[0], X_RANGE[1])
    ax.set_ylim(-1, 1)
    ax.set_title('Applied force (N)', size=40)
    #ax.set_xlabel('Applied force (N)')
    ax.xaxis.label.set_position((X_RANGE[0], 0))

    arrow = mpatches.FancyArrowPatch((0, 0.1), (a[0], 0.1),
                                     mutation_scale=200)
    ax.add_patch(arrow)
    #ax.set_ylabel('Contribution to action')
    #ax.yaxis.label.set_position((0, 0.8))
    #ax.plot(x, y)

    #ax.barh(0.1, a[0], color='r')
    #ax.quiver([0], [0], [a[0]], [0], 'red')
    #ax.quiver([0, 0], a[0], 0)

    plt.yticks([])

    #ax.scatter(o, f(o), s=200, color='r')

    #ax.spines['left'].set_position('zero')
    #ax.spines['right'].set_color('none')
    #ax.spines['bottom'].set_position('zero')
    #ax.spines['top'].set_color('none')

    # remove the ticks from the top and right edges
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks(np.arange(X_RANGE[0], X_RANGE[1], 0.005))
    ax.yaxis.set_ticks_position('left')
    #plt.show()

i = 0
for o, a in zip(obs[:33], acts[:33]):
    fig = plot(o[0], a)
    #plt.show()
    #exit()
    plt.savefig(f'./cart_action/{i}.pdf')
    i += 1

