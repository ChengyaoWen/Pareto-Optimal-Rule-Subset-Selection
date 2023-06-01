import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_size = ('100','200','500','1000','2000')
figure_df = pd.read_csv("figure_3.csv")

hvs = dict()
for _, row in figure_df.iterrows():
    stage1 = row['Stage-1']
    for size in input_size:
        hv_mean = float(row[size].split('$\pm$')[0].strip())
        hv_std = float(row[size].split('$\pm$')[1].strip())
        if not hvs.get(stage1):
            hvs[stage1] = [[], []]
        hvs[stage1][0].append(hv_mean)
        hvs[stage1][1].append(hv_std)

sr_hv_means, sr_hv_std = hvs['sr'][0], hvs['sr'][1]
te_hv_means, te_hv_std = hvs['te'][0], hvs['te'][1]

ind = np.arange(len(te_hv_means))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)

rects1 = ax.bar(ind - width, sr_hv_means, width, yerr=sr_hv_std, ecolor='black',
                label='SpectralRules', color='#98df8a')
rects2 = ax.bar(ind, te_hv_means, width, yerr=te_hv_std, ecolor='black',
                label='TreeEns', color='#ffbb78')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('HV', fontsize=15)
ax.set_xlabel('Number of Rules', fontsize=15)
ax.set_xticks(ind)
ax.set_xticklabels(input_size, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.legend(fontsize=15)

plt.rcParams["figure.figsize"] = (5,4)

leg = ax.legend()
leg.get_frame().set_edgecolor('black')

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    # textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

plt.ylim(0.4,0.7)
fig.tight_layout()
plt.savefig('sr_vs_tree.pdf', format="pdf")
