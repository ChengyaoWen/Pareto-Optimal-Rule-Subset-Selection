import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def log_to_array(log_file):
    df = pd.read_csv(log_file)
    return df.values


pors_log = "figure_emo_comp_pors.csv"
emo_log = "figure_emo_comp_emo.csv"
logs = [pors_log, emo_log]
names = [['PORS Train', 'PORS Valid', 'PORS Test'],
         ['NSGA-II Train', 'NSGA-II Valid', 'NSGA-II Test']]
results = []
for log in logs:
    res = log_to_array(log)
    results.append(res)

plt.rcParams["figure.figsize"] = (8,6)
plt.xscale('log')

plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=15)



plt.plot(results[0][:, 3], results[0][:, 0], color='#d62728', linewidth=3, label=names[0][0])
plt.plot(results[0][:, 3], results[0][:, 1], color='#ff7f0e', linewidth=3, label=names[0][1])
plt.plot(results[0][:, 3], results[0][:, 2], color='#2ca02c',linewidth=3, label=names[0][2])


plt.plot(results[1][:, 3], results[1][:, 0], color='#d62728', linestyle='--', linewidth=3, label=names[1][0])
plt.plot(results[1][:, 3], results[1][:, 1], color='#ff7f0e', linestyle='--', linewidth=3, label=names[1][1])
plt.plot(results[1][:, 3], results[1][:, 2], color='#2ca02c', linestyle='--', linewidth=3, label=names[1][2])

#plt.ylim(0.4, 0.7)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel("HV", fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.legend(loc='lower right')
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
plt.grid()
plt.savefig('emo_comp.pdf', bbox_inches='tight', format='pdf')
