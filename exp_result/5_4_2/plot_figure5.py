import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def log_to_array(log_file):
    df = pd.read_csv(log_file)
    return df.values

logfile = "figure_5.csv"
time_array = log_to_array(logfile)

#colors = ['b','r']
#linestyle = ['-','-']
iterations = np.arange(time_array.shape[0])
plt.rcParams["figure.figsize"] = (8,4)

plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=15)


plt.plot(iterations, time_array[:len(iterations), 0], color='#ffbb78', linewidth=5.0, label='PORS')
plt.plot(iterations, time_array[:len(iterations), 1], color='#98df8a', linewidth=5.0, label=r'Greedy $F_{0.1}$')
plt.plot(iterations, time_array[:len(iterations), 2], color='#ff9896', linewidth=5.0, label=r'Greedy $F_{0.2}$')
plt.plot(iterations, time_array[:len(iterations), 3], color='#c5b0d5', linewidth=5.0, label=r'Greedy $F_{0.5}$')

plt.xlabel('Iteration', fontsize=15)
plt.ylabel("Time (s)", fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.legend(loc='upper left')
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
plt.savefig('fbeta.pdf', bbox_inches='tight', format='pdf')
