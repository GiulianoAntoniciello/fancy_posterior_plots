# coding: utf-8

import itertools as it

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as colors

import seaborn as sns



# posteriors: change this to contain your actual posteriors
# each row: a sample
# each column: a parameter
posteriors = np.random.normal(loc=2, scale=1, size=(100_000, 4))


n_var = posteriors.shape[1] # counts the parameters

# set the labels for the parameters
labels = ('$C$', '$\\tau$', '$t_0$', '$\gamma$')
# set the units (leave '' for numerical parameters)
units = ('', 'd', 'd', '')

# constructs the label strings
label_and_unit = []
for i, (unit, label) in enumerate(zip(units, labels)):
        
    if unit == '':
        label_and_unit.append(labels[i])
    else:
        label_and_unit.append(labels[i] + ' (' + str(units[i]) + ')')
        


# initialize the figure as a matrix of plots
fig, ax = plt.subplots(n_var, n_var, figsize=(3*n_var, 3*n_var), 
                       sharex = 'col', sharey = False, 
                       gridspec_kw = {'hspace':0, 'wspace':0})



for i, j in it.product(range(n_var), repeat=2):
    # set the axis off for the upper half of the plot matrix
    if i < j:
        
        ax[i, j].set_axis_off()
        
    # plot the posterior histogram on the principal diagonal of the plot matrix
    elif i == j:
        
        axes = ax[i, j]
        x = posteriors[:, i]
        
        # set the location and scale parameters to
        # estimate the uncertainty from the posterior pdf
        x_median = np.median(x) # location paramtere (i. e. your best estimate)
        x_minus = np.percentile(x, 16) - x_median # lower percentile of a 1-sigma interval
        x_plus = np.percentile(x, 84) - x_median # upper percentile of a 1-sigma interval
        
        axes.hist(x, 
                  density=True, 
                  bins=20, 
                  zorder = -1, 
                  histtype = 'step', 
                  hatch='///');
        
        axes.set(xlabel='', ylabel='')
        axes.set_yticks([])
        
        # draw the posterior interval
        axes.axvline(x = x_median, lw = 1.5, color = 'r')
        axes.axvspan(x_minus + x_median, 
                     x_plus + x_median, 
                     alpha=0.2, color='r', lw = 0)
        
        # set the label
        axes.set_title(labels[i] + ' = ' +
                       '%.2f$_{%.2f}^{%.2f}$'%(x_median, x_minus, x_plus) 
                       + ' ' + label_and_unit[i])
    
    # plot the 2D histogram to represent the marginal distributions of the posteriors
    elif i > j:
        
        axes = ax[i, j]
        x = posteriors[:, j]
        y = posteriors[:, i]
        
        axes.set(xlabel=label_and_unit[j], 
                 ylabel=label_and_unit[i])
        
        if (i == (n_var - 1)):

            if j == 0:
                color = 'r'
                axes.set(xlabel=label_and_unit[j], 
                         ylabel=label_and_unit[i])
            else:
                color = 'g'
                axes.set(xlabel=label_and_unit[j], 
                         ylabel='')
                axes.set_yticklabels([])

        elif j == 0:
            color = 'y'
            axes.set(xlabel='', 
                     ylabel=label_and_unit[i])
            
        else:
            axes.set(xlabel='', ylabel='')   
            axes.set_yticklabels([])
            color = 'b'

        axes.hist2d(x, y, bins = 50, cmap='Blues', norm=colors.LogNorm(vmin = 10), lw=0)
        
        
fig.tight_layout()
        
fig.savefig('fancy_posterior_plots.pdf', dpi=600)

