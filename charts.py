# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 07:39:00 2017

@author: Isma

Module holding several charting functions for the results

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter, MultipleLocator

def plot_clustered_results(results, N=50, title="Cancelamentos por grau de risco", upperBound=None):
    """
    Plot results ordered in clusters
    return a modified results table including the cluster classification
    """
    
    groups = pd.qcut(results.prob.rank(method='first'), N)
    
    return plot_bins(results, groups, title, upperBound, plotProb=True, plotCount=False)
    
def plot_final_results(results, title=None, upperBound=None):
    """
    Function that plot the results according to our own user-defined bins
    """
    groups, bins = pd.qcut(results.prob, [0, 0.50, 0.75, 0.90, 0.98, 1], retbins=True)
    
    return plot_bins(results, groups, title, upperBound, plotProb=False, plotCount=True)

def plot_bins(results, groups, title=None, upperBound=None, plotProb=False, plotCount=False):
    
    counts = results.groupby(groups)['y'].count()
    means = results.groupby(groups)['y'].mean()
    probs = results.groupby(groups)['prob'].mean()
    
    ax = None
    if (plotProb):
        ax = probs.plot.bar(width=0.9, color='grey')
    ax = means.plot.bar(ax=ax,width=0.7, color='blue', alpha=0.5)
    ax.plot(np.ones(len(groups))*results['y'].mean(), color='black', linewidth=3, label='Média')
    h,l = ax.get_legend_handles_labels()
    
    ax.set_title(title, size=15, weight='bold')
    if (plotProb):
        ax.legend(h, ['% predicted', '% effective'], loc='best')
    else:
        ax.legend(h, ['Mean', '% effective'], loc='best')
    if (upperBound is not None):
        ax.set_ylim([0,upperBound])
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in ax.get_yticks()])
    if (len(groups) >= 20):
        ax.set_xticklabels(['%d'%x for x in range(len(groups))], rotation=45)
    else:
        ax.set_xticklabels(([chr(ord('A')+x) for x in range(len(groups))]), rotation=0)
    ax.set_xlabel('Group', size=15)
    ax.set_ylabel('% effective', size=15)

    if (plotCount):
        for (group, rect) in zip(counts.index, ax.patches):
            ax.text(rect.get_x() + rect.get_width()/2., 0.95*rect.get_height(), 
                    '{:,}\nalunos'.format(counts.loc[group]), ha='center', va='top', color='white')

    new_results = results.copy()
    new_results['cluster'] = groups
    
    return new_results


def plot_probabilities(results, zoomFactor = 0.05, title="Probabilidade de Cancelamento"):
    """
    Plot results ordered as a line, clustered
    return a modified results ordered by probability
    """    
    ax = None
    new_results = results.sort_values('prob')
    new_results.index = range(0,len(new_results))
    groups = new_results.groupby('y')
    
    ax = plt.subplot(2, 1, 1)
    for name, group in groups:
        ax.scatter(group.index, group.prob, marker='o',  s=4, label=name, alpha = .4)
        
    ax = plt.gca()
    ax.plot(np.ones(len(new_results))*results['y'].mean(), color='black', linewidth=2)
    h,l = ax.get_legend_handles_labels()
    ax.set_ylim([0,1])
    ax.set_xlim([0,len(new_results)])
    ax.set_title(title, size=15, weight='bold')
    ax.legend(h, [title], loc='best')
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in ax.get_yticks()])
    ax.set_xticks(np.arange(0, len(new_results)+1, len(new_results)/10))
    ax.set_xticklabels(['%d'%(x) for x in range(100,-1,-10)], rotation=90)
    ax.set_xlabel('Ranking', size=15)
    ax.set_ylabel('% effective', size=15)
    
    
    zoomBegin = len(new_results)*(1-zoomFactor)
    zoomEnd = len(new_results)
    
    ax2 = plt.subplot(2, 1, 2)
    for name, group in groups:
        ax2.scatter(group.index, group.prob, marker='o',  s=16, label=name, alpha = .4)

    ax2.plot(np.ones(len(new_results))*results['y'].mean(), color='black', linewidth=2)
    h,l = ax.get_legend_handles_labels()
    ax2.set_ylim([0,1])
    ax2.set_xlim([zoomBegin,zoomEnd])
    ax2.set_yticklabels(['{:3.2f}%'.format(x*100) for x in ax.get_yticks()])
    ax2.set_xticks(np.arange(zoomBegin, zoomEnd+1, (zoomEnd - zoomBegin)/10))
    ax2.set_xticklabels(['%d'%(x) for x in np.arange(100*zoomFactor,0,-zoomFactor*10)], rotation=90)
    ax2.set_xlabel('Ranking', size=15)
    ax2.set_ylabel('% effective', size=15)
        
    plt.show()
    
    return new_results

def plot_cancel_curve(all_results, title = None, cmap='Paired'):
    """
    Plot curve of cancelamentos x alunos
    """

    if (type(all_results) != dict):
        all_results = {'modelo':all_results}

    ax = None
    
    colors = iter(cm.get_cmap(cmap)(np.linspace(0, 1, len(all_results))))

    for (name, results) in all_results.items():
        sorted_results = results.sort_values('prob', ascending=False)[['y', 'prob']].cumsum()
        n_cancelamentos = results.y.sum()
        n_alunos = len(results)
        chart_data = pd.DataFrame({'alunos':np.arange(n_alunos)/n_alunos, 'y':sorted_results.y.values/n_cancelamentos})

        ax = chart_data.plot(y='y', x='alunos', label=name, marker='.', ax = ax, c=next(colors), grid=True, xticks=np.arange(0, 1, 0.1), yticks=np.arange(0,1,0.1))
        
    ax.set_title(title, size=15, weight='bold')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x,p: format(x*100, '.0f')+'%'))
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x,p: format(x*100, '.0f')+'%'))
    
    ax.set_xlabel("Total Sample (% total)", size=12)
    ax.set_ylabel("Y count (% total)", size=12)
    
    ax.scatter(x=np.arange(n_alunos)/n_alunos, y=np.min((np.ones(n_alunos), np.arange(n_alunos)/n_cancelamentos), axis=0), label='y', marker='.', s=2, color='black')
    ax.scatter(x=np.arange(n_alunos)/n_alunos, y=np.arange(n_alunos)/n_alunos, label='sem modelo', marker='.', s=2, color='0.75')

    
    
    h,l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc='lower right')
    
    
    return chart_data
    
def plot_probabilities_color(results, title=None, cmap='YlOrRd'):
    """
    Plot results ordered as a line, clustered
    return a modified results ordered by probability
    """    
    new_results = results.sort_values('prob')
    new_results.index = range(0,len(new_results))

    n_groups = (int)(len(results)/100)    
    
    groups = pd.qcut(new_results.prob.rank(method='first'), n_groups, labels=False)
    
    fig, ax = plt.subplots()

    ax = new_results['prob'].plot(color='black', linewidth=2, ax=ax)    
    
    new_results['avg_y'] = new_results.groupby(groups)['y'].mean()

    cmap = cm.get_cmap(cmap)
    
    for i in range(n_groups):
        ax.fill_between(new_results[groups==i].index, new_results[groups==i].prob, color=cmap(new_results[groups==i].y.mean()))
        
    h,l = ax.get_legend_handles_labels()
    ax.set_ylim([0,1])
    ax.set_xlim([0,len(new_results)])
    ax.set_title(title, size=15, weight='bold')
    ax.legend(h, [title], loc='best')
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x,p: format(x*100, '.0f')+'%'))
    ax.set_xlabel('Samples', size=15)
    ax.set_ylabel('Prediction', size=15)
    
    import matplotlib.colorbar as cbar
    
    cax, _ = cbar.make_axes(ax)
    cb2 = cbar.ColorbarBase(cax, cmap=cmap, label="% effective")
    
    
    
    return new_results
    