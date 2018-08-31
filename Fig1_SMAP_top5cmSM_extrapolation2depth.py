#!~/anaconda3/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.transforms import blended_transform_factory


figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
#=======================================================================
#===== Farshid Felfelani
#===== First version: 04/10/2018
#===== SMAP top 5cm SM Extrapolation
#%%=======================================================================

alpha = 1.0

plt.figure(3,figsize=(4,6))
linestyles = OrderedDict(
    [('B=4.05',               (0, ())),
     ('B=4.38',      (0, (1, 10))),
     ('B=4.9',      (0, (1, 1))),

     ('B=5.3',      (0, (5, 10))),

     ('B=7.12',      (0, (5, 1))),

     ('B=7.75',          (0, (3, 1, 1, 1))),

     ('B=8.52', (0, (3, 10, 1, 10, 1, 10))),
     ('B=10.4',         (0, (3, 5, 1, 5, 1, 5))),
     ('B=11.4', (0, (3, 1, 1, 1, 1, 1)))])



B = [4.05, 4.38, 4.9, 5.3, 7.12, 7.75, 8.52, 10.4, 11.4]

sat_deg = np.arange(0,1,0.02)

n = 9
colors = plt.cm.cool(np.linspace(0,1,n))

for i, (name, linestyle) in enumerate(linestyles.items()):
    ratio = 1-(1-sat_deg)**(B[i]*alpha)
    plt.plot(ratio,sat_deg,linestyle=linestyle,label = name,color=colors[i])
    plt.ylim(1,-0.1)
plt.legend(ncol=2,loc = 'lower left')
plt.xlabel(r'$1-(1-\theta^*)^B$')
plt.ylabel('Degree of Saturation (' + r'$\theta^*$' + ')')
    


plt.savefig(figDIR + 'Fig1_SMAP_5cmSM_Extrapolation' + '.png', dpi=600, bbox_inches='tight')
plt.savefig(figDIR + 'Fig1_SMAP_5cmSM_Extrapolation' + '.pdf', bbox_inches='tight')

