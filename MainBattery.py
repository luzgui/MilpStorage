#!/usr/bin/env python
# coding: utf-8

# In order to run: <br>
# (1) unzip the submit dataset and put the resulting csv files into the submit folder in the repository <br>
# (2) Make sure you have all the required packages including pyomo installed <br>
# (3) Make sure cplex is installed and set the path to the executable file <br>
# This post goes along with the notebook: http://energystoragesense.com/uncategorized/scheduling_batt_optimisation/

# In[57]:

from __future__ import division

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.core import Var
import pyomo.environ as en
import seaborn as sns
import time

# Since using the data-driven data for the testing, use their battery class
from battery import *
from batterymodel import *


import os
from operator import itemgetter
import scipy.io as sio
import re 


from itertools import compress


#%% Folders

cwd = os.getcwd()
#general data folder
DataFolder=cwd + '/data'


# In[58]:


# set up seaborn the way you like
sns.set_style({'axes.linewidth': 1, 'axes.edgecolor':'black', 'xtick.direction':                'out', 'xtick.major.size': 4.0, 'ytick.direction': 'out', 'ytick.major.size': 4.0,               'axes.facecolor': 'white','grid.color': '.8', 'grid.linestyle': u'-', 'grid.linewidth': 0.5})





#%%

ti=0
H=1*48
dt=30


#PV
g_df=pd.read_csv(os.path.join(DataFolder,'pv_gen.csv'), names=['PV'])
PV=4*abs(g_df.PV[ti:H].values)


#tariffs
tariffs=pd.read_csv(os.path.join(DataFolder,'issda_tariffs.csv')) 

# tar_mibel=pd.read_csv(os.path.join(DataFolder,'mibel_11_may.csv'), names=['M','tar'])
# tar=tar_mibel.tar[:]

tar=tariffs.A[0:H]
tar.name='tar' #allways have the name as tar for tariffs

# tar=pd.Series(0.10*np.ones(H),name='tar')

sellPrice = pd.Series(0.01*np.ones(H,),name='tar_sell')
priceDict1 = dict(enumerate(sellPrice))
buyPrice=tar
priceDict2 = dict(enumerate(buyPrice))

#Loads
loads= pd.read_csv(os.path.join(DataFolder,'issda_3194.csv'))
load=loads.id2001[0:H]

# Some parameters dataframes
df_param=pd.DataFrame((load,PV)).transpose()
df_param.columns=['load','PV']



#%% Instantiate a battery

batt = Battery(capacity=10.0,
           charging_power_limit=5.0,
           discharging_power_limit=-5,
           charging_efficiency=0.95,
           discharging_efficiency=0.95)



# batt2=Battery_ob()


#%% Instantiate the model

m=battmodel(H,batt,load,PV,priceDict1,priceDict2,dt)
# m=battmodelNEM(H,batt2, load,PV, sellPrice, buyPrice, dt, 2, 1.5)



#%% Solve Model

# opt = SolverFactory("gurobi")
opt = SolverFactory("glpk")
# time it for good measure
t = time.time()
# results = opt.solve(m, tee=True)
results = opt.solve(m, tee=True)
elapsed = time.time() - t
print('Time elapsed:', elapsed)



#%% get model output 
# now let's read in the value for each of the variables 
outputVars = np.zeros((9,H))
varnames=[]


j = 0
for v in m.component_objects(Var, active=True):
    # print(v.getname())
    #print varobject.get_values()
    varnames.append(v.getname())
    varobject = getattr(m, str(v))
    for index in varobject:
        # print(index)
        outputVars[j,index] = varobject[index].value
                
    j+=1
    if j>=9:
        break

df_sol=pd.merge(pd.DataFrame(outputVars.T, columns=varnames),df_param, left_index=True, right_index=True)
df_sol=pd.merge(df_sol, tar, left_index=True, right_index=True)
df_sol=pd.merge(df_sol, sellPrice, left_index=True, right_index=True)




#%% Analise costs


# get the total cost
# cost_without_batt = np.sum([(buyPrice[i]*posLoad[i] + sellPrice[i]*negLoad[i]) for i in range(H)])

# cost_with_batt = np.sum([(buyPrice[i]*outputVars[7,i] + sellPrice[i]*outputVars[8,i]) for i in range(H)])

# print('Cost without battery:', cost_without_batt)
# print('Cost with battery:', cost_with_batt)
# print('Score: %.4f'%((cost_with_batt - cost_without_batt) / np.abs(cost_without_batt)))


#%% makeplots
make_plot(df_sol,0*48,1*48)













# In[ ]:





# # Now compare to alternate formulation

# # In[95]:


# def battery_scheduler(batt, loadForecast, pvForecast, priceBuy, priceSell):

#     seriesLength=len(loadForecast)

#     ### ------------------------- ESS properties ----------------- ###
#     maxSOC = batt.capacity # Units are Wh
#     maxChg = batt.charging_power_limit/4. # W
#     maxDisChg = batt.discharging_power_limit/4. # W
#     etaChg = batt.charging_efficiency
#     etaDisChg = batt.discharging_efficiency
#     ### ---------------------------------------------------------- ###
#     expectedExports = np.zeros((seriesLength)) 

#     net = loadForecast-pvForecast          
#     expectedExports[net<0] = -net[net<0]

#     # now alter the price in the periods where there is solar available (if req.)
#     priceCharge = np.copy(priceBuy)
#     priceCharge[net<0] = priceSell[net<0]
#     priceDischarge = np.copy(priceSell)
#     priceDischarge[net>0] = priceBuy[net>0]

#     # get the storage profiles
#     SOC = np.zeros((seriesLength))
#     deltaSOC = np.zeros((seriesLength))

#     # boolean variables for charging/discharging availability
#     removeMINH = np.ones((seriesLength))
#     removeMAXH = np.ones((seriesLength))

#     demand = np.copy(net)

#     initial_charge = batt.current_charge*maxSOC
#     ###############################-----------------------------##################################
#     # now first, if there is energy stored, sell this and update schedules
    
#     if initial_charge > 0:
#         #for j in range(1):
#         while ((initial_charge>0) and (np.any(removeMAXH))):

#             matrix = np.zeros(( 2, len ( np.where(removeMAXH==True)[0] ) ))
#             matrix[0,:] = np.where(removeMAXH==True)[0]
#             matrix[1,:] = priceDischarge[ np.where(removeMAXH==True)[0] ]    

#             indici = np.where( matrix[1,:]==np.max( matrix[1,:] ) )[0]
#             maxh = np.int(matrix[0,indici[0]])
#             #print(maxh)
#             # discharge at MAXh at the maximum level and update the schedule
            
#             if priceDischarge[maxh]>0:
#                 bottleneck = np.zeros((3))
#                 bottleneck[0] = deltaSOC[maxh]-maxDisChg/etaDisChg
#                 bottleneck[1] = initial_charge
#                 ############### cannot output more than local consumption before price change
#                 if demand[maxh]>0:
#                     bottleneck[2] = (demand[maxh])/etaDisChg
#                 else:
#                     bottleneck[2] = initial_charge
#                 #print(bottleneck)
#                 actual_bottleneck = np.min(bottleneck) 

#                 # action
#                 demand[maxh] = demand[maxh] - actual_bottleneck*etaDisChg
#                 deltaSOC[maxh] = deltaSOC[maxh] - actual_bottleneck
#                 SOC[0:maxh] = SOC[0:maxh] + actual_bottleneck

#                 # account for small rounding errors
#                 if ((demand[maxh] < 0 + 0.0001) and (demand[maxh] > 0 - 0.0001)):
#                     demand[maxh] = 0
#                     # update the price
#                     priceDischarge[maxh] = priceSell[maxh]

#                 # check if at the charge or discharge operation is at capacity at either
#                 # maxh or minh and remove that hour from the price distribution.
#                 if deltaSOC[maxh] <= maxDisChg/etaDisChg+0.0001:
#                     removeMAXH[maxh] = False
#                 if deltaSOC[maxh] <= 0 - 0.0001:
#                     removeMINH[maxh] = False

#                 initial_charge = initial_charge - actual_bottleneck
#             else:
#                 removeMAXH[maxh] = False
            
#             #print removeMAXH
            
#     ###############################-----------------------------##################################

#     while np.any(removeMAXH) == True:
#         matrix = np.zeros(( 2, len ( np.where(removeMAXH==True)[0] ) ))
#         matrix[0,:] = np.where(removeMAXH==True)[0]
#         matrix[1,:] = priceDischarge[ np.where(removeMAXH==True)[0] ]

#         # find the maximum available price 
#         indici = np.where( matrix[1,:]==np.max( matrix[1,:] ) )[0]
#         maxh = np.int(matrix[0,indici[0]])
#         #print('Maxh = ', maxh)

#         # find the last hour before maxh when storage was full
#         r1 = np.where( SOC[0:maxh+1] == maxSOC )[0]
#         if r1.size == 0:
#             r1 = 0
#         else:
#             r1 = np.where( SOC[0:maxh+1] == maxSOC )[0][-1]+1
#         # find the first hour after maxh when storage is empty
#         r2 = np.where( SOC[maxh:] == 0 )[0]
#         if r2.size == 0:
#             r2 = len(SOC)
#         else:
#             r2 = np.where( SOC[maxh:] == 0 )[0][0]+maxh-1

#         #print('r1 = ', r1)
#         #print('r2 = ', r2)

#         # find the minh in the time range
#         range_price = priceCharge[r1:r2+1]
#         range_remove = removeMINH[r1:r2+1]

#         # if there is no hour in the range then remove maxh and skip to the end
#         if np.any(range_remove) == False:
#             removeMAXH[maxh]=False
#         else:
#             matrix = np.zeros(( 2, len ( np.where(range_remove==True)[0] ) ))
#             matrix[0,:] = np.where(range_remove==True)[0]
#             matrix[1,:] = range_price[ np.where(range_remove==True)[0] ]

#             indici = np.where( matrix[1,:]==np.min( matrix[1,:] ) )[0]
#             minh = np.int( matrix[0,indici[0]] + r1 )
#             # calculate the marginal cost of operation
#             MoC = priceCharge[minh]/(etaChg*etaDisChg)
#             #print('Minh = ', minh)

#             if ((MoC<priceDischarge[maxh]) and (minh!=maxh)):

#                 bottleneck = np.zeros((5))
#                 bottleneck[0] = deltaSOC[maxh]-maxDisChg/etaDisChg
#                 bottleneck[1] = maxChg*etaChg - deltaSOC[minh]
#                 if maxh > minh:
#                     bottleneck[2] = maxSOC - np.max(SOC[minh:maxh+1])
#                 else:
#                     bottleneck[2] = np.min(SOC[maxh:minh+1])

#                 ############### cannot output more than local consumption before price change
#                 if demand[maxh]>0:
#                     bottleneck[3] = demand[maxh]/etaDisChg
#                 else:
#                     bottleneck[3] = maxSOC

#                 ############## cannot charge more than export before price change
#                 if expectedExports[minh] > 0:
#                     bottleneck[4] = expectedExports[minh]*etaChg
#                 else:
#                     bottleneck[4] = maxSOC
#                 #print(bottleneck)
#                 actual_bottleneck = np.min(bottleneck)  

#                 # Update the aggregated demand
#                 demand[maxh] = demand[maxh] - actual_bottleneck*etaDisChg
#                 demand[minh] = demand[minh] + actual_bottleneck/etaChg

#                 # update the storage charging schedule
#                 expectedExports[minh] = expectedExports[minh] - actual_bottleneck/etaChg
#                 deltaSOC[minh] = deltaSOC[minh] + actual_bottleneck
#                 deltaSOC[maxh] = deltaSOC[maxh] - actual_bottleneck
#                 if maxh>minh:
#                     SOC[minh:maxh] = SOC[minh:maxh] + actual_bottleneck
#                 else:
#                     SOC[maxh:minh] = SOC[maxh:minh] - actual_bottleneck

#                 # account for small rounding errors
#                 if ((demand[maxh] < 0 + 0.0001) and (demand[maxh] > 0 - 0.0001)):
#                     demand[maxh] = 0
#                     # update the price
#                     priceDischarge[maxh] = priceSell[maxh]
#                 if expectedExports[minh] < 0 + 0.0001:
#                     expectedExports[minh] = 0
#                     # update the price
#                     priceCharge[minh] = priceBuy[minh]

#                 # check if at the charge or discharge operation is at capacity at either
#                 # maxh or minh and remove that hour from the price distribution.
#                 if deltaSOC[maxh] <= maxDisChg/etaDisChg+0.0001:
#                     removeMAXH[maxh] = False
#                     #removeMINH[maxh] = False
#                 if deltaSOC[minh] >= maxChg*etaChg-0.0001:
#                     removeMINH[minh] = False
#                     #removeMAXH[minh] = False
#                 # instead of this, let's try only allowing the hour to be selected for either charging or discharging
#                 if deltaSOC[maxh] < 0 - 0.0001:
#                     removeMINH[maxh] = False
#                 if deltaSOC[minh] > 0 + 0.0001:
#                     removeMAXH[minh] = False

#             else:
#                 # if there is no price incentive remove the hours
#                 removeMAXH[maxh] = False
#                 removeMAXH[minh] = False 

#     return demand, deltaSOC, SOC


# # In[96]:


# alternativeOutput, alternativeDeltaSOC, alternativeSOC = battery_scheduler(batt, load, PV, buyPrice, sellPrice)


# # In[97]:


# alternativeAction = np.zeros((len(buyPrice)))
# for i in range(len(buyPrice)):
#     if alternativeDeltaSOC[i]>=0:
#         alternativeAction[i] = alternativeDeltaSOC[i]/batt.charging_efficiency
#     else:    
#         alternativeAction[i] = alternativeDeltaSOC[i]*batt.discharging_efficiency


# # In[98]:


# #alternativeAction = alternativeOutput-(load-PV)


# # In[99]:


# print('Cost without battery:', cost_without_batt)
# print('Cost with battery pyomo:', cost_with_batt)
# cost_with_batt = np.sum([(buyPrice[i]*alternativeOutput[i]) for i in range(len(buyPrice)) if alternativeOutput[i]>=0])+        np.sum([(sellPrice[i]*alternativeOutput[i]) for i in range(len(buyPrice)) if alternativeOutput[i]<0])
# print('Cost with battery alternative optimum scheduling:', cost_with_batt)


# # In[100]:


# fig = plt.figure(figsize=(14,5))
# ax1 = fig.add_subplot(211)
# l1, = ax1.plot(hrs,np.sum(outputVars[3:7,:]*4/1000, axis=0),color=colors[0])
# l2, = ax1.plot(hrs,alternativeAction*4/1000,color=colors[1])
# ax1.set_xlabel('hour'), ax1.set_ylabel('action')
# ax1.legend([l1,l2],['battery action pyomo','battery action alternate'],ncol=2,loc='upper left')
# ax1.set_xlim([0,len(load)/4]);
# ax1.set_ylim([1.2*batt.discharging_power_limit/1000, 1.6*batt.charging_power_limit/1000]);

# ax2 = fig.add_subplot(212)
# l1, = ax2.plot(hrs,outputVars[0]/1000,color=colors[0])
# l2, = ax2.plot(hrs,alternativeSOC/1000,color=colors[1])
# ax2.set_xlabel('hour'), ax2.set_ylabel('action')
# ax2.legend([l1,l2],['SOC pyomo','SOC alternate '],ncol=2)
# ax2.set_xlim([0,len(load)/4]);
# ax2.set_ylim([-0.1*batt.capacity/1000, 1.5*batt.capacity/1000]);


# # In[ ]:




