# -*- coding: utf-8 -*-
"""
Created on Tuesday, November 14, 2023 

Author: Daniel Juarez Robles
Southwest Research Institute (SwRI)
Research Engineer

____________________________________________________________________________
Notes.

HPPC Baseline Test
____________________________________________________________________________

"""

import os
from os import system
from IPython import get_ipython
import csv
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
rc('mathtext', default='regular')

system('cls')
get_ipython().magic('reset -sf')


#%% File and Location folder

f_dir = 'P:\\IR&D\\03-R6439\\Data\\PEC Data\\LG E65D\\pEIS\\'
f_source = f_dir
f_save ='P:\\IR&D\\03-R6439\\Data\\PEC Data Plots\\LG E65D\\pEIS\\'


#%% File to read

Cell_ID = '03204100162'   #154 and 204
file_name = 'pEIS_'+Cell_ID
file_title = 'pEIS Baseline'
file_label = file_name + " pEIS" 
f_plots = f_save + file_name + '\\'
# os.mkdir(f_plots)

#%% Cell Specs - LG E65D

file_label = "LG E65D (032041)"
Cap0 = 65   # Ah
Vmin = 2.5  # V
Vmax = 4.2  # V
E0 = 234     # Wh (2.4*30)
Imax = 130     # A
Imin = 0.75*Imax   # A

#%% Load data DAQ Header

Data0 = pd.read_csv(f_source+file_name+'.csv', header  = None, nrows = 28, names = range(3))

#%% Load data DAQ

Data = pd.read_csv(f_source+file_name+'.csv', header = 29-1)

Step = Data['Step'].values
Cycle = Data['Cycle'].values
TT = Data['Total Time (Seconds)'].values
ST = Data['Step Time (Seconds)'].values
Voltage = Data['Voltage (V)'].values
Current = Data['Current (A)'].values
CCap = Data['Charge Capacity (mAh)'].values
DCap = Data['Discharge Capacity (mAh)'].values
CEne = Data['Charge Energy (mWh)'].values
DEne = Data['Discharge Energy (mWh)'].values
TPos = Data['K1 (°C)'].values
# TNeg = Data['K2 (°C)'].values
# TMid = Data['K3 (°C)'].values
Res = Data['DC Internal Resistance (mOhm)'].values
Temp_Time = TT[~np.isnan(Data['K1 (°C)'])][:]

Power = Voltage*Current

del Data

#%% Plot Voltage Full Test

f1 = plt.figure(figsize=(13,9))
title_font = 20
ticks_font = 16

ax1 = f1.add_subplot(1,1,1)
lns1 = ax1.plot(TT/60, Voltage, color='blue', linewidth = 2, label = 'Voltage', zorder = 2)

ax1.set_ylabel('Voltage [V]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font)
ax1.set_ylim([2.0, 4.8])
ticks = np.arange(2.0, 4.8001, 0.4)
ax1.set_yticks(ticks)

ax2 = ax1.twinx()

lns2 = ax2.plot(TT/60, Current, color='red', linewidth = 2, label = 'Current', zorder = 1)
ax2.set_ylabel('Current [A]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax2.yaxis.set_tick_params(labelsize = ticks_font)
ax2.set_ylim([-160, 400])
ticks = np.arange(-160, 400.01, 80)
ax2.set_yticks(ticks)


ax1.set_xlabel('Time [min]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([0, 500])
ticks = np.arange(0, 500.01, 50)
ax1.set_xticks(ticks)

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = title_font, title = file_name, title_fontsize = title_font, loc = 'best')

plt.show()
f1.savefig(f_plots+file_name+'_Voltage and Current vs Time, Full Test.png', bbox_inches='tight', dpi=200)
plt.close()

del f1, title_font, ticks_font, ax1, lns1, lns2, lns, labs

#%% Plot Voltage pEIS Test

f1 = plt.figure(figsize=(13,9))
title_font = 20
ticks_font = 16

ax1 = f1.add_subplot(1,1,1)
lns1 = ax1.plot(TT/60, Voltage, color='blue', linewidth = 2, label = 'Voltage', zorder = 2)

ax1.set_ylabel('Voltage [V]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font)
ax1.set_ylim([2.0, 4.4])
ticks = np.arange(2.0, 4.4001, 0.4)
ax1.set_yticks(ticks)

ax2 = ax1.twinx()

lns2 = ax2.plot(TT/60, Current, color='red', linewidth = 2, label = 'Current', zorder = 1)
ax2.set_ylabel('Current [A]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax2.yaxis.set_tick_params(labelsize = ticks_font)
ax2.set_ylim([-80, 160])
ticks = np.arange(-80, 160.01, 40)
ax2.set_yticks(ticks)


ax1.set_xlabel('Time [min]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([360, 460])
ticks = np.arange(360, 460.01, 10)
ax1.set_xticks(ticks)

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = title_font, title = file_name, title_fontsize = title_font, loc = 'best')

plt.show()
f1.savefig(f_plots+file_name+'_Voltage and Current vs Time, pEIS Test.png', bbox_inches='tight', dpi=200)
plt.close()

del f1, title_font, ticks_font, ax1, lns1, lns2, lns, labs

#%% Plot Voltage pEIS Test Close up

f1 = plt.figure(figsize=(13,9))
title_font = 20
ticks_font = 16

ax1 = f1.add_subplot(1,1,1)
lns1 = ax1.plot(TT/60, Voltage, color='blue', linewidth = 2, label = 'Voltage', zorder = 2)

ax1.set_ylabel('Voltage [V]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font)
# ax1.set_ylim([2.0, 4.8])
# ticks = np.arange(2.0, 4.8001, 0.4)
ax1.set_ylim([3.2, 3.9])
ticks = np.arange(3.2, 3.9001, 0.1)
ax1.set_yticks(ticks)

ax2 = ax1.twinx()

lns2 = ax2.plot(TT/60, Current, color='red', linewidth = 2, label = 'Current', zorder = 1)
ax2.set_ylabel('Current [A]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax2.yaxis.set_tick_params(labelsize = ticks_font)
ax2.set_ylim([-80, 200])
ticks = np.arange(-80, 200.01, 40)
ax2.set_yticks(ticks)

ax1.set_xlabel('Time [min]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([399.5, 401.5])
ticks = np.arange(399.5, 401.5001, 0.5)
ax1.set_xticks(ticks)

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = title_font, title = file_name, title_fontsize = title_font, loc = 'best')

plt.show()
f1.savefig(f_plots+file_name+'_Voltage and Current vs Time, pEIS Test Close up 4.png', bbox_inches='tight', dpi=200)
plt.close()

del f1, title_font, ticks_font, ax1, lns1, lns2, lns, labs

#%% Plot Temperatures vs Time

f1 = plt.figure(figsize=(13,9))
title_font = 20
ticks_font = 16
Tlabel = ['TC1 Positive Terminal', 'TC2 Negative Terminal','TC3 Center']

ax1 = f1.add_subplot(1,1,1)
lns1 = ax1.plot(Temp_Time/60, TPos[~np.isnan(TPos)], color = 'deeppink', linewidth = 2, label = Tlabel[0], zorder = 4)
# lns2 = ax1.plot(Temp_Time/60, TNeg[~np.isnan(TNeg)], color= 'purple', linewidth = 2, label = Tlabel[1], zorder = 3)
# lns3 = ax1.plot(Temp_Time/60, TMid[~np.isnan(TMid)], color='limegreen', linewidth = 2, label = Tlabel[2], zorder = 2)
ax1.set_ylabel('Temperature [$^oC$]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font)
ax1.set_ylim([24, 40])
ticks = np.arange(24, 40.01, 2)
ax1.set_yticks(ticks)

ax2 = ax1.twinx()

lns4 = ax2.plot(TT/60, Voltage, color='blue', linewidth = 2, label = 'Voltage', zorder = 1)
ax2.set_ylabel('Voltage [V]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax2.yaxis.set_tick_params(labelsize = ticks_font)
ax2.set_ylim([2.0, 4.8])
ticks = np.arange(2.0, 4.8001, 0.4)
ax2.set_yticks(ticks)

ax1.set_xlabel('Time [min]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([0, 500])
ticks = np.arange(0, 500.01, 50)
ax1.set_xticks(ticks)

lns = lns1 +  lns4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = title_font, title = file_name, title_fontsize = title_font, loc = 'best')

plt.show()
f1.savefig(f_plots+file_name+' Temperature and Voltage vs Time.png', bbox_inches='tight', dpi=200)
plt.close(f1)

del f1, title_font, ticks_font, ax1, lns1, lns, labs


#%% Plot Temperatures vs Time

f1 = plt.figure(figsize=(13,9))
title_font = 20
ticks_font = 16

Tlabel = ['TC1 Positive Terminal', 'TC2 Negative Terminal','TC3 Center']

disc_step = 11
ax1 = f1.add_subplot(1,1,1)
lns1 = ax1.plot(DCap[Step == disc_step]/1000, Voltage[Step == disc_step], color = 'mediumblue', linewidth = 2, label = 'Voltage', zorder = 4)
ax1.set_ylabel('Voltage [V]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font)
ax1.set_ylim([2.0, 4.8])
ticks = np.arange(2.0, 4.8001, 0.4)
ax1.set_yticks(ticks)

ax2 = ax1.twinx()

lns2 = ax2.plot(DCap[Step ==disc_step][~np.isnan(TPos[Step == disc_step])]/1000, TPos[Step == disc_step][~np.isnan(TPos[Step == disc_step])], color= 'deeppink', linewidth = 2, label = Tlabel[0], zorder = 3)
# lns3 = ax2.plot(DCap[Step ==disc_step][~np.isnan(TNeg[Step == disc_step])]/1000, TNeg[Step == disc_step][~np.isnan(TNeg[Step == disc_step])], color= 'purple', linewidth = 2, label = Tlabel[1], zorder = 2)
# lns4 = ax2.plot(DCap[Step ==disc_step][~np.isnan(TMid[Step == disc_step])]/1000, TMid[Step == disc_step][~np.isnan(TMid[Step == disc_step])], color= 'limegreen', linewidth = 2, label = Tlabel[2], zorder = 1)

ax2.set_ylabel('Temperature [$^oC$]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax2.yaxis.set_tick_params(labelsize = ticks_font)
ax2.set_ylim([20, 36])
ticks = np.arange(20, 36.01, 2)
ax2.set_yticks(ticks)

ax1.set_xlabel('Discharge Capacity [Ah]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([0, 70])
ticks = np.arange(0, 70.01, 10)
ax1.set_xticks(ticks)

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = title_font, title = file_name, title_fontsize = title_font, loc = 'best')

plt.show()
f1.savefig(f_plots+file_name+' Voltage vs Capacity.png', bbox_inches='tight', dpi=200)
plt.close(f1)

del f1, title_font, ticks_font, ax1, lns1, lns2, lns, labs


#%% Pseudo EIS Analysis

# 0  400    Rest 60 s
# 4  400    Initial CC Discharge at 1C to 1.5 V
# 5  400    Rest for 1 hour
# 7  400    CC Charge to 2.9 V
# 8  400    Rest for 1 hour
# 10 400    CC Discharge at 1C to 1.5 V       32000 mAh
# 13 400    Rest for 1 hour

# 17 401    CC Charge at 1C until C/100 capacity (320mAh)
# 18 401    Rest for 3 seconds
# 20 401    CC Discharge at 1C to C/100 for 3 seconds
# 17 402    CC Charge at 1C until C/100 capacity (320mAh)
# 18 402    Rest for 3 seconds
# 20 402    CC Discharge at 1C to C/100 for 3 seconds
# ...
# 17 517    CC Charge at 1C until C/100 capacity (320mAh)
# 18 517    Rest for 3 seconds
# 20 517    CC Discharge at 1C to C/100 for 3 seconds
# 17 518    CC Charge at 1C until C/100 capacity (320mAh)
# 18 518    Rest for 3 seconds
# 20 518    CC Discharge at 1C to C/100 for 3 seconds

# 17 519    CC Charge at 1C until C/100 capacity (320mAh) > 2.9 V
# 23        Rest for 3 seconds
# 24        Final Rest


#%% pEIS Analysis

char_step = 7
disc_step = 11
C_Pulse  = 19  # CC Charge at 1C until C/100 capacity (320mAh)
C_Rest_D = 20  # Rest for 3 seconds
D_Pulse  = 22  # CC Discharge at 1C for 3 seconds 

Cyc_min = int(min(Cycle[Cycle>0]))
Cyc_max = int(max(Cycle[Cycle>0]))
n_pulses = Cyc_max - Cyc_min + 1

print('Number of pulses = '+str(n_pulses-2)+' from '+str(Cyc_min+1)+' to '+str(Cyc_max-1))

Cell_Cap = float(max(DCap[Step == disc_step]))/1000    # Ah
# Cell_Cap = float(max(CCap[Step == char_step]))/1000    # Ah
# Cell_Cap = Cap0

OCV          = np.zeros(n_pulses)
R_Discharge  = np.zeros(n_pulses)
R_Charge     = np.zeros(n_pulses)
tP_Discharge = np.zeros(n_pulses)
tP_Charge    = np.zeros(n_pulses)

SOC_C        = np.zeros(n_pulses)
SOC_D        = np.zeros(n_pulses)
SOC_A        = np.zeros(n_pulses)
Cap_C        = np.zeros(n_pulses)
Cap_D        = np.zeros(n_pulses)
Cap_A        = np.zeros(n_pulses)


aux = 0

for ip in range (1,n_pulses-2):
    Cyc_ind = Cyc_min + ip + 1
    Cap_C[ip] = int(max(CCap[(Cycle == Cyc_ind) & (Step == C_Pulse)]))/1000
    Cap_D[ip] = int(max(DCap[(Cycle == Cyc_ind) & (Step == D_Pulse)]))/1000
    aux = Cap_D[ip-1]+aux
    Cap_A[ip] = Cap_C[ip] - aux
    
    SOC_C[ip] = 100*Cap_C[ip]/Cell_Cap
    SOC_D[ip] = 100*Cap_D[ip]/Cell_Cap
    SOC_A[ip] = 100*Cap_A[ip]/Cell_Cap
    
    OCV[ip] = float(Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1])

    V_0 = float(Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1])
    I_0 = float(Current[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1])   
    t_0 = float(TT[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1])  
    
    V_1 = float(Voltage[(Step == D_Pulse) & (Cycle == Cyc_ind)][-1])
    I_1 = float(Current[(Step == D_Pulse) & (Cycle == Cyc_ind)][-1])   
    t_1 = float(TT[(Step == D_Pulse) & (Cycle == Cyc_ind)][-1])   

    R_Discharge [ip] = 1000*np.abs((V_1-V_0)/(np.abs(I_1)-np.abs(I_0)))
    tP_Discharge[ip] = np.abs(t_1 - t_0) 

    # V_2 = float(Voltage[(Step == C_Pulse) & (Cycle == Cyc_ind)][0])
    # I_2 = float(Current[(Step == C_Pulse) & (Cycle == Cyc_ind)][0])   
    # t_2 = float(TT[(Step == C_Pulse) & (Cycle == Cyc_ind)][0])   

    V_2 = float(Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1])
    I_2 = float(Current[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1])   
    t_2 = float(TT[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1])   

    V_3 = float(Voltage[(Step == C_Pulse) & (Cycle == Cyc_ind)][-1])
    I_3 = float(Current[(Step == C_Pulse) & (Cycle == Cyc_ind)][-1])   
    t_3 = float(TT[(Step == C_Pulse) & (Cycle == Cyc_ind)][-1])  

    R_Charge [ip] = 1000*np.abs((V_3-V_2)/(np.abs(I_3)-np.abs(I_2)))
    tP_Charge[ip] = np.abs(t_3 - t_2) 



#%% Plot HPPC OCV

f1 = plt.figure(figsize=(12,8))
title_font = 16
ticks_font = 14
legend_font = 14
thickness = 1.5
n_Plots = 1

ax1 = f1.add_subplot(1,1,1)
lns1  = ax1.plot(SOC_A[OCV > 0], OCV[OCV > 0], color='red',marker='o', linewidth = thickness, label = 'OCV', zorder = 1)

plt.axhline(y = 24, color = 'silver', linewidth = 1, linestyle = '--')
plt.axhline(y = 43.2, color = 'silver', linewidth = 1, linestyle = '--')                              

ax1.set_ylabel('Voltage [V]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font)
ax1.set_ylim([2.0, 4.8])
ticks = np.arange(2.0, 4.8001, 0.4)
ax1.set_yticks(ticks)

ax1.set_xlabel('SOC [%]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([-1, 120])
ticks = np.arange(0, 120+0.001, 10)
ax1.set_xticks(ticks)

lns = []
for i in range(n_Plots):
    exec('lns += lns'+str(i+1))

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = legend_font,title=file_label,title_fontsize = title_font, loc = 'best', ncol = 1)

plt.show()
f1.savefig(f_plots+file_name+'_OCV.png', bbox_inches='tight', dpi=200)

del f1, title_font, ticks_font, legend_font, thickness, ax1, lns, labs, n_Plots


#%% Plot HPPC Discharge/Charge Resistance

f1 = plt.figure(figsize=(12,8))
title_font = 16
ticks_font = 14
legend_font = 16
thickness = 1.5
n_Plots = 2

ax1 = f1.add_subplot(1,1,1)
# lns1  = ax1.plot(SOC_A[R_Discharge > 0], R_Discharge[R_Discharge > 0], color='red', marker='v',  markersize = 8, linewidth = thickness, label = 'Discharge', zorder = 1)
lns2  = ax1.plot(SOC_A[R_Charge > 0], R_Charge[R_Charge > 0], color='red', marker='^',  markersize = 8, linewidth = thickness, label = 'Impedance Ztr Charge', zorder = 1)


ax1.set_ylabel('Transition Impedance (ZTR) [m\u03a9]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font, color='red')
ax1.tick_params(axis='y', colors='red')
ax1.set_ylim([0, 3.2])
ticks = np.arange(0, 3.2001, 0.4)
ax1.set_yticks(ticks)

ax2 = ax1.twinx()

lns1  = ax1.plot(SOC_A[R_Discharge > 0], R_Discharge[R_Discharge > 0], color='blue', marker='v',  markersize = 8, linewidth = thickness, label = 'Resistance R0* Discharge', zorder = 1)
# lns2  = ax1.plot(SOC_A[R_Charge > 0], R_Charge[R_Charge > 0], color='blue', marker='^',  markersize = 8, linewidth = thickness, label = 'Charge', zorder = 1)

ax2.set_ylabel('Resistance (R0*) [m\u03a9]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax2.yaxis.set_tick_params(labelsize = ticks_font, color='blue')
# ax2.tick_params(axis='y', colors='red')
ax2.set_ylim([0, 3.2])
ticks = np.arange(0, 3.2001, 0.4)
ax2.set_yticks(ticks)

ax1.set_xlabel('SOC [%]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([-1, 110])
ticks = np.arange(0, 110+0.001, 10)
ax1.set_xticks(ticks)

lns = []
for i in range(n_Plots):
    exec('lns += lns'+str(i+1))

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = legend_font,title=file_label,title_fontsize = title_font, loc = 'upper left', ncol = 1)

plt.show()
f1.savefig(f_plots+file_name+'_Resistance2.png', bbox_inches='tight', dpi=200)

del f1, title_font, ticks_font, legend_font, thickness, ax1, lns, labs, n_Plots
    
 
#%% Plot HPPC Charge Resistance

f1 = plt.figure(figsize=(12,8))
title_font = 16
ticks_font = 14
legend_font = 16
thickness = 1.5
n_Plots = 1

ax1 = f1.add_subplot(1,1,1)
lns1  = ax1.plot(SOC_A[R_Charge > 0], R_Charge[R_Charge > 0], color='red', marker='^',  markersize = 8, linewidth = thickness, label = 'Charge', zorder = 1)


ax1.set_ylabel('Transition Impedance (ZTR) [m\u03a9]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.yaxis.set_tick_params(labelsize = ticks_font, color='red')
# ax1.tick_params(axis='y', colors='red')
ax1.set_ylim([0, 3.2])
ticks = np.arange(0, 3.2001, 0.4)
ax1.set_yticks(ticks)


ax1.set_xlabel('SOC [%]', fontsize = title_font, fontweight = 'bold', labelpad = 15)
ax1.xaxis.set_tick_params(labelsize = ticks_font)
ax1.set_xlim([-1, 110])
ticks = np.arange(0, 110+0.001, 10)
ax1.set_xticks(ticks)

lns = []
for i in range(n_Plots):
    exec('lns += lns'+str(i+1))

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize = legend_font,title=file_label,title_fontsize = title_font, loc = 'best', ncol = 1)

plt.show()
f1.savefig(f_plots+file_name+'_Impedance.png', bbox_inches='tight', dpi=200)

del f1, title_font, ticks_font, legend_font, thickness, ax1, lns, labs, n_Plots


#%% Report Values

with open(f_save+file_name+'_Report'+'.csv', 'w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow([
    'SwRI - PDIR6439',
    'Start Date-Time',
    'End Date-Time',
    'Cell ID',
    'Data Souce', 
    'Test Title',
    'Discharge Capacity [Ah] = '])


    writer.writerow([
   'Author: Daniel Juarez Robles',
    Data0[1].values[23], 
    Data0[1].values[25], 
    Cell_ID,
    file_name+".csv", 
    file_title,
    str(Cell_Cap)])


    writer.writerow([' '])    
    writer.writerow (['SOC_C[%]','SOC_D[%]','SOC_A[%]','DOD_C[%]','OCV[V]','Charge_Resistance[Ohms]','Discharge_Resistance[Ohms]','Charge_Power[W]','Discharge_Power[W]','Charge_Pulse_Width[s]','Discharge_Pulse_Width[s]'])        
    
    for i in range(len(SOC_A)-1):
        writer.writerow ([str(SOC_C[i+1]),str(SOC_D[i+1]),str(SOC_A[i+1]),str(100-SOC_A[i+1]),str(OCV[i+1]),str(R_Charge[i]),str(R_Discharge[i]),str(tP_Charge[i]),str(tP_Discharge[i])])   
    
    



