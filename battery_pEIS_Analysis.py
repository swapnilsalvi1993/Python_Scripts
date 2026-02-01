# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 11:11:36 2026

@author: ssalvi
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
import tkinter as tk
from tkinter import filedialog
rc('mathtext', default='regular')

system('cls')
get_ipython().magic('reset -sf')


#%% GUI for File Selection

# Create a root window and hide it
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

# Open file dialog to select CSV file
print("Please select the pEIS CSV file...")
csv_file_path = filedialog.askopenfilename(
    title="Select pEIS CSV Data File",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    parent=root
)

# Check if file was selected
if not csv_file_path:
    print("No file selected. Exiting...")
    root.destroy()
    exit()

print(f"Selected file: {csv_file_path}")

# Extract directory and filename information
f_dir = os.path.dirname(csv_file_path) + '\\'
f_source = f_dir
file_name_full = os.path.basename(csv_file_path)
file_name = os.path.splitext(file_name_full)[0]  # Remove .csv extension

# Create save directory in the same location as the CSV
f_save = f_dir
f_plots = f_save + file_name + '_plots\\'

# Create plots directory if it doesn't exist
if not os.path.exists(f_plots):
    os.makedirs(f_plots)
    print(f"Created output directory: {f_plots}")
else:
    print(f"Output directory exists: {f_plots}")

root.destroy()

print(f"\nFile Directory: {f_dir}")
print(f"File Name: {file_name}")
print(f"Plots will be saved to: {f_plots}")
print("-" * 80)


#%% File to read

# Extract Cell_ID from filename - look for 10-digit number (with optional leading zero)
import re

Cell_ID = 'Unknown'

# Try multiple patterns to extract Cell ID
# Pattern 1: Look for 10-11 digit numbers in the filename
pattern1 = re.search(r'\b0?(\d{10})\b', file_name)
if pattern1:
    Cell_ID = pattern1.group(1)
    print(f"Cell ID extracted (Pattern 1 - 10 digits): {Cell_ID}")
else:
    # Pattern 2: If filename starts with 'pEIS_', extract everything after it
    if file_name.startswith('pEIS_'):
        Cell_ID = file_name.replace('pEIS_', '')
        # Clean up Cell_ID - extract just the numeric part if there are other characters
        numeric_match = re.search(r'0?(\d{10,})', Cell_ID)
        if numeric_match:
            Cell_ID = numeric_match.group(1)
            print(f"Cell ID extracted (Pattern 2 - after pEIS_): {Cell_ID}")
    else:
        # Pattern 3: Look for any sequence of 10+ digits
        pattern3 = re.search(r'0?(\d{10,})', file_name)
        if pattern3:
            Cell_ID = pattern3.group(1)
            print(f"Cell ID extracted (Pattern 3 - any 10+ digits): {Cell_ID}")
        else:
            print(f"WARNING: Could not extract Cell ID from filename: {file_name}")
            Cell_ID = 'Unknown'

file_title = 'pEIS Baseline'
file_label = file_name + " pEIS" 

print(f"Cell ID: {Cell_ID}")

#%% Cell Specs - LG E65D

file_label = "Gotion "
Cap0 = 55   # Ah
Vmin = 2.5  # V
Vmax = 4.2  # V
E0 = 203.5     # Wh (2.4*30)
# Imax = 130     # A
# Imin = 0.75*Imax   # A

#%% Load data DAQ Header

Data0 = pd.read_csv(csv_file_path, header=None, nrows=28, names=range(3))
print("\nLoading data header... Done")

#%% Load data DAQ

Data = pd.read_csv(csv_file_path, header=29-1)
print("Loading main data... Done")

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

print(f"Data points loaded: {len(TT)}")
print(f"Test duration: {TT[-1]/3600:.2f} hours")
print("-" * 80)

#%% Plot Voltage Full Test

print("\nGenerating Plot 1: Voltage and Current vs Time (Full Test)...")
f1 = plt.figure(figsize=(13,9))
title_font = 20
ticks_font = 16

ax1 = f1.add_subplot(1,1,1)
lns1 = ax1.plot(TT/60, Voltage, color='blue', linewidth=2, label='Voltage', zorder=2)

ax1.set_ylabel('Voltage [V]', fontsize=title_font, fontweight='bold', labelpad=15)
ax1.grid(color='gray', linestyle='--', linewidth=0.5)
ax1.yaxis.set_tick_params(labelsize=ticks_font)
ax1.set_ylim([2.0, 4.8])
ticks = np.arange(2.0, 4.8001, 0.4)
ax1.set_yticks(ticks)

ax2 = ax1.twinx()

lns2 = ax2.plot(TT/60, Current, color='red', linewidth=2, label='Current', zorder=1)
ax2.set_ylabel('Current [A]', fontsize=title_font, fontweight='bold', labelpad=15)
ax2.grid(color='gray', linestyle='--', linewidth=0.5)
ax2.yaxis.set_tick_params(labelsize=ticks_font)
ax2.set_ylim([-160, 400])
ticks = np.arange(-160, 400.01, 80)
ax2.set_yticks(ticks)


ax1.set_xlabel('Time [min]', fontsize=title_font, fontweight='bold', labelpad=15)
ax1.xaxis.set_tick_params(labelsize=ticks_font)
ax1.set_xlim([0, 1200])
ticks = np.arange(0, 1200.01, 100)
ax1.set_xticks(ticks)

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=title_font, title=file_name, title_fontsize=title_font, loc='best')

plt.show()
f1.savefig(f_plots+file_name+'_Voltage and Current vs Time, Full Test.png', bbox_inches='tight', dpi=200)
plt.close()

del f1, title_font, ticks_font, ax1, lns1, lns2, lns, labs

print("Saved: Voltage and Current vs Time, Full Test.png")

#%% pEIS Analysis

print("\n" + "="*80)
print("Starting pEIS Analysis...")
print("="*80)

# Automatically identify discharge and charge steps for capacity calculation
print("\nIdentifying capacity calculation steps...")

# Find the discharge step with maximum discharge capacity (typically the baseline discharge)
unique_steps = np.unique(Step[Step > 0])
max_dcap_per_step = []
max_ccap_per_step = []

for step_num in unique_steps:
    step_mask = Step == step_num
    dcap_in_step = DCap[step_mask]
    ccap_in_step = CCap[step_mask]
    
    max_dcap = np.max(dcap_in_step[~np.isnan(dcap_in_step)]) if len(dcap_in_step[~np.isnan(dcap_in_step)]) > 0 else 0
    max_ccap = np.max(ccap_in_step[~np.isnan(ccap_in_step)]) if len(ccap_in_step[~np.isnan(ccap_in_step)]) > 0 else 0
    
    max_dcap_per_step.append((step_num, max_dcap))
    max_ccap_per_step.append((step_num, max_ccap))

# Sort by discharge capacity to find the step with maximum discharge
max_dcap_per_step.sort(key=lambda x: x[1], reverse=True)
max_ccap_per_step.sort(key=lambda x: x[1], reverse=True)

# The step with highest discharge capacity is likely the baseline discharge
if len(max_dcap_per_step) > 0 and max_dcap_per_step[0][1] > 1000:  # At least 1 Ah
    disc_step = int(max_dcap_per_step[0][0])
    print(f"Found disc_step (Baseline Discharge): Step {disc_step}")
    print(f"  Max Discharge Capacity: {max_dcap_per_step[0][1]/1000:.2f} Ah")
else:
    print("WARNING: Could not auto-identify disc_step. Using default value 11.")
    disc_step = 11

# The step with highest charge capacity is likely the baseline charge
if len(max_ccap_per_step) > 0 and max_ccap_per_step[0][1] > 1000:  # At least 1 Ah
    char_step = int(max_ccap_per_step[0][0])
    print(f"Found char_step (Baseline Charge): Step {char_step}")
    print(f"  Max Charge Capacity: {max_ccap_per_step[0][1]/1000:.2f} Ah")
else:
    print("WARNING: Could not auto-identify char_step. Using default value 7.")
    char_step = 7

print(f"\nCapacity Calculation Steps:")
print(f"  char_step = {char_step}")
print(f"  disc_step = {disc_step}")
print("-" * 80)

# Automatically identify step numbers based on characteristics
print("\nIdentifying pEIS step numbers...")

Cyc_min = int(min(Cycle[Cycle>0]))
Cyc_max = int(max(Cycle[Cycle>0]))

# Focus on pulse cycles (typically cycle 401 onwards based on your protocol)
# Use dynamic range based on total cycles
pulse_start = Cyc_min + 1
pulse_end = Cyc_max

pulse_cycles = Cycle[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
pulse_steps = Step[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
pulse_current = Current[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
pulse_ST = ST[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
pulse_CCap = CCap[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
pulse_DCap = DCap[(Cycle >= pulse_start) & (Cycle <= pulse_end)]

# Get unique steps in the pulse region
unique_pulse_steps = np.unique(pulse_steps)

print(f"Unique steps in pulse region (Cycles {pulse_start}-{pulse_end}): {unique_pulse_steps}")

# Identify C_Pulse: CC Charge at 1C (positive current, charge capacity increasing)
# This step should have positive current and significant charge capacity accumulation
C_Pulse = None
C_Rest_D = None  # Initialize here to avoid NameError
D_Pulse = None    # Initialize here as well

# First attempt: Look for charge step with capacity accumulation
for step_num in unique_pulse_steps:
    step_mask = pulse_steps == step_num
    step_current = pulse_current[step_mask]
    step_ccap = pulse_CCap[step_mask]
    
    # Check if it's a charge step (positive current) with capacity accumulation
    if len(step_current) > 0 and len(step_ccap) > 0:
        valid_current = step_current[~np.isnan(step_current)]
        valid_ccap = step_ccap[~np.isnan(step_ccap)]
        
        if len(valid_current) > 0 and len(valid_ccap) > 0:
            avg_current = np.mean(valid_current)
            ccap_range = np.max(valid_ccap) - np.min(valid_ccap)
            
            # Charge pulse: positive current and charge capacity increases
            # Relaxed thresholds for different cell capacities
            if avg_current > 5 and ccap_range > 50 and ccap_range < 2000:
                C_Pulse = int(step_num)
                print(f"Found C_Pulse (CC Charge at 1C until C/100): Step {C_Pulse}")
                print(f"  Avg Current: {avg_current:.2f} A, Charge Cap Range: {ccap_range:.2f} mAh")
                break

# Second attempt: Look for charge step with highest average positive current
if C_Pulse is None:
    print("Attempting to identify C_Pulse based on highest positive current...")
    
    charge_candidates = []
    for step_num in unique_pulse_steps:
        step_mask = pulse_steps == step_num
        step_current = pulse_current[step_mask]
        step_st = pulse_ST[step_mask]
        
        if len(step_current) > 0 and len(step_st) > 0:
            valid_current = step_current[~np.isnan(step_current)]
            valid_st = step_st[~np.isnan(step_st)]
            
            if len(valid_current) > 0 and len(valid_st) > 0:
                avg_current = np.mean(valid_current)
                max_step_time = np.max(valid_st)
                
                # Charge step: positive current and longer duration (not a short pulse)
                if avg_current > 5 and max_step_time > 10:
                    charge_candidates.append((step_num, avg_current))
    
    if charge_candidates:
        # Sort by average current and pick the highest
        charge_candidates.sort(key=lambda x: x[1], reverse=True)
        C_Pulse = int(charge_candidates[0][0])
        print(f"Found C_Pulse (CC Charge at 1C until C/100): Step {C_Pulse}")
        print(f"  Avg Current: {charge_candidates[0][1]:.2f} A (highest positive current)")

if C_Pulse is None:
    print("WARNING: Could not auto-identify C_Pulse step. Using default value 19.")
    C_Pulse = 19

# Identify C_Rest_D: Rest for 3 seconds (near-zero current, short duration)
for step_num in unique_pulse_steps:
    if step_num <= C_Pulse:  # Rest should come after charge pulse
        continue
    
    step_mask = pulse_steps == step_num
    step_current = pulse_current[step_mask]
    step_st = pulse_ST[step_mask]
    
    if len(step_current) > 0 and len(step_st) > 0:
        valid_current = step_current[~np.isnan(step_current)]
        valid_st = step_st[~np.isnan(step_st)]
        
        if len(valid_current) > 0 and len(valid_st) > 0:
            avg_current = np.mean(np.abs(valid_current))
            max_step_time = np.max(valid_st)
            
            # Rest: near-zero current and duration around 3 seconds
            if avg_current < 5 and max_step_time > 2 and max_step_time < 10:
                C_Rest_D = int(step_num)
                print(f"Found C_Rest_D (Rest for 3 seconds): Step {C_Rest_D}")
                print(f"  Avg Current: {avg_current:.2f} A, Max Step Time: {max_step_time:.2f} s")
                break

# Third attempt for C_Pulse: Use sequence logic if C_Rest_D was found
if C_Pulse == 19 and C_Rest_D is not None:  # Only if using default value
    print("Attempting to refine C_Pulse based on sequence before C_Rest_D...")
    
    # Find the step that comes immediately before C_Rest_D in the pulse sequence
    for step_num in unique_pulse_steps:
        if step_num >= C_Rest_D:
            continue
        
        step_mask = pulse_steps == step_num
        step_current = pulse_current[step_mask]
        
        if len(step_current) > 0:
            valid_current = step_current[~np.isnan(step_current)]
            
            if len(valid_current) > 0:
                avg_current = np.mean(valid_current)
                
                # Charge pulse: positive current, comes before rest
                if avg_current > 5:
                    # Check if this step appears in the same cycles as C_Rest_D
                    step_cycles = pulse_cycles[step_mask]
                    rest_cycles = pulse_cycles[pulse_steps == C_Rest_D]
                    
                    # Check for overlap in cycles
                    if len(np.intersect1d(step_cycles, rest_cycles)) > 10:  # At least 10 common cycles
                        C_Pulse = int(step_num)
                        print(f"Refined C_Pulse (CC Charge at 1C until C/100): Step {C_Pulse}")
                        print(f"  Avg Current: {avg_current:.2f} A (identified by sequence)")
                        break

if C_Rest_D is None:
    print("WARNING: Could not auto-identify C_Rest_D step. Using default value 20.")
    C_Rest_D = 20

# Identify D_Pulse: CC Discharge at 1C for 3 seconds (negative current, short duration)
for step_num in unique_pulse_steps:
    if step_num <= C_Rest_D:  # Discharge should come after rest
        continue
    
    step_mask = pulse_steps == step_num
    step_current = pulse_current[step_mask]
    step_dcap = pulse_DCap[step_mask]
    step_st = pulse_ST[step_mask]
    
    if len(step_current) > 0 and len(step_dcap) > 0 and len(step_st) > 0:
        valid_current = step_current[~np.isnan(step_current)]
        valid_dcap = step_dcap[~np.isnan(step_dcap)]
        valid_st = step_st[~np.isnan(step_st)]
        
        if len(valid_current) > 0 and len(valid_dcap) > 0 and len(valid_st) > 0:
            avg_current = np.mean(valid_current)
            dcap_range = np.max(valid_dcap) - np.min(valid_dcap)
            max_step_time = np.max(valid_st)
            
            # Discharge pulse: negative current, short duration (around 3s), small discharge capacity
            if avg_current < -10 and max_step_time > 2 and max_step_time < 10 and dcap_range > 10:
                D_Pulse = int(step_num)
                print(f"Found D_Pulse (CC Discharge at 1C for 3 seconds): Step {D_Pulse}")
                print(f"  Avg Current: {avg_current:.2f} A, Max Step Time: {max_step_time:.2f} s, Discharge Cap Range: {dcap_range:.2f} mAh")
                break

if D_Pulse is None:
    print("WARNING: Could not auto-identify D_Pulse step. Using default value 22.")
    D_Pulse = 22

print(f"\nStep Identification Complete:")
print(f"  C_Pulse  = {C_Pulse}  (CC Charge at 1C until C/100 capacity)")
print(f"  C_Rest_D = {C_Rest_D}  (Rest for 3 seconds)")
print(f"  D_Pulse  = {D_Pulse}  (CC Discharge at 1C for 3 seconds)")
print("="*80)

# Identify C_Rest_D: Rest for 3 seconds (near-zero current, short duration)
C_Rest_D = None
for step_num in unique_pulse_steps:
    if step_num <= C_Pulse:  # Rest should come after charge pulse
        continue
    
    step_mask = pulse_steps == step_num
    step_current = pulse_current[step_mask]
    step_st = pulse_ST[step_mask]
    
    if len(step_current) > 0 and len(step_st) > 0:
        valid_current = step_current[~np.isnan(step_current)]
        valid_st = step_st[~np.isnan(step_st)]
        
        if len(valid_current) > 0 and len(valid_st) > 0:
            avg_current = np.mean(np.abs(valid_current))
            max_step_time = np.max(valid_st)
            
            # Rest: near-zero current and duration around 3 seconds
            if avg_current < 5 and max_step_time > 2 and max_step_time < 10:
                C_Rest_D = int(step_num)
                print(f"Found C_Rest_D (Rest for 3 seconds): Step {C_Rest_D}")
                print(f"  Avg Current: {avg_current:.2f} A, Max Step Time: {max_step_time:.2f} s")
                break

if C_Rest_D is None:
    print("WARNING: Could not auto-identify C_Rest_D step. Using default value 20.")
    C_Rest_D = 20

# Identify D_Pulse: CC Discharge at 1C for 3 seconds (negative current, short duration)
D_Pulse = None
for step_num in unique_pulse_steps:
    if step_num <= C_Rest_D:  # Discharge should come after rest
        continue
    
    step_mask = pulse_steps == step_num
    step_current = pulse_current[step_mask]
    step_dcap = pulse_DCap[step_mask]
    step_st = pulse_ST[step_mask]
    
    if len(step_current) > 0 and len(step_dcap) > 0 and len(step_st) > 0:
        valid_current = step_current[~np.isnan(step_current)]
        valid_dcap = step_dcap[~np.isnan(step_dcap)]
        valid_st = step_st[~np.isnan(step_st)]
        
        if len(valid_current) > 0 and len(valid_dcap) > 0 and len(valid_st) > 0:
            avg_current = np.mean(valid_current)
            dcap_range = np.max(valid_dcap) - np.min(valid_dcap)
            max_step_time = np.max(valid_st)
            
            # Discharge pulse: negative current, short duration (around 3s), small discharge capacity
            if avg_current < -10 and max_step_time > 2 and max_step_time < 10 and dcap_range > 10:
                D_Pulse = int(step_num)
                print(f"Found D_Pulse (CC Discharge at 1C for 3 seconds): Step {D_Pulse}")
                print(f"  Avg Current: {avg_current:.2f} A, Max Step Time: {max_step_time:.2f} s, Discharge Cap Range: {dcap_range:.2f} mAh")
                break

if D_Pulse is None:
    print("WARNING: Could not auto-identify D_Pulse step. Using default value 22.")
    D_Pulse = 22

print(f"\nStep Identification Complete:")
print(f"  C_Pulse  = {C_Pulse}  (CC Charge at 1C until C/100 capacity)")
print(f"  C_Rest_D = {C_Rest_D}  (Rest for 3 seconds)")
print(f"  D_Pulse  = {D_Pulse}  (CC Discharge at 1C for 3 seconds)")
print("="*80)

n_pulses = Cyc_max - Cyc_min + 1

print('Number of pulses = '+str(n_pulses-2)+' from '+str(Cyc_min+1)+' to '+str(Cyc_max-1))

# Calculate Cell Capacity with error handling
try:
    discharge_caps = DCap[Step == disc_step]
    if len(discharge_caps) > 0:
        valid_discharge_caps = discharge_caps[~np.isnan(discharge_caps)]
        if len(valid_discharge_caps) > 0:
            Cell_Cap = float(max(valid_discharge_caps))/1000    # Ah
        else:
            raise ValueError("No valid discharge capacity data found")
    else:
        raise ValueError("No data found for discharge step")
except (ValueError, Exception) as e:
    print(f"\nWARNING: Could not calculate Cell_Cap from disc_step {disc_step}: {e}")
    print("Trying alternative method using char_step...")
    
    try:
        charge_caps = CCap[Step == char_step]
        if len(charge_caps) > 0:
            valid_charge_caps = charge_caps[~np.isnan(charge_caps)]
            if len(valid_charge_caps) > 0:
                Cell_Cap = float(max(valid_charge_caps))/1000    # Ah
            else:
                raise ValueError("No valid charge capacity data found")
        else:
            raise ValueError("No data found for charge step")
    except (ValueError, Exception) as e2:
        print(f"WARNING: Could not calculate Cell_Cap from char_step {char_step}: {e2}")
        print(f"Using default capacity: Cap0 = {Cap0} Ah")
        Cell_Cap = Cap0

print(f'Cell Capacity = {Cell_Cap:.2f} Ah')

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
    
    try:
        # Safe extraction with error handling
        ccap_data = CCap[(Cycle == Cyc_ind) & (Step == C_Pulse)]
        dcap_data = DCap[(Cycle == Cyc_ind) & (Step == D_Pulse)]
        
        if len(ccap_data) > 0:
            Cap_C[ip] = int(max(ccap_data[~np.isnan(ccap_data)]))/1000 if len(ccap_data[~np.isnan(ccap_data)]) > 0 else 0
        else:
            Cap_C[ip] = 0
            
        if len(dcap_data) > 0:
            Cap_D[ip] = int(max(dcap_data[~np.isnan(dcap_data)]))/1000 if len(dcap_data[~np.isnan(dcap_data)]) > 0 else 0
        else:
            Cap_D[ip] = 0
        
        aux = Cap_D[ip-1]+aux
        Cap_A[ip] = Cap_C[ip] - aux
        
        SOC_C[ip] = 100*Cap_C[ip]/Cell_Cap
        SOC_D[ip] = 100*Cap_D[ip]/Cell_Cap
        SOC_A[ip] = 100*Cap_A[ip]/Cell_Cap
        
        # OCV extraction with safety check
        ocv_data = Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)]
        if len(ocv_data) > 0:
            OCV[ip] = float(ocv_data[-1])
        
        V_0 = float(Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1]) if len(Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)]) > 0 else 0
        I_0 = float(Current[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1]) if len(Current[(Step == C_Rest_D) & (Cycle == Cyc_ind)]) > 0 else 0
        t_0 = float(TT[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1]) if len(TT[(Step == C_Rest_D) & (Cycle == Cyc_ind)]) > 0 else 0
        
        V_1 = float(Voltage[(Step == D_Pulse) & (Cycle == Cyc_ind)][-1]) if len(Voltage[(Step == D_Pulse) & (Cycle == Cyc_ind)]) > 0 else 0
        I_1 = float(Current[(Step == D_Pulse) & (Cycle == Cyc_ind)][-1]) if len(Current[(Step == D_Pulse) & (Cycle == Cyc_ind)]) > 0 else 0
        t_1 = float(TT[(Step == D_Pulse) & (Cycle == Cyc_ind)][-1]) if len(TT[(Step == D_Pulse) & (Cycle == Cyc_ind)]) > 0 else 0

        R_Discharge [ip] = 1000*np.abs((V_1-V_0)/(np.abs(I_1)-np.abs(I_0))) if (np.abs(I_1)-np.abs(I_0)) != 0 else 0
        tP_Discharge[ip] = np.abs(t_1 - t_0) 

        V_2 = float(Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1]) if len(Voltage[(Step == C_Rest_D) & (Cycle == Cyc_ind)]) > 0 else 0
        I_2 = float(Current[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1]) if len(Current[(Step == C_Rest_D) & (Cycle == Cyc_ind)]) > 0 else 0
        t_2 = float(TT[(Step == C_Rest_D) & (Cycle == Cyc_ind)][-1]) if len(TT[(Step == C_Rest_D) & (Cycle == Cyc_ind)]) > 0 else 0

        V_3 = float(Voltage[(Step == C_Pulse) & (Cycle == Cyc_ind)][-1]) if len(Voltage[(Step == C_Pulse) & (Cycle == Cyc_ind)]) > 0 else 0
        I_3 = float(Current[(Step == C_Pulse) & (Cycle == Cyc_ind)][-1]) if len(Current[(Step == C_Pulse) & (Cycle == Cyc_ind)]) > 0 else 0
        t_3 = float(TT[(Step == C_Pulse) & (Cycle == Cyc_ind)][-1]) if len(TT[(Step == C_Pulse) & (Cycle == Cyc_ind)]) > 0 else 0

        R_Charge [ip] = 1000*np.abs((V_3-V_2)/(np.abs(I_3)-np.abs(I_2))) if (np.abs(I_3)-np.abs(I_2)) != 0 else 0
        tP_Charge[ip] = np.abs(t_3 - t_2) 
        
    except Exception as e:
        print(f"Warning: Error processing cycle {Cyc_ind}: {e}")
        continue

print(f'Analysis complete. Processed {n_pulses-2} pulses.')
print("="*80)

#%% Plot HPPC OCV

print("\nGenerating Plot 6: OCV vs SOC...")
f1 = plt.figure(figsize=(12,8))
title_font = 16
ticks_font = 14
legend_font = 14
thickness = 1.5
n_Plots = 1

ax1 = f1.add_subplot(1,1,1)
lns1  = ax1.plot(SOC_A[OCV > 0], OCV[OCV > 0], color='red',marker='o', linewidth=thickness, label='OCV', zorder=1)

plt.axhline(y=24, color='silver', linewidth=1, linestyle='--')
plt.axhline(y=43.2, color='silver', linewidth=1, linestyle='--')                              

ax1.set_ylabel('Voltage [V]', fontsize=title_font, fontweight='bold', labelpad=15)
ax1.grid(color='gray', linestyle='--', linewidth=0.5)
ax1.yaxis.set_tick_params(labelsize=ticks_font)
ax1.set_ylim([2.0, 4.8])
ticks = np.arange(2.0, 4.8001, 0.4)
ax1.set_yticks(ticks)

ax1.set_xlabel('SOC [%]', fontsize=title_font, fontweight='bold', labelpad=15)
ax1.xaxis.set_tick_params(labelsize=ticks_font)
ax1.set_xlim([-1, 120])
ticks = np.arange(0, 120+0.001, 10)
ax1.set_xticks(ticks)

lns = []
for i in range(n_Plots):
    exec('lns += lns'+str(i+1))

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=legend_font,title=file_label,title_fontsize=title_font, loc='best', ncol=1)

plt.show()
f1.savefig(f_plots+file_name+'_OCV.png', bbox_inches='tight', dpi=200)

del f1, title_font, ticks_font, legend_font, thickness, ax1, lns, labs, n_Plots

print("Saved: OCV.png")

#%% Plot HPPC Charge Resistance

print("Generating Plot 8: Charge Impedance vs SOC...")
f1 = plt.figure(figsize=(12,8))
title_font = 16
ticks_font = 14
legend_font = 16
thickness = 1.5
n_Plots = 1

ax1 = f1.add_subplot(1,1,1)

# Filter data and remove the last point
valid_mask = R_Charge > 0
SOC_A_valid = SOC_A[valid_mask]
R_Charge_valid = R_Charge[valid_mask]

# Remove the last data point
if len(SOC_A_valid) > 1:
    SOC_A_plot = SOC_A_valid[:-1]
    R_Charge_plot = R_Charge_valid[:-1]
else:
    SOC_A_plot = SOC_A_valid
    R_Charge_plot = R_Charge_valid

lns1  = ax1.plot(SOC_A_plot, R_Charge_plot, color='red', marker='^',  markersize=8, linewidth=thickness, label='Charge', zorder=1)


ax1.set_ylabel('Transition Impedance (ZTR) [m\u03a9]', fontsize=title_font, fontweight='bold', labelpad=15)
ax1.grid(color='gray', linestyle='--', linewidth=0.5)
ax1.yaxis.set_tick_params(labelsize=ticks_font, color='red')
# ax1.tick_params(axis='y', colors='red')
ax1.set_ylim([0, 3.2])
ticks = np.arange(0, 3.2001, 0.4)
ax1.set_yticks(ticks)


ax1.set_xlabel('SOC [%]', fontsize=title_font, fontweight='bold', labelpad=15)
ax1.xaxis.set_tick_params(labelsize=ticks_font)
ax1.set_xlim([-1, 110])
ticks = np.arange(0, 110+0.001, 10)
ax1.set_xticks(ticks)

lns = []
for i in range(n_Plots):
    exec('lns += lns'+str(i+1))

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=legend_font,title=file_label,title_fontsize=title_font, loc='best', ncol=1)

plt.show()
f1.savefig(f_plots+file_name+'_Impedance.png', bbox_inches='tight', dpi=200)

del f1, title_font, ticks_font, legend_font, thickness, ax1, lns, labs, n_Plots

print("Saved: Impedance.png")
