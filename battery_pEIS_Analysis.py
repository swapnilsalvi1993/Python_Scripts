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
from tkinter import filedialog, ttk, messagebox
import re
rc('mathtext', default='regular')

system('cls')
get_ipython().magic('reset -sf')


#%% GUI for File Selection and Parameter Input

class pEISConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("pEIS Analysis Configuration")
        self.root.geometry("650x700")
        self.root.resizable(False, False)
        
        # Variables to store user inputs
        self.csv_file_path = None
        self.cell_name = tk.StringVar(value="Unknown Cell")
        self.cell_id = tk.StringVar(value="Auto-detect")
        self.nominal_capacity = tk.StringVar(value="55")
        self.v_min = tk.StringVar(value="2.5")
        self.v_max = tk.StringVar(value="4.2")
        self.energy_capacity = tk.StringVar(value="203.5")
        
        # Step identification variables
        self.auto_detect_steps = tk.BooleanVar(value=True)
        self.char_step = tk.StringVar(value="7")
        self.disc_step = tk.StringVar(value="11")
        self.c_pulse_step = tk.StringVar(value="19")
        self.c_rest_step = tk.StringVar(value="20")
        self.d_pulse_step = tk.StringVar(value="22")
        
        # Plot options
        self.plot_full_test = tk.BooleanVar(value=True)
        self.plot_ocv = tk.BooleanVar(value=True)
        self.plot_impedance = tk.BooleanVar(value=True)
        self.generate_report = tk.BooleanVar(value=True)
        
        self.result = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main container frame
        container = ttk.Frame(self.root)
        container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create canvas
        canvas = tk.Canvas(container, highlightthickness=0)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Create scrollable frame
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrollable frame
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind canvas width to frame width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_frame, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', bind_mousewheel)
        canvas.bind('<Leave>', unbind_mousewheel)
        
        # Now use scrollable_frame for all widgets
        main_frame = scrollable_frame
        
        row = 0
        
        # Title
        title_label = ttk.Label(main_frame, text="pEIS Analysis Configuration", 
                                font=('Arial', 14, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=10, sticky='w')
        row += 1
        
        # File Selection Section
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="1. DATA FILE", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="CSV File:").grid(row=row, column=0, sticky='w', padx=5)
        self.file_label = ttk.Label(main_frame, text="No file selected", foreground="gray", wraplength=300)
        self.file_label.grid(row=row, column=1, sticky='w', padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_file).grid(row=row, column=2, padx=5, sticky='e')
        row += 1
        
        # Cell Information Section
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="2. CELL INFORMATION", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Cell Name/Type:").grid(row=row, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(main_frame, textvariable=self.cell_name, width=35).grid(row=row, column=1, columnspan=2, sticky='w', padx=5)
        row += 1
        
        ttk.Label(main_frame, text="Cell ID:").grid(row=row, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(main_frame, textvariable=self.cell_id, width=35).grid(row=row, column=1, columnspan=2, sticky='w', padx=5)
        row += 1
        ttk.Label(main_frame, text="(Leave as 'Auto-detect' to extract from filename)", 
                  font=('Arial', 8), foreground='gray').grid(row=row, column=1, columnspan=2, sticky='w', padx=5)
        row += 1
        
        # Cell Specifications Section
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="3. CELL SPECIFICATIONS", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Nominal Capacity (Ah):").grid(row=row, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(main_frame, textvariable=self.nominal_capacity, width=15).grid(row=row, column=1, sticky='w', padx=5)
        row += 1
        
        ttk.Label(main_frame, text="Min Voltage (V):").grid(row=row, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(main_frame, textvariable=self.v_min, width=15).grid(row=row, column=1, sticky='w', padx=5)
        row += 1
        
        ttk.Label(main_frame, text="Max Voltage (V):").grid(row=row, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(main_frame, textvariable=self.v_max, width=15).grid(row=row, column=1, sticky='w', padx=5)
        row += 1
        
        ttk.Label(main_frame, text="Energy Capacity (Wh):").grid(row=row, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(main_frame, textvariable=self.energy_capacity, width=15).grid(row=row, column=1, sticky='w', padx=5)
        row += 1
        
        # Step Identification Section
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="4. STEP IDENTIFICATION", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=5)
        row += 1
        
        ttk.Checkbutton(main_frame, text="Auto-detect step numbers from data", 
                        variable=self.auto_detect_steps, 
                        command=self.toggle_step_entries).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=3)
        row += 1
        
        # Manual step entry frame
        self.step_frame = ttk.Frame(main_frame)
        self.step_frame.grid(row=row, column=0, columnspan=3, sticky='w', padx=20, pady=5)
        
        step_row = 0
        ttk.Label(self.step_frame, text="Charge Step:").grid(row=step_row, column=0, sticky='w', padx=5, pady=2)
        self.char_step_entry = ttk.Entry(self.step_frame, textvariable=self.char_step, width=10)
        self.char_step_entry.grid(row=step_row, column=1, sticky='w', padx=5)
        ttk.Label(self.step_frame, text="(Baseline charge step)", font=('Arial', 8), foreground='gray').grid(row=step_row, column=2, sticky='w', padx=5)
        step_row += 1
        
        ttk.Label(self.step_frame, text="Discharge Step:").grid(row=step_row, column=0, sticky='w', padx=5, pady=2)
        self.disc_step_entry = ttk.Entry(self.step_frame, textvariable=self.disc_step, width=10)
        self.disc_step_entry.grid(row=step_row, column=1, sticky='w', padx=5)
        ttk.Label(self.step_frame, text="(Baseline discharge step)", font=('Arial', 8), foreground='gray').grid(row=step_row, column=2, sticky='w', padx=5)
        step_row += 1
        
        ttk.Label(self.step_frame, text="C_Pulse Step:").grid(row=step_row, column=0, sticky='w', padx=5, pady=2)
        self.c_pulse_entry = ttk.Entry(self.step_frame, textvariable=self.c_pulse_step, width=10)
        self.c_pulse_entry.grid(row=step_row, column=1, sticky='w', padx=5)
        ttk.Label(self.step_frame, text="(Charge pulse)", font=('Arial', 8), foreground='gray').grid(row=step_row, column=2, sticky='w', padx=5)
        step_row += 1
        
        ttk.Label(self.step_frame, text="C_Rest Step:").grid(row=step_row, column=0, sticky='w', padx=5, pady=2)
        self.c_rest_entry = ttk.Entry(self.step_frame, textvariable=self.c_rest_step, width=10)
        self.c_rest_entry.grid(row=step_row, column=1, sticky='w', padx=5)
        ttk.Label(self.step_frame, text="(Rest after charge)", font=('Arial', 8), foreground='gray').grid(row=step_row, column=2, sticky='w', padx=5)
        step_row += 1
        
        ttk.Label(self.step_frame, text="D_Pulse Step:").grid(row=step_row, column=0, sticky='w', padx=5, pady=2)
        self.d_pulse_entry = ttk.Entry(self.step_frame, textvariable=self.d_pulse_step, width=10)
        self.d_pulse_entry.grid(row=step_row, column=1, sticky='w', padx=5)
        ttk.Label(self.step_frame, text="(Discharge pulse)", font=('Arial', 8), foreground='gray').grid(row=step_row, column=2, sticky='w', padx=5)
        
        row += 1
        
        # Plot Options Section
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(main_frame, text="5. OUTPUT OPTIONS", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=5)
        row += 1
        
        ttk.Checkbutton(main_frame, text="Generate Full Test Plot (Voltage & Current vs Time)", 
                        variable=self.plot_full_test).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        row += 1
        
        ttk.Checkbutton(main_frame, text="Generate OCV vs SOC Plot", 
                        variable=self.plot_ocv).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        row += 1
        
        ttk.Checkbutton(main_frame, text="Generate Impedance vs SOC Plot", 
                        variable=self.plot_impedance).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        row += 1
        
        ttk.Checkbutton(main_frame, text="Generate CSV Report", 
                        variable=self.generate_report).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        row += 1
        
        # Add some spacing before buttons
        ttk.Label(main_frame, text="").grid(row=row, column=0, pady=10)
        row += 1
        
        # Buttons - Fixed at bottom (outside scrollable area)
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=15)
        
        ttk.Button(button_frame, text="Start Analysis", command=self.start_analysis, 
                   width=15).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.cancel, 
                   width=15).grid(row=0, column=1, padx=10)
        
        # Add padding at bottom
        ttk.Label(main_frame, text="").grid(row=row+1, column=0, pady=10)
        
        # Initially disable step entries if auto-detect is on
        self.toggle_step_entries()
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select pEIS CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.csv_file_path = file_path
            file_name = os.path.basename(file_path)
            self.file_label.config(text=file_name, foreground="black")
            
            # Auto-populate cell name if empty or default
            if self.cell_name.get() in ["Unknown Cell", ""]:
                # Extract a meaningful name from filename
                name_without_ext = os.path.splitext(file_name)[0]
                # Remove common prefixes and clean up
                clean_name = name_without_ext.replace('pEIS_', '').replace('_', ' ').strip()
                if clean_name:
                    self.cell_name.set(clean_name)
                else:
                    self.cell_name.set("Cell from " + file_name[:20])
            
            # Try to auto-extract Cell ID if set to auto-detect
            if self.cell_id.get() == "Auto-detect":
                extracted_id = self.extract_cell_id(file_name)
                if extracted_id != "Unknown":
                    self.cell_id.set(extracted_id)
    
    def extract_cell_id(self, filename):
        """Extract 10-digit Cell ID from filename"""
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Pattern 1: Look for 10-11 digit numbers
        pattern1 = re.search(r'\b0?(\d{10})\b', name_without_ext)
        if pattern1:
            return pattern1.group(1)
        
        # Pattern 2: Any sequence of 10+ digits
        pattern2 = re.search(r'0?(\d{10,})', name_without_ext)
        if pattern2:
            return pattern2.group(1)[:10]
        
        return "Unknown"
    
    def toggle_step_entries(self):
        """Enable/disable step entry fields based on auto-detect checkbox"""
        if self.auto_detect_steps.get():
            state = 'disabled'
        else:
            state = 'normal'
        
        self.char_step_entry.config(state=state)
        self.disc_step_entry.config(state=state)
        self.c_pulse_entry.config(state=state)
        self.c_rest_entry.config(state=state)
        self.d_pulse_entry.config(state=state)
    
    def validate_inputs(self):
        """Validate all user inputs"""
        if not self.csv_file_path:
            messagebox.showerror("Error", "Please select a CSV file")
            return False
        
        if not os.path.exists(self.csv_file_path):
            messagebox.showerror("Error", "Selected file does not exist")
            return False
        
        # Validate numeric inputs
        try:
            float(self.nominal_capacity.get())
            float(self.v_min.get())
            float(self.v_max.get())
            float(self.energy_capacity.get())
        except ValueError:
            messagebox.showerror("Error", "Cell specifications must be valid numbers")
            return False
        
        # Validate step numbers if not auto-detecting
        if not self.auto_detect_steps.get():
            try:
                int(self.char_step.get())
                int(self.disc_step.get())
                int(self.c_pulse_step.get())
                int(self.c_rest_step.get())
                int(self.d_pulse_step.get())
            except ValueError:
                messagebox.showerror("Error", "Step numbers must be valid integers")
                return False
        
        return True
    
    def start_analysis(self):
        """Collect all inputs and close the GUI"""
        if not self.validate_inputs():
            return
        
        self.result = {
            'csv_file_path': self.csv_file_path,
            'cell_name': self.cell_name.get(),
            'cell_id': self.cell_id.get(),
            'nominal_capacity': float(self.nominal_capacity.get()),
            'v_min': float(self.v_min.get()),
            'v_max': float(self.v_max.get()),
            'energy_capacity': float(self.energy_capacity.get()),
            'auto_detect_steps': self.auto_detect_steps.get(),
            'char_step': int(self.char_step.get()) if not self.auto_detect_steps.get() else None,
            'disc_step': int(self.disc_step.get()) if not self.auto_detect_steps.get() else None,
            'c_pulse_step': int(self.c_pulse_step.get()) if not self.auto_detect_steps.get() else None,
            'c_rest_step': int(self.c_rest_step.get()) if not self.auto_detect_steps.get() else None,
            'd_pulse_step': int(self.d_pulse_step.get()) if not self.auto_detect_steps.get() else None,
            'plot_full_test': self.plot_full_test.get(),
            'plot_ocv': self.plot_ocv.get(),
            'plot_impedance': self.plot_impedance.get(),
            'generate_report': self.generate_report.get()
        }
        
        self.root.quit()
        self.root.destroy()
    
    def cancel(self):
        """Cancel and exit"""
        self.result = None
        self.root.quit()
        self.root.destroy()


# Create and run the GUI
root = tk.Tk()
root.attributes('-topmost', True)
gui = pEISConfigGUI(root)
root.mainloop()

# Check if user cancelled
if gui.result is None:
    print("Analysis cancelled by user. Exiting...")
    exit()

# Extract configuration
config = gui.result
csv_file_path = config['csv_file_path']
Cell_ID = config['cell_id']
file_label = config['cell_name']
Cap0 = config['nominal_capacity']
Vmin = config['v_min']
Vmax = config['v_max']
E0 = config['energy_capacity']

# DEBUG: Print what file_label is
print(f"\n*** DEBUG: file_label = '{file_label}' ***\n")

print("\n" + "="*80)
print("pEIS ANALYSIS CONFIGURATION")
print("="*80)
print(f"File: {csv_file_path}")
print(f"Cell Name: {file_label}")
print(f"Cell ID: {Cell_ID}")
print(f"Nominal Capacity: {Cap0} Ah")
print(f"Voltage Range: {Vmin} - {Vmax} V")
print(f"Energy Capacity: {E0} Wh")
print(f"Auto-detect Steps: {config['auto_detect_steps']}")
print(f"Generate Full Test Plot: {config['plot_full_test']}")
print(f"Generate OCV Plot: {config['plot_ocv']}")
print(f"Generate Impedance Plot: {config['plot_impedance']}")
print(f"Generate Report: {config['generate_report']}")
print("="*80 + "\n")

# Extract directory and filename information
f_dir = os.path.dirname(csv_file_path) + '\\'
f_source = f_dir
file_name_full = os.path.basename(csv_file_path)
file_name = os.path.splitext(file_name_full)[0]

# Create save directory in the same location as the CSV
f_save = f_dir
f_plots = f_save + file_name + '_plots\\'

# Create plots directory if it doesn't exist
if not os.path.exists(f_plots):
    os.makedirs(f_plots)
    print(f"Created output directory: {f_plots}")
else:
    print(f"Output directory exists: {f_plots}")

print(f"\nFile Directory: {f_dir}")
print(f"File Name: {file_name}")
print(f"Plots will be saved to: {f_plots}")
print("-" * 80)

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
Res = Data['DC Internal Resistance (mOhm)'].values
Temp_Time = TT[~np.isnan(Data['K1 (°C)'])][:]

Power = Voltage*Current

del Data

print(f"Data points loaded: {len(TT)}")
print(f"Test duration: {TT[-1]/3600:.2f} hours")
print("-" * 80)

#%% Plot Voltage Full Test

if config['plot_full_test']:
    print("\nGenerating Plot 1: Voltage and Current vs Time (Full Test)...")
    f1 = plt.figure(figsize=(13,9))
    title_font = 20
    ticks_font = 16

    ax1 = f1.add_subplot(1,1,1)
    lns1 = ax1.plot(TT/60, Voltage, color='blue', linewidth=2, label='Voltage', zorder=2)

    ax1.set_ylabel('Voltage [V]', fontsize=title_font, fontweight='bold', labelpad=15)
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_tick_params(labelsize=ticks_font)
    ax1.set_ylim([Vmin-0.5, Vmax+0.6])
    ticks = np.arange(Vmin-0.5, Vmax+0.601, 0.4)
    ax1.set_yticks(ticks)

    ax2 = ax1.twinx()

    lns2 = ax2.plot(TT/60, Current, color='red', linewidth=2, label='Current', zorder=1)
    ax2.set_ylabel('Current [A]', fontsize=title_font, fontweight='bold', labelpad=15)
    ax2.grid(color='gray', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_tick_params(labelsize=ticks_font)
    
    # Auto-scale current axis
    current_max = np.ceil(np.max(Current)/10)*10
    current_min = np.floor(np.min(Current)/10)*10
    ax2.set_ylim([current_min, current_max])
    ticks = np.arange(current_min, current_max+0.01, (current_max-current_min)/8)
    ax2.set_yticks(ticks)

    ax1.set_xlabel('Time [min]', fontsize=title_font, fontweight='bold', labelpad=15)
    ax1.xaxis.set_tick_params(labelsize=ticks_font)
    
    # Auto-scale time axis
    time_max = np.ceil(TT[-1]/60/100)*100
    ax1.set_xlim([0, time_max])
    ticks = np.arange(0, time_max+0.01, time_max/10)
    ax1.set_xticks(ticks)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=title_font, title=f"Cell ID: {Cell_ID}", title_fontsize=title_font, loc='best')

    plt.show()
    f1.savefig(f_plots+file_name+'_Voltage and Current vs Time, Full Test.png', bbox_inches='tight', dpi=200)
    plt.close()

    del f1, title_font, ticks_font, ax1, lns1, lns2, lns, labs

    print("Saved: Voltage and Current vs Time, Full Test.png")
else:
    print("\nSkipping Plot 1: Full Test Plot (disabled in options)")

#%% pEIS Analysis

print("\n" + "="*80)
print("Starting pEIS Analysis...")
print("="*80)

# Use manual or auto-detect based on user choice
if config['auto_detect_steps']:
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
    if len(max_dcap_per_step) > 0 and max_dcap_per_step[0][1] > 1000:
        disc_step = int(max_dcap_per_step[0][0])
        print(f"Found disc_step (Baseline Discharge): Step {disc_step}")
        print(f"  Max Discharge Capacity: {max_dcap_per_step[0][1]/1000:.2f} Ah")
    else:
        print("WARNING: Could not auto-identify disc_step. Using default value 11.")
        disc_step = 11

    # The step with highest charge capacity is likely the baseline charge
    if len(max_ccap_per_step) > 0 and max_ccap_per_step[0][1] > 1000:
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

    pulse_start = Cyc_min + 1
    pulse_end = Cyc_max

    pulse_cycles = Cycle[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
    pulse_steps = Step[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
    pulse_current = Current[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
    pulse_ST = ST[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
    pulse_CCap = CCap[(Cycle >= pulse_start) & (Cycle <= pulse_end)]
    pulse_DCap = DCap[(Cycle >= pulse_start) & (Cycle <= pulse_end)]

    unique_pulse_steps = np.unique(pulse_steps)

    print(f"Unique steps in pulse region (Cycles {pulse_start}-{pulse_end}): {unique_pulse_steps}")

    # Initialize variables
    C_Pulse = None
    C_Rest_D = None
    D_Pulse = None

    # First attempt: Look for charge step with capacity accumulation
    for step_num in unique_pulse_steps:
        step_mask = pulse_steps == step_num
        step_current = pulse_current[step_mask]
        step_ccap = pulse_CCap[step_mask]
        
        if len(step_current) > 0 and len(step_ccap) > 0:
            valid_current = step_current[~np.isnan(step_current)]
            valid_ccap = step_ccap[~np.isnan(step_ccap)]
            
            if len(valid_current) > 0 and len(valid_ccap) > 0:
                avg_current = np.mean(valid_current)
                ccap_range = np.max(valid_ccap) - np.min(valid_ccap)
                
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
                    
                    if avg_current > 5 and max_step_time > 10:
                        charge_candidates.append((step_num, avg_current))
        
        if charge_candidates:
            charge_candidates.sort(key=lambda x: x[1], reverse=True)
            C_Pulse = int(charge_candidates[0][0])
            print(f"Found C_Pulse (CC Charge at 1C until C/100): Step {C_Pulse}")
            print(f"  Avg Current: {charge_candidates[0][1]:.2f} A (highest positive current)")

    if C_Pulse is None:
        print("WARNING: Could not auto-identify C_Pulse step. Using default value 19.")
        C_Pulse = 19

    # Identify C_Rest_D
    for step_num in unique_pulse_steps:
        if step_num <= C_Pulse:
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
                
                if avg_current < 5 and max_step_time > 2 and max_step_time < 10:
                    C_Rest_D = int(step_num)
                    print(f"Found C_Rest_D (Rest for 3 seconds): Step {C_Rest_D}")
                    print(f"  Avg Current: {avg_current:.2f} A, Max Step Time: {max_step_time:.2f} s")
                    break

    # Third attempt for C_Pulse: Use sequence logic if C_Rest_D was found
    if C_Pulse == 19 and C_Rest_D is not None:
        print("Attempting to refine C_Pulse based on sequence before C_Rest_D...")
        
        for step_num in unique_pulse_steps:
            if step_num >= C_Rest_D:
                continue
            
            step_mask = pulse_steps == step_num
            step_current = pulse_current[step_mask]
            
            if len(step_current) > 0:
                valid_current = step_current[~np.isnan(step_current)]
                
                if len(valid_current) > 0:
                    avg_current = np.mean(valid_current)
                    
                    if avg_current > 5:
                        step_cycles = pulse_cycles[step_mask]
                        rest_cycles = pulse_cycles[pulse_steps == C_Rest_D]
                        
                        if len(np.intersect1d(step_cycles, rest_cycles)) > 10:
                            C_Pulse = int(step_num)
                            print(f"Refined C_Pulse (CC Charge at 1C until C/100): Step {C_Pulse}")
                            print(f"  Avg Current: {avg_current:.2f} A (identified by sequence)")
                            break

    if C_Rest_D is None:
        print("WARNING: Could not auto-identify C_Rest_D step. Using default value 20.")
        C_Rest_D = 20

    # Identify D_Pulse
    for step_num in unique_pulse_steps:
        if step_num <= C_Rest_D:
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

else:
    # Use manual step numbers from GUI
    char_step = config['char_step']
    disc_step = config['disc_step']
    C_Pulse = config['c_pulse_step']
    C_Rest_D = config['c_rest_step']
    D_Pulse = config['d_pulse_step']
    
    print("\nUsing manual step numbers:")
    print(f"  char_step = {char_step}")
    print(f"  disc_step = {disc_step}")
    print(f"  C_Pulse  = {C_Pulse}  (CC Charge at 1C until C/100 capacity)")
    print(f"  C_Rest_D = {C_Rest_D}  (Rest for 3 seconds)")
    print(f"  D_Pulse  = {D_Pulse}  (CC Discharge at 1C for 3 seconds)")
    print("="*80)

Cyc_min = int(min(Cycle[Cycle>0]))
Cyc_max = int(max(Cycle[Cycle>0]))
n_pulses = Cyc_max - Cyc_min + 1

print('Number of pulses = '+str(n_pulses-2)+' from '+str(Cyc_min+1)+' to '+str(Cyc_max-1))

# Calculate Cell Capacity with error handling
try:
    discharge_caps = DCap[Step == disc_step]
    if len(discharge_caps) > 0:
        valid_discharge_caps = discharge_caps[~np.isnan(discharge_caps)]
        if len(valid_discharge_caps) > 0:
            Cell_Cap = float(max(valid_discharge_caps))/1000
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
                Cell_Cap = float(max(valid_charge_caps))/1000
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

if config['plot_ocv']:
    print("\nGenerating Plot: OCV vs SOC...")
    f1 = plt.figure(figsize=(12,8))
    title_font = 16
    ticks_font = 14
    legend_font = 14
    thickness = 1.5
    n_Plots = 1

    ax1 = f1.add_subplot(1,1,1)
    lns1  = ax1.plot(SOC_A[OCV > 0], OCV[OCV > 0], color='red',marker='o', linewidth=thickness, label='OCV', zorder=1)

    ax1.set_ylabel('Voltage [V]', fontsize=title_font, fontweight='bold', labelpad=15)
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_tick_params(labelsize=ticks_font)
    ax1.set_ylim([Vmin-0.5, Vmax+0.6])
    ticks = np.arange(Vmin-0.5, Vmax+0.601, 0.4)
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
    ax1.legend(lns, labs, fontsize=legend_font,title=f"Cell ID: {Cell_ID}",title_fontsize=title_font, loc='best', ncol=1)

    plt.show()
    f1.savefig(f_plots+file_name+'_OCV.png', bbox_inches='tight', dpi=200)

    del f1, title_font, ticks_font, legend_font, thickness, ax1, lns, labs, n_Plots

    print("Saved: OCV.png")
else:
    print("\nSkipping OCV Plot (disabled in options)")

#%% Plot HPPC Charge Resistance

if config['plot_impedance']:
    print("Generating Plot: Charge Impedance vs SOC...")
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
    
    # Auto-scale y-axis
    if len(R_Charge_plot) > 0:
        r_max = np.ceil(np.max(R_Charge_plot)*1.1/0.4)*0.4
        ax1.set_ylim([0, r_max])
        ticks = np.arange(0, r_max+0.001, 0.4)
        ax1.set_yticks(ticks)
    else:
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
    ax1.legend(lns, labs, fontsize=legend_font,title=f"Cell ID: {Cell_ID}",title_fontsize=title_font, loc='best', ncol=1)

    plt.show()
    f1.savefig(f_plots+file_name+'_Impedance.png', bbox_inches='tight', dpi=200)

    del f1, title_font, ticks_font, legend_font, thickness, ax1, lns, labs, n_Plots

    print("Saved: Impedance.png")
else:
    print("\nSkipping Impedance Plot (disabled in options)")

#%% Report Values

if config['generate_report']:
    print("\nGenerating CSV Report...")
    report_file = f_save + file_name + '_Report.csv'

    with open(report_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Project', 'Start Date-Time', 'End Date-Time', 'Cell ID', 'Cell Name', 'Data Source', 'Test Title', 'Discharge Capacity [Ah]'])
        
        writer.writerow(['pEIS Analysis', 
                        Data0[1].values[23] if len(Data0[1].values) > 23 else 'N/A', 
                        Data0[1].values[25] if len(Data0[1].values) > 25 else 'N/A', 
                        Cell_ID,
                        file_label,
                        file_name_full, 
                        f"{Cell_Cap:.2f}"])

        writer.writerow([' '])    
        writer.writerow(['SOC_C[%]','SOC_D[%]','SOC_A[%]','DOD[%]','OCV[V]','Charge_Impedance[mOhms]','Discharge_Resistance[mOhms]','Charge_Pulse_Width[s]','Discharge_Pulse_Width[s]'])        
        
        for i in range(len(SOC_A)-1):
            if SOC_A[i+1] != 0 or OCV[i+1] != 0:
                writer.writerow([f"{SOC_C[i+1]:.2f}", 
                               f"{SOC_D[i+1]:.2f}", 
                               f"{SOC_A[i+1]:.2f}", 
                               f"{100-SOC_A[i+1]:.2f}", 
                               f"{OCV[i+1]:.4f}", 
                               f"{R_Charge[i]:.4f}", 
                               f"{R_Discharge[i]:.4f}", 
                               f"{tP_Charge[i]:.2f}", 
                               f"{tP_Discharge[i]:.2f}"])

    print(f"Saved: {report_file}")
else:
    print("\nSkipping CSV Report generation (disabled in options)")

#%% Summary

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"Cell Name: {file_label}")
print(f"Cell ID: {Cell_ID}")
print(f"Cell Capacity: {Cell_Cap:.2f} Ah")
print(f"Total Pulses Analyzed: {len(SOC_A[SOC_A > 0])}")
if len(SOC_A[SOC_A > 0]) > 0:
    print(f"SOC Range: {np.min(SOC_A[SOC_A > 0]):.1f}% - {np.max(SOC_A[SOC_A > 0]):.1f}%")
if len(R_Charge[R_Charge > 0]) > 0:
    print(f"Charge Impedance Range: {np.min(R_Charge[R_Charge > 0]):.3f} - {np.max(R_Charge[R_Charge > 0]):.3f} mΩ")
if len(R_Discharge[R_Discharge > 0]) > 0:
    print(f"Discharge Resistance Range: {np.min(R_Discharge[R_Discharge > 0]):.3f} - {np.max(R_Discharge[R_Discharge > 0]):.3f} mΩ")
print(f"\nAll files saved to:")
print(f"  Data: {f_save}")
print(f"  Plots: {f_plots}")
print("="*80)
