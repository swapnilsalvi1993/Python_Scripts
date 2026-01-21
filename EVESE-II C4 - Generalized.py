# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 17:31:44 2025

@author: ssalvi
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from tkinter import filedialog
from tkinter import Tk

def process_csv_data(file_path):
    """
    Processes a single CSV file by finding the start of the test based on
    a synchronization signal and adding relative time columns.
    """
    print(f"Reading data from: {file_path}")
    
    # try:
    #     data = pd.read_csv(file_path, low_memory=False)
    #     sync_column_name = 'Actuator Sync'
    #     required_columns = ['TIME 20 Hz', sync_column_name]

    #     if not all(col in data.columns for col in required_columns):
    #         print(f"!!ERROR!! - Missing required columns in '{os.path.basename(file_path)}'. Skipping.")
    #         print(f"Expected columns: {required_columns}")
    #         print(f"Found columns: {list(data.columns)}")
    #         return

    #     change_indices = data[(data[sync_column_name] > 8) & (data[sync_column_name].shift(1) <= 8)].index
        
    #     if len(change_indices) == 0:
    #         print(f"!!WARNING!! - Synchronization signal change not found in '{os.path.basename(file_path)}'. Skipping.")
    #         return
            
    #     change_index = change_indices[0]
    #     datum_time_sec = data.loc[change_index, 'TIME 20 Hz']
        
    #     data['Time (sec)'] = (data['TIME 20 Hz'] - datum_time_sec).round(3)
    #     data['Time (min)'] = (data['Time (sec)'] / 60).round(3)

    #     data.to_csv(file_path, index=False)
        
    #     print(f"\nSuccessfully processed '{os.path.basename(file_path)}'. Relative time columns added.")
        
    # except Exception as e:
    #     print(f"An error occurred while processing '{os.path.basename(file_path)}': {e}")

# --- Main Script ---

if __name__ == '__main__':
    # Initialize tkinter and hide the root window
    root = Tk()
    root.withdraw()
    
    # Prompt the user to select the directory
    directory = filedialog.askdirectory(title="Select the Directory with CSV Files")
    
    if not directory:
        print("!!ERROR!! - No directory selected. Exiting.")
    else:
        # Change the current working directory to the selected folder
        os.chdir(directory)

        # Now, you can simply use the file names without the full path
        csv_files = glob.glob("*.csv")

        if not csv_files:
            print("!!ERROR!! - No CSV files found in the specified directory.")
        else:
            for csv_file in csv_files:
                process_csv_data(csv_file)
            
            if csv_files:
                d = pd.read_csv(csv_files[0], low_memory=False)
                print("\n\nFirst processed file loaded into DataFrame 'd' for further analysis.")
                print(d.head())
                
                


#%% #######Producing Temperature (all) plots in Minutes#####################################################################

# Creating plot with dataset
fig, ax1 = plt.subplots()

color1 = 'tab:red'
color2 = 'tab:pink'
color3 = 'tab:purple'
color4 = 'tab:green'
color5 = 'tab:blue' 

ax1.set_xlabel('time (min)')
ax1.set_ylabel('Temperature (degC)')
ax1.plot(d['Time (min)'],d['/RTAC Data/TC1  Cell Positive'], label = 'Cell Positive', color = color2)
ax1.plot(d['Time (min)'],d['/RTAC Data/TC2  Cell Negative'], label = 'Cell Negative', color = color3)     
ax1.plot(d['Time (min)'],d['/RTAC Data/TC3  Cell Front Center'], label = 'Cell Front Center', color = color4)
ax1.plot(d['Time (min)'],d['/RTAC Data/TC4  Cell Back Center'],'--', label = 'Cell Back Center', color = color4)
ax1.plot(d['Time (min)'],d['/RTAC Data/TC5  Cell Vent'], label = 'Cell Vent', color = color1)
ax1.plot(d['Time (min)'],d['/RTAC Data/TC6  Enclosure Ambient'], label = 'Enclosure Ambient', color = color5)
ax1.plot(d['Time (min)'],d['/RTAC Data/TC7  Cell Side Positive'],'--', label = 'Cell Side Positive', color = color2)
ax1.plot(d['Time (min)'],d['/RTAC Data/TC8  Enclosure Ambient Wire'],'--', label = 'Enclosure Ambient Wire', color = color3)   
ax1.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

plt.xlim([-1, 65])

ax1.xaxis.set_major_locator(plt.MultipleLocator(5))


# Adding Twin Axes to plot using dataset_2

ax2 = ax1.twinx()
 
color6 = 'tab:gray'
color7 = 'black'

ax2.set_ylabel('Cell Voltage & Actuator Signal (V)')
ax2.plot(d['Time (min)'],d['/RTAC Data/Actuator Sync'], ':', label = 'Nail Penetration', color = color6)
ax2.plot(d['Time (min)'],d['/RTAC Data/Cell_V'], label = 'Cell Voltage', color = color7)
ax2.tick_params(axis ='y')
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
plt.ylim([0, 12])

ax1.set_ylim(0, 815)
plt.savefig("EVESE_C4 Temperatures - all.png", dpi=500, bbox_inches= 'tight')   
# ax1.set_xlim(-0.1, 2)
# ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
# ax1.set_ylim(0, 815)
# plt.savefig("EVESE_C4 Temperatures - all_zoomed.png", dpi=500, bbox_inches= 'tight')    


#%% #######Producing Temperature (2 miutes) plots in seconds#####################################################################

# Creating plot with dataset
fig, ax1 = plt.subplots()
 
color1 = 'tab:red'
color2 = 'tab:pink'
color3 = 'tab:purple'
color4 = 'tab:green'
color5 = 'tab:blue' 

ax1.set_xlabel('time (sec)')
ax1.set_ylabel('Temperature (degC)')
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC1  Cell Positive'], label = 'Cell Positive', color = color2)
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC2  Cell Negative'], label = 'Cell Negative', color = color3)     
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC3  Cell Front Center'], label = 'Cell Front Center', color = color4)
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC4  Cell Back Center'],'--', label = 'Cell Back Center', color = color4)
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC5  Cell Vent'], label = 'Cell Vent', color = color1)
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC6  Enclosure Ambient'], label = 'Enclosure Ambient', color = color5)
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC7  Cell Side Positive'],'--', label = 'Cell Side Positive', color = color2)
ax1.plot(d['Time (sec)'],d['/RTAC Data/TC8  Enclosure Ambient Wire'],'--', label = 'Enclosure Ambient Wire', color = color3) 
ax1.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

plt.ylim([0, 815])
ax1.xaxis.set_major_locator(plt.MultipleLocator(30))

# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()
color6 = 'tab:gray'
color7 = 'black'
ax2.set_ylabel('Cell Voltage & Actuator Signal (V)')
ax2.plot(d['Time (sec)'],d['/RTAC Data/Actuator Sync'], ':', label = 'Nail Penetration', color = color6)
ax2.plot(d['Time (sec)'],d['/RTAC Data/Cell_V'], label = 'Cell Voltage', color = color7)
ax2.tick_params(axis ='y')
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
plt.ylim([0, 12])


plt.xlim([-6, 120])
plt.savefig("EVESE_C4 Temperatures - 2 min.png", dpi=500, bbox_inches= 'tight')    

plt.xlim([-6, 300])
plt.savefig("EVESE_C4 Temperatures - 5 min.png", dpi=500, bbox_inches= 'tight')    

#%% Plotting alll Individual Parameters

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (sec)')
ax1.set_ylabel('Pressure (PSIG)')
ax1.plot(d['Time (sec)'],d['/RTAC Data/1000 Pressure'], 'r', label = 'Enclosure Pressure')
ax1.plot(d['Time (sec)'],d['/RTAC Data/Air Pressure'], 'b', label = 'Actuator Pressure')
plt.ylim([0, 250])
ax1.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
ax2 = ax1.twinx()
color6 = 'tab:gray'
color7 = 'black'
ax2.set_ylabel('Cell Voltage & Actuator Signal (V)')
ax2.plot(d['Time (sec)'],d['/RTAC Data/Actuator Sync'], ':', label = 'Nail Penetration', color = color6)
ax2.plot(d['Time (sec)'],d['/RTAC Data/Cell_V'], label = 'Cell Voltage', color = color7)
ax2.tick_params(axis ='y')
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
plt.ylim([0, 12])
 
plt.xlim([-6, 60])    
plt.savefig('EVESE_C4 pressures - 1min.png', dpi=300, bbox_inches= 'tight')
plt.xlim([-6, 120])    
plt.savefig('EVESE_C4 pressures - 2min.png', dpi=300, bbox_inches= 'tight')
plt.xlim([-6, 300])    
plt.savefig('EVESE_C4 pressures - 5min.png', dpi=300, bbox_inches= 'tight')



fig, ax1 = plt.subplots()
ax1.set_xlabel('time (sec)')
ax1.set_ylabel('Nail Travel (mm)')
ax1.plot(d['Time (sec)'],d['/RTAC Data/Dispacement_LVIT'], 'g', label = 'Nail Displacement')
plt.ylim([-5, 80])
ax1.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
ax2 = ax1.twinx()
color6 = 'tab:gray'
color7 = 'black'
ax2.set_ylabel('Cell Voltage & Actuator Signal (V)')
ax2.plot(d['Time (sec)'],d['/RTAC Data/Actuator Sync'], ':', label = 'Nail Penetration', color = color6)
ax2.plot(d['Time (sec)'],d['/RTAC Data/Cell_V'], label = 'Cell Voltage', color = color7)
ax2.tick_params(axis ='y')
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
plt.ylim([0, 12])
 
plt.xlim([-2, 2])    
plt.savefig('EVESE_C4 displacement - 2sec.png', dpi=300, bbox_inches= 'tight')
plt.xlim([-2, 3])    
plt.savefig('EVESE_C4 displacement - 3sec.png', dpi=300, bbox_inches= 'tight')
plt.xlim([-2, 5])    
plt.savefig('EVESE_C4 displacement - 5sec.png', dpi=300, bbox_inches= 'tight')

# #%%  ###################### FOR MaxT on the Article #########################

# # List of parameters you want to analyze
# parameters = ['Hot_Blank', 'Left_Blank_Top', 'Left_Blank_Bottom', 'Right_Blank_Top', 'Right_Blank_Bottom', 'Left_Fluid', 'Right_Fluid']

# # List to store max temperature values and corresponding times for each parameter
# max_data = []

# # Iterate over each parameter to find max values and corresponding times
# for param in parameters:
#     input_list = d[param]
#     max_value = max(input_list)
#     x1 = []

#     # Find all indices where the parameter equals its max value
#     for i in range(len(input_list)):
#         if input_list[i] == max_value:
#             x1.append(i)

#     # Collect max temperature values and corresponding times
#     max_temperatures = []
#     times = []
#     max_temperatures.append(input_list[x1[0]])
#     times.append(d['Time (sec)'][x1[0]])  # Assuming 'Time (sec)' exists

#     # Append the results for this parameter to the max_data list
#     for i in range(len(max_temperatures)):
#         max_data.append({
#             'Parameter': param,
#             'Time (sec)': times[i],
#             'Max Temperature (degC)': max_temperatures[i]
#         })

# # Create a DataFrame from the collected data
# max_data_df = pd.DataFrame(max_data)

# # Append data to CSV (if file exists, it appends; otherwise, it creates a new file)
# existing_columns = d.columns.tolist()
# empty_columns = pd.DataFrame('', index=d.index, columns=[''])  # Adding empty column for the gap
# d = pd.concat([d, empty_columns, max_data_df], axis=1)

# d.to_csv(csv_file, index=False)

# #%%  ###################### Time (sec) to 200C #########################

# # Find indices where 'Hot_Blank' exceeds 200
# indices = [index for index, item in enumerate(d['Hot_Blank']) if item > 200]

# # Get the last index where the value exceeds 200
# if indices:
#     x1 = indices[-1]
#     # Get the corresponding time for this parameter when it exceeds 200
#     time_to_200 = d['Time (min)'][x1]*60
#     t_to_200 = {'Parameter': 'Hot_Blank', 'Time to 200 C (sec)': time_to_200}
    
# t_to_200_df = pd.DataFrame([t_to_200])

# # Append data to CSV (if file exists, it appends; otherwise, it creates a new file)
# existing_columns = d.columns.tolist()
# empty_columns = pd.DataFrame('', index=d.index, columns=[''])  # Adding empty column for the gap
# d = pd.concat([d, empty_columns, t_to_200_df], axis=1)

# d.to_csv(csv_file, index=False)

    
# #%%  ###################### Time (min) to 40C #########################

# # List to store results for each parameter
# t_to_40 = []

# # Iterate over each parameter
# for param in parameters:
#     indices = [index for index, item in enumerate(d[param]) if item > 40]
    
#     # Get the last index where the value is greater than 40
#     if indices:
#         x1 = indices[-1]
#         # Get the corresponding time for this parameter when it exceeds 40
#         time_to_40 = d['Time (min)'][x1]  # Assuming 'Time (min)' column exists
        
#         # Append the result to the max_data list (store parameter name and time)
#         t_to_40.append({'Parameter': param, 'Time to 40 degC (min)': time_to_40})
#     else:
#         # If no value exceeds 40, append NaN or any other value as required
#         t_to_40.append({'Parameter': param, 'Time to 40 degC (min)': None})

# # Create a DataFrame from the collected data
# t_to_40_df = pd.DataFrame(t_to_40)

# # Append data to CSV (if file exists, it appends; otherwise, it creates a new file)
# existing_columns = d.columns.tolist()
# empty_columns = pd.DataFrame('', index=d.index, columns=[''])  # Adding empty column for the gap
# d = pd.concat([d, empty_columns, t_to_40_df], axis=1)

# d.to_csv(csv_file, index=False)




#%% #######Producing FTIR (all) plots in Minutes#####################################################################

# # Creating plot with dataset
# fig, ax1 = plt.subplots()

# color = 'black'
# color1 = 'tab:red'
# color2 = 'tab:pink'
# color3 = 'tab:purple'
# color4 = 'tab:green'
# color5 = 'tab:blue' 
# color6 = 'tab:orange'
# color7 = 'tab:brown'
# color8 = 'tab:cyan'
# color9 = 'tab:olive'
# color10 = 'tab:gray'

# ax1.set_xlabel('time (min)')
# ax1.set_ylabel('H2O (ppm)')
# ax1.plot(d['Time (min)'],d['H2O'], label = 'H2O', color = color5)
# ax1.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
#            ncol=3, mode="expand", borderaxespad=0.)
# plt.ylim(bottom=0)
# # plt.ylim([0, 30000])

# # Adding Twin Axes to plot using dataset_2
# ax2 = ax1.twinx()
# ax2.set_ylabel('CO2:H, CO2:L, CO:H & CO:L (ppm)')
# ax2.plot(d['Time (min)'],d['CO2_H'],':', label = 'CO2:H', color = color1)     
# ax2.plot(d['Time (min)'],d['CO2_L'],'--', label = 'CO2:L', color = color1)
# ax2.plot(d['Time (min)'],d['CO_H'],':', label = 'CO:H', color = color2)
# ax2.plot(d['Time (min)'],d['CO_L'],'--', label = 'CO:L', color = color2)
# ax2.legend(bbox_to_anchor=(0., 1.22, 1., .102), loc='lower left',
#            ncol=4, mode="expand", borderaxespad=0.)
# plt.ylim(bottom=0)
# # plt.ylim([0, 1000])

# # Adding Twin Axes to plot using dataset_2
# ax3 = ax1.twinx()
# ax3.set_ylabel('Ethylene Glycol & rest all species from FTIR (ppm)')
# ax3.plot(d['Time (min)'],d['NO'], label = 'NO', color = color4)
# ax3.plot(d['Time (min)'],d['NO2'],'--', label = 'NO2', color = color4)
# ax3.plot(d['Time (min)'],d['N2O'], label = 'N2O', color = color9)
# ax3.plot(d['Time (min)'],d['NH3'],'--', label = 'NH3', color = color9)
# ax3.plot(d['Time (min)'],d['SO2'],':', label = 'SO2', color = color10)
# ax3.plot(d['Time (min)'],d['CH4'],':', label = 'CH4', color = color3)
# ax3.plot(d['Time (min)'],d['C2H6'], label = 'C2H6', color = color3)
# ax3.plot(d['Time (min)'],d['C2H4'],'--', label = 'C2H4', color = color3)
# ax3.plot(d['Time (min)'],d['CH2O'], label = 'CH2O', color = color8)
# ax3.plot(d['Time (min)'],d['CH2O'],'--', label = 'CH2O', color = color8)
# ax3.plot(d['Time (min)'],d['Ethylene Glycol'], label = 'Ethylene Glycol', color = color1)
# ax3.plot(d['Time (min)'],d['HCl'], label = 'HCL', color = color6)
# ax3.plot(d['Time (min)'],d['HCN'],'--', label = 'HCN', color = color6)
# ax3.plot(d['Time (min)'],d['HF'],':', label = 'HF', color = color6)
# ax3.legend(bbox_to_anchor=(0., 1.32, 1., .102), loc='lower left',
#            ncol=4, mode="expand", borderaxespad=0.)
# ax3.spines['right'].set_position(('outward', 55))  # Adjust the position of ax4
# plt.ylim(bottom=0)
# # plt.ylim([0, 350])


# # Adding Twin Axes to plot using dataset_2
# ax4 = ax1.twinx()
# ax4.set_ylabel('Submersion Signal')
# ax4.plot(d['Time (min)'],d['Actuator Sync'], ':', label = 'Submersion', color = color)
# ax4.tick_params(axis ='y')
# ax4.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#             ncol=1, mode="expand", borderaxespad=0.)
# ax4.spines['right'].set_position(('outward', 105))  # Adjust the position of ax4
# plt.ylim([0, 12])

# plt.xlim([-1, 30])
# ax1.xaxis.set_major_locator(plt.MultipleLocator(5))

# plt.savefig("EVESE_C4  FTIR - all.png", dpi=500, bbox_inches= 'tight')    
