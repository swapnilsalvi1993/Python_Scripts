import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import sys
import re
from datetime import datetime, timedelta

# Create a GUI to select the CSV file
def select_csv_file():
    """Open a file dialog to select CSV file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    print("Please select your CSV file...")
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def save_plot_dialog(default_filename):
    """Open a dialog to save the plot"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.asksaveasfilename(
        title="Save Plot As",
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        initialfile=default_filename
    )
    
    root.destroy()
    return file_path

def extract_start_time(file_path):
    """
    Extract start time from the CSV file metadata section (before the data header).
    Looks for "Start Time:" in the first column and reads the adjacent cell.
    
    Parameters:
    - file_path: Path to the CSV file
    
    Returns:
    - datetime object or None if not found
    """
    try:
        # Read the entire file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Detect delimiter by looking at early lines
        delimiter = '\t'  # Default to tab
        for line in lines[:50]:
            tab_count = line.count('\t')
            comma_count = line.count(',')
            
            if tab_count > comma_count and tab_count > 2:
                delimiter = '\t'
                break
            elif comma_count > tab_count and comma_count > 2:
                delimiter = ','
                break
        
        delimiter_name = 'tab' if delimiter == '\t' else 'comma'
        print(f"Detected delimiter for metadata: {delimiter_name}")
        
        # Search through lines (especially before the data header)
        for line_num, line in enumerate(lines):
            # Stop searching once we hit the data header
            if 'Test' in line and 'Rack' in line and 'Shelf' in line and 'Current' in line:
                print(f"Reached data header at line {line_num + 1}, stopping search")
                break
            
            # Check if this line contains "Start Time:" (not "Start Time Format")
            if 'Start Time:' in line and 'Format' not in line:
                print(f"Found 'Start Time:' at line {line_num + 1}")
                
                # Split by detected delimiter
                parts = line.split(delimiter)
                
                time_str = None
                
                # The format should be: "Start Time:\t7/16/2025 16:50\t\t\t..."
                # So parts[0] = "Start Time:", parts[1] = "7/16/2025 16:50"
                
                if len(parts) >= 2:
                    # parts[0] should contain "Start Time:"
                    # parts[1] should contain the datetime
                    time_str = parts[1].strip()
                    
                    if time_str:
                        print(f"Found Start Time string: '{time_str}' at line {line_num + 1}")
                        
                        # Try to parse the datetime with various formats
                        formats_to_try = [
                            '%m/%d/%Y %H:%M',
                            '%m/%d/%Y %H:%M:%S',
                            '%Y-%m-%d %H:%M',
                            '%Y-%m-%d %H:%M:%S',
                            '%m-%d-%Y %H:%M',
                            '%m-%d-%Y %H:%M:%S',
                            '%d/%m/%Y %H:%M',
                            '%d/%m/%Y %H:%M:%S',
                            '%Y/%m/%d %H:%M',
                            '%Y/%m/%d %H:%M:%S'
                        ]
                        
                        for fmt in formats_to_try:
                            try:
                                start_time = datetime.strptime(time_str, fmt)
                                print(f"Successfully parsed Start Time: {start_time} using format: {fmt}")
                                return start_time
                            except ValueError:
                                continue
                        
                        print(f"Warning: Could not parse Start Time format: '{time_str}'")
                        print(f"Tried formats: {formats_to_try}")
                    else:
                        print(f"Warning: Empty time string found in parts[1]")
                else:
                    print(f"Warning: Line split resulted in {len(parts)} parts (expected at least 2)")
                    print(f"Line content: {line[:100]}")
        
        print("Warning: 'Start Time:' not found in metadata section")
                        
    except Exception as e:
        print(f"Error extracting start time: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def extract_title_info_from_filename(filename):
    """
    Extract test number, temperature, cycle number, and file number from filename.
    
    Parameters:
    - filename: String containing the filename
    
    Returns:
    - Dictionary with extracted information
    """
    # Get just the filename without path
    base_filename = os.path.basename(filename)
    
    info = {
        'test_num': None,
        'temperature': None,
        'cycle_num': None,
        'file_num': None
    }
    
    # Extract Test Number (T01, T02, etc.)
    test_match = re.search(r'T(\d+)', base_filename, re.IGNORECASE)
    if test_match:
        info['test_num'] = f"T{test_match.group(1)}"
    
    # Extract Temperature (35degC, 45degC, 35C, etc.)
    temp_match = re.search(r'(\d+)\s*(?:deg)?C', base_filename, re.IGNORECASE)
    if temp_match:
        info['temperature'] = f"{temp_match.group(1)}°C"
    
    # Extract Cycle Number (Cyc00, Cyc01, etc.)
    cycle_match = re.search(r'Cyc(\d+)', base_filename, re.IGNORECASE)
    if cycle_match:
        info['cycle_num'] = f"Cyc{cycle_match.group(1)}"
    
    # Extract File Number (number inside parentheses)
    file_match = re.search(r'\((\d+)\)', base_filename)
    if file_match:
        info['file_num'] = f"File {file_match.group(1)}"
    
    return info

def create_plot_title(filename):
    """
    Create plot title from filename information.
    
    Parameters:
    - filename: String containing the filename
    
    Returns:
    - Formatted title string
    """
    info = extract_title_info_from_filename(filename)
    
    # Build title from extracted components
    title_parts = []
    
    if info['test_num']:
        title_parts.append(info['test_num'])
    
    if info['temperature']:
        title_parts.append(info['temperature'])
    
    if info['cycle_num']:
        title_parts.append(info['cycle_num'])
    
    if info['file_num']:
        title_parts.append(info['file_num'])
    
    if title_parts:
        main_title = ' - '.join(title_parts)
    else:
        main_title = 'Discharge Capacity Analysis'
    
    return main_title

def get_average_temperature(df, target_index, temp_cols, max_search_distance=50):
    """
    Get average temperature at target index. If NaN, search nearby rows for valid values.
    
    Parameters:
    - df: DataFrame
    - target_index: Index where we want temperature
    - temp_cols: List of temperature column names
    - max_search_distance: Maximum number of rows to search before/after
    
    Returns:
    - Average temperature (float) or np.nan if no valid values found
    """
    # Try to get temperature at target index first
    temp_values = []
    for temp_col in temp_cols:
        temp_val = df.iloc[target_index][temp_col]
        if pd.notna(temp_val):
            temp_values.append(temp_val)
    
    # If we have valid values, return the average
    if temp_values:
        return np.mean(temp_values)
    
    # If all are NaN, search nearby rows
    # Search both forward and backward alternately
    for offset in range(1, max_search_distance + 1):
        # Check forward
        forward_idx = target_index + offset
        if forward_idx < len(df):
            temp_values = []
            for temp_col in temp_cols:
                temp_val = df.iloc[forward_idx][temp_col]
                if pd.notna(temp_val):
                    temp_values.append(temp_val)
            if temp_values:
                return np.mean(temp_values)
        
        # Check backward
        backward_idx = target_index - offset
        if backward_idx >= 0:
            temp_values = []
            for temp_col in temp_cols:
                temp_val = df.iloc[backward_idx][temp_col]
                if pd.notna(temp_val):
                    temp_values.append(temp_val)
            if temp_values:
                return np.mean(temp_values)
    
    # If still no valid temperature found, return NaN
    return np.nan

class ProgressWindow:
    """Progress bar window for showing pulse detection progress"""
    def __init__(self, title="Processing"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("500x150")
        self.root.attributes('-topmost', True)
        
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Create widgets
        self.label = tk.Label(self.root, text="Searching for discharge pulses...", font=("Arial", 11))
        self.label.pack(pady=10)
        
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        self.status_label = tk.Label(self.root, text="", font=("Arial", 9), fg="blue")
        self.status_label.pack(pady=5)
        
        self.pulse_label = tk.Label(self.root, text="Pulses detected: 0", font=("Arial", 10, "bold"))
        self.pulse_label.pack(pady=5)
        
    def update(self, current, total, pulses_found, last_pulse_info=""):
        """Update progress bar"""
        percentage = (current / total) * 100
        self.progress['value'] = percentage
        self.status_label.config(text=f"Processing: {current}/{total} rows ({percentage:.1f}%)")
        self.pulse_label.config(text=f"Pulses detected: {pulses_found}")
        if last_pulse_info:
            self.label.config(text=last_pulse_info)
        self.root.update()
        
    def close(self):
        """Close the progress window"""
        self.root.destroy()

# Get the file path from user
file_path = select_csv_file()

if not file_path:
    print("No file selected. Exiting...")
    sys.exit()

print(f"Selected file: {file_path}")
print(f"Filename: {os.path.basename(file_path)}")

# Extract start time from file BEFORE processing data
print("\nExtracting Start Time from file...")
start_time = extract_start_time(file_path)
if start_time is None:
    print("\n!!! WARNING: Could not extract start time from file !!!")
    print("Please check that the file contains 'Start Time:' in the header.")
    response = messagebox.askyesno(
        "Start Time Not Found",
        "Could not find 'Start Time:' in the file.\n\nDo you want to continue using current date/time as reference?",
    )
    if not response:
        print("Exiting...")
        sys.exit()
    start_time = datetime.now()
    print(f"Using current time as reference: {start_time}")

# Extract title information from filename
title_info = extract_title_info_from_filename(file_path)
print(f"\nExtracted title information:")
print(f"  Test Number: {title_info['test_num']}")
print(f"  Temperature: {title_info['temperature']}")
print(f"  Cycle Number: {title_info['cycle_num']}")
print(f"  File Number: {title_info['file_num']}")

# First, find where the actual data starts (after "Test	Rack	Shelf..." header)
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    header_row = None
    for i, line in enumerate(lines):
        if 'Test' in line and 'Rack' in line and 'Shelf' in line and 'Current' in line:
            header_row = i
            break

if header_row is None:
    print("Could not find data header row. Please check your CSV file format.")
    sys.exit()

print(f"\nData header found at row {header_row}")

# Detect delimiter - check if it's comma or tab separated
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if i == header_row:
            if '\t' in line and line.count('\t') > 5:
                delimiter = '\t'
                print("Detected tab-separated file")
            else:
                delimiter = ','
                print("Detected comma-separated file")
            break

# Read the CSV starting from the header row
df = pd.read_csv(file_path, sep=delimiter, skiprows=header_row, encoding='utf-8')

# Clean column names (remove extra spaces and special characters)
df.columns = df.columns.str.strip()

print(f"Loaded {len(df)} rows of data")
print(f"\nColumns found ({len(df.columns)} total):")
for idx, col in enumerate(df.columns[:15]):  # Show first 15 columns
    print(f"  {idx}: '{col}'")
if len(df.columns) > 15:
    print(f"  ... and {len(df.columns) - 15} more columns")

# Find the correct column names (case-insensitive search)
current_col = None
discharge_cap_col = None
time_col = None
k1_col = None
k2_col = None
k3_col = None

for col in df.columns:
    col_lower = col.lower()
    if 'current' in col_lower and '(a)' in col_lower:
        current_col = col
    elif 'discharge capacity' in col_lower and 'mah' in col_lower:
        discharge_cap_col = col
    elif 'total time' in col_lower and 'second' in col_lower:
        time_col = col
    elif 'k1' in col_lower and 'c' in col_lower:
        k1_col = col
    elif 'k2' in col_lower and 'c' in col_lower:
        k2_col = col
    elif 'k3' in col_lower and 'c' in col_lower:
        k3_col = col

if not current_col or not discharge_cap_col or not time_col:
    print("\nError: Could not find required columns!")
    print(f"Current column: {current_col}")
    print(f"Discharge Capacity column: {discharge_cap_col}")
    print(f"Total Time column: {time_col}")
    sys.exit()

print(f"\nUsing columns:")
print(f"  Current: '{current_col}'")
print(f"  Discharge Capacity: '{discharge_cap_col}'")
print(f"  Total Time: '{time_col}'")
print(f"  Temperature K1: '{k1_col}'")
print(f"  Temperature K2: '{k2_col}'")
print(f"  Temperature K3: '{k3_col}'")

# Convert columns to numeric (in case they're read as strings)
df[current_col] = pd.to_numeric(df[current_col], errors='coerce')
df[discharge_cap_col] = pd.to_numeric(df[discharge_cap_col], errors='coerce')
df[time_col] = pd.to_numeric(df[time_col], errors='coerce')

# Convert temperature columns if they exist
temp_cols_available = []
if k1_col:
    df[k1_col] = pd.to_numeric(df[k1_col], errors='coerce')
    temp_cols_available.append(k1_col)
if k2_col:
    df[k2_col] = pd.to_numeric(df[k2_col], errors='coerce')
    temp_cols_available.append(k2_col)
if k3_col:
    df[k3_col] = pd.to_numeric(df[k3_col], errors='coerce')
    temp_cols_available.append(k3_col)

print(f"\nTemperature columns available: {len(temp_cols_available)}")

# Remove rows with NaN values in critical columns
required_cols = [current_col, discharge_cap_col, time_col]
df_clean = df.dropna(subset=required_cols)
print(f"\nData rows with valid values: {len(df_clean)}")

# Calculate datetime for each row using the extracted start time
print(f"\nCalculating DateTime using Start Time: {start_time.strftime('%m/%d/%Y %H:%M:%S')}")
df_clean['DateTime'] = df_clean[time_col].apply(lambda x: start_time + timedelta(seconds=x))

# Verify first few datetime calculations
print(f"First row: Time(s)={df_clean.iloc[0][time_col]:.2f}, DateTime={df_clean.iloc[0]['DateTime']}")
if len(df_clean) > 1:
    print(f"Second row: Time(s)={df_clean.iloc[1][time_col]:.2f}, DateTime={df_clean.iloc[1]['DateTime']}")

# Find points where current drops to ~0 from -40A (fast discharge)
# Strategy: Look for transitions where current goes from around -40A to near 0
discharge_points = []

# Define thresholds
discharge_threshold = -30  # Current less than -30A is considered fast discharge (from ~-40A)
zero_threshold = 5  # Current between -5 and 5 is considered ~zero
skip_points = 4  # Skip 4 points, then take the 5th (index 4 means 5th point)
skip_initial_pulses = 4  # Skip first 4 pulses to avoid initial transients

print("\n" + "="*110)
print("PULSE DETECTION SETTINGS")
print("="*110)
print(f"Start Time: {start_time.strftime('%m/%d/%Y %H:%M:%S')}")
print(f"Discharge threshold: {discharge_threshold}A")
print(f"Zero threshold: ±{zero_threshold}A")
print(f"Taking 5th datapoint after current reaches zero")
print(f"Skipping first {skip_initial_pulses} pulses to avoid initial transients")
print(f"Temperature: Average of K1, K2, K3 (searching nearest valid value if NaN)")
print(f"Current range in data: {df_clean[current_col].min():.2f}A to {df_clean[current_col].max():.2f}A")
print("="*110)

# Create progress window
progress_win = ProgressWindow("Analyzing Discharge Pulses")

i = 0
pulse_count = 0
temp_searches = 0  # Count how many times we had to search for temperature
update_interval = max(1, len(df_clean) // 100)  # Update progress bar 100 times

print("\nDetected Pulses:")
print("-" * 120)
if temp_cols_available:
    print(f"{'Pulse #':<10} {'Row':<10} {'DateTime':<25} {'Capacity (Ah)':<20} {'Temp (°C)':<15} {'Temp Source':<15}")
else:
    print(f"{'Pulse #':<10} {'Row':<10} {'DateTime':<25} {'Capacity (Ah)':<20}")
print("-" * 120)

while i < len(df_clean) - skip_points - 1:
    # Update progress bar periodically
    if i % update_interval == 0:
        progress_win.update(i, len(df_clean), pulse_count)
    
    # Check if current point is in fast discharge range
    current_val = df_clean.iloc[i][current_col]
    
    if current_val < discharge_threshold:
        # Look ahead to find when it drops to zero
        for j in range(i + 1, min(i + 30, len(df_clean))):  # Look within next 30 points
            next_current = df_clean.iloc[j][current_col]
            
            if abs(next_current) < zero_threshold:
                # Found transition to zero, skip 4 points and take the 5th reading
                target_index = j + skip_points
                if target_index < len(df_clean):
                    pulse_count += 1
                    
                    # Skip the first few pulses
                    if pulse_count <= skip_initial_pulses:
                        if pulse_count == 1:
                            print(f">>> Skipping first {skip_initial_pulses} pulses (initial transients)...")
                        i = target_index + 1
                        break
                    
                    discharge_capacity_mah = df_clean.iloc[target_index][discharge_cap_col]
                    total_time_sec = df_clean.iloc[target_index][time_col]
                    date_time = df_clean.iloc[target_index]['DateTime']
                    
                    # Calculate average temperature with smart NaN handling
                    avg_temp = np.nan
                    temp_source = "N/A"
                    if temp_cols_available:
                        # Try direct value first
                        temp_values = []
                        for temp_col in temp_cols_available:
                            temp_val = df_clean.iloc[target_index][temp_col]
                            if pd.notna(temp_val):
                                temp_values.append(temp_val)
                        
                        if temp_values:
                            avg_temp = np.mean(temp_values)
                            temp_source = "Direct"
                        else:
                            # Search for nearest valid temperature
                            avg_temp = get_average_temperature(df_clean, target_index, temp_cols_available, max_search_distance=50)
                            if pd.notna(avg_temp):
                                temp_source = "Nearest"
                                temp_searches += 1
                            else:
                                temp_source = "Not Found"
                    
                    discharge_capacity_ah = discharge_capacity_mah / 1000  # Convert mAh to Ah
                    time_hours = total_time_sec / 3600
                    
                    discharge_points.append({
                        'DateTime': date_time,
                        'Time (s)': total_time_sec,
                        'Time (hours)': time_hours,
                        'Discharge Capacity (Ah)': discharge_capacity_ah,
                        'Temperature (°C)': avg_temp,
                        'Temp Source': temp_source,
                        'Row Index': target_index,
                        'Pulse Number': pulse_count
                    })
                    
                    # Print all pulse detections (after skipping initial ones)
                    datetime_str = date_time.strftime('%m/%d/%Y %H:%M:%S')
                    if temp_cols_available:
                        if pd.notna(avg_temp):
                            print(f"{pulse_count:<10} {target_index:<10} {datetime_str:<25} {discharge_capacity_ah:<20.4f} {avg_temp:<15.2f} {temp_source:<15}")
                            last_info = f"Last pulse: #{pulse_count} at {datetime_str}, Cap={discharge_capacity_ah:.3f}Ah, Temp={avg_temp:.1f}°C"
                        else:
                            print(f"{pulse_count:<10} {target_index:<10} {datetime_str:<25} {discharge_capacity_ah:<20.4f} {'N/A':<15} {temp_source:<15}")
                            last_info = f"Last pulse: #{pulse_count} at {datetime_str}, Cap={discharge_capacity_ah:.3f}Ah, Temp=N/A"
                    else:
                        print(f"{pulse_count:<10} {target_index:<10} {datetime_str:<25} {discharge_capacity_ah:<20.4f}")
                        last_info = f"Last pulse: #{pulse_count} at {datetime_str}, Cap={discharge_capacity_ah:.3f}Ah"
                    
                    # Update progress window with last pulse info
                    progress_win.update(i, len(df_clean), pulse_count, last_info)
                
                # Skip ahead to avoid capturing the same pulse multiple times
                i = target_index + 1
                break
        else:
            i += 1
    else:
        i += 1

# Final progress update
progress_win.update(len(df_clean), len(df_clean), pulse_count, f"Analysis complete! Found {pulse_count} pulses")
progress_win.root.after(1000, progress_win.close)  # Close after 1 second

print("-" * 120)
print(f"Total pulses detected: {pulse_count}")
print(f"Pulses plotted: {pulse_count - skip_initial_pulses} (skipped first {skip_initial_pulses})")
if temp_cols_available:
    print(f"Temperature values found via nearest neighbor search: {temp_searches}")
print("-" * 120)

# Convert to DataFrame for easier plotting
plot_df = pd.DataFrame(discharge_points)

if len(plot_df) == 0:
    print("\nWarning: No discharge pulses detected (after skipping initial transients)!")
    print("You may need to adjust the thresholds:")
    print(f"  - Current discharge_threshold: {discharge_threshold}A")
    print(f"  - Current zero_threshold: ±{zero_threshold}A")
    
    # Show some sample current values
    print("\nSample current values (first 20 rows):")
    print(df_clean[[current_col]].head(20))
    sys.exit()

# Create the scatter plot with dual Y-axes
fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)

# Left Y-axis - Discharge Capacity
color1 = 'steelblue'
ax1.set_xlabel('Date/Time', fontsize=28, fontweight='bold')
ax1.set_ylabel('Discharge Capacity (Ah)', fontsize=28, fontweight='bold', color=color1)
ax1.scatter(plot_df['DateTime'], plot_df['Discharge Capacity (Ah)'], 
           s=80, alpha=0.7, edgecolors='black', linewidth=0.8, color=color1, label='Discharge Capacity')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=22)
ax1.tick_params(axis='x', labelsize=20, rotation=45)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Set fixed Y-axis range for discharge capacity
ax1.set_ylim(15, 20)

# Format x-axis for datetime
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y\n%H:%M'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate()  # Auto-format date labels to prevent overlap

# Right Y-axis - Temperature
if temp_cols_available and not plot_df['Temperature (°C)'].isna().all():
    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.set_ylabel('Temperature (°C)', fontsize=28, fontweight='bold', color=color2)
    ax2.scatter(plot_df['DateTime'], plot_df['Temperature (°C)'], 
               s=80, alpha=0.7, edgecolors='black', linewidth=0.8, color=color2, 
               marker='s', label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=22)
    
    # Set fixed Y-axis range for temperature
    ax2.set_ylim(31, 37.5)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=22, framealpha=0.9)
else:
    ax1.legend(loc='upper left', fontsize=22, framealpha=0.9)

# Create title from filename - simplified without subtitle
plot_title = create_plot_title(file_path)
ax1.set_title(plot_title, fontsize=32, fontweight='bold', pad=20)

plt.tight_layout()

# Show the plot
plt.show(block=False)

# Print summary
print(f"\n{'='*110}")
print(f"RESULTS SUMMARY")
print(f"{'='*110}")
print(f"Total number of discharge pulses detected: {pulse_count}")
print(f"Pulses plotted: {len(plot_df)} (first {skip_initial_pulses} skipped)")
print(f"\nFirst 10 plotted data points:")
print(plot_df[['Pulse Number', 'DateTime', 'Discharge Capacity (Ah)', 'Temperature (°C)', 'Temp Source']].head(10).to_string(index=False))
if len(plot_df) > 10:
    print(f"\nLast 10 plotted data points:")
    print(plot_df[['Pulse Number', 'DateTime', 'Discharge Capacity (Ah)', 'Temperature (°C)', 'Temp Source']].tail(10).to_string(index=False))
print(f"{'='*110}")

# Statistics
print(f"\nDISCHARGE CAPACITY STATISTICS:")
print(f"  Average: {plot_df['Discharge Capacity (Ah)'].mean():.4f} Ah")
print(f"  Min: {plot_df['Discharge Capacity (Ah)'].min():.4f} Ah")
print(f"  Max: {plot_df['Discharge Capacity (Ah)'].max():.4f} Ah")
print(f"  Standard Deviation: {plot_df['Discharge Capacity (Ah)'].std():.4f} Ah")

if temp_cols_available and not plot_df['Temperature (°C)'].isna().all():
    valid_temps = plot_df['Temperature (°C)'].dropna()
    print(f"\nTEMPERATURE STATISTICS:")
    print(f"  Average: {valid_temps.mean():.2f} °C")
    print(f"  Min: {valid_temps.min():.2f} °C")
    print(f"  Max: {valid_temps.max():.2f} °C")
    print(f"  Standard Deviation: {valid_temps.std():.2f} °C")
    print(f"  Valid temperature readings: {len(valid_temps)}/{len(plot_df)}")

print(f"\nTEST DURATION:")
print(f"  Start: {plot_df['DateTime'].min().strftime('%m/%d/%Y %H:%M:%S')}")
print(f"  End: {plot_df['DateTime'].max().strftime('%m/%d/%Y %H:%M:%S')}")
print(f"  Total: {plot_df['Time (hours)'].max():.2f} hours ({plot_df['Time (hours)'].max()/24:.2f} days)")
print(f"{'='*110}")

# Ask user if they want to save the plot
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

save_response = messagebox.askyesno(
    "Save Plot",
    f"Detected {pulse_count} discharge pulses.\nPlotted {len(plot_df)} pulses (skipped first {skip_initial_pulses}).\n\nDo you want to save this plot?",
    parent=root
)

if save_response:
    # Generate default filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    default_filename = f"{base_name}_discharge_capacity_temp_plot.png"
    
    # Open save dialog
    save_path = save_plot_dialog(default_filename)
    
    if save_path:
        # Save the figure with high DPI
        fig.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        messagebox.showinfo("Success", f"Plot saved successfully to:\n{save_path}", parent=root)
    else:
        print("\nPlot save cancelled.")
else:
    print("\nPlot not saved.")

# Ask if user wants to save the data CSV
save_csv_response = messagebox.askyesno(
    "Save Data",
    "Do you want to save the extracted data points to a CSV file?",
    parent=root
)

if save_csv_response:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    csv_default_name = f"{base_name}_discharge_points.csv"
    
    csv_save_path = filedialog.asksaveasfilename(
        title="Save Data As CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=csv_default_name,
        parent=root
    )
    
    if csv_save_path:
        # Format DateTime as string for CSV export and exclude Temp Source
        export_df = plot_df[['Pulse Number', 'DateTime', 'Time (s)', 'Time (hours)', 
                             'Discharge Capacity (Ah)', 'Temperature (°C)', 'Row Index']].copy()
        export_df['DateTime'] = export_df['DateTime'].dt.strftime('%m/%d/%Y %H:%M:%S')
        export_df.to_csv(csv_save_path, index=False)
        print(f"Data saved to: {csv_save_path}")
        messagebox.showinfo("Success", f"Data saved successfully to:\n{csv_save_path}", parent=root)

root.destroy()

# Keep the plot window open until user closes it
plt.show()

print("\nDone!")