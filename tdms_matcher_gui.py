import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nptdms import TdmsFile
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import re

class TDMSMatcher:
    def __init__(self, csv_file_path, tdms_folder_path):
        """
        Initialize the TDMS Matcher
        
        Args:
            csv_file_path: Path to the CSV summary file
            tdms_folder_path: Path to folder containing TDMS files
        """
        self.csv_file_path = csv_file_path
        self.tdms_folder_path = tdms_folder_path
        self.summary_df = None
        self.tdms_data_combined = None
        self.group_name = None  # Store the actual group name
        
    def read_summary_csv(self):
        """Read the summary CSV file"""
        # Try multiple encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                self.summary_df = pd.read_csv(self.csv_file_path, encoding=encoding)
                print(f"✓ CSV read successfully using encoding: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                if encoding == encodings[-1]:  # Last encoding attempt
                    raise ValueError(f"Could not read CSV file with any encoding. Error: {e}")
        
        # Try multiple datetime formats
        datetime_formats = [
            '%m/%d/%Y %H:%M:%S',  # With seconds: 7/16/2025 23:21:24
            '%m/%d/%Y %H:%M',     # Without seconds: 7/16/2025 23:21
            '%Y-%m-%d %H:%M:%S',  # ISO format with seconds
            '%Y-%m-%d %H:%M',     # ISO format without seconds
            '%d/%m/%Y %H:%M:%S',  # DD/MM/YYYY with seconds
            '%d/%m/%Y %H:%M',     # DD/MM/YYYY without seconds
        ]
        
        parsed = False
        for fmt in datetime_formats:
            try:
                self.summary_df['DateTime'] = pd.to_datetime(
                    self.summary_df['DateTime'], 
                    format=fmt
                )
                print(f"✓ DateTime parsed successfully using format: {fmt}")
                parsed = True
                break
            except (ValueError, TypeError) as e:
                continue
        
        if not parsed:
            # If all formats fail, let pandas infer the format
            try:
                self.summary_df['DateTime'] = pd.to_datetime(
                    self.summary_df['DateTime'], 
                    infer_datetime_format=True
                )
                print(f"✓ DateTime parsed successfully using auto-detection")
            except Exception as e:
                raise ValueError(f"Could not parse DateTime column. Error: {e}")
        
        print(f"Summary CSV loaded: {len(self.summary_df)} rows")
        print(f"DateTime Range: {self.summary_df['DateTime'].min()} to {self.summary_df['DateTime'].max()}")
        return self.summary_df
    
    def parse_tdms_filename(self, filename):
        """
        Parse TDMS filename to extract datetime
        
        Args:
            filename: TDMS filename (e.g., '20250715_095042_BambiData_0001.tdms')
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            base_name = os.path.basename(filename)
            datetime_str = '_'.join(base_name.split('_')[:2])
            file_datetime = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
            return file_datetime
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None
    
    def is_tdms_file_in_range(self, tdms_datetime, start_time, end_time):
        """
        Check if TDMS file is within 12 hours of the datetime range
        
        Args:
            tdms_datetime: datetime object from TDMS filename
            start_time: Start of datetime range
            end_time: End of datetime range
            
        Returns:
            Boolean indicating if file should be processed
        """
        if tdms_datetime is None:
            return False
        
        # Expand range by 12 hours on both sides
        range_start = start_time - timedelta(hours=12)
        range_end = end_time + timedelta(hours=12)
        
        return range_start <= tdms_datetime <= range_end
    
    def find_data_group(self, tdms_file):
        """
        Find the group that contains the actual data
        
        Args:
            tdms_file: TdmsFile object
            
        Returns:
            Group name or None
        """
        groups = tdms_file.groups()
        
        # Try to find a group with channels (not Root)
        for group in groups:
            group_name = group.name
            if group_name and group_name.lower() != 'root':
                # Check if this group has channels
                if len(group.channels()) > 0:
                    print(f"  Found data group: '{group_name}' with {len(group.channels())} channels")
                    return group_name
        
        return None
    
    def read_tdms_file(self, tdms_file_path, channels_needed):
        """
        Read TDMS file and extract specific channels
        
        Args:
            tdms_file_path: Path to TDMS file
            channels_needed: List of channel names to extract
            
        Returns:
            DataFrame with selected TDMS data
        """
        try:
            tdms_file = TdmsFile.read(tdms_file_path)
            
            # Find the data group if not already set
            if self.group_name is None:
                self.group_name = self.find_data_group(tdms_file)
                if self.group_name is None:
                    print(f"  Error: No data group found in {os.path.basename(tdms_file_path)}")
                    return None
            
            group = tdms_file[self.group_name]
            
            data_dict = {}
            
            # Extract only the channels we need
            for channel_name in channels_needed:
                try:
                    channel = group[channel_name]
                    data_dict[channel_name] = channel[:]
                except KeyError:
                    print(f"  Warning: Channel '{channel_name}' not found in {os.path.basename(tdms_file_path)}")
            
            if not data_dict:
                return None
            
            df = pd.DataFrame(data_dict)
            
            # Convert Excel format datetime to readable format
            if 'Date/Time (Excel Format)' in df.columns:
                # Excel datetime: days since 1899-12-30
                df['DateTime'] = pd.to_datetime(
                    df['Date/Time (Excel Format)'], 
                    unit='D', 
                    origin='1899-12-30'
                )
            
            print(f"  Successfully read {os.path.basename(tdms_file_path)}: {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"  Error reading {tdms_file_path}: {e}")
            return None
    
    def load_all_relevant_tdms_data(self, channels_needed):
        """
        Load and combine all relevant TDMS files
        
        Args:
            channels_needed: List of channel names to extract
            
        Returns:
            Combined DataFrame with all TDMS data
        """
        csv_start = self.summary_df['DateTime'].min()
        csv_end = self.summary_df['DateTime'].max()
        
        # Get all TDMS files
        tdms_files = glob.glob(os.path.join(self.tdms_folder_path, '*.tdms'))
        print(f"\nFound {len(tdms_files)} TDMS files")
        
        all_dataframes = []
        processed_count = 0
        
        for tdms_file in tdms_files:
            tdms_datetime = self.parse_tdms_filename(tdms_file)
            
            if tdms_datetime:
                print(f"\nChecking {os.path.basename(tdms_file)} (Start: {tdms_datetime})")
            
            if self.is_tdms_file_in_range(tdms_datetime, csv_start, csv_end):
                print(f"  ✓ Within range - Reading...")
                df = self.read_tdms_file(tdms_file, channels_needed)
                
                if df is not None and 'DateTime' in df.columns:
                    all_dataframes.append(df)
                    processed_count += 1
            else:
                print(f"  ✗ Outside range - Skipping")
        
        print(f"\n{'='*60}")
        print(f"Processed {processed_count} TDMS files")
        print(f"{'='*60}\n")
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            # Sort by DateTime for efficient matching
            combined_df = combined_df.sort_values('DateTime').reset_index(drop=True)
            print(f"Combined TDMS data: {len(combined_df)} total rows")
            print(f"TDMS DateTime Range: {combined_df['DateTime'].min()} to {combined_df['DateTime'].max()}")
            return combined_df
        else:
            print("No TDMS data loaded!")
            return None
    
    def find_nearest_match(self, target_datetime, tdms_df, tolerance_seconds=60):
        """
        Find the nearest matching TDMS data point for a given datetime
        
        Args:
            target_datetime: DateTime to match
            tdms_df: DataFrame with TDMS data
            tolerance_seconds: Maximum time difference in seconds for a valid match
            
        Returns:
            Matched row data or None if no match within tolerance
        """
        # Calculate time differences
        time_diffs = abs((tdms_df['DateTime'] - target_datetime).dt.total_seconds())
        
        # Find the index of minimum difference
        min_idx = time_diffs.idxmin()
        min_diff = time_diffs[min_idx]
        
        # Check if within tolerance
        if min_diff <= tolerance_seconds:
            return tdms_df.loc[min_idx], min_diff
        else:
            return None, min_diff
    
    def match_and_add_multiple_tdms_data(self, channel_mappings, tolerance_seconds=60):
        """
        Match multiple TDMS channels to summary CSV and add as new columns
        
        Args:
            channel_mappings: List of tuples (tdms_channel_name, new_column_name)
            tolerance_seconds: Maximum time difference in seconds for a valid match
        """
        # Load summary CSV
        self.read_summary_csv()
        
        # Collect all channel names needed
        all_channels = ['Date/Time (Excel Format)'] + [ch[0] for ch in channel_mappings]
        
        # Load all relevant TDMS data
        self.tdms_data_combined = self.load_all_relevant_tdms_data(all_channels)
        
        if self.tdms_data_combined is None:
            print(f"Error: Could not load TDMS data")
            return
        
        # Process each channel mapping
        for tdms_channel_name, new_column_name in channel_mappings:
            if tdms_channel_name not in self.tdms_data_combined.columns:
                print(f"Warning: Channel '{tdms_channel_name}' not found in TDMS data, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Matching TDMS channel: {tdms_channel_name}")
            print(f"To new column: {new_column_name}")
            print(f"Tolerance: ±{tolerance_seconds} seconds")
            print(f"{'='*60}\n")
            
            matched_values = []
            match_time_diffs = []
            match_count = 0
            no_match_count = 0
            
            for idx, row in self.summary_df.iterrows():
                target_dt = row['DateTime']
                
                matched_row, time_diff = self.find_nearest_match(
                    target_dt, 
                    self.tdms_data_combined, 
                    tolerance_seconds
                )
                
                if matched_row is not None:
                    value = matched_row[tdms_channel_name]
                    matched_values.append(value)
                    match_time_diffs.append(time_diff)
                    match_count += 1
                    print(f"Row {idx+1}: {target_dt} → Matched (Δ={time_diff:.1f}s, Value={value:.2f})")
                else:
                    matched_values.append(np.nan)
                    match_time_diffs.append(time_diff)
                    no_match_count += 1
                    print(f"Row {idx+1}: {target_dt} → No match (closest: {time_diff:.1f}s away)")
            
            # Add new columns to summary DataFrame
            self.summary_df[new_column_name] = matched_values
            self.summary_df[f'{new_column_name}_TimeDiff_s'] = match_time_diffs
            
            print(f"\n{'='*60}")
            print(f"Matching Summary for {new_column_name}:")
            print(f"  Total rows: {len(self.summary_df)}")
            print(f"  Matched: {match_count}")
            print(f"  No match: {no_match_count}")
            print(f"  Match rate: {match_count/len(self.summary_df)*100:.1f}%")
            if match_count > 0:
                print(f"  Avg time difference: {np.mean([d for d in match_time_diffs if not np.isnan(d)]):.2f}s")
                print(f"  Max time difference: {max([d for d in match_time_diffs if not np.isnan(d)]):.2f}s")
            print(f"{'='*60}\n")
        
        return self.summary_df
    
    def save_updated_csv(self, output_path=None):
        """
        Save the updated summary CSV with new TDMS data
        
        Args:
            output_path: Path for output file (default: overwrite original)
        """
        if self.summary_df is None:
            print("No data to save!")
            return
        
        if output_path is None:
            output_path = self.csv_file_path
        
        # Save with UTF-8 encoding
        self.summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ Updated CSV saved to: {output_path}")
        print(f"  Total columns: {len(self.summary_df.columns)}")
        print(f"  Column names: {list(self.summary_df.columns)}")


class DataPlotterWindow:
    def __init__(self, parent, csv_path):
        self.window = tk.Toplevel(parent)
        self.window.title("Data Plotter")
        self.window.geometry("1400x900")
        
        self.csv_path = csv_path
        self.df = None
        self.load_data()
        
        self.setup_ui()
        
    def load_data(self):
        """Load the CSV data"""
        # Try multiple encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"✓ CSV loaded successfully with encoding: {encoding}")
                # Parse DateTime
                self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
                return
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                if encoding == encodings[-1]:  # Last encoding attempt
                    messagebox.showerror("Error", f"Could not load CSV file:\n{e}")
                    self.window.destroy()
                    return
    
    def setup_ui(self):
        """Setup the plotter UI with tabs"""
        # Create notebook (tab container)
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create two tabs
        self.controls_tab = ttk.Frame(self.notebook)
        self.plot_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.controls_tab, text="⚙ Controls & Settings")
        self.notebook.add(self.plot_tab, text="📊 Plot Preview")
        
        # Setup controls tab
        self.setup_controls_tab()
        
        # Setup plot tab
        self.setup_plot_tab()
        
    def setup_controls_tab(self):
        """Setup the controls tab with all settings"""
        # Make the controls scrollable
        canvas = tk.Canvas(self.controls_tab)
        scrollbar = ttk.Scrollbar(self.controls_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        control_frame = ttk.Frame(scrollable_frame, padding=10)
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(control_frame, text="Data Plotter Configuration", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Column selection frame
        selection_frame = tk.LabelFrame(control_frame, text="Select Columns to Plot", padx=10, pady=10)
        selection_frame.pack(fill=tk.X, pady=10)
        
        # Get numeric columns (exclude DateTime and string columns)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Left Y-axis selection
        left_frame = tk.Frame(selection_frame)
        left_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(left_frame, text="Left Y-Axis (Temperature)", font=("Arial", 10, "bold")).pack()
        
        self.left_vars = []
        self.left_checkboxes = []
        for col in numeric_cols:
            if 'temp' in col.lower() or 't1' in col.lower() or 't19' in col.lower() or '°c' in col.lower() or 'inlet' in col.lower() or 'degc' in col.lower():
                var = tk.BooleanVar(value=False)
                cb = tk.Checkbutton(left_frame, text=col, variable=var)
                cb.pack(anchor='w')
                self.left_vars.append((col, var))
                self.left_checkboxes.append(cb)
        
        # Right Y-axis selection
        right_frame = tk.Frame(selection_frame)
        right_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(right_frame, text="Right Y-Axis (Capacity/Other)", font=("Arial", 10, "bold")).pack()
        
        self.right_vars = []
        self.right_checkboxes = []
        for col in numeric_cols:
            if 'capacity' in col.lower() or 'ah' in col.lower() or 'discharge' in col.lower():
                var = tk.BooleanVar(value=False)
                cb = tk.Checkbutton(right_frame, text=col, variable=var)
                cb.pack(anchor='w')
                self.right_vars.append((col, var))
                self.right_checkboxes.append(cb)
        
        # Plot Controls Frame
        controls_frame = tk.LabelFrame(control_frame, text="Plot Controls", padx=10, pady=10)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Row 1: Font, Marker, Plot Size, X-axis Angle
        row1_frame = tk.Frame(controls_frame)
        row1_frame.pack(fill=tk.X, pady=5)
        
        # Font Size Control
        font_control_frame = tk.Frame(row1_frame)
        font_control_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(font_control_frame, text="Font Size:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.font_size_var = tk.IntVar(value=5)
        tk.Spinbox(font_control_frame, from_=3, to=24, textvariable=self.font_size_var, width=6).pack(side=tk.LEFT)
        
        # Marker Size Control
        marker_control_frame = tk.Frame(row1_frame)
        marker_control_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(marker_control_frame, text="Marker Size:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.marker_size_var = tk.IntVar(value=21)
        tk.Spinbox(marker_control_frame, from_=10, to=300, increment=5, textvariable=self.marker_size_var, width=6).pack(side=tk.LEFT)
        
        # Plot Size Control
        size_control_frame = tk.Frame(row1_frame)
        size_control_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(size_control_frame, text="Plot Size (W×H):", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.plot_width_var = tk.IntVar(value=6)
        tk.Spinbox(size_control_frame, from_=4, to=20, textvariable=self.plot_width_var, width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(size_control_frame, text="×", font=("Arial", 9)).pack(side=tk.LEFT)
        self.plot_height_var = tk.IntVar(value=4)
        tk.Spinbox(size_control_frame, from_=3, to=15, textvariable=self.plot_height_var, width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(size_control_frame, text="inches", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        
        # X-axis Tick Angle Control
        xangle_control_frame = tk.Frame(row1_frame)
        xangle_control_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(xangle_control_frame, text="X-Tick Angle:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.x_tick_angle_var = tk.IntVar(value=45)
        tk.Spinbox(xangle_control_frame, from_=0, to=90, increment=5, textvariable=self.x_tick_angle_var, width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(xangle_control_frame, text="°", font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Row 2: Legend Controls
        row2_frame = tk.Frame(controls_frame)
        row2_frame.pack(fill=tk.X, pady=5)
        
        # Legend Preset Position Control
        legend_preset_frame = tk.Frame(row2_frame)
        legend_preset_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(legend_preset_frame, text="Legend Preset:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.legend_position_var = tk.StringVar(value="top")
        legend_dropdown = ttk.Combobox(legend_preset_frame, textvariable=self.legend_position_var, 
                                      values=["custom", "bottom", "top", "right", "left", "upper right", "upper left", 
                                              "lower right", "lower left"], state='readonly', width=12)
        legend_dropdown.pack(side=tk.LEFT)
        legend_dropdown.current(2)  # Default to "top"
        legend_dropdown.bind('<<ComboboxSelected>>', self.on_legend_preset_change)
        
        # Advanced Legend Position (X, Y coordinates)
        legend_advanced_frame = tk.Frame(row2_frame)
        legend_advanced_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(legend_advanced_frame, text="Legend XY:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Label(legend_advanced_frame, text="X:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.legend_x_var = tk.StringVar(value="0.5")
        tk.Entry(legend_advanced_frame, textvariable=self.legend_x_var, width=5).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_advanced_frame, text="Y:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.legend_y_var = tk.StringVar(value="1.12")
        tk.Entry(legend_advanced_frame, textvariable=self.legend_y_var, width=5).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_advanced_frame, text="(0-1 range)", font=("Arial", 7), fg="gray").pack(side=tk.LEFT, padx=2)
        
        # Row 3: Axis Limits
        row3_frame = tk.Frame(controls_frame)
        row3_frame.pack(fill=tk.X, pady=5)
        
        # X-axis limits
        x_limit_frame = tk.Frame(row3_frame)
        x_limit_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(x_limit_frame, text="X-Axis (rows):", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Label(x_limit_frame, text="Min:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.x_min_var = tk.StringVar(value="")
        tk.Entry(x_limit_frame, textvariable=self.x_min_var, width=6).pack(side=tk.LEFT, padx=2)
        tk.Label(x_limit_frame, text="Max:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.x_max_var = tk.StringVar(value="")
        tk.Entry(x_limit_frame, textvariable=self.x_max_var, width=6).pack(side=tk.LEFT, padx=2)
        
        # Left Y-axis limits
        y1_limit_frame = tk.Frame(row3_frame)
        y1_limit_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(y1_limit_frame, text="Left Y-Axis:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Label(y1_limit_frame, text="Min:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.y1_min_var = tk.StringVar(value="")
        tk.Entry(y1_limit_frame, textvariable=self.y1_min_var, width=6).pack(side=tk.LEFT, padx=2)
        tk.Label(y1_limit_frame, text="Max:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.y1_max_var = tk.StringVar(value="")
        tk.Entry(y1_limit_frame, textvariable=self.y1_max_var, width=6).pack(side=tk.LEFT, padx=2)
        
        # Right Y-axis limits
        y2_limit_frame = tk.Frame(row3_frame)
        y2_limit_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(y2_limit_frame, text="Right Y-Axis:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Label(y2_limit_frame, text="Min:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.y2_min_var = tk.StringVar(value="")
        tk.Entry(y2_limit_frame, textvariable=self.y2_min_var, width=6).pack(side=tk.LEFT, padx=2)
        tk.Label(y2_limit_frame, text="Max:", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.y2_max_var = tk.StringVar(value="")
        tk.Entry(y2_limit_frame, textvariable=self.y2_max_var, width=6).pack(side=tk.LEFT, padx=2)
        
        # Button frame
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=20)
        
        # Row 1 of buttons
        button_row1 = tk.Frame(button_frame)
        button_row1.pack(pady=5)
        
        tk.Button(button_row1, text="Generate Plot", command=self.generate_plot, 
                 bg="#27ae60", fg="white", font=("Arial", 11, "bold"), width=15, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_row1, text="Save Plot", command=self.save_plot,
                 bg="#3498db", fg="white", font=("Arial", 11, "bold"), width=15, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_row1, text="Reset All", command=self.reset_all,
                 bg="#f39c12", fg="white", font=("Arial", 11, "bold"), width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        # Row 2 of buttons
        button_row2 = tk.Frame(button_frame)
        button_row2.pack(pady=5)
        
        tk.Button(button_row2, text="Save Configuration", command=self.save_configuration,
                 bg="#9b59b6", fg="white", font=("Arial", 11, "bold"), width=15, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_row2, text="Load Configuration", command=self.load_configuration,
                 bg="#8e44ad", fg="white", font=("Arial", 11, "bold"), width=15, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_row2, text="Close", command=self.window.destroy,
                 bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), width=15, height=2).pack(side=tk.LEFT, padx=5)
        
    def setup_plot_tab(self):
        """Setup the plot preview tab"""
        # Info label
        info_frame = tk.Frame(self.plot_tab, bg="#ecf0f1", height=40)
        info_frame.pack(fill=tk.X)
        info_frame.pack_propagate(False)
        
        tk.Label(info_frame, text="Configure settings in the 'Controls & Settings' tab, then click 'Generate Plot' to view here", 
                font=("Arial", 10), bg="#ecf0f1", fg="#34495e").pack(pady=10)
        
        # Plot frame with proper scaling
        self.plot_frame = tk.Frame(self.plot_tab, bg="white")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = None
        self.canvas = None
        
        # Initial placeholder
        self.placeholder = tk.Label(self.plot_frame, text="No plot generated yet\n\nGo to Controls & Settings tab and click 'Generate Plot'", 
                              font=("Arial", 14), fg="gray", bg="white")
        self.placeholder.pack(expand=True)
    
    def on_legend_preset_change(self, event=None):
        """Update legend X,Y values when preset changes"""
        preset = self.legend_position_var.get()
        
        preset_positions = {
            "top": ("0.5", "1.12"),
            "bottom": ("0.5", "-0.20"),
            "right": ("1.15", "0.5"),
            "left": ("-0.15", "0.5"),
            "upper right": ("0.98", "0.98"),
            "upper left": ("0.02", "0.98"),
            "lower right": ("0.98", "0.02"),
            "lower left": ("0.02", "0.02"),
            "custom": (self.legend_x_var.get(), self.legend_y_var.get())
        }
        
        if preset in preset_positions:
            x, y = preset_positions[preset]
            self.legend_x_var.set(x)
            self.legend_y_var.set(y)
    
    def save_configuration(self):
        """Save current plot configuration to a text file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration As",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Get selected columns
            left_cols = [col for col, var in self.left_vars if var.get()]
            right_cols = [col for col, var in self.right_vars if var.get()]
            
            # Create configuration text
            config_text = "=" * 70 + "\n"
            config_text += "DATA PLOTTER CONFIGURATION\n"
            config_text += f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            config_text += "=" * 70 + "\n\n"
            
            config_text += "DATA SOURCE:\n"
            config_text += "-" * 70 + "\n"
            config_text += f"CSV File: {os.path.basename(self.csv_path)}\n"
            config_text += f"Full Path: {self.csv_path}\n\n"
            
            config_text += "SELECTED COLUMNS:\n"
            config_text += "-" * 70 + "\n"
            config_text += f"Left Y-Axis (Temperature): {', '.join(left_cols) if left_cols else 'None'}\n"
            config_text += f"Right Y-Axis (Capacity): {', '.join(right_cols) if right_cols else 'None'}\n\n"
            
            config_text += "PLOT APPEARANCE:\n"
            config_text += "-" * 70 + "\n"
            config_text += f"Font Size: {self.font_size_var.get()}\n"
            config_text += f"Marker Size: {self.marker_size_var.get()}\n"
            config_text += f"Plot Size: {self.plot_width_var.get()}\" × {self.plot_height_var.get()}\"\n"
            config_text += f"X-Axis Tick Angle: {self.x_tick_angle_var.get()}°\n\n"
            
            config_text += "LEGEND SETTINGS:\n"
            config_text += "-" * 70 + "\n"
            config_text += f"Legend Preset: {self.legend_position_var.get()}\n"
            config_text += f"Legend X Position: {self.legend_x_var.get()}\n"
            config_text += f"Legend Y Position: {self.legend_y_var.get()}\n\n"
            
            config_text += "AXIS LIMITS:\n"
            config_text += "-" * 70 + "\n"
            config_text += f"X-Axis Min (rows): {self.x_min_var.get() if self.x_min_var.get() else 'Auto'}\n"
            config_text += f"X-Axis Max (rows): {self.x_max_var.get() if self.x_max_var.get() else 'Auto'}\n"
            config_text += f"Left Y-Axis Min: {self.y1_min_var.get() if self.y1_min_var.get() else 'Auto'}\n"
            config_text += f"Left Y-Axis Max: {self.y1_max_var.get() if self.y1_max_var.get() else 'Auto'}\n"
            config_text += f"Right Y-Axis Min: {self.y2_min_var.get() if self.y2_min_var.get() else 'Auto'}\n"
            config_text += f"Right Y-Axis Max: {self.y2_max_var.get() if self.y2_max_var.get() else 'Auto'}\n\n"
            
            config_text += "=" * 70 + "\n"
            config_text += "END OF CONFIGURATION\n"
            config_text += "=" * 70 + "\n"
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(config_text)
            
            messagebox.showinfo("Success", f"Configuration saved successfully to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save configuration:\n{e}")
    
    def load_configuration(self):
        """Load plot configuration from a text file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config_text = f.read()
            
            # Parse configuration file
            config = {}
            
            # Extract values using regex
            patterns = {
                'left_cols': r'Left Y-Axis \(Temperature\): (.+)',
                'right_cols': r'Right Y-Axis \(Capacity\): (.+)',
                'font_size': r'Font Size: (\d+)',
                'marker_size': r'Marker Size: (\d+)',
                'plot_width': r'Plot Size: (\d+)" × (\d+)"',
                'x_tick_angle': r'X-Axis Tick Angle: (\d+)°',
                'legend_preset': r'Legend Preset: (.+)',
                'legend_x': r'Legend X Position: (.+)',
                'legend_y': r'Legend Y Position: (.+)',
                'x_min': r'X-Axis Min \(rows\): (.+)',
                'x_max': r'X-Axis Max \(rows\): (.+)',
                'y1_min': r'Left Y-Axis Min: (.+)',
                'y1_max': r'Left Y-Axis Max: (.+)',
                'y2_min': r'Right Y-Axis Min: (.+)',
                'y2_max': r'Right Y-Axis Max: (.+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, config_text)
                if match:
                    if key == 'plot_width':
                        config['plot_width'] = match.group(1)
                        config['plot_height'] = match.group(2)
                    else:
                        config[key] = match.group(1).strip()
            
            # Apply configuration
            loaded_count = 0
            
            # Set column selections
            if 'left_cols' in config and config['left_cols'] != 'None':
                left_cols = [c.strip() for c in config['left_cols'].split(',')]
                for col, var in self.left_vars:
                    var.set(col in left_cols)
                    if col in left_cols:
                        loaded_count += 1
            
            if 'right_cols' in config and config['right_cols'] != 'None':
                right_cols = [c.strip() for c in config['right_cols'].split(',')]
                for col, var in self.right_vars:
                    var.set(col in right_cols)
                    if col in right_cols:
                        loaded_count += 1
            
            # Set plot appearance
            if 'font_size' in config:
                self.font_size_var.set(int(config['font_size']))
                loaded_count += 1
            
            if 'marker_size' in config:
                self.marker_size_var.set(int(config['marker_size']))
                loaded_count += 1
            
            if 'plot_width' in config:
                self.plot_width_var.set(int(config['plot_width']))
                loaded_count += 1
            
            if 'plot_height' in config:
                self.plot_height_var.set(int(config['plot_height']))
                loaded_count += 1
            
            if 'x_tick_angle' in config:
                self.x_tick_angle_var.set(int(config['x_tick_angle']))
                loaded_count += 1
            
            # Set legend settings
            if 'legend_preset' in config:
                self.legend_position_var.set(config['legend_preset'])
                loaded_count += 1
            
            if 'legend_x' in config:
                self.legend_x_var.set(config['legend_x'])
                loaded_count += 1
            
            if 'legend_y' in config:
                self.legend_y_var.set(config['legend_y'])
                loaded_count += 1
            
            # Set axis limits
            if 'x_min' in config:
                self.x_min_var.set("" if config['x_min'] == 'Auto' else config['x_min'])
                loaded_count += 1
            
            if 'x_max' in config:
                self.x_max_var.set("" if config['x_max'] == 'Auto' else config['x_max'])
                loaded_count += 1
            
            if 'y1_min' in config:
                self.y1_min_var.set("" if config['y1_min'] == 'Auto' else config['y1_min'])
                loaded_count += 1
            
            if 'y1_max' in config:
                self.y1_max_var.set("" if config['y1_max'] == 'Auto' else config['y1_max'])
                loaded_count += 1
            
            if 'y2_min' in config:
                self.y2_min_var.set("" if config['y2_min'] == 'Auto' else config['y2_min'])
                loaded_count += 1
            
            if 'y2_max' in config:
                self.y2_max_var.set("" if config['y2_max'] == 'Auto' else config['y2_max'])
                loaded_count += 1
            
            messagebox.showinfo("Success", f"Configuration loaded successfully!\n\n{loaded_count} settings applied.\n\nFrom: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load configuration:\n{e}")
    
    def reset_all(self):
        """Reset all controls to defaults"""
        self.x_min_var.set("")
        self.x_max_var.set("")
        self.y1_min_var.set("")
        self.y1_max_var.set("")
        self.y2_min_var.set("")
        self.y2_max_var.set("")
        self.font_size_var.set(5)
        self.marker_size_var.set(21)
        self.plot_width_var.set(6)
        self.plot_height_var.set(4)
        self.x_tick_angle_var.set(45)
        self.legend_position_var.set("top")
        self.legend_x_var.set("0.5")
        self.legend_y_var.set("1.12")
        
    def generate_plot(self):
        """Generate the scatter plot"""
        # Get selected columns
        left_cols = [col for col, var in self.left_vars if var.get()]
        right_cols = [col for col, var in self.right_vars if var.get()]
        
        if not left_cols and not right_cols:
            messagebox.showwarning("Warning", "Please select at least one column to plot!")
            return
        
        # Clear previous plot
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Clear placeholder if exists
        if hasattr(self, 'placeholder'):
            self.placeholder.destroy()
        
        # Get plot size from controls
        plot_width = self.plot_width_var.get()
        plot_height = self.plot_height_var.get()
        
        # Create figure with tight layout that fits in frame
        self.fig = Figure(figsize=(plot_width, plot_height), dpi=100, tight_layout=True)
        ax1 = self.fig.add_subplot(111)
        
        # Colors for different series
        temp_colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
        capacity_color = '#e74c3c'
        
        # Get marker size from control
        marker_size = self.marker_size_var.get()
        
        # Get base font size from control
        base_font = self.font_size_var.get()
        title_fontsize = base_font * 1.5
        label_fontsize = base_font * 1.3
        tick_fontsize = base_font
        legend_fontsize = base_font * 0.9
        
        # Get x-axis tick angle
        x_tick_angle = self.x_tick_angle_var.get()
        
        # Plot left Y-axis data (Temperature)
        for idx, col in enumerate(left_cols):
            color = temp_colors[idx % len(temp_colors)]
            ax1.scatter(self.df['DateTime'], self.df[col], 
                       marker='^', s=marker_size, alpha=0.7, 
                       color=color, edgecolors='black', linewidth=0.5,
                       label=col)
        
        ax1.set_xlabel('Time', fontsize=label_fontsize, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=label_fontsize, fontweight='bold', color='black')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis with custom angle
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=x_tick_angle, ha='right')
        
        # Apply X-axis limits
        try:
            x_min = int(self.x_min_var.get()) if self.x_min_var.get() else None
            x_max = int(self.x_max_var.get()) if self.x_max_var.get() else None
            
            if x_min is not None or x_max is not None:
                if x_min is None:
                    x_min = 0
                if x_max is None:
                    x_max = len(self.df) - 1
                
                x_min_date = self.df['DateTime'].iloc[min(max(0, x_min), len(self.df)-1)]
                x_max_date = self.df['DateTime'].iloc[min(max(0, x_max), len(self.df)-1)]
                ax1.set_xlim(x_min_date, x_max_date)
        except ValueError:
            pass
        
        # Apply Left Y-axis limits
        try:
            y1_min = float(self.y1_min_var.get()) if self.y1_min_var.get() else None
            y1_max = float(self.y1_max_var.get()) if self.y1_max_var.get() else None
            
            if y1_min is not None or y1_max is not None:
                current_ylim = ax1.get_ylim()
                if y1_min is None:
                    y1_min = current_ylim[0]
                if y1_max is None:
                    y1_max = current_ylim[1]
                ax1.set_ylim(y1_min, y1_max)
        except ValueError:
            pass
        
        # Plot right Y-axis data (Capacity)
        ax2 = None
        if right_cols:
            ax2 = ax1.twinx()
            for col in right_cols:
                ax2.scatter(self.df['DateTime'], self.df[col],
                           marker='o', s=marker_size, alpha=0.8,
                           color=capacity_color, edgecolors='darkred', linewidth=0.5,
                           label=col)
            
            ax2.set_ylabel('Discharge Capacity (Ah)', fontsize=label_fontsize, fontweight='bold', color=capacity_color)
            ax2.tick_params(axis='y', labelcolor=capacity_color, labelsize=tick_fontsize)
            
            # Apply Right Y-axis limits
            try:
                y2_min = float(self.y2_min_var.get()) if self.y2_min_var.get() else None
                y2_max = float(self.y2_max_var.get()) if self.y2_max_var.get() else None
                
                if y2_min is not None or y2_max is not None:
                    current_ylim = ax2.get_ylim()
                    if y2_min is None:
                        y2_min = current_ylim[0]
                    if y2_max is None:
                        y2_max = current_ylim[1]
                    ax2.set_ylim(y2_min, y2_max)
            except ValueError:
                pass
        
        # Title - MANUAL CHANGE: Split into two lines with \n
        title = f"Temperature-Capacity Correlation \n {os.path.basename(self.csv_path)}"
        ax1.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        
        # Get legend position
        legend_preset = self.legend_position_var.get()
        
        try:
            legend_x = float(self.legend_x_var.get())
            legend_y = float(self.legend_y_var.get())
        except ValueError:
            legend_x = 0.5
            legend_y = 1.12
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if right_cols and ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      loc='center', 
                      bbox_to_anchor=(legend_x, legend_y),
                      ncol=3,
                      frameon=True, 
                      shadow=True, 
                      fontsize=legend_fontsize,
                      fancybox=True)
        else:
            ax1.legend(loc='center', 
                      bbox_to_anchor=(legend_x, legend_y),
                      ncol=3,
                      frameon=True, 
                      shadow=True, 
                      fontsize=legend_fontsize,
                      fancybox=True)
        
        # Embed in tkinter with proper scaling
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Switch to plot tab to show the result
        self.notebook.select(self.plot_tab)
        
    def save_plot(self):
        """Save the plot to file"""
        if self.fig is None:
            messagebox.showwarning("Warning", "Please generate a plot first!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Plot As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            plot_width = self.plot_width_var.get()
            plot_height = self.plot_height_var.get()
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to:\n{filename}\n\nSize: {plot_width}\" × {plot_height}\"\nDPI: 300")


class TDMSMatcherGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TDMS Data Matcher")
        self.root.geometry("750x850")
        self.root.resizable(True, True)
        
        # Variables
        self.csv_path = tk.StringVar()
        self.tdms_folder = tk.StringVar()
        self.tolerance = tk.IntVar(value=1)
        self.output_path = tk.StringVar()
        
        self.available_channels = []
        self.group_name = None
        self.matcher = None
        
        # Channel mappings list: [(tdms_channel, column_name), ...]
        self.channel_mappings = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="TDMS Data Matcher", 
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Main content frame with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_main_frame = tk.Frame(main_canvas)
        
        scrollable_main_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_main_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_main_frame
        
        # CSV File Selection
        csv_frame = tk.LabelFrame(main_frame, text="1. Select Summary CSV File", font=("Arial", 10, "bold"), padx=10, pady=10)
        csv_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Entry(csv_frame, textvariable=self.csv_path, width=60, state='readonly').pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(csv_frame, text="Browse...", command=self.browse_csv, width=12).pack(side=tk.LEFT)
        
        # TDMS Folder Selection
        tdms_frame = tk.LabelFrame(main_frame, text="2. Select TDMS Folder", font=("Arial", 10, "bold"), padx=10, pady=10)
        tdms_frame.pack(fill=tk.X, pady=(0, 10))
        
        folder_entry_frame = tk.Frame(tdms_frame)
        folder_entry_frame.pack(fill=tk.X)
        
        tk.Entry(folder_entry_frame, textvariable=self.tdms_folder, width=60, state='readonly').pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(folder_entry_frame, text="Browse...", command=self.browse_folder, width=12).pack(side=tk.LEFT)
        
        # Load channels button
        self.load_channels_button = tk.Button(
            tdms_frame, 
            text="⟳ Load Available Channels", 
            command=self.load_channels,
            state=tk.DISABLED,
            bg="#3498db",
            fg="white",
            font=("Arial", 9, "bold")
        )
        self.load_channels_button.pack(pady=(10, 0))
        
        # Configuration Frame
        config_frame = tk.LabelFrame(main_frame, text="3. Configuration - TDMS Channels", font=("Arial", 10, "bold"), padx=10, pady=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Channel mappings container
        self.channel_container = tk.Frame(config_frame)
        self.channel_container.pack(fill=tk.X, pady=5)
        
        # Add channel button
        add_channel_btn = tk.Button(
            config_frame,
            text="➕ Add TDMS Channel",
            command=self.add_channel_row,
            bg="#27ae60",
            fg="white",
            font=("Arial", 9, "bold"),
            state=tk.DISABLED
        )
        add_channel_btn.pack(pady=5)
        self.add_channel_button = add_channel_btn
        
        # Tolerance (Universal)
        tolerance_frame = tk.Frame(config_frame)
        tolerance_frame.pack(fill=tk.X, pady=10)
        tk.Label(tolerance_frame, text="Tolerance (seconds) - applies to all channels:", anchor='w', font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Spinbox(tolerance_frame, from_=1, to=300, textvariable=self.tolerance, width=10).pack(side=tk.LEFT)
        tk.Label(tolerance_frame, text="(±seconds for matching)").pack(side=tk.LEFT, padx=5)
        
        # Output File Selection
        output_frame = tk.LabelFrame(main_frame, text="4. Output Location (Optional - leave blank to overwrite)", font=("Arial", 10, "bold"), padx=10, pady=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Entry(output_frame, textvariable=self.output_path, width=60).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(output_frame, text="Browse...", command=self.browse_output, width=12).pack(side=tk.LEFT)
        
        # Progress Frame
        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = tk.Label(progress_frame, text="Ready", fg="blue")
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X)
        
        # Button Frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.process_button = tk.Button(
            button_frame, 
            text="Process Data", 
            command=self.process_data,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            cursor="hand2"
        )
        self.process_button.grid(row=0, column=0, padx=5)
        
        self.plot_button = tk.Button(
            button_frame, 
            text="📊 Plot Data", 
            command=self.open_plotter,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.plot_button.grid(row=0, column=1, padx=5)
        
        # Status text
        status_frame = tk.LabelFrame(main_frame, text="Status", font=("Arial", 10, "bold"))
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = tk.Text(status_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        status_scrollbar = tk.Scrollbar(self.status_text)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=status_scrollbar.set)
        status_scrollbar.config(command=self.status_text.yview)
    
    def add_channel_row(self):
        """Add a new channel mapping row"""
        row_frame = tk.Frame(self.channel_container, relief=tk.RIDGE, borderwidth=2, padx=5, pady=5)
        row_frame.pack(fill=tk.X, pady=5)
        
        # Row number label
        row_num = len(self.channel_mappings) + 1
        tk.Label(row_frame, text=f"#{row_num}", font=("Arial", 9, "bold"), width=3).grid(row=0, column=0, padx=5)
        
        # TDMS Channel dropdown
        tk.Label(row_frame, text="TDMS Channel:", anchor='w').grid(row=0, column=1, sticky='w', padx=5)
        channel_var = tk.StringVar()
        channel_dropdown = ttk.Combobox(
            row_frame, 
            textvariable=channel_var, 
            width=35,
            state='readonly',
            values=self.available_channels
        )
        channel_dropdown.grid(row=0, column=2, sticky='w', padx=5)
        
        # Auto-set default if available
        if self.available_channels:
            default_channel = None
            for ch in self.available_channels:
                if 'T19' in ch or '9211_6 TC2' in ch:
                    default_channel = ch
                    break
            if default_channel:
                channel_dropdown.set(default_channel)
            else:
                channel_dropdown.current(0)
        
        # New Column Name
        tk.Label(row_frame, text="New Column Name:", anchor='w').grid(row=0, column=3, sticky='w', padx=5)
        column_name_var = tk.StringVar(value=f"TDMS_Channel_{row_num}")
        tk.Entry(row_frame, textvariable=column_name_var, width=25).grid(row=0, column=4, sticky='w', padx=5)
        
        # Remove button
        remove_btn = tk.Button(
            row_frame,
            text="✕",
            command=lambda: self.remove_channel_row(row_frame, channel_var, column_name_var),
            bg="#e74c3c",
            fg="white",
            font=("Arial", 9, "bold"),
            width=3
        )
        remove_btn.grid(row=0, column=5, padx=5)
        
        # Store the mapping
        self.channel_mappings.append({
            'frame': row_frame,
            'channel_var': channel_var,
            'column_name_var': column_name_var
        })
        
        self.log_status(f"Added channel mapping row #{row_num}")
    
    def remove_channel_row(self, frame, channel_var, column_name_var):
        """Remove a channel mapping row"""
        # Find and remove from list
        for mapping in self.channel_mappings:
            if mapping['frame'] == frame:
                self.channel_mappings.remove(mapping)
                break
        
        # Destroy the frame
        frame.destroy()
        
        # Renumber remaining rows
        for idx, mapping in enumerate(self.channel_mappings):
            # Update row number label
            for widget in mapping['frame'].winfo_children():
                if isinstance(widget, tk.Label) and widget.cget("text").startswith("#"):
                    widget.config(text=f"#{idx+1}")
                    break
        
        self.log_status(f"Removed channel mapping row")
        
    def browse_csv(self):
        """Open file dialog to select CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Summary CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path.set(filename)
            self.log_status(f"CSV selected: {os.path.basename(filename)}")
            self.plot_button.config(state=tk.NORMAL)
            
    def browse_folder(self):
        """Open folder dialog to select TDMS folder"""
        folder = filedialog.askdirectory(
            title="Select TDMS Folder"
        )
        if folder:
            self.tdms_folder.set(folder)
            tdms_count = len(glob.glob(os.path.join(folder, "*.tdms")))
            self.log_status(f"TDMS folder selected: {tdms_count} TDMS files found")
            self.load_channels_button.config(state=tk.NORMAL)
            self.log_status("Click 'Load Available Channels' to read channel list")
            
    def load_channels(self):
        """Load channel names from a TDMS file"""
        tdms_files = glob.glob(os.path.join(self.tdms_folder.get(), "*.tdms"))
        
        if not tdms_files:
            messagebox.showerror("Error", "No TDMS files found in the selected folder!")
            return
        
        self.log_status(f"Reading channels from {os.path.basename(tdms_files[0])}...")
        
        try:
            tdms_file = TdmsFile.read(tdms_files[0])
            groups = tdms_file.groups()
            self.log_status(f"Found {len(groups)} group(s) in TDMS file")
            
            data_group = None
            for group in groups:
                group_name = group.name
                self.log_status(f"  Group: '{group_name}' ({len(group.channels())} channels)")
                
                if group_name and group_name.lower() != 'root' and len(group.channels()) > 0:
                    data_group = group
                    self.group_name = group_name
                    break
            
            if data_group is None:
                messagebox.showerror("Error", "No data group with channels found in TDMS file!")
                return
            
            all_channels = [channel.name for channel in data_group.channels()]
            self.available_channels = [ch for ch in all_channels if 'Date/Time' not in ch]
            
            if self.available_channels:
                self.add_channel_button.config(state=tk.NORMAL)
                
                self.log_status(f"✓ Found {len(self.available_channels)} data channels in group '{self.group_name}'")
                messagebox.showinfo(
                    "Success", 
                    f"Loaded {len(self.available_channels)} channels from TDMS file!\n\nGroup: '{self.group_name}'\n\nClick '➕ Add TDMS Channel' to add channel mappings."
                )
            else:
                messagebox.showerror("Error", "No data channels found in TDMS file!")
                
        except Exception as e:
            self.log_status(f"✗ Error loading channels: {str(e)}")
            messagebox.showerror("Error", f"Could not read TDMS file:\n\n{str(e)}")
    
    def browse_output(self):
        """Open file dialog to select output location"""
        filename = filedialog.asksaveasfilename(
            title="Save Output CSV As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
            self.log_status(f"Output location: {os.path.basename(filename)}")
    
    def log_status(self, message):
        """Add message to status text box"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def validate_inputs(self):
        """Validate user inputs"""
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please select a CSV file!")
            return False
        
        if not self.tdms_folder.get():
            messagebox.showerror("Error", "Please select a TDMS folder!")
            return False
        
        if not os.path.exists(self.csv_path.get()):
            messagebox.showerror("Error", "CSV file does not exist!")
            return False
        
        if not os.path.exists(self.tdms_folder.get()):
            messagebox.showerror("Error", "TDMS folder does not exist!")
            return False
        
        if not self.channel_mappings:
            messagebox.showerror("Error", "Please add at least one TDMS channel mapping!\n\nClick '➕ Add TDMS Channel' first.")
            return False
        
        # Validate each mapping
        for idx, mapping in enumerate(self.channel_mappings):
            channel = mapping['channel_var'].get()
            col_name = mapping['column_name_var'].get()
            
            if not channel:
                messagebox.showerror("Error", f"Channel mapping #{idx+1}: Please select a TDMS channel!")
                return False
            
            if not col_name:
                messagebox.showerror("Error", f"Channel mapping #{idx+1}: Please enter a column name!")
                return False
        
        return True
    
    def process_data_thread(self):
        """Process data in separate thread to prevent GUI freezing"""
        try:
            self.progress_bar.start(10)
            self.process_button.config(state=tk.DISABLED)
            self.progress_label.config(text="Processing...", fg="orange")
            
            self.log_status("\n" + "="*60)
            self.log_status("Starting data processing...")
            self.log_status("="*60)
            
            self.matcher = TDMSMatcher(
                self.csv_path.get(),
                self.tdms_folder.get()
            )
            
            self.matcher.group_name = self.group_name
            
            # Prepare channel mappings
            mappings = [
                (mapping['channel_var'].get(), mapping['column_name_var'].get())
                for mapping in self.channel_mappings
            ]
            
            self.log_status(f"Processing {len(mappings)} TDMS channels:")
            for tdms_ch, col_name in mappings:
                self.log_status(f"  • {tdms_ch} → {col_name}")
            self.log_status(f"Tolerance: ±{self.tolerance.get()} seconds")
            
            updated_df = self.matcher.match_and_add_multiple_tdms_data(
                channel_mappings=mappings,
                tolerance_seconds=self.tolerance.get()
            )
            
            output = self.output_path.get() if self.output_path.get() else None
            self.matcher.save_updated_csv(output_path=output)
            
            if output:
                self.csv_path.set(output)
            
            self.log_status("="*60)
            self.log_status("✓ Processing completed successfully!")
            self.log_status("="*60)
            
            self.progress_label.config(text="Completed!", fg="green")
            
            final_path = output if output else self.csv_path.get()
            messagebox.showinfo(
                "Success", 
                f"Data processed successfully!\n\n{len(mappings)} TDMS channels added to CSV.\n\nOutput saved to:\n{final_path}\n\nYou can now click 'Plot Data' to visualize!"
            )
            
            self.plot_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log_status(f"\n✗ Error: {str(e)}")
            self.progress_label.config(text="Error occurred!", fg="red")
            messagebox.showerror("Error", f"An error occurred:\n\n{str(e)}")
        
        finally:
            self.progress_bar.stop()
            self.process_button.config(state=tk.NORMAL)
    
    def process_data(self):
        """Validate and start processing"""
        if not self.validate_inputs():
            return
        
        self.status_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.process_data_thread)
        thread.daemon = True
        thread.start()
    
    def open_plotter(self):
        """Open the data plotter window"""
        if not self.csv_path.get() or not os.path.exists(self.csv_path.get()):
            messagebox.showerror("Error", "Please select a valid CSV file first!")
            return
        
        DataPlotterWindow(self.root, self.csv_path.get())
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


# Run the GUI application
if __name__ == "__main__":
    app = TDMSMatcherGUI()
    app.run()
