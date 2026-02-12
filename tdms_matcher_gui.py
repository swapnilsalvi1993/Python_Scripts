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
    
    def match_and_add_tdms_data(self, tdms_channel_name, new_column_name=None, tolerance_seconds=60):
        """
        Match TDMS data to summary CSV and add as new column
        
        Args:
            tdms_channel_name: Name of the TDMS channel to extract (e.g., '9211_6 TC2 T19 (C)')
            new_column_name: Name for the new column in summary CSV (default: same as channel name)
            tolerance_seconds: Maximum time difference in seconds for a valid match
        """
        if new_column_name is None:
            new_column_name = tdms_channel_name
        
        # Load summary CSV
        self.read_summary_csv()
        
        # Load all relevant TDMS data
        channels_needed = ['Date/Time (Excel Format)', tdms_channel_name]
        self.tdms_data_combined = self.load_all_relevant_tdms_data(channels_needed)
        
        if self.tdms_data_combined is None or tdms_channel_name not in self.tdms_data_combined.columns:
            print(f"Error: Could not load TDMS data for channel '{tdms_channel_name}'")
            return
        
        # Match each row in summary CSV
        print(f"\n{'='*60}")
        print(f"Matching TDMS data to Summary CSV...")
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
                print(f"Row {idx+1}: {target_dt} → Matched (Δ={time_diff:.1f}s, Value={value:.2f}°C)")
            else:
                matched_values.append(np.nan)
                match_time_diffs.append(time_diff)
                no_match_count += 1
                print(f"Row {idx+1}: {target_dt} → No match (closest: {time_diff:.1f}s away)")
        
        # Add new columns to summary DataFrame
        self.summary_df[new_column_name] = matched_values
        self.summary_df[f'{new_column_name}_TimeDiff_s'] = match_time_diffs
        
        print(f"\n{'='*60}")
        print(f"Matching Summary:")
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
        self.window.geometry("1400x1050")
        
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
        """Setup the plotter UI"""
        # Control Panel
        control_frame = tk.Frame(self.window, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(control_frame, text="Data Plotter", font=("Arial", 14, "bold")).pack()
        
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
        
        # Row 1: Font, Marker, Plot Size
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
        
        # X-axis limits (using index-based for simplicity)
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
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Generate Plot", command=self.generate_plot, 
                 bg="#27ae60", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Plot", command=self.save_plot,
                 bg="#3498db", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset All", command=self.reset_all,
                 bg="#f39c12", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=self.window.destroy,
                 bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=5)
        
        # Plot frame
        self.plot_frame = tk.Frame(self.window)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = None
        self.canvas = None
    
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
        
        # Get plot size from controls
        plot_width = self.plot_width_var.get()
        plot_height = self.plot_height_var.get()
        
        # Create figure with specified size and DPI
        self.fig, ax1 = plt.subplots(figsize=(plot_width, plot_height), dpi=300)
        
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
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Apply X-axis limits (based on row index)
        try:
            x_min = int(self.x_min_var.get()) if self.x_min_var.get() else None
            x_max = int(self.x_max_var.get()) if self.x_max_var.get() else None
            
            if x_min is not None or x_max is not None:
                if x_min is None:
                    x_min = 0
                if x_max is None:
                    x_max = len(self.df) - 1
                
                # Convert row indices to datetime
                x_min_date = self.df['DateTime'].iloc[min(max(0, x_min), len(self.df)-1)]
                x_max_date = self.df['DateTime'].iloc[min(max(0, x_max), len(self.df)-1)]
                ax1.set_xlim(x_min_date, x_max_date)
        except ValueError:
            pass  # Invalid input, use auto limits
        
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
            pass  # Invalid input, use auto limits
        
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
                pass  # Invalid input, use auto limits
        
        # Title
        title = f"Temperature-Capacity Correlation - {os.path.basename(self.csv_path)}"
        plt.title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        
        # Get legend position (use custom X,Y values)
        legend_preset = self.legend_position_var.get()
        
        try:
            legend_x = float(self.legend_x_var.get())
            legend_y = float(self.legend_y_var.get())
        except ValueError:
            legend_x = 0.5
            legend_y = 1.12
        
        # Combine legends with custom position
        lines1, labels1 = ax1.get_legend_handles_labels()
        if right_cols and ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            
            if legend_preset == "custom" or legend_preset not in ["bottom", "top", "right", "left"]:
                # Use custom X,Y coordinates
                ax1.legend(lines1 + lines2, labels1 + labels2, 
                          loc='center', 
                          bbox_to_anchor=(legend_x, legend_y),
                          ncol=3,
                          frameon=True, 
                          shadow=True, 
                          fontsize=legend_fontsize,
                          fancybox=True)
            else:
                # Use preset position
                if legend_preset == "bottom":
                    ax1.legend(lines1 + lines2, labels1 + labels2, 
                              loc='upper center', 
                              bbox_to_anchor=(legend_x, legend_y),
                              ncol=3,
                              frameon=True, 
                              shadow=True, 
                              fontsize=legend_fontsize,
                              fancybox=True)
                    plt.subplots_adjust(bottom=0.20, top=0.93, left=0.10, right=0.90)
                elif legend_preset == "top":
                    ax1.legend(lines1 + lines2, labels1 + labels2, 
                              loc='upper center', 
                              bbox_to_anchor=(legend_x, legend_y),
                              ncol=3,
                              frameon=True, 
                              shadow=True, 
                              fontsize=legend_fontsize,
                              fancybox=True)
                    plt.subplots_adjust(bottom=0.12, top=0.85, left=0.10, right=0.90)
                elif legend_preset == "right":
                    ax1.legend(lines1 + lines2, labels1 + labels2, 
                              loc='center left', 
                              bbox_to_anchor=(legend_x, legend_y),
                              frameon=True, 
                              shadow=True, 
                              fontsize=legend_fontsize,
                              fancybox=True)
                    plt.subplots_adjust(bottom=0.12, top=0.93, left=0.10, right=0.80)
                elif legend_preset == "left":
                    ax1.legend(lines1 + lines2, labels1 + labels2, 
                              loc='center right', 
                              bbox_to_anchor=(legend_x, legend_y),
                              frameon=True, 
                              shadow=True, 
                              fontsize=legend_fontsize,
                              fancybox=True)
                    plt.subplots_adjust(bottom=0.12, top=0.93, left=0.25, right=0.90)
                else:
                    # For built-in positions
                    ax1.legend(lines1 + lines2, labels1 + labels2, 
                              loc=legend_preset,
                              frameon=True, 
                              shadow=True, 
                              fontsize=legend_fontsize,
                              fancybox=True)
                    plt.subplots_adjust(bottom=0.12, top=0.93, left=0.10, right=0.90)
        else:
            ax1.legend(loc='center', 
                      bbox_to_anchor=(legend_x, legend_y),
                      ncol=3,
                      frameon=True, 
                      shadow=True, 
                      fontsize=legend_fontsize,
                      fancybox=True)
            plt.subplots_adjust(bottom=0.12, top=0.93, left=0.10, right=0.90)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
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
            # Save with current size at 300 DPI
            plot_width = self.plot_width_var.get()
            plot_height = self.plot_height_var.get()
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to:\n{filename}\n\nSize: {plot_width}\" × {plot_height}\"\nDPI: 300")


class TDMSMatcherGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TDMS Data Matcher")
        self.root.geometry("700x750")
        self.root.resizable(False, False)
        
        # Variables
        self.csv_path = tk.StringVar()
        self.tdms_folder = tk.StringVar()
        self.tdms_channel = tk.StringVar()
        self.new_column_name = tk.StringVar(value="Inlet Temperature (degC)")
        self.tolerance = tk.IntVar(value=1)
        self.output_path = tk.StringVar()
        
        self.available_channels = []
        self.group_name = None
        self.matcher = None
        
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
        
        # Main content frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
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
        config_frame = tk.LabelFrame(main_frame, text="3. Configuration", font=("Arial", 10, "bold"), padx=10, pady=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # TDMS Channel Dropdown
        tk.Label(config_frame, text="TDMS Channel:", anchor='w').grid(row=0, column=0, sticky='w', pady=5)
        channel_frame = tk.Frame(config_frame)
        channel_frame.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        self.channel_dropdown = ttk.Combobox(
            channel_frame, 
            textvariable=self.tdms_channel, 
            width=45,
            state='readonly'
        )
        self.channel_dropdown.pack(side=tk.LEFT)
        self.channel_dropdown['values'] = ["Select TDMS folder first..."]
        self.channel_dropdown.current(0)
        
        # New Column Name
        tk.Label(config_frame, text="New Column Name:", anchor='w').grid(row=1, column=0, sticky='w', pady=5)
        tk.Entry(config_frame, textvariable=self.new_column_name, width=47).grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Tolerance
        tk.Label(config_frame, text="Tolerance (seconds):", anchor='w').grid(row=2, column=0, sticky='w', pady=5)
        tolerance_frame = tk.Frame(config_frame)
        tolerance_frame.grid(row=2, column=1, sticky='w', padx=10, pady=5)
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
        
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
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
            
            self.available_channels = [channel.name for channel in data_group.channels()]
            display_channels = [ch for ch in self.available_channels if 'Date/Time' not in ch]
            
            if display_channels:
                self.channel_dropdown['values'] = display_channels
                
                default_channel = None
                for ch in display_channels:
                    if 'T19' in ch or '9211_6 TC2' in ch:
                        default_channel = ch
                        break
                
                if default_channel:
                    self.channel_dropdown.set(default_channel)
                    self.new_column_name.set("Inlet Temperature (degC)")
                else:
                    self.channel_dropdown.current(0)
                
                self.log_status(f"✓ Found {len(display_channels)} data channels in group '{self.group_name}'")
                messagebox.showinfo(
                    "Success", 
                    f"Loaded {len(display_channels)} channels from TDMS file!\n\nGroup: '{self.group_name}'\n\nSelect a channel from the dropdown."
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
        
        if not self.tdms_channel.get() or self.tdms_channel.get() == "Select TDMS folder first...":
            messagebox.showerror("Error", "Please select a TDMS channel!\n\nClick 'Load Available Channels' first.")
            return False
        
        if not self.new_column_name.get():
            messagebox.showerror("Error", "Please enter a new column name!")
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
            
            self.log_status(f"Channel: {self.tdms_channel.get()}")
            self.log_status(f"Tolerance: ±{self.tolerance.get()} seconds")
            
            updated_df = self.matcher.match_and_add_tdms_data(
                tdms_channel_name=self.tdms_channel.get(),
                new_column_name=self.new_column_name.get(),
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
                f"Data processed successfully!\n\nOutput saved to:\n{final_path}\n\nYou can now click 'Plot Data' to visualize!"
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
