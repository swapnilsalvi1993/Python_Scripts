from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd


EXCEL_EPOCH = "1899-12-30"  # origin for Excel serial dates (Windows Excel)
NI_TIME_SUBSTRING = "Date/Time (Excel Format)"


# -------------------------
# PEC parsing helpers
# -------------------------

def find_pec_header_row(lines: list[str], required_cols: tuple[str, ...] = ("Total Time (Seconds)",)) -> int:
    """
    Find the row index (0-based) in the raw file where the actual PEC table header begins.
    Detect a tab- or comma-separated header line containing required columns.
    """
    for i, raw in enumerate(lines):
        tokens = [t.strip() for t in re.split(r"[\t,]", raw.rstrip("\n\r")) if t.strip() != ""]
        if not tokens:
            continue

        token_set = set(tokens)
        if all(col in token_set for col in required_cols) and len(tokens) >= 10:
            return i

    raise ValueError("Could not locate PEC data header row (line containing 'Total Time (Seconds)').")


def parse_pec_start_time(lines: list[str]) -> datetime:
    """
    Extract Start Time from PEC metadata area:
      Start Time:\t3/3/2026 22:43
    """
    pat = re.compile(r"^\s*Start Time:\s*[\t,]\s*(.+?)\s*$", re.IGNORECASE)

    for raw in lines[:250]:
        m = pat.match(raw.rstrip("\n\r"))
        if not m:
            continue

        value = m.group(1).strip()
        fmts = [
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%y %H:%M:%S",
            "%m/%d/%y %H:%M",
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %I:%M %p",
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass

        # last resort: pandas parser
        try:
            dt = pd.to_datetime(value, errors="raise")
            return dt.to_pydatetime()
        except Exception as e:
            raise ValueError(f"Found Start Time but couldn't parse it: {value!r}") from e

    raise ValueError("Could not find 'Start Time:' in PEC file metadata.")


def read_pec_file(path: str | Path) -> pd.DataFrame:
    """
    Read PEC file: detect header row, parse data table, add pec_timestamp.
    """
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines(True)

    start_time = parse_pec_start_time(lines)
    header_row = find_pec_header_row(lines)

    header_line = lines[header_row]
    sep = "\t" if "\t" in header_line else ","

    data_text = "".join(lines[header_row:])
    df = pd.read_csv(io.StringIO(data_text), sep=sep)

    if "Total Time (Seconds)" not in df.columns:
        raise ValueError("PEC data table does not include 'Total Time (Seconds)' column after parsing.")

    df["Total Time (Seconds)"] = pd.to_numeric(df["Total Time (Seconds)"], errors="coerce")
    df["pec_timestamp"] = pd.to_datetime(start_time) + pd.to_timedelta(df["Total Time (Seconds)"], unit="s")

    df = df.sort_values("pec_timestamp").reset_index(drop=True)
    return df


# -------------------------
# NI DAQ parsing helpers
# -------------------------

def detect_ni_excel_datetime_column(columns: list[str]) -> str:
    """
    Detect NI DAQ Excel date/time column by substring match, regardless of prefix like 'Untitled/'.
    If multiple columns match, prefer the shortest name (often the 'cleanest').
    """
    matches = [c for c in columns if NI_TIME_SUBSTRING.lower() in str(c).lower()]
    if not matches:
        raise ValueError(
            f"Could not find NI DAQ time column containing substring: {NI_TIME_SUBSTRING!r}. "
            f"Available columns (first 20): {columns[:20]}"
        )
    # choose the shortest (usually least prefixed) then stable sort
    matches.sort(key=lambda s: (len(s), s))
    return matches[0]


def read_ni_daq_file(path: str | Path) -> pd.DataFrame:
    """
    Read NI DAQ file and add ni_timestamp from Excel serial date column.
    Auto-detect delimiter (tab vs comma).
    """
    path = Path(path)
    first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(path, sep=sep)

    col = detect_ni_excel_datetime_column(list(df.columns))

    df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ni_timestamp"] = pd.to_datetime(df[col], unit="D", origin=EXCEL_EPOCH, errors="coerce")

    df = df.sort_values("ni_timestamp").reset_index(drop=True)
    return df


# -------------------------
# Merge
# -------------------------

def merge_files(
    ni_path: str | Path,
    pec_path: str | Path,
    tolerance_seconds: float = 0.5,
    direction: str = "nearest",  # 'nearest', 'backward', 'forward'
) -> pd.DataFrame:
    ni_df = read_ni_daq_file(ni_path)
    pec_df = read_pec_file(pec_path)

    merged = pd.merge_asof(
        ni_df,
        pec_df,
        left_on="ni_timestamp",
        right_on="pec_timestamp",
        direction=direction,
        tolerance=pd.Timedelta(seconds=float(tolerance_seconds)),
        suffixes=("_ni", "_pec"),
    )

    merged["match_delta_seconds"] = (merged["ni_timestamp"] - merged["pec_timestamp"]).dt.total_seconds()
    return merged


# -------------------------
# Tkinter GUI
# -------------------------

@dataclass
class AppState:
    ni_path: str | None = None
    pec_path: str | None = None
    output_path: str | None = None


class MergeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NI DAQ + PEC Cycler CSV Merger")
        self.geometry("900x420")

        self.state_data = AppState()
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        self.ni_var = tk.StringVar(value="")
        self.pec_var = tk.StringVar(value="")
        self.out_var = tk.StringVar(value="")

        row = 0
        ttk.Label(frm, text="NI DAQ CSV:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.ni_var, width=90).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="Browse...", command=self.pick_ni).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(frm, text="PEC Cycler CSV:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.pec_var, width=90).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="Browse...", command=self.pick_pec).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(frm, text="Output CSV:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_var, width=90).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="Save As...", command=self.pick_out).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Separator(frm).grid(row=row, column=0, columnspan=3, sticky="we", pady=10)

        row += 1
        opt = ttk.Frame(frm)
        opt.grid(row=row, column=0, columnspan=3, sticky="we")

        self.tol_var = tk.StringVar(value="0.5")
        self.dir_var = tk.StringVar(value="nearest")

        ttk.Label(opt, text="Tolerance (seconds):").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(opt, textvariable=self.tol_var, width=10).grid(row=0, column=1, sticky="w", padx=(0, 18))

        ttk.Label(opt, text="Merge direction:").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Combobox(
            opt,
            textvariable=self.dir_var,
            values=["nearest", "backward", "forward"],
            width=10,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=(0, 18))

        ttk.Label(opt, text=f"NI time column detection: contains '{NI_TIME_SUBSTRING}'").grid(
            row=0, column=4, sticky="w"
        )

        row += 1
        self.run_btn = ttk.Button(frm, text="Merge and Save", command=self.run_merge)
        self.run_btn.grid(row=row, column=0, sticky="w", pady=(10, 0))

        self.progress = ttk.Progressbar(frm, mode="indeterminate")
        self.progress.grid(row=row, column=1, columnspan=2, sticky="we", pady=(10, 0), padx=(10, 0))

        row += 1
        ttk.Label(frm, text="Log:").grid(row=row, column=0, sticky="nw", pady=(10, 0))

        self.log = tk.Text(frm, height=10, wrap="word")
        self.log.grid(row=row, column=1, columnspan=2, sticky="nsew", pady=(10, 0))

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)

        self._log("Select both CSV files, choose output path, set tolerance (default 0.5s), then click 'Merge and Save'.")

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def pick_ni(self) -> None:
        p = filedialog.askopenfilename(title="Select NI DAQ CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.ni_var.set(p)
            self._log(f"NI DAQ file: {p}")

    def pick_pec(self) -> None:
        p = filedialog.askopenfilename(title="Select PEC Cycler CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.pec_var.set(p)
            self._log(f"PEC file: {p}")

    def pick_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Save merged CSV as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if p:
            self.out_var.set(p)
            self._log(f"Output will be saved to: {p}")

    def run_merge(self) -> None:
        ni_path = self.ni_var.get().strip()
        pec_path = self.pec_var.get().strip()
        out_path = self.out_var.get().strip()

        if not ni_path or not Path(ni_path).exists():
            messagebox.showerror("Missing file", "Please select a valid NI DAQ CSV file.")
            return
        if not pec_path or not Path(pec_path).exists():
            messagebox.showerror("Missing file", "Please select a valid PEC cycler CSV file.")
            return
        if not out_path:
            messagebox.showerror("Missing output", "Please choose an output CSV path.")
            return

        try:
            tol = float(self.tol_var.get().strip())
            if tol < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid tolerance", "Tolerance must be a non-negative number (seconds).")
            return

        direction = self.dir_var.get().strip()
        if direction not in ("nearest", "backward", "forward"):
            messagebox.showerror("Invalid direction", "Direction must be one of: nearest, backward, forward.")
            return

        self.progress.start(10)
        self.run_btn.configure(state="disabled")
        self.update_idletasks()

        try:
            self._log("Reading NI DAQ + PEC files...")
            merged = merge_files(
                ni_path=ni_path,
                pec_path=pec_path,
                tolerance_seconds=tol,
                direction=direction,
            )

            matched = merged["pec_timestamp"].notna().sum()
            total = len(merged)
            self._log(f"Merged rows: {total}. Matched within tolerance: {matched} ({matched/total:.1%}).")

            merged.to_csv(out_path, index=False)
            self._log(f"Saved merged CSV: {out_path}")
            messagebox.showinfo("Done", f"Merged file saved:\n{out_path}")

        except Exception as e:
            self._log(f"ERROR: {e}")
            messagebox.showerror("Merge failed", str(e))
        finally:
            self.progress.stop()
            self.run_btn.configure(state="normal")


def main() -> None:
    app = MergeApp()
    app.mainloop()


if __name__ == "__main__":
    main()