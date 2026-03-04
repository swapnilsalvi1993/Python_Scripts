from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd


EXCEL_EPOCH = "1899-12-30"
NI_TIME_SUBSTRING = "Date/Time (Excel Format)"


# -------------------------
# PEC parsing helpers
# -------------------------

def find_pec_header_row(lines: list[str], required_cols: tuple[str, ...] = ("Total Time (Seconds)",)) -> int:
    for i, raw in enumerate(lines):
        tokens = [t.strip() for t in re.split(r"[\t,]", raw.rstrip("\n\r")) if t.strip() != ""]
        if not tokens:
            continue
        token_set = set(tokens)
        if all(col in token_set for col in required_cols) and len(tokens) >= 10:
            return i
    raise ValueError("Could not locate PEC data header row (line containing 'Total Time (Seconds)').")


def parse_pec_start_time(lines: list[str]) -> datetime:
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

        try:
            dt = pd.to_datetime(value, errors="raise")
            return dt.to_pydatetime()
        except Exception as e:
            raise ValueError(f"Found Start Time but couldn't parse it: {value!r}") from e

    raise ValueError("Could not find 'Start Time:' in PEC file metadata.")


def read_pec_file(path: str | Path) -> pd.DataFrame:
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


@dataclass(frozen=True)
class OverlapInfo:
    file_a: str
    file_b: str
    a_end: pd.Timestamp
    b_start: pd.Timestamp
    overlap: pd.Timedelta


def compute_pec_bounds(paths: list[str | Path]) -> pd.DataFrame:
    bounds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for p in paths:
        p = Path(p)
        df = read_pec_file(p)
        lo = df["pec_timestamp"].min()
        hi = df["pec_timestamp"].max()
        bounds.append((p.name, lo, hi))

    bdf = pd.DataFrame(bounds, columns=["file", "start", "end"])
    bdf = bdf.sort_values("start").reset_index(drop=True)
    return bdf


def find_overlaps(bounds_df: pd.DataFrame) -> list[OverlapInfo]:
    overlaps: list[OverlapInfo] = []
    if bounds_df.empty:
        return overlaps

    for i in range(1, len(bounds_df)):
        prev = bounds_df.loc[i - 1]
        cur = bounds_df.loc[i]
        if pd.isna(prev["end"]) or pd.isna(cur["start"]):
            continue
        if cur["start"] <= prev["end"]:
            overlap = prev["end"] - cur["start"]
            overlaps.append(
                OverlapInfo(
                    file_a=str(prev["file"]),
                    file_b=str(cur["file"]),
                    a_end=prev["end"],
                    b_start=cur["start"],
                    overlap=overlap,
                )
            )
    return overlaps


def read_multiple_pec_files(paths: list[str | Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("No PEC files provided.")

    frames: list[pd.DataFrame] = []
    for p in paths:
        p = Path(p)
        df = read_pec_file(p)
        df["pec_source_file"] = p.name
        frames.append(df)

    pec_all = pd.concat(frames, ignore_index=True)
    pec_all = pec_all.sort_values("pec_timestamp").reset_index(drop=True)
    return pec_all


# -------------------------
# NI DAQ parsing helpers
# -------------------------

def detect_ni_excel_datetime_column(columns: list[str]) -> str:
    matches = [c for c in columns if NI_TIME_SUBSTRING.lower() in str(c).lower()]
    if not matches:
        raise ValueError(
            f"Could not find NI DAQ time column containing substring: {NI_TIME_SUBSTRING!r}. "
            f"Available columns (first 20): {columns[:20]}"
        )
    matches.sort(key=lambda s: (len(s), s))
    return matches[0]


def read_ni_daq_file(path: str | Path) -> pd.DataFrame:
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
    pec_paths: list[str | Path],
    tolerance_seconds: float = 0.5,
    direction: str = "nearest",
) -> pd.DataFrame:
    ni_df = read_ni_daq_file(ni_path)
    pec_df = read_multiple_pec_files(pec_paths)

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
    pec_paths: list[str] = field(default_factory=list)  # always a real list
    output_path: str | None = None


class MergeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NI DAQ + PEC Cycler CSV Merger (Multi-PEC)")
        self.geometry("980x520")

        self.state_data = AppState()
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        self.ni_var = tk.StringVar(value="")
        self.out_var = tk.StringVar(value="")

        row = 0
        ttk.Label(frm, text="NI DAQ CSV:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.ni_var, width=100).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="Browse...", command=self.pick_ni).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(frm, text="PEC Cycler CSVs (multiple):").grid(row=row, column=0, sticky="nw")

        pec_frame = ttk.Frame(frm)
        pec_frame.grid(row=row, column=1, columnspan=2, sticky="nsew")

        self.pec_list = tk.Listbox(pec_frame, height=7, selectmode="extended")
        self.pec_list.grid(row=0, column=0, sticky="nsew")
        pec_scroll = ttk.Scrollbar(pec_frame, orient="vertical", command=self.pec_list.yview)
        pec_scroll.grid(row=0, column=1, sticky="ns")
        self.pec_list.configure(yscrollcommand=pec_scroll.set)

        btns = ttk.Frame(pec_frame)
        btns.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))

        ttk.Button(btns, text="Add PEC files...", command=self.add_pec_files).pack(side="left")
        ttk.Button(btns, text="Remove selected", command=self.remove_selected_pec).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Clear", command=self.clear_pec).pack(side="left", padx=(8, 0))

        pec_frame.columnconfigure(0, weight=1)
        pec_frame.rowconfigure(0, weight=1)

        row += 1
        ttk.Label(frm, text="Output CSV:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_var, width=100).grid(row=row, column=1, sticky="we")
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

        self._log("Select NI DAQ CSV, add 1+ PEC CSVs, choose output path, then click 'Merge and Save'.")

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def pick_ni(self) -> None:
        p = filedialog.askopenfilename(title="Select NI DAQ CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.ni_var.set(p)
            self._log(f"NI DAQ file: {p}")

    def add_pec_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select one or more PEC Cycler CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not paths:
            return

        existing = set(self.state_data.pec_paths)
        added = 0
        for p in paths:
            if p not in existing:
                self.state_data.pec_paths.append(p)  # FIX: append to the real list
                self.pec_list.insert("end", p)
                existing.add(p)
                added += 1

        self._log(f"Added {added} PEC file(s). Total PEC files: {len(self.state_data.pec_paths)}")

    def remove_selected_pec(self) -> None:
        sel = list(self.pec_list.curselection())
        if not sel:
            return
        sel.reverse()
        for idx in sel:
            path = self.pec_list.get(idx)
            self.pec_list.delete(idx)
            if path in self.state_data.pec_paths:
                self.state_data.pec_paths.remove(path)

        self._log(f"Removed selected PEC file(s). Total PEC files: {len(self.state_data.pec_paths)}")

    def clear_pec(self) -> None:
        self.pec_list.delete(0, "end")
        self.state_data.pec_paths.clear()
        self._log("Cleared PEC file list.")

    def pick_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Save merged CSV as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if p:
            self.out_var.set(p)
            self._log(f"Output will be saved to: {p}")

    @staticmethod
    def _format_timedelta(td: pd.Timedelta) -> str:
        total_seconds = abs(float(td.total_seconds()))
        if total_seconds < 60:
            return f"{total_seconds:.3f} s"
        if total_seconds < 3600:
            return f"{total_seconds/60:.3f} min"
        return f"{total_seconds/3600:.3f} hr"

    def _check_overlaps_and_confirm(self, pec_paths: list[str]) -> bool:
        self._log("Checking PEC file time ranges for overlap...")
        bounds_df = compute_pec_bounds(pec_paths)
        overlaps = find_overlaps(bounds_df)

        if not overlaps:
            self._log("No PEC overlaps detected.")
            return True

        lines = [
            "WARNING: Overlapping PEC file time ranges detected.",
            "",
            "This can cause ambiguous matching when merging.",
            "",
            "Overlaps found:",
        ]

        for o in overlaps:
            lines.append(f"- {o.file_a} and {o.file_b}")
            lines.append(f"  {o.file_a} ends:   {o.a_end}")
            lines.append(f"  {o.file_b} starts: {o.b_start}")
            lines.append(f"  Overlap amount:    {self._format_timedelta(o.overlap)}")
            lines.append("")

        lines.append("Do you want to continue anyway?")

        msg = "\n".join(lines)
        self._log("PEC overlap detected; asking user whether to continue.")
        return messagebox.askyesno("PEC Overlap Warning", msg)

    def run_merge(self) -> None:
        ni_path = self.ni_var.get().strip()
        pec_paths = list(self.state_data.pec_paths)  # real list now
        out_path = self.out_var.get().strip()

        if not ni_path or not Path(ni_path).exists():
            messagebox.showerror("Missing file", "Please select a valid NI DAQ CSV file.")
            return
        if not pec_paths:
            messagebox.showerror("Missing PEC files", "Please add at least one PEC cycler CSV file.")
            return
        missing = [p for p in pec_paths if not Path(p).exists()]
        if missing:
            messagebox.showerror("Missing PEC files", "Some PEC paths no longer exist:\n" + "\n".join(missing))
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

        try:
            if not self._check_overlaps_and_confirm(pec_paths):
                self._log("User cancelled due to PEC overlap warning.")
                return
        except Exception as e:
            self._log(f"ERROR during overlap check: {e}")
            messagebox.showerror("Overlap check failed", str(e))
            return

        self.progress.start(10)
        self.run_btn.configure(state="disabled")
        self.update_idletasks()

        try:
            self._log("Reading NI DAQ file and merging with PEC data...")
            merged = merge_files(
                ni_path=ni_path,
                pec_paths=pec_paths,
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
