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
        raise ValueError(f"{path.name}: missing 'Total Time (Seconds)' after parsing.")

    df["Total Time (Seconds)"] = pd.to_numeric(df["Total Time (Seconds)"], errors="coerce")
    df["pec_timestamp"] = pd.to_datetime(start_time) + pd.to_timedelta(df["Total Time (Seconds)"], unit="s")

    # Stable sort helps with duplicate timestamps
    df = df.sort_values("pec_timestamp", kind="mergesort").reset_index(drop=True)
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
    bdf = bdf.sort_values("start", kind="mergesort").reset_index(drop=True)
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


def read_multiple_pec_files(paths: list[str | Path], battery_id: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        p = Path(p)
        df = read_pec_file(p)
        df["pec_source_file"] = p.name
        df["battery_id"] = battery_id
        frames.append(df)

    pec_all = pd.concat(frames, ignore_index=True)
    pec_all = pec_all.sort_values("pec_timestamp", kind="mergesort").reset_index(drop=True)
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

    # add unique row id to prevent many-to-many merges later
    df = df.reset_index(drop=True)
    df.insert(0, "ni_row_id", df.index.astype("int64"))

    # stable sort for merge_asof, keep row_id for later reordering
    df = df.sort_values(["ni_timestamp", "ni_row_id"], kind="mergesort").reset_index(drop=True)
    return df


# -------------------------
# Merge (multi-battery wide) - FIXED
# -------------------------

def merge_multi_battery_wide(
    ni_path: str | Path,
    battery_to_pec_paths: dict[str, list[str]],
    tolerance_seconds: float = 0.5,
    direction: str = "nearest",
) -> pd.DataFrame:
    ni = read_ni_daq_file(ni_path)

    # We'll build output from NI, then attach per-battery PEC columns by ni_row_id (unique).
    out = ni.copy()

    left_base = out[["ni_row_id", "ni_timestamp"]].copy()
    left_base = left_base.sort_values(["ni_timestamp", "ni_row_id"], kind="mergesort").reset_index(drop=True)

    for battery_id, paths in battery_to_pec_paths.items():
        pec = read_multiple_pec_files(paths, battery_id=battery_id)
        pec = pec.sort_values("pec_timestamp", kind="mergesort").reset_index(drop=True)

        merged = pd.merge_asof(
            left_base,
            pec,
            left_on="ni_timestamp",
            right_on="pec_timestamp",
            direction=direction,
            tolerance=pd.Timedelta(seconds=float(tolerance_seconds)),
        )

        merged["match_delta_seconds"] = (merged["ni_timestamp"] - merged["pec_timestamp"]).dt.total_seconds()

        # Prefix ALL PEC-derived columns (everything except ni_row_id/ni_timestamp)
        keep = {"ni_row_id", "ni_timestamp"}
        rename = {c: f"{battery_id}__{c}" for c in merged.columns if c not in keep}
        merged = merged.rename(columns=rename)

        # Only keep ni_row_id + prefixed columns for join
        cols_to_join = ["ni_row_id"] + [c for c in merged.columns if c.startswith(f"{battery_id}__")]
        merged_small = merged[cols_to_join].copy()

        # Join back by unique ni_row_id => guaranteed 1:1, no row multiplication
        out = out.merge(merged_small, on="ni_row_id", how="left")

    # Restore original NI order by ni_row_id (so your output matches input order)
    out = out.sort_values("ni_row_id", kind="mergesort").reset_index(drop=True)
    return out


# -------------------------
# Tkinter GUI (multi-battery)
# -------------------------

@dataclass
class BatterySelection:
    battery_id: str
    pec_paths: list[str] = field(default_factory=list)


@dataclass
class AppState:
    batteries: list[BatterySelection] = field(default_factory=list)


class BatteryRow(ttk.Frame):
    def __init__(self, master, app: "MergeApp", battery: BatterySelection):
        super().__init__(master)
        self.app = app
        self.battery = battery
        self.id_var = tk.StringVar(value=battery.battery_id)

        ttk.Label(self, text="Battery ID (4 digits):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.id_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(self, text="PEC files:").grid(row=0, column=2, sticky="w")

        self.listbox = tk.Listbox(self, height=5, selectmode="extended")
        self.listbox.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(4, 0))

        scroll = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=1, column=4, sticky="ns", pady=(4, 0))
        self.listbox.configure(yscrollcommand=scroll.set)

        btns = ttk.Frame(self)
        btns.grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Button(btns, text="Add PEC files...", command=self.add_files).pack(side="left")
        ttk.Button(btns, text="Remove selected", command=self.remove_selected).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Remove Battery", command=self.remove_battery).pack(side="left", padx=(16, 0))

        self.columnconfigure(3, weight=1)
        self.rowconfigure(1, weight=1)

    def _validate_and_set_battery_id(self) -> str | None:
        bid = self.id_var.get().strip()
        if not re.fullmatch(r"\d{4}", bid):
            messagebox.showerror("Invalid Battery ID", "Battery ID must be exactly 4 digits (e.g., 2196).")
            return None
        self.battery.battery_id = bid
        return bid

    def add_files(self) -> None:
        if self._validate_and_set_battery_id() is None:
            return
        paths = filedialog.askopenfilenames(
            title=f"Select PEC CSV files for Battery {self.battery.battery_id}",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not paths:
            return

        existing = set(self.battery.pec_paths)
        added = 0
        for p in paths:
            if p not in existing:
                self.battery.pec_paths.append(p)
                self.listbox.insert("end", p)
                existing.add(p)
                added += 1

        self.app._log(f"Battery {self.battery.battery_id}: added {added} file(s). Total: {len(self.battery.pec_paths)}")

    def remove_selected(self) -> None:
        sel = list(self.listbox.curselection())
        if not sel:
            return
        sel.reverse()
        for idx in sel:
            path = self.listbox.get(idx)
            self.listbox.delete(idx)
            if path in self.battery.pec_paths:
                self.battery.pec_paths.remove(path)
        self.app._log(f"Battery {self.battery.battery_id}: total files now {len(self.battery.pec_paths)}")

    def clear(self) -> None:
        self.listbox.delete(0, "end")
        self.battery.pec_paths.clear()
        self.app._log(f"Battery {self.battery.battery_id}: cleared files.")

    def remove_battery(self) -> None:
        self.app.remove_battery_row(self)


class MergeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NI DAQ + PEC Cycler CSV Merger (Multi-Battery)")
        self.geometry("1100x700")

        self.state = AppState()
        self.battery_rows: list[BatteryRow] = []

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, **pad)

        self.ni_var = tk.StringVar(value="")
        self.out_var = tk.StringVar(value="")
        self.tol_var = tk.StringVar(value="0.5")
        self.dir_var = tk.StringVar(value="nearest")

        r = 0
        ttk.Label(root, text="NI DAQ CSV:").grid(row=r, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.ni_var, width=110).grid(row=r, column=1, sticky="we")
        ttk.Button(root, text="Browse...", command=self.pick_ni).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(root, text="Output CSV:").grid(row=r, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.out_var, width=110).grid(row=r, column=1, sticky="we")
        ttk.Button(root, text="Save As...", command=self.pick_out).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Separator(root).grid(row=r, column=0, columnspan=3, sticky="we", pady=10)

        r += 1
        opt = ttk.Frame(root)
        opt.grid(row=r, column=0, columnspan=3, sticky="we")
        ttk.Label(opt, text="Tolerance (seconds):").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(opt, textvariable=self.tol_var, width=10).grid(row=0, column=1, sticky="w", padx=(0, 18))
        ttk.Label(opt, text="Merge direction:").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Combobox(opt, textvariable=self.dir_var, values=["nearest", "backward", "forward"], width=10, state="readonly")\
            .grid(row=0, column=3, sticky="w", padx=(0, 18))
        ttk.Label(opt, text=f"NI time column detection: contains '{NI_TIME_SUBSTRING}'").grid(row=0, column=4, sticky="w")

        r += 1
        ttk.Separator(root).grid(row=r, column=0, columnspan=3, sticky="we", pady=10)

        r += 1
        header = ttk.Frame(root)
        header.grid(row=r, column=0, columnspan=3, sticky="we")
        ttk.Label(header, text="Batteries:").pack(side="left")
        ttk.Button(header, text="Add Battery...", command=self.add_battery).pack(side="right")

        r += 1
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.grid(row=r, column=0, columnspan=3, sticky="nsew")
        sb = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        sb.grid(row=r, column=3, sticky="ns")
        self.canvas.configure(yscrollcommand=sb.set)

        self.container = ttk.Frame(self.canvas)
        self.container_win = self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        def _on_container_configure(_evt):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_configure(evt):
            self.canvas.itemconfigure(self.container_win, width=evt.width)

        self.container.bind("<Configure>", _on_container_configure)
        self.canvas.bind("<Configure>", _on_canvas_configure)

        r += 1
        actions = ttk.Frame(root)
        actions.grid(row=r, column=0, columnspan=3, sticky="we", pady=(10, 0))
        self.run_btn = ttk.Button(actions, text="Merge and Save", command=self.run_merge)
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(10, 0))

        r += 1
        ttk.Label(root, text="Log:").grid(row=r, column=0, sticky="nw", pady=(10, 0))
        self.log = tk.Text(root, height=10, wrap="word")
        self.log.grid(row=r, column=1, columnspan=2, sticky="nsew", pady=(10, 0))

        root.columnconfigure(1, weight=1)
        root.rowconfigure(5, weight=1)
        root.rowconfigure(r, weight=1)

        self.add_battery()
        self._log("Fixed: no more row explosion with duplicate NI timestamps (joins use ni_row_id).")

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def pick_ni(self) -> None:
        p = filedialog.askopenfilename(title="Select NI DAQ CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.ni_var.set(p)
            self._log(f"NI DAQ file: {p}")

    def pick_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Save merged CSV as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if p:
            self.out_var.set(p)
            self._log(f"Output will be saved to: {p}")

    def add_battery(self) -> None:
        b = BatterySelection(battery_id="0000")
        self.state.batteries.append(b)
        row = BatteryRow(self.container, app=self, battery=b)
        row.pack(fill="x", expand=True, pady=(0, 10))
        self.battery_rows.append(row)
        self._log("Added battery section. Enter 4-digit ID and select PEC files.")

    def remove_battery_row(self, row: BatteryRow) -> None:
        if row not in self.battery_rows:
            return
        idx = self.battery_rows.index(row)
        bid = row.id_var.get().strip()
        self.battery_rows.pop(idx)
        self.state.batteries.pop(idx)
        row.destroy()
        self._log(f"Removed battery section ({bid or 'unknown'}).")

    @staticmethod
    def _format_timedelta(td: pd.Timedelta) -> str:
        s = abs(float(td.total_seconds()))
        if s < 60:
            return f"{s:.3f} s"
        if s < 3600:
            return f"{s/60:.3f} min"
        return f"{s/3600:.3f} hr"

    def _validate_batteries(self) -> dict[str, list[str]] | None:
        battery_map: dict[str, list[str]] = {}
        seen: set[str] = set()

        for row in self.battery_rows:
            bid = row.id_var.get().strip()
            if not re.fullmatch(r"\d{4}", bid):
                messagebox.showerror("Invalid Battery ID", f"Battery ID must be exactly 4 digits. Got: {bid!r}")
                return None
            if bid in seen:
                messagebox.showerror("Duplicate Battery ID", f"Battery ID {bid} is used more than once.")
                return None
            seen.add(bid)

            if not row.battery.pec_paths:
                messagebox.showerror("Missing PEC files", f"Please add at least one PEC CSV file for battery {bid}.")
                return None

            missing = [p for p in row.battery.pec_paths if not Path(p).exists()]
            if missing:
                messagebox.showerror(
                    "Missing PEC files",
                    f"Some PEC paths no longer exist for battery {bid}:\n" + "\n".join(missing),
                )
                return None

            battery_map[bid] = list(row.battery.pec_paths)

        return battery_map

    def _check_overlaps_for_battery_and_confirm(self, battery_id: str, pec_paths: list[str]) -> bool:
        self._log(f"Checking overlaps for battery {battery_id}...")
        bounds_df = compute_pec_bounds(pec_paths)
        overlaps = find_overlaps(bounds_df)
        if not overlaps:
            self._log(f"Battery {battery_id}: no overlaps detected.")
            return True

        lines = [
            f"WARNING: Overlapping PEC file time ranges detected for battery {battery_id}.",
            "",
            "Overlaps found:",
        ]
        for o in overlaps:
            lines.append(f"- {o.file_a} and {o.file_b}")
            lines.append(f"  {o.file_a} ends:   {o.a_end}")
            lines.append(f"  {o.file_b} starts: {o.b_start}")
            lines.append(f"  Overlap amount:    {self._format_timedelta(o.overlap)}")
            lines.append("")
        lines.append("Do you want to continue anyway for this battery?")

        return messagebox.askyesno(f"PEC Overlap Warning (Battery {battery_id})", "\n".join(lines))

    def run_merge(self) -> None:
        ni_path = self.ni_var.get().strip()
        out_path = self.out_var.get().strip()

        if not ni_path or not Path(ni_path).exists():
            messagebox.showerror("Missing file", "Please select a valid NI DAQ CSV file.")
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

        battery_map = self._validate_batteries()
        if battery_map is None:
            return

        for bid, paths in battery_map.items():
            try:
                if not self._check_overlaps_for_battery_and_confirm(bid, paths):
                    self._log(f"User cancelled merge due to overlap warning for battery {bid}.")
                    return
            except Exception as e:
                self._log(f"ERROR during overlap check for battery {bid}: {e}")
                messagebox.showerror("Overlap check failed", f"Battery {bid}: {e}")
                return

        self.progress.start(10)
        self.run_btn.configure(state="disabled")
        self.update_idletasks()

        try:
            self._log(f"Merging {len(battery_map)} battery(ies) onto NI timeline...")
            merged = merge_multi_battery_wide(
                ni_path=ni_path,
                battery_to_pec_paths=battery_map,
                tolerance_seconds=tol,
                direction=direction,
            )

            for bid in battery_map.keys():
                ts_col = f"{bid}__pec_timestamp"
                if ts_col in merged.columns:
                    matched = merged[ts_col].notna().sum()
                    total = len(merged)
                    self._log(f"Battery {bid}: matched within tolerance: {matched}/{total} ({matched/total:.1%})")

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
