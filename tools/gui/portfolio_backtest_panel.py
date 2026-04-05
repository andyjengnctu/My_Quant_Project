from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

from tools.portfolio_sim.reporting import DASHBOARD_HTML_PATH, OUTPUT_DIR, REPORT_XLSX_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_APP_PATH = PROJECT_ROOT / "apps" / "portfolio_sim.py"


class PortfolioBacktestPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6, style="Workbench.TFrame")
        self._status_var = tk.StringVar(value="尚未執行投組回測")
        self._max_pos_var = tk.StringVar(value="10")
        self._start_year_var = tk.StringVar(value="2015")
        self._benchmark_var = tk.StringVar(value="0050")
        self._rotation_var = tk.BooleanVar(value=False)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        controls = ttk.LabelFrame(self, text="投組回測參數", padding=8, style="Workbench.TLabelframe")
        controls.pack(fill="x", pady=(0, 6))

        ttk.Label(controls, text="最大持倉", style="Workbench.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._max_pos_var, width=10, style="Workbench.TEntry").grid(row=0, column=1, padx=(6, 12), sticky="w")
        ttk.Label(controls, text="開始年份", style="Workbench.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self._start_year_var, width=10, style="Workbench.TEntry").grid(row=0, column=3, padx=(6, 12), sticky="w")
        ttk.Label(controls, text="大盤標的", style="Workbench.TLabel").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self._benchmark_var, width=10, style="Workbench.TEntry").grid(row=0, column=5, padx=(6, 12), sticky="w")
        ttk.Checkbutton(controls, text="啟用汰弱換股", variable=self._rotation_var, style="Workbench.TCheckbutton").grid(row=0, column=6, padx=(0, 12), sticky="w")
        ttk.Button(controls, text="執行投組回測", command=self._run_portfolio_sim, style="Workbench.TButton").grid(row=0, column=7, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟投組儀表板", command=lambda: self._open_path(DASHBOARD_HTML_PATH), style="Workbench.TButton").grid(row=0, column=8, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟投組 Excel", command=lambda: self._open_path(REPORT_XLSX_PATH), style="Workbench.TButton").grid(row=0, column=9, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟輸出資料夾", command=lambda: self._open_path(OUTPUT_DIR), style="Workbench.TButton").grid(row=0, column=10, sticky="w")

        console_frame = ttk.Frame(self, style="Workbench.TFrame")
        console_frame.pack(fill="both", expand=True)
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)
        self._console = tk.Text(console_frame, wrap="word", bg="#040a12", fg="#f7fbff", insertbackground="#f7fbff", relief="flat", bd=0, font=("Consolas", 10))
        self._console.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self._console.yview, style="Workbench.Vertical.TScrollbar")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._console.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(self, style="Workbench.TFrame")
        footer.pack(fill="x", pady=(4, 0))
        ttk.Label(footer, textvariable=self._status_var, style="Workbench.TLabel").pack(anchor="w")

    def _append_console(self, text):
        self._console.insert("end", text)
        self._console.see("end")
        self.update_idletasks()

    def _set_status(self, text):
        self._status_var.set(text)
        self.update_idletasks()

    def _run_portfolio_sim(self):
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("股票工具工作台", "投組回測執行中，請稍候。")
            return
        self._console.delete("1.0", "end")
        rotation_choice = "Y" if self._rotation_var.get() else "N"
        input_text = (
            f"{rotation_choice}\n"
            f"{self._max_pos_var.get().strip() or '10'}\n"
            f"{self._start_year_var.get().strip() or '2015'}\n"
            f"{self._benchmark_var.get().strip() or '0050'}\n"
        )
        self._set_status("執行中：投組回測")

        def _worker():
            try:
                completed = subprocess.run(
                    [sys.executable, str(PORTFOLIO_APP_PATH)],
                    cwd=str(PROJECT_ROOT),
                    input=input_text,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                output_text = (completed.stdout or "") + (completed.stderr or "")
                self.after(0, lambda: self._append_console(output_text or "(無 console 輸出)\n"))
                self.after(0, lambda: self._set_status(f"投組回測完成：exit_code={completed.returncode}"))
            except Exception as exc:
                self.after(0, lambda: self._append_console(f"執行失敗：{type(exc).__name__}: {exc}\n"))
                self.after(0, lambda: self._set_status(f"投組回測失敗：{type(exc).__name__}: {exc}"))

        self._worker = threading.Thread(target=_worker, daemon=True)
        self._worker.start()

    def _open_path(self, path):
        normalized_path = os.path.abspath(str(path))
        if not os.path.exists(normalized_path):
            messagebox.showerror("股票工具工作台", f"找不到檔案或資料夾：{normalized_path}")
            return
        try:
            if os.name == "nt":
                os.startfile(normalized_path)  # type: ignore[attr-defined]
                return
            if sys.platform == "darwin":
                subprocess.Popen(["open", normalized_path])
                return
            webbrowser.open(f"file://{normalized_path}")
        except Exception as exc:
            messagebox.showerror("股票工具工作台", f"開啟失敗：{type(exc).__name__}: {exc}")
