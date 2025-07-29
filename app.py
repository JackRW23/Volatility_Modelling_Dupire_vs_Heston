import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys

###############################################################################
# Configuration
###############################################################################
# Scale factor (< 1 shrinks, > 1 enlarges) applied to **all** displayed plots
OUTPUT_SCALE = 0.60        # 0.60 ≈ 40 % reduction

###############################################################################
# Add src/ to sys.path so we can import our vol_modelling package
###############################################################################
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

###############################################################################
# Import project modules
###############################################################################
import vol_modelling.parameters  as parameters
import vol_modelling.task1_heston as task1
import vol_modelling.task2_surface as task2
import vol_modelling.task3_iv as task3
import vol_modelling.task4_dupire as task4
import vol_modelling.task5_reprice as task5
from   vol_modelling.common import HestonParams

###############################################################################
# Optional PIL import (needed for high‑quality image scaling)
###############################################################################
try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

###############################################################################
# Simple tooltip helper
###############################################################################
class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text   = text
        self.tip    = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self.text, background="#ffffe0",
            relief="solid", borderwidth=1, font=("TkDefaultFont", 8)
        ).pack(ipadx=1, ipady=1)

    def _hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

###############################################################################
# Main application
###############################################################################
class VolatilityApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Volatility Modelling GUI")

        # ────────────────────────────────────────────────────────────────────
        # Task list (title, module, images, optional caption)
        # ────────────────────────────────────────────────────────────────────
        self.tasks = [
            ("Task 1 - Heston Path Simulation",
             task1,
             ["task1_heston_paths.png"],
             "First 20 simulated spot paths"),

            ("Task 2 - Heston Price Surface",
             task2,
             ["task2_price_surface.png", "task2_price_surface_3d.png"],
             None),

            ("Task 3 - Implied Volatility Surface",
             task3,
             ["task3_iv_surface.png", "task3_iv_surface_3d.png"],
             None),

            ("Task 4 - Dupire Local-Vol Surface",
             task4,
             ["task4_local_vol_surface.png", "task4_local_vol_surface_3d.png"],
             None),

            ("Task 5 - Dupire vs Heston Repricing",
             task5,
             ["task5_repricing.png"],
             "Repricing comparison across snapshot times"),
        ]

        # ────────────────────────────────────────────────────────────────────
        # Defaults from coursework spec
        # ────────────────────────────────────────────────────────────────────
        self.default_params         = HestonParams()
        self.default_strike_min     = parameters.STRIKE_MIN
        self.default_strike_max     = parameters.STRIKE_MAX
        self.default_maturity_min   = parameters.MATURITY_MIN
        self.default_maturity_max   = parameters.MATURITY_MAX
        self.default_grid_size      = parameters.GRID_SIZE
        self.default_test_strike    = parameters.TEST_STRIKE   # for Task 4
        self.default_test_maturity  = parameters.TEST_MATURITY # for Task 4

        # ────────────────────────────────────────────────────────────────────
        # Tk variables
        # ────────────────────────────────────────────────────────────────────
        self.S0_var       = tk.DoubleVar(value=float(self.default_params.S0))
        self.v0_var       = tk.DoubleVar(value=float(self.default_params.v0))
        self.kappa_var    = tk.DoubleVar(value=float(self.default_params.kappa))
        self.theta_var    = tk.DoubleVar(value=float(self.default_params.theta))
        self.xi_var       = tk.DoubleVar(value=float(self.default_params.xi))
        self.rho_var      = tk.DoubleVar(value=float(self.default_params.rho))
        self.r_var        = tk.DoubleVar(value=float(self.default_params.r))
        self.q_var        = tk.DoubleVar(value=float(self.default_params.q))
        self.strike_min_var   = tk.DoubleVar(value=float(self.default_strike_min))
        self.strike_max_var   = tk.DoubleVar(value=float(self.default_strike_max))
        self.maturity_min_var = tk.DoubleVar(value=float(self.default_maturity_min))
        self.maturity_max_var = tk.DoubleVar(value=float(self.default_maturity_max))
        self.grid_size_var    = tk.IntVar   (value=int  (self.default_grid_size))
        self.test_strike_var  = tk.StringVar(value=str  (self.default_test_strike))
        self.test_maturity_var= tk.StringVar(value=str  (self.default_test_maturity))

        # ────────────────────────────────────────────────────────────────────
        # Progress bar + tick row
        # ────────────────────────────────────────────────────────────────────
        prog_fr = tk.Frame(root, pady=4); prog_fr.pack(fill=tk.X)
        tick_fr = tk.Frame(prog_fr);      tick_fr.pack()

        self.task_vars = []
        for c, (title, *_ ) in enumerate(self.tasks):
            var = tk.StringVar(value=f"◻ {title}")
            tk.Label(tick_fr, textvariable=var, anchor="w", padx=4)\
              .grid(row=0, column=c, sticky="w")
            self.task_vars.append(var)

        self.progress_bar = ttk.Progressbar(
            prog_fr, orient="horizontal", mode="determinate",
            maximum=len(self.tasks)
        )
        self.progress_bar.pack(fill=tk.X, pady=(4, 0))

        # ────────────────────────────────────────────────────────────────────
        # 1‑row Heston parameter grid
        # ────────────────────────────────────────────────────────────────────
        heston_fr = tk.LabelFrame(root, text="Heston Parameters", padx=10, pady=6)
        heston_fr.pack(fill=tk.X, expand=False)

        h_fields = [
            ("Spot price S₀ (>0)",   self.S0_var,   "Initial spot price"),
            ("Initial var v₀ (≥0)",  self.v0_var,   "Starting variance"),
            ("Mean‑rev κ (≥0)",      self.kappa_var,"Speed of mean‑reversion"),
            ("Long‑run θ (≥0)",      self.theta_var,"Long‑term variance level"),
            ("Vol‑of‑vol ξ (≥0)",    self.xi_var,   "Volatility of variance"),
            ("Correlation ρ (‑1…1)", self.rho_var,  "Correlation (‑1…1)"),
        ]
        for i, (lbl, var, tip) in enumerate(h_fields):
            lab = tk.Label(heston_fr, text=lbl + ":")
            ent = tk.Entry(heston_fr, textvariable=var, width=8)
            lab.grid(row=0, column=2*i,   sticky="w", padx=(0,2))
            ent.grid(row=0, column=2*i+1, sticky="w", padx=(0,8))
            ToolTip(lab, tip)

        # ────────────────────────────────────────────────────────────────────
        # Market conditions (single row)
        # ────────────────────────────────────────────────────────────────────
        market_fr = tk.LabelFrame(root, text="Market Conditions", padx=10, pady=6)
        market_fr.pack(fill=tk.X, expand=False)

        lab_r = tk.Label(market_fr, text="Risk‑free rate r (≥0):")
        ent_r = tk.Entry(market_fr, textvariable=self.r_var, width=8)
        lab_q = tk.Label(market_fr, text="Dividend yield q (≥0):")
        ent_q = tk.Entry(market_fr, textvariable=self.q_var, width=8)

        lab_r.grid(row=0, column=0, sticky="w", padx=(0,2))
        ent_r.grid(row=0, column=1, sticky="w", padx=(0,8))
        lab_q.grid(row=0, column=2, sticky="w", padx=(0,2))
        ent_q.grid(row=0, column=3, sticky="w", padx=(0,8))

        ToolTip(lab_r, "Continuous compounding risk‑free rate (non‑negative)")
        ToolTip(lab_q, "Continuous dividend yield (non‑negative)")

        # ────────────────────────────────────────────────────────────────────
        # Grid settings
        # ────────────────────────────────────────────────────────────────────
        grid_fr = tk.LabelFrame(root, text="Grid Settings", padx=10, pady=6)
        grid_fr.pack(fill=tk.X, expand=False)

        lab_s = tk.Label(grid_fr, text="Strike range [min, max] (min<max):")
        lab_s.grid(row=0, column=0, sticky="w")
        tk.Entry(grid_fr, textvariable=self.strike_min_var, width=8)\
            .grid(row=0, column=1, sticky="w", padx=(0,4))
        tk.Entry(grid_fr, textvariable=self.strike_max_var, width=8)\
            .grid(row=0, column=2, sticky="w", padx=(0,8))
        ToolTip(lab_s, "Minimum & maximum strike")

        lab_t = tk.Label(grid_fr, text="Maturity range [min, max] years (min>0):")
        lab_t.grid(row=1, column=0, sticky="w")
        tk.Entry(grid_fr, textvariable=self.maturity_min_var, width=8)\
            .grid(row=1, column=1, sticky="w", padx=(0,4))
        tk.Entry(grid_fr, textvariable=self.maturity_max_var, width=8)\
            .grid(row=1, column=2, sticky="w", padx=(0,8))
        ToolTip(lab_t, "Minimum & maximum maturity in years")

        lab_g = tk.Label(grid_fr, text="Grid size (≥2):")
        lab_g.grid(row=2, column=0, sticky="w")
        tk.Entry(grid_fr, textvariable=self.grid_size_var, width=8)\
            .grid(row=2, column=1, sticky="w")
        ToolTip(lab_g, "Points per axis for strikes & maturities")

        # ────────────────────────────────────────────────────────────────────
        # Task 4 off‑grid option inputs
        # ────────────────────────────────────────────────────────────────────
        t4_fr = tk.LabelFrame(root, text="Task 4 Inputs", padx=10, pady=6)
        t4_fr.pack(fill=tk.X, expand=False)

        lab_K = tk.Label(t4_fr, text="Off‑grid strike K:")
        lab_T = tk.Label(t4_fr, text="Time to maturity T (years):")
        ent_K = tk.Entry(t4_fr, textvariable=self.test_strike_var,   width=8)
        ent_T = tk.Entry(t4_fr, textvariable=self.test_maturity_var, width=8)

        lab_K.grid(row=0, column=0, sticky="w", padx=(0,2))
        ent_K.grid(row=0, column=1, sticky="w", padx=(0,8))
        lab_T.grid(row=0, column=2, sticky="w", padx=(0,2))
        ent_T.grid(row=0, column=3, sticky="w", padx=(0,8))

        ToolTip(lab_K, "Strike of option repriced in Task 4 (blank → default)")
        ToolTip(lab_T, "Maturity of option repriced in Task 4 (blank → default)")

        # ────────────────────────────────────────────────────────────────────
        # Run & Reset buttons
        # ────────────────────────────────────────────────────────────────────
        btn_fr = tk.Frame(root, pady=8); btn_fr.pack(fill=tk.X)
        tk.Button(btn_fr, text="Run All Tasks", command=self.run_tasks)\
            .pack(side=tk.LEFT, padx=5)
        tk.Button(btn_fr, text="Reset to Defaults", command=self.reset_defaults)\
            .pack(side=tk.LEFT, padx=5)

        # ────────────────────────────────────────────────────────────────────
        # Output area (canvas + scrollbars)
        # ────────────────────────────────────────────────────────────────────
        cvs_fr = tk.Frame(root); cvs_fr.pack(fill=tk.BOTH, expand=True)
        self.output_canvas = tk.Canvas(cvs_fr); self.output_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = tk.Scrollbar(cvs_fr, orient="vertical", command=self.output_canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb = tk.Scrollbar(root, orient="horizontal", command=self.output_canvas.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.output_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.output_frame = tk.Frame(self.output_canvas)
        self.output_canvas.create_window((0,0), window=self.output_frame, anchor="nw")
        self.output_frame.bind(
            "<Configure>",
            lambda e: self.output_canvas.configure(scrollregion=self.output_canvas.bbox("all"))
        )

        # Mouse wheel support
        self.output_canvas.bind("<Enter>", self._bind_mousewheel)
        self.output_canvas.bind("<Leave>", self._unbind_mousewheel)

        # Store image references to prevent GC
        self.image_refs = []

    # ------------------------------------------------------------------  #
    # Mouse‑wheel helpers                                                 #
    # ------------------------------------------------------------------  #
    def _on_mousewheel(self, event):
        if event.delta:                       # Windows / macOS
            self.output_canvas.yview_scroll(int(-event.delta/120), "units")
        else:                                # X11: Button‑4/5
            self.output_canvas.yview_scroll(-1 if event.num==4 else 1, "units")

    def _bind_mousewheel(self, _e=None):
        self.output_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.output_canvas.bind_all("<Button-4>",   self._on_mousewheel)
        self.output_canvas.bind_all("<Button-5>",   self._on_mousewheel)

    def _unbind_mousewheel(self, _e=None):
        self.output_canvas.unbind_all("<MouseWheel>")
        self.output_canvas.unbind_all("<Button-4>")
        self.output_canvas.unbind_all("<Button-5>")

    # ------------------------------------------------------------------  #
    # Main workflow (validation, run tasks, display)                      #
    # ------------------------------------------------------------------  #
    def run_tasks(self):
        # ---- collect + basic validation -----------------------------------
        strike_input  = self.test_strike_var.get().strip() or str(self.default_test_strike)
        matur_input   = self.test_maturity_var.get().strip() or str(self.default_test_maturity)
        try:
            S0, v0, kappa, theta, xi, rho = map(float, (
                self.S0_var.get(), self.v0_var.get(), self.kappa_var.get(),
                self.theta_var.get(), self.xi_var.get(), self.rho_var.get()
            ))
            r  = float(self.r_var.get());  q = float(self.q_var.get())
            smin = float(self.strike_min_var.get()); smax = float(self.strike_max_var.get())
            tmin = float(self.maturity_min_var.get()); tmax = float(self.maturity_max_var.get())
            N    = int  (self.grid_size_var.get())
            Ktest = float(strike_input);    Ttest = float(matur_input)
        except Exception as e:
            messagebox.showerror("Invalid input", f"Numeric conversion failed:\n{e}")
            return

        if min(S0, kappa, theta, xi) <= 0 or v0 < 0:
            messagebox.showerror("Invalid input", "Model parameters must be positive (except v₀≥0, ρ).")
            return
        if not -1 <= rho <= 1:   messagebox.showerror("Invalid input","ρ must be within [‑1,1]"); return
        if min(r, q) < 0:        messagebox.showerror("Invalid input","Rates r,q ≥0");           return
        if smax <= smin:         messagebox.showerror("Invalid input","Strike max ≤ min");       return
        if tmax <= tmin or tmin <= 0: messagebox.showerror("Invalid input","Check maturity range"); return
        if N < 2:                messagebox.showerror("Invalid input","Grid size must be ≥2");   return
        if Ktest<=0 or Ttest<=0: messagebox.showerror("Invalid input","Off‑grid K,T must be >0"); return

        # ---- override global parameters -----------------------------------
        parameters.override_heston_params(
            S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, r=r, q=q
        )
        parameters.override_strike_bounds(smin, smax)
        parameters.override_maturity_bounds(tmin, tmax)
        parameters.override_grid_size(N)
        parameters.override_test_option(Ktest, Ttest)

        # ---- clear output --------------------------------------------------
        for w in self.output_frame.winfo_children(): w.destroy()
        self.image_refs.clear()
        self.output_canvas.yview_moveto(0); self.output_canvas.xview_moveto(0)
        for var, (title,*_) in zip(self.task_vars, self.tasks): var.set(f"◻ {title}")
        self.progress_bar['value'] = 0; self.root.update_idletasks()

        # ---- run tasks -----------------------------------------------------
        for idx,(title,module,files,caption) in enumerate(self.tasks, start=1):
            self.task_vars[idx-1].set(f"⏳ {title}"); self.root.update_idletasks()
            try: module.run_task()
            except Exception as e:
                self.task_vars[idx-1].set(f"❌ {title} – failed")
                messagebox.showerror("Task error", f"{title} failed:\n{e}")
                self.progress_bar['value'] = idx; return
            self.task_vars[idx-1].set(f"✅ {title}")
            self.progress_bar['value'] = idx; self.root.update_idletasks()

            tk.Label(self.output_frame, text=title, font=("TkDefaultFont",10,"bold"))\
              .pack(anchor="w", pady=(10,0))

            for fname in files:
                p = (module.OUTPUT_DIR / fname) if hasattr(module,"OUTPUT_DIR") else (module.PLOT_DIR / fname)
                if not p.exists(): continue
                try:
                    if _PIL_AVAILABLE:
                        img = Image.open(p)
                        if OUTPUT_SCALE!=1: img=img.resize((int(img.width*OUTPUT_SCALE),int(img.height*OUTPUT_SCALE)),Image.LANCZOS)
                        ph = ImageTk.PhotoImage(img)
                    else:
                        ph = tk.PhotoImage(file=str(p))
                        if OUTPUT_SCALE<=0.5:
                            f=int(round(1/OUTPUT_SCALE)); ph=ph.subsample(f,f)
                except Exception as e:
                    messagebox.showwarning("Image error", f"Could not load {p}:\n{e}"); continue
                tk.Label(self.output_frame,image=ph).pack(pady=5); self.image_refs.append(ph)
                if caption and len(files)==1:
                    tk.Label(self.output_frame,text=caption).pack(pady=(0,10))

    # ---------------------------------------------------------------------#
    # Reset
    # ---------------------------------------------------------------------#
    def reset_defaults(self):
        self.S0_var.set(float(self.default_params.S0))
        self.v0_var.set(float(self.default_params.v0))
        self.kappa_var.set(float(self.default_params.kappa))
        self.theta_var.set(float(self.default_params.theta))
        self.xi_var.set(float(self.default_params.xi))
        self.rho_var.set(float(self.default_params.rho))
        self.r_var.set(float(self.default_params.r))
        self.q_var.set(float(self.default_params.q))
        self.strike_min_var.set(float(self.default_strike_min))
        self.strike_max_var.set(float(self.default_strike_max))
        self.maturity_min_var.set(float(self.default_maturity_min))
        self.maturity_max_var.set(float(self.default_maturity_max))
        self.grid_size_var.set(int(self.default_grid_size))
        self.test_strike_var.set(str(self.default_test_strike))
        self.test_maturity_var.set(str(self.default_test_maturity))

        parameters.override_heston_params(**self.default_params.__dict__)
        parameters.override_strike_bounds(self.default_strike_min, self.default_strike_max)
        parameters.override_maturity_bounds(self.default_maturity_min, self.default_maturity_max)
        parameters.override_grid_size(self.default_grid_size)
        parameters.override_test_option(self.default_test_strike, self.default_test_maturity)

###############################################################################
# Entrypoint
###############################################################################
if __name__ == "__main__":
    tk.Tk.report_callback_exception = \
        (lambda *exc: messagebox.showerror("Error", f"Uncaught exception:\n{exc[1]}"))
    root = tk.Tk()
    VolatilityApp(root)
    root.mainloop()
