# Volatility Modelling Coursework  
### Heston Stochastic Volatility & Dupire Local Volatility Toolkit

---

## Description
This project is a self-contained Python toolkit that fulfils the five-part coursework brief:

1. **Simulate** the Heston stochastic-volatility model.  
2. **Build** a European-call price surface under Heston.  
3. **Invert** that surface to an implied-volatility (IV) surface.  
4. **Derive** a Dupire local-volatility (LV) surface from the IVs and price an **off-grid** option.  
5. **Re-price** the same option at later dates (updated Heston spots) and compare LV vs “true” Heston prices.

All heavy lifting is handled with open-source scientific libraries—**QuantLib-Python**, **NumPy**, **Pandas**, and **Matplotlib**—so the code stays concise, reproducible, and easy to extend.

The **ENTRY POINT** is the **main.py** file. Running this completes all tasks.

Alternatively, the user may launch **app.py** which sits at the same level as the main.py.
This was used by me to adjust bounds & experiment with different model parameters, but is not required by the CW or expected to be marked.
   - (Depending on hardware may be slow, advise to start with low grid size and slowly increase)

---

## Features
- **Monte-Carlo Heston simulation** with QuantLib’s multi-path generator.  
- **Closed-form Heston pricing** (QuantLib `AnalyticHestonEngine`) cached for speed.  
- **Robust implied-vol solver** with automatic bounds expansion & error trapping.  
- **Dupire LV surface** built via second-order central differences and box-clamped bilinear interpolation.  
- **Log-Euler LV Monte-Carlo** (guarantees \(S_t>0\) even for huge vols).  
- **Data-driven plots** (heat-maps & 3-D wire-frame surfaces) using the perceptually-uniform **“turbo”** colormap.  
- **Two UIs**:  
  * CLI pipeline (`python main.py`) for one-shot batch runs.  
  * Tkinter GUI (`python app.py`) for interactive parameter tweaking & live plots.
- **clean_workspace.py** A simple tool to ensure plots generated on new iterations are new
  * Plots used in report are included in this code submission, but can be wiped and regenerated using in-built seed 

---

## Python 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]