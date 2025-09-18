#!/usr/bin/env python3
"""
doe_pipeline.py
Pipeline completa:
 - Preprocessing e feature extraction
 - Test di white noise
 - Estrazione steady-state
 - DoE regression + ANOVA
 - Leave-One-Experiment-Out CV
 - Grafici e salvataggi

Usage:
  python doe_pipeline.py /percorso/alla/cartella_con_i_csv
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# Parametri modificabili
# -------------------------
FS = 100                    # sampling rate sensor (Hz)
#STEADY_SAMPLES = 3000       # ultimi N campioni per steady-state
ACF_LAGS = [1, 5, 10, 20]   # lag da riportare
BG_LAGS = 10                # numero lag per Breusch-Godfrey (per residui)

TARGET_COL = "Acc_Z [g]"

def fit_factorial_and_anova(df_features):
    # we expect columns 'y', 'T_c', 'RH_c' numeric coded (-1/0/1)
    df = df_features.copy()
    # formula with numeric coded factors
    formula = 'y ~ T_c * RH_c'
    model = ols(formula, data=df).fit()
    try:
        anova_table = sm.stats.anova_lm(model, typ=2)
    except Exception:
        anova_table = sm.stats.anova_lm(model, typ=1)
    coeffs = model.params.to_dict()
    return model, anova_table, coeffs

def parse_filename_for_factors(fname):
    """
    Parse filename and return (T_label, RH_label, run, kind)
    Labels: 'low','high','center'
    kind: 'corner' or 'center'
    Examples:
      TempBassa_UmBassa_Run_1.csv -> ('low','low',1,'corner')
      TempAlta_UmAlta_Run_2.csv  -> ('high','high',2,'corner')
      CenterPoints_Run_3.csv     -> ('center','center',3,'center')
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")

    # center points
    if parts[0].lower().startswith("centerpoints"):
        run = None
        for p in parts:
            if p.lower().startswith("run"):
                try:
                    run = int(p.split("_")[-1]) if "_" in p else int(parts[2])
                except Exception:
                    try:
                        run = int(parts[-1])
                    except Exception:
                        run = 0
                break
        if run is None:
            try:
                run = int(parts[-1])
            except:
                run = 0
        return ('center', 'center', int(run), 'center')

    # corner points
    T_label = None
    RH_label = None
    run = None
    for p in parts:
        low_p = p.lower()
        if low_p.startswith("temp"):
            if "bassa" in low_p:
                T_label = 'low'
            elif "alta" in low_p:
                T_label = 'high'
        elif low_p.startswith("um") or low_p.startswith("rh"):
            if "bassa" in low_p:
                RH_label = 'low'
            elif "alta" in low_p:
                RH_label = 'high'
        elif low_p.startswith("run"):
            try:
                run = int(low_p.split("_")[-1])
            except:
                pass

    if run is None:
        for token in reversed(parts):
            try:
                run = int(token)
                break
            except:
                pass
        if run is None:
            run = 0

    return (T_label, RH_label, int(run), 'corner')

def label_to_code(label):
    return {'low': -1, 'center': 0, 'high': 1}[label]

def plot_time_series(arr, fname, outdir):
    """
    Plotta la serie temporale (tutti i campioni) e salva come PNG.
    arr: array numpy con i dati
    fname: nome file sorgente (es. TempAlta_UmAlta_Run_1.csv)
    outdir: cartella di output per i grafici
    """
    plt.figure(figsize=(10,4))
    plt.plot(arr, linewidth=0.7)
    plt.title(f"Time Series - {fname}")
    plt.xlabel("Samples")
    plt.ylabel("Acc_Z [g]")
    plt.grid(True, alpha=0.3)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, fname.replace(".csv",".png"))
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot salvato in {outpath}")

def load_csv_skip2(path, colname=TARGET_COL):
    """Leggi CSV con skiprows=2, decimal=',' e ritorna DataFrame."""
    return pd.read_csv(path, sep=",", decimal=",", skiprows=2)

def extract_column_array(df, colname=TARGET_COL):
    """Estrai colonna numerica di interesse, fallback se non trovata."""
    if colname not in df.columns:
        candidates = [c for c in df.columns if 'acc' in c.lower() and 'z' in c.lower()]
        if len(candidates) > 0:
            colname = candidates[0]
        else:
            raise ValueError(
                f"Colonna {colname} non trovata. Colonne disponibili: {df.columns.tolist()}"
            )
    arr = pd.to_numeric(df[colname], errors='coerce').to_numpy()
    arr = arr[~np.isnan(arr)]
    return arr

def compute_basic_stats(arr):
    """Statistiche base sull'intera serie temporale (tutti i campioni)."""
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=0)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'skew': float(stats.skew(arr)),
        'kurtosis': float(stats.kurtosis(arr))  # excess kurtosis
    }

def sample_acf(arr, nlags):
    """ACF fino a nlags."""
    acfs = acf(arr, nlags=nlags, fft=False, missing='conservative')
    return acfs

def portmanteau_stats(arr, h, model_params=0):
    """Calcola Box–Pierce e Ljung–Box."""
    n = len(arr)
    acfs = sample_acf(arr, nlags=h)
    rhos = acfs[1:h+1]
    Q_BP = n * np.sum(rhos**2)
    Q_LB = n * (n + 2) * np.sum([(rhos[k-1]**2) / (n - k) for k in range(1, h+1)])
    df = max(h - model_params, 1)
    p_bp = chi2.sf(Q_BP, df)
    p_lb = chi2.sf(Q_LB, df)
    return {
        'Q_BP': float(Q_BP),
        'p_BP': float(p_bp),
        'Q_LB': float(Q_LB),
        'p_LB': float(p_lb),
        'h': h,
        'df': df
    }

def durbin_watson_stat(arr):
    """Calcola Durbin-Watson sui residui della serie centrata."""
    return float(durbin_watson(arr))

def plot_experimental_scatter(features_df, savepath=None):
    """Scatter plot dei punti sperimentali (T_c vs RH_c, colore = y)."""
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(
        features_df['T_c'], features_df['RH_c'],
        c=features_df['y'], cmap='viridis', s=120,
        edgecolors='k', marker='o'
    )
    # evidenzio i center points con un marker diverso
    centers = features_df[features_df['kind']=='center']
    if not centers.empty:
        plt.scatter(centers['T_c'], centers['RH_c'], 
                    c=centers['y'], cmap='viridis', 
                    s=160, edgecolors='red', marker='D', label='Center')
    plt.colorbar(scatter, label='y (risposta media)')
    plt.xlabel("T_c (codificato)")
    plt.ylabel("RH_c (codificato)")
    plt.title("Scatter dei punti sperimentali")
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()

def plot_response_surface(model, features_df, savepath=None):
    """Surface plot della regressione con corner e center points."""
    T_range = np.linspace(-1, 1, 50)
    RH_range = np.linspace(-1, 1, 50)
    T_grid, RH_grid = np.meshgrid(T_range, RH_range)
    df_grid = pd.DataFrame({'T_c': T_grid.ravel(), 'RH_c': RH_grid.ravel()})
    y_pred = model.predict(df_grid).values.reshape(T_grid.shape)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_grid, RH_grid, y_pred, alpha=0.6, cmap='viridis')

    # aggiungo i punti sperimentali
    ax.scatter(
        features_df['T_c'], features_df['RH_c'], features_df['y'],
        c='r', s=60, depthshade=True, label='Sperimentali'
    )
    ax.set_xlabel("T_c (codificato)")
    ax.set_ylabel("RH_c (codificato)")
    ax.set_zlabel("y (risposta media)")
    ax.set_title("Superficie stimata + punti sperimentali")
    ax.legend()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()

def leave_one_experiment_out_cv(df_design_features):
    df = df_design_features.reset_index(drop=True).copy()
    N = len(df)
    preds = []
    trues = []
    resid = []
    for i in range(N):
        train = df.drop(i)
        test = df.loc[[i]]
        m = ols('y ~ T_c * RH_c', data=train).fit()
        yhat = m.predict(test).iloc[0]
        ytrue = test['y'].values[0]
        preds.append(yhat)
        trues.append(ytrue)
        resid.append(yhat - ytrue)
    preds = np.array(preds)
    trues = np.array(trues)
    resid = np.array(resid)
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    r2 = float(1 - np.sum((trues - preds)**2) / np.sum((trues - np.mean(trues))**2)) if np.var(trues)>0 else np.nan
    results_df = df.copy()
    results_df['yhat_loeo'] = preds
    results_df['resid_loeo'] = resid
    metrics = {'rmse_loeo': rmse, 'mae_loeo': mae, 'bias_loeo': bias, 'r2_loeo': r2}
    return results_df, metrics

def run_analysis_on_file(path):
    """
    Analizza un singolo CSV:
      - statistiche base su tutti i campioni
      - auto-correlazioni (ACF)
      - test white noise: Box-Pierce, Ljung-Box
      - Durbin-Watson
      - Breusch-Godfrey
      - media come risposta y
    """
    df = load_csv_skip2(path)
    arr = extract_column_array(df, TARGET_COL)
    if len(arr) == 0:
        raise ValueError(f"{path} non contiene dati utili nella colonna {TARGET_COL}")

    # media reale della serie (usata come y)
    y_value = float(np.mean(arr))

    # serie centrata per test statistici
    arr_c = arr - y_value  

    # statistiche base
    stats_basic = compute_basic_stats(arr)

    # ACF
    nlags_for_acf = max(ACF_LAGS) if len(ACF_LAGS) > 0 else 20
    acfs = sample_acf(arr_c, nlags=nlags_for_acf)
    acf_dict = {f'acf_lag{lag}': float(acfs[lag]) if lag < len(acfs) else np.nan for lag in ACF_LAGS}

    # Portmanteau statistics -> white noise
    h = int(math.sqrt(len(arr_c)))
    wn_stats = portmanteau_stats(arr_c, h=h, model_params=0)

    # Durbin-Watson
    dw = durbin_watson_stat(arr_c)

    # Breusch-Godfrey su costante
    Xc = sm.add_constant(np.ones(len(arr_c)))
    ols_c = sm.OLS(arr_c, Xc).fit()
    try:
        bg_res = acorr_breusch_godfrey(ols_c, nlags=BG_LAGS)
        bg_lm, bg_pval = float(bg_res[0]), float(bg_res[1])
    except Exception:
        bg_lm, bg_pval = np.nan, np.nan

    result = {
        'file': os.path.basename(path),
        'n_samples_total': len(arr),
        'dw_stat': dw,
        'bg_lm': bg_lm,
        'bg_pval': bg_pval,
        'y': y_value
    }
    result.update(stats_basic)
    result.update(acf_dict)
    result.update({
        'Q_BP': wn_stats['Q_BP'],
        'p_BP': wn_stats['p_BP'],
        'Q_LB': wn_stats['Q_LB'],
        'p_LB': wn_stats['p_LB'],
        'h_wn': wn_stats['h'],
        'df_wn': wn_stats['df']
    })
    return result

def main(data_folder):
    # -------------------------
    # 0. Cartella output
    # -------------------------
    outdir = os.path.join(data_folder, "results")
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # 1. Scansione file CSV
    # -------------------------
    files = [
        f for f in os.listdir(data_folder)
        if f.lower().endswith('.csv')
        and not f.lower().startswith(('features_summary', 'anova_table', 'cv_results'))
    ]
    if not files:
        raise FileNotFoundError("Nessun CSV trovato nella cartella.")

    records = []
    for fname in sorted(files):
        fpath = os.path.join(data_folder, fname)
        try:
            T_label, RH_label, run, kind = parse_filename_for_factors(fname)
        except Exception as e:
            print(f"[WARN] Parsing filename {fname}: {e}. Skip.")
            continue

        try:
            # Analisi file singolo
            res = run_analysis_on_file(fpath)

            # Serie per plottaggio
            arr = extract_column_array(load_csv_skip2(fpath), TARGET_COL)
            plot_time_series(arr, fname, outdir)

        except Exception as e:
            print(f"[ERROR] File {fname} -> {e}. Skip.")
            continue

        # Aggiungi label e codifica fattori
        res['T_label'] = T_label
        res['RH_label'] = RH_label
        res['run'] = run
        res['kind'] = kind
        res['T_c'] = label_to_code(T_label)
        res['RH_c'] = label_to_code(RH_label)
        records.append(res)

    features_df = pd.DataFrame(records)
    if features_df.empty:
        raise RuntimeError("Nessun record valido prodotto.")

    print(features_df[['file','kind','y']])

    # -------------------------
    # 2. DoE: corner points -> regressione + ANOVA
    # -------------------------
    df_corner = features_df[features_df['kind'] == 'corner'].copy()
    if df_corner.empty:
        raise RuntimeError("Nessun corner point trovato per ANOVA.")

    df_corner['T_c'] = pd.to_numeric(df_corner['T_c'], errors='coerce')
    df_corner['RH_c'] = pd.to_numeric(df_corner['RH_c'], errors='coerce')

    model, anova_table, coeffs = fit_factorial_and_anova(df_corner[['y','T_c','RH_c']])

    features_df['resid_ols'] = np.nan
    features_df.loc[df_corner.index, 'resid_ols'] = model.resid.values

    # -------------------------
    # 3. Salvataggi
    # -------------------------
    out_features = os.path.join(outdir, 'features_summary.csv')
    features_df.to_csv(out_features, index=False)
    print(f"Features + residui OLS + statistiche white noise salvati in: {out_features}")

    anova_path = os.path.join(outdir, 'anova_table.csv')
    anova_table.to_csv(anova_path, index=True)

    summary_path = os.path.join(outdir, 'doe_regression_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("OLS SUMMARY\n")
        f.write(str(model.summary()))
        f.write("\n\nANOVA (Type II)\n")
        f.write(anova_table.to_string())
    print(f"Summary salvato in {summary_path}")

    # -------------------------
    # 4. LOEO CV
    # -------------------------
    df_for_cv = features_df[['y','T_c','RH_c']].copy()
    cv_df, cv_metrics = leave_one_experiment_out_cv(df_for_cv)

    cv_out = os.path.join(outdir, 'cv_results.csv')
    cv_df.to_csv(cv_out, index=False)
    print(f"CV results + residui LOEO salvati in {cv_out}")
    print("CV metrics:", cv_metrics)

    # -------------------------
    # 5. Grafici globali
    # -------------------------
    plot_experimental_scatter(features_df, savepath=os.path.join(outdir, "scatter_points.png"))
    plot_response_surface(model, features_df, savepath=os.path.join(outdir, "surface_plot.png"))

    return {
        'features_df': features_df,
        'model': model,
        'anova_table': anova_table,
        'cv_metrics': cv_metrics
    }
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python doe_pipeline.py /percorso/alla/cartella_con_CSV")
        sys.exit(1)
    data_folder = sys.argv[1]
    results = main(data_folder)
    print("\nFatto.")