#!/usr/bin/env python3
"""
doe_pipeline.py
Pipeline per: preprocessing -> test di bianchezza -> DoE regression + ANOVA + LOEO CV

Usage:
  python doe_pipeline.py /percorso/alla/cartella_con_i_csv

Assume file con header alla riga 3 (skiprows=2) e colonna "Acc_Z [g]" presente.
File example names:
  TempBassa_UmBassa_Run_1.csv
  TempAlta_UmBassa_Run_2.csv
  CenterPoints_Run_1.csv
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TARGET_COL = "Acc_Z [g]"

# -------------------------
# FUNZIONI DI SUPPORTO
# -------------------------

def load_csv_skip2(path, colname=TARGET_COL):
    """Leggi CSV con skiprows=2, decimal=',' e ritorna DataFrame."""
    return pd.read_csv(path, sep=",", decimal=",", skiprows=2)

def extract_column_array(df, colname=TARGET_COL):
    if colname not in df.columns:
        candidates = [c for c in df.columns if 'acc' in c.lower() and 'z' in c.lower()]
        if len(candidates) > 0:
            colname = candidates[0]
        else:
            raise ValueError(f"Colonna {colname} non trovata. Colonne disponibili: {df.columns.tolist()}")
    arr = pd.to_numeric(df[colname], errors='coerce').to_numpy()
    arr = arr[~np.isnan(arr)]
    return arr

def parse_filename_for_factors(fname):
    """
    Parse filename e restituisce (T_label, RH_label, run, kind).
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if parts[0].lower().startswith("centerpoints"):
        return ('center', 'center', 1, 'center')
    if "TempBassa" in base: T_label = 'low'
    elif "TempAlta" in base: T_label = 'high'
    else: T_label = 'center'
    if "UmBassa" in base: RH_label = 'low'
    elif "UmAlta" in base: RH_label = 'high'
    else: RH_label = 'center'
    run = int(parts[-1]) if parts[-1].isdigit() else 1
    kind = 'corner' if 'center' not in (T_label, RH_label) else 'center'
    return (T_label, RH_label, run, kind)

def label_to_code(label):
    return {'low': -1, 'center': 0, 'high': 1}[label]

def leave_one_experiment_out_cv(df_design_features):
    df = df_design_features.reset_index(drop=True).copy()
    N = len(df)
    preds, trues, resid = [], [], []
    for i in range(N):
        train = df.drop(i)
        test = df.loc[[i]]
        m = ols('y ~ T_c * RH_c', data=train).fit()
        yhat = m.predict(test).iloc[0]
        ytrue = test['y'].values[0]
        preds.append(yhat)
        trues.append(ytrue)
        resid.append(yhat - ytrue)
    preds, trues, resid = np.array(preds), np.array(trues), np.array(resid)
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    r2 = float(1 - np.sum((trues - preds)**2) / np.sum((trues - np.mean(trues))**2))
    return pd.DataFrame({'y': trues, 'yhat_loeo': preds, 'resid_loeo': resid}), {
        'rmse_loeo': rmse, 'mae_loeo': mae, 'bias_loeo': bias, 'r2_loeo': r2
    }

# -------------------------
# MAIN ANALISI
# -------------------------

def main(data_folder):
    files = [
        f for f in os.listdir(data_folder)
        if f.lower().endswith('.csv')
        and ("run" in f.lower() or "center" in f.lower())
        and not f.lower().startswith(("features_summary", "anova_table", "cv_results", "design"))
    ]
    if len(files) == 0:
        raise FileNotFoundError("Nessun CSV trovato nella cartella.")

    records = []
    for fname in sorted(files):
        fpath = os.path.join(data_folder, fname)
        T_label, RH_label, run, kind = parse_filename_for_factors(fname)
        df = load_csv_skip2(fpath)
        arr = extract_column_array(df, TARGET_COL)
        y_value = float(np.mean(arr))  # media
        T_real = float(df['Temperature [°C]'].iloc[0]) if 'Temperature [°C]' in df.columns else np.nan
        RH_real = float(df['Humidity [%]'].iloc[0]) if 'Humidity [%]' in df.columns else np.nan
        records.append({
            'file': fname, 'y': y_value,
            'T_label': T_label, 'RH_label': RH_label,
            'T_c': label_to_code(T_label), 'RH_c': label_to_code(RH_label),
            'T_c_real': T_real, 'RH_c_real': RH_real,
            'run': run, 'kind': kind
        })

    df_all = pd.DataFrame(records)
    print("\n=== MEDIE ESPERIMENTI ===")
    print(df_all[['file', 'y']])

    # ---- Calcolo corner point sums
    df_corner = df_all[df_all['kind'] == 'corner']
    corner_means = df_corner.groupby(['T_label', 'RH_label'])['y'].sum().to_dict()
    sums = {
        '(1)': corner_means.get(('low','low'), 0),
        'a':   corner_means.get(('high','low'), 0),
        'b':   corner_means.get(('low','high'), 0),
        'ab':  corner_means.get(('high','high'), 0)
    }
    n = 2  # due run per corner
    A = (sums['ab'] + sums['a'] - sums['b'] - sums['(1)']) / (2*n)
    B = (sums['ab'] + sums['b'] - sums['a'] - sums['(1)']) / (2*n)
    AB = (sums['ab'] + sums['(1)'] - sums['a'] - sums['b']) / (2*n)

    print("\n=== SOMME CORNER POINTS ===")
    print(sums)
    print(f"A = {A}, B = {B}, AB = {AB}")

    # ---- ANOVA sui corner
    model = ols('y ~ T_c * RH_c', data=df_corner).fit()
    anova_table = anova_lm(model, typ=2)
    print("\n=== ANOVA (corner points) ===")
    print(anova_table)

    # ---- Coefficienti di regressione
    coeffs = model.params.to_dict()
    b0_theoretical = (sums['(1)'] + sums['a'] + sums['b'] + sums['ab']) / (4*n)
    b1_theoretical = A/2
    b2_theoretical = B/2
    b12_theoretical = AB/2
    print("\n=== COEFFICIENTI REGRESSIONE ===")
    print("Stimati:", coeffs)
    print("Teorici:", {"b0": b0_theoretical, "b1": b1_theoretical,
                       "b2": b2_theoretical, "b12": b12_theoretical})

    # ---- Metriche modello
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    print(f"\nR² = {r2:.4f}, Adjusted R² = {r2_adj:.4f}")

    # ---- Response surface con valori reali
    T_vals = np.linspace(df_all['T_c_real'].min(), df_all['T_c_real'].max(), 30)
    RH_vals = np.linspace(df_all['RH_c_real'].min(), df_all['RH_c_real'].max(), 30)
    T_grid, RH_grid = np.meshgrid(T_vals, RH_vals)
    #df_pred = pd.DataFrame({'T_c': np.sign(T_grid.ravel()-np.mean(T_vals)),
    #                        'RH_c': np.sign(RH_grid.ravel()-np.mean(RH_vals))})

    # codifica continua da valori reali a [-1,1] basata su min/max corner points
    T_min, T_max = df_corner['T_c_real'].min(), df_corner['T_c_real'].max()
    RH_min, RH_max = df_corner['RH_c_real'].min(), df_corner['RH_c_real'].max()

    T_c_grid = 2 * (T_grid.ravel() - T_min) / (T_max - T_min) - 1
    RH_c_grid = 2 * (RH_grid.ravel() - RH_min) / (RH_max - RH_min) - 1

    df_pred = pd.DataFrame({'T_c': T_c_grid, 'RH_c': RH_c_grid})

    Y_pred = model.predict(df_pred).values.reshape(T_grid.shape)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_grid, RH_grid, Y_pred, alpha=0.6, cmap='viridis')
    ax.scatter(df_all['T_c_real'], df_all['RH_c_real'], df_all['y'],
               c='r', s=60, depthshade=True, label='Punti sperimentali')
    ax.set_xlabel("A (Temperature [°C])")
    ax.set_ylabel("B (Relative Humidity [%])")
    ax.set_zlabel("y (avg) [g]")
    ax.set_title("Response Surface Plot")
    ax.legend()
    plt.show()

    # ---- LOEO CV su tutti i 12 esperimenti
    cv_df, cv_metrics = leave_one_experiment_out_cv(df_all[['y','T_c','RH_c']])
    print("\n=== LOEO METRICHE ===")
    print(cv_metrics)

    print("\n=== MODEL SUMMARY ===")
    print(model.summary())

# Salvo anche su file
    summary_path = os.path.join(data_folder, 'doe_regression_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== MODEL SUMMARY ===\n")
        f.write(str(model.summary()))
        f.write("\n\n=== ANOVA TABLE (Type II) ===\n")
        f.write(anova_table.to_string())
    print(f"Summary salvato in {summary_path}")



    return {
        'features_df': df_all,
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
