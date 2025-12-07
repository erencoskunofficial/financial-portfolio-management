import os
import numpy as np
import pandas as pd

DATA_DIR = "data"

def get_metrics(values):
    pv = np.array(values)
    ret = np.diff(pv) / pv[:-1]
    cumulative_return = pv[-1] / pv[0] - 1
    sharpe_ratio = ret.mean() / (ret.std() + 1e-8)
    max_curve = np.maximum.accumulate(pv)
    drawdown = (pv - max_curve) / max_curve
    max_drawdown = drawdown.min()
    
    # pasif strateji hesapla
    prices = pd.read_csv(os.path.join(DATA_DIR, "test_prices.csv"), index_col=0)
    benchmark = (prices.pct_change().fillna(0).mean(axis=1) + 1).cumprod()
    passive_return = benchmark.values[-1] - 1
    passive_final = pv[0] * benchmark.values[-1]
    
    return {
        "Baslangic Sermayesi": f"{pv[0]:,.0f} TL",
        "DRL Son Deger": f"{pv[-1]:,.0f} TL",
        "DRL Kar/Zarar": f"{pv[-1] - pv[0]:+,.0f} TL",
        "DRL Getiri": f"%{cumulative_return * 100:.2f}",
        "Pasif Son Deger": f"{passive_final:,.0f} TL",
        "Pasif Kar/Zarar": f"{passive_final - pv[0]:+,.0f} TL",
        "Pasif Getiri": f"%{passive_return * 100:.2f}",
        "Sharpe Orani": round(sharpe_ratio, 4),
        "Maksimum Dusus": f"%{max_drawdown * 100:.2f}",
    }

def evaluate(values):
    metrics = get_metrics(values)
    print("\nBasari Metrikleri")
    print(metrics)
    return metrics
