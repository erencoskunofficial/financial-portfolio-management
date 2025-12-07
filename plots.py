import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

plt.style.use("seaborn-v0_8")

DATA_DIR = "data"
PLOTS_DIR = "plots"

def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_results(values):
    _ensure_plots_dir()
    values = np.array(values)

    plt.figure(figsize=(10,5), dpi=120)
    plt.plot(values, linewidth=2)
    plt.title("Portfoy Degerinin Zaman Icinde Gelisimi", fontsize=12)
    plt.xlabel("Gun")
    plt.ylabel("Portfoy Degeri")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "portfoy_degeri.png"), bbox_inches="tight")
    plt.close()


def plot_benchmark(values):
    _ensure_plots_dir()
    # test verisini kullan
    prices = pd.read_csv(os.path.join(DATA_DIR, "test_prices.csv"), index_col=0)
    benchmark = (prices.pct_change().fillna(0).mean(axis=1) + 1).cumprod()

    values = np.array(values)
    # ayni baslangic noktasina normalize et
    benchmark_scaled = benchmark.values * (values[0] / benchmark.values[0])

    plt.figure(figsize=(10,5), dpi=120)
    plt.plot(values, label="DRL Portfoy", linewidth=2)
    plt.plot(benchmark_scaled, label="Pasif Portfoy (33-33-33)", linestyle="--")
    plt.legend()
    plt.title("DRL Stratejisi vs Pasif Yatirim Stratejisi TEST PERİYODU")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "karsilastirma.png"), bbox_inches="tight")
    plt.close()


def plot_monthly_weights(weights):
    _ensure_plots_dir()
    weights = np.array(weights)

    df = pd.DataFrame(weights, columns=["Agirlik 1", "Agirlik 2", "Agirlik 3"])
    # test donemi 2024'ten basliyor
    df["Month"] = pd.date_range(start="2024-01-01", periods=len(weights), freq="D").to_period("M")
    monthly = df.groupby("Month").mean()

    plt.figure(figsize=(10,5), dpi=120)
    for col in ["Agirlik 1", "Agirlik 2", "Agirlik 3"]:
        plt.plot(monthly.index.astype(str), monthly[col], marker="o", label=col)

    plt.title("Aylik Ortalama Portfoy Ağirliklari Grafiği", fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel("Ay")
    plt.ylabel("Agirlik")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "aylik_agirliklar.png"), bbox_inches="tight")
    plt.close()


def plot_rewards(rewards):
    _ensure_plots_dir()
    rewards = np.array(rewards)

    # daha anlasilir hale getirmek icin yumusatma uygula
    smoothed = uniform_filter1d(rewards, size=15)

    plt.figure(figsize=(10,4), dpi=120)
    plt.plot(rewards, alpha=0.4, label="Ham Odul")
    plt.plot(smoothed, linewidth=2, label="Duzgunlestirilmis Odul")
    plt.title("Ogrenme Sureci - Odul Eğrisi", fontsize=12)
    plt.xlabel("Adim")
    plt.ylabel("Odul")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "odul_egrisi.png"), bbox_inches="tight")
    plt.close()
