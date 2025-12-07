import os
import pandas as pd
from data import load_data
from train import train_model
from test import test_model
from metrics import get_metrics
from plots import plot_results, plot_benchmark, plot_rewards, plot_monthly_weights
from report import generate_report

# klasor yollari
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
PLOTS_DIR = "plots"
REPORTS_DIR = "reports"

tickers = ["ASELS.IS", "SISE.IS", "TUPRS.IS"]
INITIAL_CAPITAL = 300  # baslangic sermayesi (TL)

def create_directories():
    """gerekli klasorleri olustur"""
    for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR]:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    
    # klasorleri olustur
    create_directories()

    # veri dosyalari yoksa indir
    train_path = os.path.join(DATA_DIR, "train_prices.csv")
    test_path = os.path.join(DATA_DIR, "test_prices.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Veriler indirdim.")
        load_data(tickers)
    else:
        print("Egitim ve test verileri zaten var.")
        

    # model yoksa egit
    model_path = os.path.join(MODELS_DIR, "ppo_model.zip")
    if not os.path.exists(model_path):
        print("\nModel bulunamadi eğitimi başlatıyorum...")
        train_model()
    else:
        print("\nZaten kayıtlı model var, eğitimi atladım.")

    # test et ve cikti degerlerini al
    values, rewards, weights = test_model(initial_capital=INITIAL_CAPITAL)

    # metrikleri hesapla
    metrics = get_metrics(values)

    # grafikleri olustur
    plot_rewards(rewards)
    plot_results(values)
    plot_benchmark(values)
    plot_monthly_weights(weights)

    # pdf rapor olustur
    generate_report(metrics)

    print("\n" + "="*50)
    print("Tum işlemler tamamlandı ")
 
