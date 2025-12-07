import os
from stable_baselines3 import PPO
from environment import PortfolioEnv
import pandas as pd

DATA_DIR = "data"
MODELS_DIR = "models"

def train_model(initial_capital=100_000):
   
    prices = pd.read_csv(os.path.join(DATA_DIR, "train_prices.csv"), index_col=0)
    env = PortfolioEnv(prices, initial_capital=initial_capital)

    print(f"Egitim verisi: {len(prices)} gun")
    print(f"PPO DRLF  modeli egitiliyor... (Baslangic Sermayesi: {initial_capital:,} TL)")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)

    # modeli kaydet
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "ppo_model")
    model.save(model_path)
    print(f"Model kaydedildi: {model_path}.zip")
