import os
from stable_baselines3 import PPO
from environment import PortfolioEnv
import pandas as pd
import numpy as np

DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

def test_model(initial_capital=100_000):
    # test verisini yuklüyorum.
    prices = pd.read_csv(os.path.join(DATA_DIR, "test_prices.csv"), index_col=0)
    env = PortfolioEnv(prices, initial_capital=initial_capital)

    print(f"Test verisi: {len(prices)} gun (modelin hiç görmediği veriler kullanıldı)")
    
    model = PPO.load(os.path.join(MODELS_DIR, "ppo_model"))

    obs = env.reset()
    done = False

    values = [env.portfolio_value]
    rewards = []
    weights = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        values.append(info["portfolio_value"])
        rewards.append(reward)
        weights.append(env.current_weights.copy())


    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, "portfolio_values.csv")
    pd.DataFrame(values, columns=["PortfolioValue"]).to_csv(output_path, index=False)

   
    benchmark = (prices.pct_change().fillna(0).mean(axis=1) + 1).cumprod()
    passive_final = initial_capital * benchmark.values[-1]
    passive_kar = passive_final - initial_capital


    drl_kar = values[-1] - values[0]
    print(f"\n{'='*50}")
    print(f"DRL Stratejisi:    {values[0]:,.0f} TL -> {values[-1]:,.0f} TL ({drl_kar:+,.0f} TL)")
    print(f"Pasif Strateji:    {initial_capital:,.0f} TL -> {passive_final:,.0f} TL ({passive_kar:+,.0f} TL)")
    print(f"{'='*50}")
    
    if values[-1] > passive_final:
        fark = values[-1] - passive_final
        print(f"DRL, pasif stratejiden {fark:,.0f} TL daha iyi!")
    else:
        fark = passive_final - values[-1]
        print(f"Pasif strateji, DRL'den {fark:,.0f} TL daha iyi!")
    
    print(f"Kaydedildi: {output_path}")

    return values, rewards, weights
