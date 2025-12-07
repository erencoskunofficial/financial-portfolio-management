# Derin Pekistirmeli Ogrenme ile Finansal Portfoy Yonetimi

Bu proje, **Proximal Policy Optimization (PPO)** algoritmasi kullanarak Borsa Istanbul'daki hisse senetleri arasinda optimal portfoy dagilimini ogrenen bir yapay zeka sistemi olusturmak amaciyla gelistirilmistir.

## Proje Amaci

Bir yatirim portfoyunde hangi hisseye ne kadar yatirim yapilacagini otomatik olarak belirlemek. Sistem gecmis fiyat verilerini analiz ederek karı maksimize ederken ve riski minimize ederek dengeyi sağlar.

---

## Neden PPO Secildi?

### Alternatif Algoritmalar

| Algoritma | Aksiyon Tipi | Bu Problem Icin Uygunluk |
|-----------|--------------|--------------------------|
| Q-Learning | Ayrik (discrete) | Uygun degil |
| DQN | Ayrik (discrete) | Uygun degil |
| **PPO** | Surekli (continuous) | **En uygun** |

### PPO'nun Avantajlari

1. **Surekli Aksiyon Destegi**: Portfoy agirliklari %0-100 arasi surekli degerlerdir. Ayrik aksiyonlar bu problem için uygun olmayacaktır.

2. **Stabilite**: PPO, politika guncellemelerini sinirlandirarak egitim sirasinda ani performans dususlerini onler.

3. **Orneklem Verimliligi**: Ayni veriyi birden fazla kez kullanabilir, bu da finansal verilerin sinirli oldugu durumlarda onemlidir.

4. **Guvenilirlik**: OpenAI tarafindan gelistirilmis ve bircok karmasik problemde basarili oldugu kanitlanmistir.



## Sistem Mimarisi

```
+------------------------------------------------------------------+
|                           GIRDI                                   |
|  - Son 30 gunluk fiyat getirileri (logaritmik)                   |
|  - Mevcut portfoy agirliklari                                    |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        PPO MODELI                                 |
|  - MLP (Multi-Layer Perceptron) Politikasi                       |
|  - Actor-Critic Mimarisi                                         |
|  - 50.000 adim egitim                                            |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                           CIKTI                                   |
|  - 3 hisse icin agirlik dagilimi [w1, w2, w3]                    |
|  - Toplam = 1.0 (%100)                                           |
+------------------------------------------------------------------+
```

---

## Nasil Calisir?

1. **Veri Toplama**: Yahoo Finance'den belirlenen hisse senetlerinin gecmis fiyat verileri indirilir
2. **Egitim**: PPO modeli, 2020-2023 verileriyle egitilir (50.000 adim)
3. **Test**: Model hic gormedigi 2024 verileriyle test edilir
4. **Karsilastirma**: DRL stratejisi, pasif strateji (%33-33-33 esit dagilim) ile karsilastirilir

---

## Veri Seti

| Ozellik | Deger |
|---------|-------|
| Kaynak | Yahoo Finance |
| Hisseler | ASELS.IS, SISE.IS, TUPRS.IS |
| Egitim Donemi | 2020-01-01 - 2023-12-31 (~1000 gun) |
| Test Donemi | 2024-01-01 - 2024-12-31 (~250 gun) |
| Veri Tipi | Gunluk kapanis fiyatlari (Adjusted Close) |


## Odul Fonksiyonu

Model, asagidaki odul fonksiyonuyla egitilir:

```
odul = portfoy_getirisi - (risk_katsayisi x volatilite) - islem_maliyeti
```

### Bilesenler

| Bilesen | Formul | Amac |
|---------|--------|------|
| **Portfoy Getirisi** | sum(agirlik_i x getiri_i) | Kar maksimizasyonu (+) |
| **Risk Cezasi** | 0.4 x std(getiriler) | Asiri riskten kacinma (-) |
| **Islem Maliyeti** | 0.001 x agirlik_degisimi | Gereksiz alim-satimi onleme (-) |

### Neden Bu Odul Fonksiyonu?

1. **Sadece kar**: Model cok riskli yatirimlar yapabilir
2. **Kar - Risk**: Dengeli strateji ogrenir
3. **Kar - Risk - Maliyet**: Gercekci, uygulanabilir strateji ogrenir (secilen)

---

## Sonuclar

### Test Donemi Performansi (2024)

| Metrik | DRL Stratejisi | Pasif Strateji |
|--------|----------------|----------------|
| Baslangic | 300 TL | 300 TL |
| Son Deger | ~1,467 TL | ~892 TL |
| Toplam Getiri | ~%389 | ~%197 |
| Kar | +1,167 TL | +592 TL |

DRL stratejisi, pasif stratejiden yaklasik **575 TL daha fazla** kar elde etmistir.

---

## Grafikler

### 1. Portfoy Degeri Grafigi
`plots/portfoy_degeri.png`

Portfoy degerinin test donemi boyunca nasil degistigini gosterir.

### 2. DRL vs Pasif Strateji Karsilastirmasi
`plots/karsilastirma.png`

DRL stratejisinin pasif stratejiye (%33-33-33 esit dagilim) gore performansini karsilastirir. Surekli cizgi DRL, kesikli cizgi pasif stratejiyi gosterir.

### 3. Aylik Portfoy Agirliklari
`plots/aylik_agirliklar.png`

Modelin her ay hangi hisseye ne kadar agirlik verdigini gosterir. Model, piyasa kosullarina gore dinamik olarak agirlik degistirir.

### 4. Odul Egrisi
`plots/odul_egrisi.png`

Test sirasinda modelin aldigi odullerin zamanla nasil degistigini gosterir. Yumusatilmis egri, genel trendi gosterir.

---

## Teknik Detaylar

### Hiperparametreler

| Parametre | Deger | Aciklama |
|-----------|-------|----------|
| window_size | 30 | Gozlem penceresi (gun) |
| risk_aversion | 0.4 | Risk ceza katsayisi |
| transaction_cost | 0.001 | Islem maliyeti (%0.1) |
| total_timesteps | 50,000 | Egitim adim sayisi |
| policy | MlpPolicy | Cok katmanli algilayici |

### Gozlem Uzayi (Observation Space)

```
obs = [son_30_gun_getirileri] + [mevcut_agirliklar]
    = [90 deger]              + [3 deger]
    = 93 boyutlu vektor
```

### Aksiyon Uzayi (Action Space)

```
action = [w1, w2, w3]  // her biri 0-1 arasi
// Normalize edilir: w1 + w2 + w3 = 1
```

---

## Kurulum

```bash
# repoyu klonla
git clone <repo-url>
cd finansal-portfoy-yonetimi

# gereksinimleri yukle
pip install -r requirements.txt
```

## Kullanim

```bash
python main.py
```

Bu komut sirasiyla:
1. Verileri indirir (yoksa)
2. Modeli egitir (yoksa)
3. Test yapar
4. Grafikleri olusturur
5. PDF rapor uretir

---

## Ayarlar

`main.py` dosyasinda asagidaki parametreleri degistirebilirsiniz:

```python
tickers = ["ASELS.IS", "SISE.IS", "TUPRS.IS"]  # hisse senetleri
INITIAL_CAPITAL = 300  # baslangic sermayesi (TL)
```

`data.py` dosyasinda tarih araligini degistirebilirsiniz:

```python
start="2020-01-01"      # veri baslangic tarihi
end="2025-01-01"        # veri bitis tarihi  
train_end="2024-01-01"  # egitim/test ayrim tarihi
```

---

## Ciktilar

### Metrikler
- Baslangic Sermayesi
- DRL Son Deger / Kar-Zarar / Getiri
- Pasif Strateji Son Deger / Kar-Zarar / Getiri
- Sharpe Orani
- Maksimum Dusus (Drawdown)

### Cikti Dosyalari
- `outputs/portfolio_values.csv` - Gunluk portfoy degerleri
- `plots/*.png` - Performans grafikleri
- `reports/rapor.pdf` - PDF performans raporu

---

## Notlar

- Bu proje Bursa Teknik Universitesi Derin ve Pekistirmeli Ogrenme Vize Odevi kapsaminda gelistirilmistir.
