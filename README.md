# Cryptocurrency Price Forecasting with a Hybrid LSTM-Transformer Architecture

This repository contains the implementation and experimental artifacts for the research project **"Cryptocurrency Price Forecasting with a Hybrid LSTM-Transformer Architecture"**. The work proposes a hybrid deep learning model that combines the strengths of Long Short-Term Memory (LSTM) networks and Transformer architectures through a learned fusion gate to improve multi-horizon cryptocurrency price forecasting.

The model is evaluated on three major cryptocurrencies — **Bitcoin (BTC)**, **Ethereum (ETH)**, and **Litecoin (LTC)** — using consolidated, scale-invariant feature representations derived from OHLCV data and technical indicators.

---

## 📌 Key Contributions

- A **hybrid LSTM–Transformer architecture** that captures both local sequential dependencies and long-range temporal relationships.

- A **learned fusion gate** that adaptively balances the contributions of the LSTM and Transformer branches.

- A **multi-cryptocurrency training setup** that enables cross-asset generalization across BTC, ETH, and LTC.

- Comprehensive evaluation across **1-day, 7-day, and 30-day forecasting horizons** using MAPE, RMSE, MAE, R², and directional accuracy.

---

## 📊 Dataset & Features

- **Data source** Kaggle – Cryptocurrency Timeseries 2020 (Gemini exchange)

- **Assets** Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC)

- **Granularity** Minute-level OHLCV data aggregated into daily candles

- **Features** 15 currency-agnostic technical indicators (momentum, volatility, trend, volume)

- **Sequence length** 30 days

All features are Min–Max normalized per asset to ensure robustness against differing price scales.

---

## 🧠 Model Overview

The proposed architecture consists of:

1. **LSTM Branch** – Captures short- and medium-term temporal dependencies.

2. **Transformer Branch** – Models global dependencies using multi-head self-attention.

3. **Fusion Gate** – A learnable sigmoid gate that adaptively weights both branches.

This design enables stable and accurate forecasting across multiple horizons and assets.

---

## 📈 Results Summary

The hybrid model consistently outperforms LSTM, BiLSTM, GRU, and standalone Transformer baselines in terms of:

- Lower Mean Absolute Percentage Error (MAPE)

- Higher directional accuracy

- Improved robustness across longer forecasting horizons

These results highlight the effectiveness of combining recurrent and attention-based architectures for volatile financial time series.

---

## 📜 License

This repository is licensed under the **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- ✔️ Academic and research use is allowed

- ❌ Commercial or personal financial use is prohibited

- 📌 Proper attribution and citation are required for any use

See the `LICENSE` file for full details.

---

## 📚 Citation (APA)

If you use this work in academic or research contexts, please cite:

Hritom, R., Aqib, H. M. H. J., & Roy, N. (2026). *Cryptocurrency Price Forecasting with a Hybrid LSTM-Transformer Architecture* https://github.com/rezwanhritom/Cryptocurrency-Price-Forecasting-with-a-Hybrid-LSTM-Transformer-Architecture

---

## 💬 Connect

If you find this repository helpful or want to discuss about it, feel free to connect or open an issue/discussion.

Created and maintained by: [Rezwanur Rahman](https://github.com/rezwanhritom)













