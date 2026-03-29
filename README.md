## Supply Chain Analytics & Profit Prediction using Ensemble Regressor

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.x-yellow" />
  <img src="https://img.shields.io/badge/Machine-Learning-blue" />
  <img src="https://img.shields.io/badge/Supply--Chain-Analytics-green" />
  <img src="https://img.shields.io/badge/Regression-Analysis-orange" />
  <img src="https://img.shields.io/badge/Metric-MAE%20%2F%20MSE-red" />
</p>

Di era Industri 4.0, efisiensi rantai pasok (supply chain) sangat bergantung pada kemampuan perusahaan dalam mengolah data besar (Big Data). Proyek ini bertujuan untuk melakukan analisis mendalam dan prediksi metrik finansial pada sistem rantai pasok global menggunakan teknik Machine Learning untuk mengoptimalkan pengambilan keputusan operasional.

### Table of Contents

- [Dataset](#dataset)
- [Libraries](#libraries)
- [Moving Average](#moving-average)
- [Sales Trend Analysis](#sales-trend-analysis)
- [Models](#models)
- [Results](#results)


### Dataset
Dataset yang digunakan dalam proyek ini adalah DataCo Smart Supply Chain, yang dapat diakses melalui [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis). .

Dataset ini terdiri dari sekitar 181.000 baris data dengan 52 kolom yang mencakup aspek logistik, informasi pelanggan, dan transaksi finansial. Beberapa kolom kunci meliputi:

* Days for shipping (real) : Durasi pengiriman aktual.
* Benefit per order : Keuntungan yang diperoleh per pesanan.
* Sales : Total nilai penjualan.
* Delivery Status : Status keberhasilan pengiriman.
* Late_delivery_risk : Indikator risiko keterlambatan.
* Order Country/City : Lokasi geografis pemesanan.

### Libraries
Proyek ini menggunakan ekosistem data science Python untuk pemrosesan dan pemodelan:

<pre>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
</pre>

### Data Cleaning & Preprocessing
Sebelum masuk ke tahap pemodelan, data melalui proses cleaning yang ketat:
* Currency Parsing: Membersihkan simbol mata uang ($) dan karakter non-numerik pada kolom finansial agar dapat diolah secara matematis.
* Outlier Handling: Menggunakan metode Interquartile Range (IQR) untuk mendeteksi anomali. Sebanyak 124 titik data ekstrem pada kolom permintaan dihapus untuk menjaga stabilitas model.
* Feature Engineering: Transformasi data waktu (order date) ke format datetime dan standarisasi fitur menggunakan StandardScaler.

<img src="./img/line.png" alt="Sales Trend" width="100%">

<br>
*The following figure shows the sales trend in the dataset over time for each car brand, providing an overview of overall patterns and fluctuations.*
<img src="./img/trend.png" alt="Sales Trend" width="100%">

### Models
This project applies an ensemble learning approach with bias correction to improve sales forecasting performance, focusing on the TOYOTA sales column. Instead of relying on a single model, multiple forecasting models are combined to capture different patterns in the data and reduce prediction bias.

The ensemble consists of three models:
* ARIMA, which captures linear patterns and temporal dependencies in time series data
* Prophet, which models trend and seasonality effectively
* LightGBM, which learns complex and non-linear relationships from the data

### Results
The table below presents the final sales prediction results for each automotive brand based on the ensemble learning model:

| Brand | Final Prediction |
| :--- | :---: |
| DAIHATSU | 7,412 |
| HONDA | 4,429 |
| MITSUBISHI | 6,739 |
| SUZUKI | 3,492 |
| TOYOTA | 13,313 |

The forecasting performance was evaluated using Symmetric Mean Absolute Percentage Error (SMAPE), resulting in a value of 28.22%.