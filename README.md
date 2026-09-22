# Prediksi Harga Saham dengan LSTM

Soal 1 ujian akhir mata kuliah **Deep Learning** (DTSC6007001), BINUS University.

Model LSTM yang memprediksi harga penutupan (Close) satu hari ke depan untuk dua saham, **AAPL** (Apple) dan **AMD**, dari harga lima hari sebelumnya. Isi utamanya adalah eksperimen bertahap: apa yang benar-benar membuat model lebih baik, dan apa yang ternyata tidak.

![Prediksi dibanding harga aktual di data uji](saham-prediksi-vs-aktual.png)

## Hasil

Data uji adalah satu tahun terakhir (April 2019 sampai April 2020), yang mencakup crash awal pandemi COVID-19.

| Saham | Model | RMSE (USD) | MAE (USD) | MAPE |
|---|---|---|---|---|
| AAPL | Baseline: LSTM(50, ReLU), target harga | 18,85 | 12,44 | 4,66% |
| AAPL | **Final: LSTM(50, tanh), target selisih harga** | **6,32** | **3,94** | **1,61%** |
| AMD | Baseline | 1,44 | 0,94 | 2,51% |
| AMD | **Final** | **1,38** | **0,92** | **2,46%** |

## Temuan dari eksperimen

Sepuluh varian diuji pada kedua saham, supaya kesimpulannya tidak hanya kebetulan cocok untuk satu saham.

**Model yang lebih besar tidak membantu.** Menumpuk dua layer LSTM, menambah dropout, atau menambah layer Dense tidak konsisten membaik, dan varian paling kompleks justru paling buruk untuk AMD (RMSE 3,28 dibanding 1,44). Satu-satunya perubahan arsitektur yang membaik di kedua saham adalah yang paling sederhana: mengganti aktivasi ReLU menjadi tanh.

**Yang membantu adalah mengubah apa yang diprediksi.** Harga saham tidak stasioner. Uji Augmented Dickey-Fuller memberi p-value 0,999 untuk harga AAPL, sedangkan untuk selisih harga harian p-value-nya di bawah 0,001. Karena itu model dilatih memprediksi selisih terhadap harga hari terakhir di jendela input, lalu hasilnya dijumlahkan kembali dengan harga itu. Dengan arsitektur yang sama persis, RMSE AAPL turun dari 11,37 menjadi 6,32.

**Perbaikan terbesar ada di saham yang paling tidak stasioner.** AAPL naik ratusan kali lipat selama periode data, sedangkan AMD bergerak naik turun. Perubahan target membantu AAPL jauh lebih banyak daripada AMD.

**Error membesar saat pasar bergejolak.** Residual model tersebar di sekitar nol tanpa pola terhadap waktu, kecuali di akhir data uji. Pada periode crash Maret 2020, error AAPL melebar sampai sekitar 35 USD.

## Cara kerja

1. Hanya kolom Date dan Close yang dipakai.
2. Data dibagi berurutan waktu: satu tahun terakhir untuk uji, sisanya untuk latih. Data tidak diacak.
3. `MinMaxScaler` hanya di-fit pada data latih, supaya informasi data uji tidak bocor ke model.
4. Setiap sampel berisi harga 5 hari berturut-turut sebagai input dan harga hari berikutnya sebagai target.
5. 10% terakhir data latih dipakai sebagai validasi untuk early stopping.
6. Metrik dihitung setelah prediksi dikembalikan ke skala USD.

## Keterbatasan

- Model belum dibandingkan dengan patokan paling sederhana untuk data saham, yaitu "harga besok sama dengan harga hari ini". Model final memprediksi selisih dari harga terakhir, sehingga perbandingan ini perlu untuk menunjukkan seberapa besar nilai tambahnya.
- Hanya memakai harga penutupan. Volume, harga saham lain, dan berita tidak dipakai.
- Satu kali pembagian data latih dan uji. Hasil bisa berbeda di periode lain.

## Menjalankan

Notebook ditulis untuk Google Colab dengan TensorFlow. Letakkan `AAPL.csv` dan `AMD.csv` (kolom `Date` dan `Close`, harga harian) di folder yang sama dengan notebook, lalu jalankan semua sel.

Library: pandas, numpy, matplotlib, scikit-learn, statsmodels, TensorFlow/Keras.
