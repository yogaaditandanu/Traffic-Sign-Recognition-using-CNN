# Traffic-Sign-Recognition-using-CNN

# Proyek Klasifikasi Gambar: German Traffic Sign Recognition (GTSRB)

Proyek ini adalah submission akhir untuk kelas Machine Learning. Tujuan utamanya adalah membangun model **Convolutional Neural Network (CNN)** yang mampu mengklasifikasikan **43 jenis rambu lalu lintas** dari dataset GTSRB dengan akurasi tinggi.

## 👨‍💻 Profil Pengembang
* **Nama:** Yoga Adi Tandanu
* **Email:** yogaaditandanu@gmail.com
* **ID Dicoding:** yogaaditandanu

---

## 📊 Deskripsi Dataset
Dataset yang digunakan adalah **German Traffic Sign Recognition Benchmark (GTSRB)**.
* **Sumber:** Kaggle
* **Jumlah Gambar:** > 50.000 citra
* **Jumlah Kelas:** 43 Kategori Rambu Lalu Lintas
* **Resolusi:** Beragam (di-resize menjadi 30x30 piksel untuk model ini)

> **Catatan:** Dataset ini memenuhi kriteria "Minimal 1000 gambar" dan "Bukan dataset RPS/X-Ray".

---

## 🛠️ Struktur Direktori Submission
Berikut adalah struktur folder yang dikirimkan dalam file `.zip`:

submission
├───tfjs_model
| ├───group1-shard1of1.bin
| └───model.json
├───tflite
| ├───model.tflite
| └───label.txt
├───saved_model
| ├───saved_model.pb
| └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt


---

## 🚀 Langkah-Langkah Pembuatan Model (Metodologi)

1.  **Data Loading & Splitting:**
    * Menggunakan library `split-folders` untuk membagi dataset secara fisik menjadi 3 bagian wajib:
        * **Train Set (80%)**: Untuk melatih model.
        * **Validation Set (10%)**: Untuk validasi saat training.
        * **Test Set (10%)**: Untuk evaluasi akhir (Unseen Data).

2.  **Data Preprocessing & Augmentation:**
    * **Rescaling:** Normalisasi nilai piksel (1./255).
    * **Augmentasi:** Diterapkan **HANYA pada Train Set** (Rotation, Zoom, Shift) untuk mencegah overfitting dan memperkaya variasi data.

3.  **Arsitektur Model (CNN):**
    * Model dibangun menggunakan **TensorFlow Keras Sequential API**.
    * Terdiri dari 3 blok konvolusi (`Conv2D` + `MaxPooling2D`).
    * Hidden Layer dengan 512 neuron (`Dense`).
    * Output Layer dengan 43 neuron dan aktivasi `Softmax`.
    * Menggunakan `Dropout(0.5)` untuk regularisasi.

4.  **Training & Evaluasi:**
    * Optimizer: `Adam`
    * Loss Function: `Categorical Crossentropy`
    * Metrics: `Accuracy`
    * **Callback:** Menggunakan `EarlyStopping` atau custom callback untuk menghentikan training jika akurasi > 96%.

---

## 📈 Hasil & Performa
Berdasarkan proses training selama 15 epoch, model berhasil mencapai performa berikut:

* **Training Accuracy:** ~97%
* **Validation Accuracy:** ~95%
* **Test Set Accuracy:** **> 95%**

> **Hasil ini jauh melampaui target minimal kelulusan yaitu 85%.**

Plot grafik Akurasi dan Loss juga disertakan di dalam notebook untuk menunjukkan bahwa model **Good Fit** (tidak underfitting/overfitting parah).

---

## 📱 Format Model (Deployment)
Model telah diekspor ke dalam tiga format standar industri:
1.  **SavedModel:** Format default TensorFlow.
2.  **TF-Lite:** Ukuran lebih kecil, siap digunakan di Android/iOS.
3.  **TFJS:** Siap digunakan di browser menggunakan JavaScript.

---

## ⚙️ Cara Menjalankan (Dependencies)
Untuk menjalankan notebook ini, pastikan library berikut terinstal (lihat `requirements.txt`):
* tensorflow
* pandas
* numpy
* matplotlib
* split-folders
* tensorflowjs

Jalankan perintah berikut untuk menginstal dependensi:
```bash
pip install -r requirements.txt
