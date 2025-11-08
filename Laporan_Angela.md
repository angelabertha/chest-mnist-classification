**Nama:** Angela Bertha  
**NIM:** 122430137  
**Mata Kuliah:** Kecerdasan Buatan  

---

### 1. Mencoba Arsitektur Dasar
Langkah pertama yang saya lakukan adalah menjalankan seluruh arsitektur dasar dari repository asli untuk melihat apakah dapat berjalan di perangkat saya dan melihat akurasi akhirnya, yaitu file:
- `datareader.py`
- `model.py`
- `train.py`

Setelah menjalankan ketiganya, saya memperoleh hasil awal seperti yang terlihat pada Gambar berikut.

![alt text](<Screenshot (421)-1.png>)

---

### 2. Mencoba EfficientNet
Selanjutnya, saya mencoba mengganti model dasar dengan **EfficientNet**. Berdasarkan hasil pencarian saya pada laman pytorch weight, arsitektur Efficient Net merupakan salah satu arsitektur yang ringan, mengingat keterbatasa perangkat saya. Pada proses ini saya hanya mengganti arsitektur tanpa mengubah hyperparameter. 
Namun hasil akurasi justru mengalami penurunan dibandingkan model dasar sebelumnya.  
Untuk hasil menggunakan arsitektur ini tidak sempat saya dokumentasikan persentase pastinya karena tidak langsung melakukan commit setelah training selesai.

---

### 3. Mencoba ResNet18, Augmentasi Data, dan Mengubah Hyperparameter
Selanjutnya, saya mencoba langsung menggunakan **ResNet18** sebagai arsitektur model utama.  
Selain itu, saya juga menambahkan **augmentasi data** pada `datareader.py` dengan menambahkan transformasi berikut:

```python
def get_data_loaders(batch_size):
    # Augmentasi hanya untuk data training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),             
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
```

Saya menambahkan augmentasi data untuk meningkatkan kemampuan generalisasi model.
Dataset Chest MNIST memiliki variasi citra yang terbatas, baik dari segi rotasi, pencahayaan, maupun posisi objek. Dengan augmentasi seperti horizontal flip, rotation, dan affine transformation, model dapat belajar mengenali pola dari berbagai sudut pandang dan kondisi yang berbeda. Hal ini penting agar model tidak hanya hafal pada data training (overfitting), tetapi juga mampu beradaptasi dengan data baru saat validasi atau pengujian.

Selain itu, saya juga melakukan beberapa penyesuaian pada hyperparameter di file train.py untuk menyesuaikan karakteristik ResNet18 dan proses training yang lebih stabil. Pemilihan hyperparameter ini juga saya sesuaikan dengan perangkat saya yang terbatas.

### Hyperparameter
- `EPOCHS = 15`
- `BATCH_SIZE = 16`
- `LEARNING_RATE = 1e-4`
- `OPTIMIZER_TYPE = "Adam"  # bisa "SGD", "RMSprop", atau "Adam"`

Pemilihan Adam optimizer didasarkan pada kemampuannya menyesuaikan laju pembelajaran secara adaptif untuk setiap parameter, sehingga konvergensi bisa lebih cepat dan stabil dibandingkan optimizer lain seperti SGD. Learning rate sebesar 1e-4 dipilih agar perubahan bobot tidak terlalu agresif dan dapat menghindari osilasi pada proses training. EPOCHS sebesar 15 dipilih mengingat arsitektur resnet termasuk berat untuk perangkat saya, sehingga untuk EPOCHS 15 itu tidak memakan waktu terlalu lama.

### 4. Hasil Akhir
Setelah menjalankan proses training selama kurang lebih 6 jam, model menghasilkan performa akhir sebagai berikut:

![alt text](<Screenshot (416).png>)

- `Train Loss: 0.0859`
- `Train Accuracy: 95.58%`
- `Validation Loss: 0.3043`
- `Validation Accuracy: 87.98%`

Hasil ini menunjukkan adanya peningkatan dibandingkan model CNN sederhana sebelumnya.
Model ResNet18 mampu mengekstraksi fitur yang lebih kompleks dan mendalam, sementara augmentasi data membantu mengurangi overfitting dengan memperkaya variasi data training.