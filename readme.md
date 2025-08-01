Tentu, ini adalah penjelasan yang telah dirapikan dan diformat dalam format Markdown agar mudah dibaca di `README.md`.

---

# Penjelasan Kode: rtspv2.py

Skrip `rtspv2.py` ini dirancang untuk melakukan deteksi objek (khususnya orang) secara *real-time* dari stream video RTSP atau webcam. Sistem ini mampu melacak individu yang terdeteksi, menyimpan bukti gambar, dan mengirimkan notifikasi melalui API eksternal secara asinkron.

Seluruh perilaku skrip, mulai dari model yang digunakan, sumber video, hingga target API, diatur melalui file konfigurasi `config.yaml`.

---

## Fungsionalitas Inti

* **Deteksi Real-time**: Menggunakan model YOLO untuk mendeteksi objek pada setiap frame video.
* **Pelacakan Objek**: Memberikan `track_id` unik untuk setiap objek yang terdeteksi agar dapat dilacak pergerakannya antar frame.
* **Region of Interest (ROI)**: Memungkinkan deteksi hanya pada area yang telah ditentukan untuk efisiensi dan fokus.
* **Notifikasi & Logging Asinkron**: Pengiriman gambar bukti dan data log ke API eksternal (WhatsApp, Server ZAI) dilakukan di *thread* terpisah agar tidak menghambat proses deteksi utama.
* **Konfigurasi Terpusat**: Semua parameter penting diatur dalam satu file `config.yaml`.
* **Logging Detail**: Mencatat semua aktivitas penting ke dalam file log untuk kemudahan *debugging* dan audit.

---

## Alur Kerja Rinci

Berikut adalah penjelasan detail mengenai setiap komponen dalam kode.

### 1. Setup dan Inisialisasi

Bagian ini mempersiapkan semua yang dibutuhkan sebelum deteksi dimulai.
* **Impor Library**: Mengimpor semua pustaka yang diperlukan, seperti `OpenCV` untuk video, `YOLO` untuk deteksi, `threading` untuk tugas asinkron, dan `requests` untuk komunikasi API.
* **Konfigurasi Logging**: Menggunakan fungsi `setup_logger` untuk membuat sistem logging yang mencatat output ke file (`.log`) dan menampilkannya di konsol secara bersamaan.

### 2. Kelas `RealTimeDetector`

Ini adalah kelas utama yang membungkus semua logika deteksi.
* **Muat Konfigurasi**: Membaca semua pengaturan dari file `config.yaml`, termasuk path model YOLO, *device* (`cpu`/`gpu`), *threshold* deteksi, path penyimpanan, konfigurasi API, dan definisi ROI.
* **Inisialisasi Model**: Memuat model deteksi objek YOLO dan memindahkannya ke *device* yang dipilih.
* **Manajemen Status**: Menyiapkan struktur data untuk mengelola status setiap orang yang terlacak (`track_id`), termasuk kapan pertama dan terakhir kali terlihat, serta status notifikasi untuk menghindari pengiriman berulang.
* **Executor Asinkron**: Membuat `ThreadPoolExecutor` dengan maksimal 5 *thread* untuk menangani tugas-tugas yang memakan waktu (seperti penyimpanan file dan pengiriman API).

### 3. Loop Utama (Metode `run`)

Metode ini adalah jantung dari proses deteksi *real-time*.
1.  **Pilih Sumber Video**: Membuka stream video dari RTSP atau webcam sesuai dengan konfigurasi.
2.  **Loop Baca Frame**: Terus-menerus membaca frame dari sumber video. Jika koneksi gagal, skrip akan mencoba menyambung kembali setelah 5 detik.
3.  **Penanganan ROI**: Jika ROI aktif, frame akan di-masking sehingga deteksi hanya dilakukan pada area yang telah ditentukan.
4.  **Deteksi Objek**: Menjalankan deteksi dan pelacakan objek menggunakan `self.model.track()` pada frame yang telah diproses.
5.  **Proses & Tampilkan**: Hasil deteksi diproses lebih lanjut oleh `_process_detections` dan divisualisasikan (misalnya, dengan kotak pembatas) pada frame yang akan ditampilkan di jendela.
6.  **Keluar**: Loop akan berhenti jika pengguna menekan tombol **'q'**.
7.  **Pembersihan**: Setelah loop selesai, semua sumber daya (video capture, window, thread executor) akan dilepaskan dengan benar.

### 4. Proses Deteksi (Metode `_process_detections`)

Metode ini dipanggil di setiap frame untuk memproses setiap objek yang terdeteksi oleh YOLO.
1.  **Filter Deteksi**: Memeriksa `confidence score` dan kelas objek (misalnya, hanya memproses `person`).
2.  **Filter ROI**: Memastikan pusat objek berada di dalam *Region of Interest* (jika fitur ini diaktifkan).
3.  **Update Status**: Memperbarui waktu pertama dan terakhir kali sebuah `track_id` terlihat.
4.  **Pemicu Notifikasi**: Jika sebuah `track_id` terdeteksi secara terus-menerus melebihi durasi *threshold* yang ditentukan (misal: 3 detik) dan belum pernah dikirimi notifikasi, maka tugas asinkron `_handle_persistent_detection` akan dijalankan untuk `track_id` tersebut.

### 5. Tugas Asinkron (`_handle_persistent_detection`)

Fungsi ini berjalan di *background* agar tidak mengganggu loop deteksi utama.
* **Simpan Gambar Bukti**:
    * **`captures`**: Menyimpan gambar asli (tanpa kotak deteksi) atau hasil *crop* dari objek yang terdeteksi.
    * **`framerecord`**: Jika diaktifkan, menyimpan seluruh frame (lengkap dengan kotak deteksi) sebagai konteks tambahan.
* **Kirim Notifikasi WhatsApp**: Jika diaktifkan, mengirim gambar bukti dari direktori `captures` beserta pesan ke API WhatsApp.
* **Kirim Log ke Server**: Jika diaktifkan, mengirim data log kejadian dalam format JSON ke server ZAI.

### 6. Pengiriman API (`_send_api_request`)

Fungsi utilitas ini bertanggung jawab untuk semua komunikasi dengan API eksternal.
* **Robust**: Dirancang untuk menangani pengiriman data `JSON` dan file gambar (`multipart/form-data`).
* **Mekanisme Retry**: Dilengkapi dengan logika *retry* (mencoba ulang hingga 3 kali) jika terjadi kegagalan koneksi atau *timeout* untuk memastikan pengiriman yang andal.