from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
import requests

# Import Library

# Mengimpor YOLO dari ultralytics untuk deteksi objek.
# Mengimpor OpenCV (cv2) untuk pengolahan gambar/video.
# Mengimpor modul waktu, OS, datetime, zona waktu, dan requests untuk HTTP.
# Inisialisasi Model dan RTSP

# Model YOLO dimuat dari file yolo11m.pt.
# RTSP stream diakses menggunakan URL kamera CCTV.
# Persiapan Direktori

# Membuat folder capturerealtime untuk menyimpan frame realtime.
# Tracking dan Waktu

# Menyimpan ID orang yang sudah terdeteksi dan waktu pertama kali terlihat.
# Loop Utama (Deteksi Real-Time)

# Membaca frame dari stream RTSP.
# Jika gagal, program berhenti.
# Deteksi dan Tracking

# Menggunakan YOLO untuk mendeteksi dan tracking objek pada frame.
# Hanya memproses jika ada kotak deteksi (boxes).
# Visualisasi Deteksi

# Untuk setiap box, jika kelas adalah "person" (cls_id == 0), confidence ≥ 0.6, dan ada track_id:
# Membuat kotak hijau dan label pada frame.
# Simpan Frame Realtime

# Setiap detik, menyimpan frame asli dan frame dengan kotak ke folder capturerealtime.
# Simpan Deteksi Person

# Untuk setiap orang yang terdeteksi lebih dari 2 detik dan belum pernah disimpan:
# Menyimpan frame ke folder bbox-<tanggal>.
# Nama file berisi track_id dan timestamp.
# Mengirim alert WhatsApp (WA) via API (endpoint kosong, perlu diisi).
# Mengirim log capture ke server ZAI (endpoint kosong, perlu diisi).
# Keluar Program

# Jika tombol 'q' ditekan, program berhenti dan melepaskan resource kamera.
# Kesimpulan:
# Kode ini melakukan deteksi dan tracking orang secara real-time dari stream RTSP, menyimpan gambar setiap detik, dan menyimpan gambar orang yang terdeteksi lebih dari 2 detik. Selain itu, kode mencoba mengirim notifikasi WA dan log ke server eksternal.

# Muat model YOLO
model = YOLO("yolo11m.pt")

# RTSP stream URL
rtsp_url = "rtsp://admin:KAQSML@172.16.6.77:554/"
cap = cv2.VideoCapture(rtsp_url)

# Direktori penyimpanan
base_dir = "imagerecord"
realtime_dir = os.path.join(base_dir, "capturerealtime")
os.makedirs(realtime_dir, exist_ok=True)

# Tracking ID dan waktu
seen_person_ids = set()
person_entry_times = {}

# Konfigurasi capture
capture_interval = 1
last_capture_time = time.time()

print("🎥 Mulai deteksi dari RTSP...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame dari RTSP.")
        break

    results = model.track(
        source=frame,
        persist=True,
        save=False,
        stream=False
    )

    result = results[0]
    boxes = result.boxes
    if boxes is None:
        continue

    current_time = time.time()
    frame_with_box = frame.copy()

    # Buat kotak dan label hanya untuk person
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        track_id = int(box.id[0].item()) if box.id is not None else -1

        if cls_id == 0 and conf >= 0.6 and track_id != -1:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = f"Person ({track_id}) - {conf*100:.1f}%"

            cv2.putText(frame_with_box, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, lineType=cv2.LINE_AA)
            
            cv2.rectangle(frame_with_box, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

    # Simpan realtime frame setiap detik
    if current_time - last_capture_time >= capture_interval:
        realtime_path = os.path.join(realtime_dir, "realtime.jpg")
        box_path = os.path.join(realtime_dir, "realtime-withbox.jpg")
        cv2.imwrite(realtime_path, frame)
        cv2.imwrite(box_path, frame_with_box)
        print(f"[📸] Realtime frame diperbarui: {realtime_path}")
        print(f"[🟩] Realtime with box disimpan: {box_path}")
        last_capture_time = current_time

    # Cek dan simpan jika orang sudah terlihat > 2 detik
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        track_id = int(box.id[0].item()) if box.id is not None else -1

        if cls_id != 0 or conf < 0.6 or track_id == -1:
            continue

        timestamp_now = time.time()
        if track_id not in person_entry_times:
            person_entry_times[track_id] = timestamp_now

        duration_seen = timestamp_now - person_entry_times[track_id]

        if duration_seen >= 2 and track_id not in seen_person_ids:
            seen_person_ids.add(track_id)

            now = datetime.now(ZoneInfo("Asia/Jakarta"))
            tanggal_str = now.strftime("%d-%m-%Y")
            waktu_str = now.strftime("%d-%m-%Y_%H-%M-%S")
            alert_time_str = now.strftime("%d-%m-%Y %H:%M:%S")

            folder_path = os.path.join(base_dir, f"bbox-{tanggal_str}")
            os.makedirs(folder_path, exist_ok=True)

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = f"Person ({track_id}) - {conf*100:.1f}%"

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, lineType=cv2.LINE_AA)


            filename = f"person_{track_id}_{waktu_str}.jpg"
            filepath = os.path.join(folder_path, filename)
            cv2.imwrite(filepath, frame)
            print(f"[💾] Frame disimpan: {filepath}")

            # Kirim alert WA
            alert_payload = {
                "chatId": "6282392905992@c.us",
                "reply_to": None,
                "text": f"Terdeteksi Orang di Ruangan pada jam {alert_time_str}",
                "linkPreview": True,
                "linkPreviewHighQuality": False,
                "session": "default"
            }

            try:
                response = requests.post(
                    "",
                    headers={
                        "x-api-key": "wahauhuyaa",
                        "Content-Type": "application/json"
                    },
                    json=alert_payload,
                    timeout=5
                )
                if response.status_code == 200:
                    print("📲 Alert WA terkirim.")
                else:
                    print(f"⚠️ Gagal kirim WA. Status: {response.status_code}")
            except Exception as e:
                print(f"❌ Error kirim WA: {e}")

            # Kirim log capture ke server ZAI
            try:
                log_response = requests.post(
                    "",
                    headers={"Accept": "application/json"},
                    files={
                        "filename": (None, filename),
                        "directory": (None, f"bbox-{tanggal_str}")
                    },
                    timeout=5
                )
                if log_response.status_code == 200:
                    print("📝 Log capture berhasil dikirim.")
                else:
                    print(f"⚠️ Gagal kirim log capture. Status: {log_response.status_code}")
            except Exception as e:
                print(f"❌ Error saat mengirim log capture: {e}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("❌ Keluar dari program.")
        break

cap.release()
cv2.destroyAllWindows()
