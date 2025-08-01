import cv2
import yaml
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
import requests
from ultralytics import YOLO
# Import dan Setup

# Mengimpor berbagai library untuk video, logging, file, waktu, threading, numpy, requests, dan YOLO dari ultralytics.
# Fungsi setup_logger membuat logger yang menulis ke file dan menampilkan ke konsol.
# Kelas RealTimeDetector

# Kelas utama untuk deteksi objek real-time dari stream RTSP, dengan pemrosesan asinkron untuk I/O (menyimpan file, kirim API).
# Inisialisasi (__init__)

# Memuat konfigurasi dari file YAML (config.yaml).
# Inisialisasi logger, model YOLO, dan device (CPU/GPU).
# Konfigurasi ROI (Region of Interest) jika diaktifkan.
# State management untuk tracking orang (waktu pertama dan terakhir terlihat, status notifikasi).
# Membuat ThreadPoolExecutor untuk tugas asinkron (max 5 thread).
# Menyimpan threshold confidence, target class, dan durasi persistence dari config.
# Proses Deteksi (_process_detections)

# Untuk setiap box hasil deteksi:
# Dicek apakah confidence cukup dan kelas sesuai target (misal: person).
# Dicek apakah ada track_id.
# Dicek apakah berada di dalam ROI (jika diaktifkan).
# Gambar kotak dan label pada frame.
# Update waktu pertama dan terakhir terlihat untuk track_id.
# Jika orang sudah terlihat lebih lama dari threshold dan belum pernah dinotifikasi, submit tugas asinkron untuk menyimpan gambar dan kirim notifikasi.
# Tugas Asinkron (_handle_persistent_detection)

# Menyimpan gambar (full/crop sesuai config) ke folder berdasarkan tanggal.
# Jika fitur WhatsApp aktif, mengirim notifikasi dengan gambar ke API WhatsApp.
# Jika fitur log server aktif, mengirim log ke server ZAI.
# Semua proses ini dilakukan di thread terpisah agar tidak menghambat deteksi utama.
# Kirim API (_send_api_request)

# Mengirim request ke API dengan mekanisme retry (maksimal 3 kali).
# Mendukung pengiriman file (gambar) dan data JSON.
# Loop Utama (run)

# Membuka stream RTSP (atau webcam jika belum diatur).
# Membuat ROI mask jika diaktifkan.
# Loop membaca frame:
# Jika gagal, mencoba reconnect.
# Proses frame (mask ROI jika ada).
# Deteksi dengan YOLO dan proses hasilnya.
# Gambar ROI di frame jika diaktifkan.
# Tampilkan frame ke window.
# Keluar jika tombol 'q' ditekan.
# Setelah selesai, release semua resource dan shutdown thread executor.
# Main Program

# Membuat instance RealTimeDetector dan menjalankan deteksi.
# Menangani error jika file config tidak ditemukan atau error lain saat inisialisasi.
# Kesimpulan:
# Kode ini mendeteksi orang secara real-time dari stream RTSP, menandai dan tracking orang, menyimpan gambar jika terdeteksi cukup lama, serta mengirim notifikasi dan log ke API eksternal secara asinkron. Semua konfigurasi (model, threshold, API, storage, ROI) diatur melalui file YAML. Logging detail tersedia untuk debugging dan audit.
# --- Konfigurasi Logging ---
def setup_logger(log_path_str: str):
    """Menginisialisasi logger untuk menyimpan log ke file dan menampilkan di konsol."""
    log_path = Path(log_path_str)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("RealTimeDetector")
    logger.setLevel(logging.DEBUG)
    
    # Mencegah duplikasi handler jika fungsi ini dipanggil lagi
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler untuk konsol
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    
    # Handler untuk file
    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

class RealTimeDetector:
    """
    Kelas utama untuk menjalankan deteksi objek secara real-time dari stream RTSP.
    Menggunakan pemrosesan asinkron untuk tugas I/O (menyimpan file, mengirim API).
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config['storage']['log_path'])
        
        self.logger.info("🚀 Memulai inisialisasi sistem deteksi...")
        
        # Inisialisasi Model
        self.device = self.config['processing']['device']
        self.model = YOLO(self.config['model']['path'])
        self.model.to(self.device)
        self.logger.info(f"Model '{self.config['model']['path']}' dimuat ke perangkat '{self.device}'.")

        # Konfigurasi ROI
        self.roi_points = np.array(self.config['processing']['roi_points'], dtype=np.int32) if self.config['processing']['enable_roi'] else None

        # State Management
        self.tracked_persons = defaultdict(lambda: {"first_seen": None, "last_seen": None, "notified": False})
        
        # Asynchronous Executor
        self.executor = ThreadPoolExecutor(max_workers=5) 

        self.confidence_threshold = self.config['model']['confidence_threshold']
        self.target_class = self.config['model']['target_class']
        self.persistence_threshold = self.config['tracking']['persistence_threshold_sec']

    def _load_config(self, path: str):
        """Memuat konfigurasi dari file YAML."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _process_detections(self, frame: np.ndarray, results: list):
        """Memproses hasil deteksi dari model YOLO."""
        if results[0].boxes is None:
            return

        for box in results[0].boxes:
            if box.conf[0] < self.confidence_threshold or int(box.cls[0]) != self.target_class:
                continue
            if box.id is None:
                continue
            
            track_id = int(box.id[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if self.roi_points is not None and cv2.pointPolygonTest(self.roi_points, center_point, False) < 0:
                continue 
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            current_time = time.time()
            if self.tracked_persons[track_id]["first_seen"] is None:
                self.tracked_persons[track_id]["first_seen"] = current_time
            
            self.tracked_persons[track_id]["last_seen"] = current_time
            
            detection_duration = current_time - self.tracked_persons[track_id]["first_seen"]
            if detection_duration >= self.persistence_threshold and not self.tracked_persons[track_id]["notified"]:
                self.logger.info(f"✅ Deteksi valid untuk ID: {track_id}. Durasi: {detection_duration:.2f}s. Mengirim tugas notifikasi.")
                
                self.tracked_persons[track_id]["notified"] = True
                frame_copy = frame.copy()
                crop_img = frame_copy[y1:y2, x1:x2]

                self.executor.submit(self._handle_persistent_detection, frame_copy, crop_img, track_id, box.conf[0])

    def _handle_persistent_detection(self, frame: np.ndarray, crop_img: np.ndarray, track_id: int, confidence: float):
        """
        Tugas asinkron untuk menyimpan gambar dan mengirim notifikasi sesuai konfigurasi on/off.
        """
        try:
            timestamp = datetime.now()
            date_folder = timestamp.strftime('%Y-%m-%d')
            capture_dir = Path(self.config['storage']['base_capture_path']) / date_folder
            capture_dir.mkdir(parents=True, exist_ok=True)
            
            img_to_save = crop_img if self.config['storage']['save_crop'] else frame
            filename = f"capture_id_{track_id}_{timestamp.strftime('%H%M%S')}.jpg"
            filepath = capture_dir / filename
            
            cv2.imwrite(str(filepath), img_to_save)
            self.logger.info(f"🖼️ Gambar disimpan: {filepath}")

            if self.config['api']['whatsapp']['enabled']:
                self.logger.debug("Fitur WhatsApp aktif, mencoba mengirim notifikasi.")
                self._send_api_request(
                    url=self.config['api']['whatsapp']['endpoint'],
                    json_data={
                        "recipient": "PHONE_NUMBER", # Ganti dengan nomor tujuan
                        "message": f"🔴 Peringatan Keamanan! 🔴\nTerdeteksi seseorang (ID: {track_id}) pada {timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
                    },
                    files={"attachment": open(filepath, 'rb')},
                    service_name="WhatsApp"
                )
            else:
                self.logger.info("Fitur notifikasi WhatsApp dinonaktifkan di config.")

            if self.config['api']['log_server']['enabled']:
                self.logger.debug("Fitur Log Server aktif, mencoba mengirim log.")
                self._send_api_request(
                    url=self.config['api']['log_server']['endpoint'],
                    json_data={
                        "event": "person_detected", "track_id": track_id,
                        "timestamp": timestamp.isoformat(), "confidence": f"{confidence:.2f}",
                        "image_path": str(filepath)
                    },
                    service_name="ZAI Log Server"
                )
            else:
                self.logger.info("Fitur Log Server dinonaktifkan di config.")
        except Exception as e:
            self.logger.error(f"Error pada _handle_persistent_detection untuk ID {track_id}: {e}")

    def _send_api_request(self, url: str, json_data: dict, service_name: str, files: dict = None, retries: int = 3):
        """Mengirim request API dengan mekanisme retry."""
        headers = {"Authorization": f"Bearer {self.config['api']['api_key']}"}
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=json_data, files=files, timeout=10)
                response.raise_for_status() 
                self.logger.info(f"✔️ Notifikasi {service_name} berhasil dikirim. Status: {response.status_code}")
                return
            except requests.RequestException as e:
                self.logger.warning(f"Gagal mengirim notifikasi {service_name} (Percobaan {attempt + 1}/{retries}): {e}")
                time.sleep(2 ** attempt)
        self.logger.error(f"❌ Gagal total mengirim notifikasi {service_name} setelah {retries} percobaan.")

    def run(self):
        """Loop utama untuk menangkap dan memproses stream video dari sumber yang dipilih."""
        
        # --- Memilih Sumber Video Berdasarkan Konfigurasi ---
        source_type = self.config['camera']['source_type'].lower()
        video_source = None

        if source_type == 'rtsp':
            video_source = self.config['camera']['rtsp_url']
            self.logger.info(f"📹 Menggunakan sumber video RTSP: {video_source}")
        elif source_type == 'webcam':
            video_source = self.config['camera']['webcam_id']
            self.logger.info(f"📷 Menggunakan sumber video Webcam dengan ID: {video_source}")
        else:
            self.logger.error(f"Tipe sumber '{source_type}' tidak valid. Menggunakan webcam default (ID: 0) sebagai fallback.")
            video_source = 0
            
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.critical(f"❌ Gagal total membuka sumber video: {video_source}")
            return
            
        self.logger.info("✅ Sumber video berhasil dibuka. Memulai deteksi...")
        
        roi_mask = None
        if self.roi_points is not None:
            ret, frame = cap.read()
            if ret:
                roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(roi_mask, [self.roi_points], 255)
            else:
                self.logger.error("Gagal membaca frame pertama untuk membuat ROI mask.")

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning(f"Frame kosong dari sumber {video_source}. Mencoba menyambung ulang dalam 5 detik...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(video_source) # Coba sambung ulang ke sumber yang sama
                continue

            display_frame = frame.copy()

            if roi_mask is not None:
                processing_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            else:
                processing_frame = frame

            results = self.model.track(
                processing_frame, persist=True, verbose=False, tracker="bytetrack.yaml"
            )
            
            self._process_detections(display_frame, results)

            if self.roi_points is not None:
                cv2.polylines(display_frame, [self.roi_points], isClosed=True, color=(255, 255, 0), thickness=2)

            cv2.imshow("Real-Time Person Detection (Tekan 'q' untuk keluar)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("Tombol 'q' ditekan. Menghentikan program...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=True)
        self.logger.info("👋 Sistem berhenti. Semua resource telah dilepaskan.")

if __name__ == "__main__":
    try:
        detector = RealTimeDetector(config_path="config.yaml")
        detector.run()
    except FileNotFoundError:
        print("FATAL ERROR: File 'config.yaml' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
    except Exception as e:
        print(f"FATAL ERROR: Terjadi kesalahan saat inisialisasi: {e}")