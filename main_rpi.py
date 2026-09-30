"""
SIPARTA — Edge Computing Script (Raspberry Pi 3 B+)
=====================================================
Script ini berjalan di perangkat IoT edge (RPi) dan bertugas:

  1. Membaca tegangan dari 4 sensor gas via ADC ADS1115 (I2C)
  2. Menjalankan inferensi ANN (TFLite) untuk klasifikasi gas
  3. Mengontrol aktuator (LED RGB, Buzzer)
  4. Mengirim laporan insiden ke FastAPI Backend (HTTP POST multipart)
     → Backend yang kemudian menangani: Supabase + Gemini AI + Blockchain

Alur Lengkap (sesuai architecture_design.md Section 8):
  [RPi] → POST /api/v1/incidents/report → [FastAPI]
         → [Supabase] + [Gemini AI] + [relay.ts → Polygon Amoy]

Koneksi I2C Sensor → ADS1115 → RPi:
  ADS1115 P0 → MICS-5524  (Gas umum / CO)
  ADS1115 P1 → TGS2600    (Air Quality / Propana)
  ADS1115 P2 → MQ-2       (Asap / Gas mudah terbakar)
  ADS1115 P3 → MQ-135     (NH3, Benzena, CO2)
"""

import time
import json
import logging
import os
import requests
import numpy as np
import cv2
from datetime import datetime, timezone
from pathlib import Path

# ─── Load .env dari project root ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_models.inference import run_inference

# ─── Deteksi environment (RPi fisik atau mode simulasi PC) ──────────────────
try:
    import RPi.GPIO as GPIO
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    IS_RPI = True
except ImportError:
    IS_RPI = False
    print("[WARN] Library RPi.GPIO / Adafruit tidak ditemukan. Mode Simulasi (PC) aktif.")

# ============================================================
# KONFIGURASI GPIO (BCM Numbering)
# ============================================================

BTN_PIN    = 17   # Tactile push button (power toggle)
LED_GREEN  = 27   # Status: AMAN
LED_YELLOW = 22   # Status: WASPADA
LED_RED    = 23   # Status: BAHAYA
BUZZER     = 24   # Active buzzer alarm

# ============================================================
# KONFIGURASI BACKEND & AUTENTIKASI
# ============================================================

# URL FastAPI Backend — ganti dengan IP server produksi atau Render URL
_API_BASE = os.getenv("SIPARTA_BACKEND_URL", "https://siparta-backend.onrender.com")
API_URL = f"{_API_BASE}/api/v1/incidents/report"

# API Key untuk autentikasi ke backend (harus sama dengan DEVICE_API_KEY di backend .env)
DEVICE_API_KEY = os.getenv("DEVICE_API_KEY", "")

# ID perangkat ini — diisi setelah mendaftarkan RPi di tabel iot_devices Supabase
DEVICE_ID = os.getenv("DEVICE_ID", "")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("siparta.rpi")


# ============================================================
# INISIALISASI HARDWARE
# ============================================================

def setup_gpio():
    """Inisialisasi pin GPIO BCM. Tidak beroperasi di mode simulasi."""
    if not IS_RPI:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    for pin in [LED_GREEN, LED_YELLOW, LED_RED, BUZZER]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def setup_adc():
    """Inisialisasi I2C bus dan modul ADC ADS1115."""
    if not IS_RPI:
        return None, None, None, None
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1115(i2c)
    ch0 = AnalogIn(ads, ADS.P0)  # MICS-5524
    ch1 = AnalogIn(ads, ADS.P1)  # TGS2600
    ch2 = AnalogIn(ads, ADS.P2)  # MQ-2
    ch3 = AnalogIn(ads, ADS.P3)  # MQ-135
    return ch0, ch1, ch2, ch3


# ============================================================
# KALIBRASI & WARM-UP
# ============================================================

def warm_up_routine(duration: int = 60):
    """
    Sensor gas memerlukan waktu pemanasan sebelum pembacaan stabil.
    60 detik adalah standar minimum untuk MICS-5524 dan TGS2600.
    """
    logger.info(f"Memulai kalibrasi dan warm-up sensor ({duration} detik)...")
    if IS_RPI:
        GPIO.output(LED_YELLOW, GPIO.HIGH)

    for i in range(duration, 0, -1):
        if i % 10 == 0 or i <= 5:
            logger.info(f"  Warm-up tersisa: {i} detik")
        time.sleep(1)

    if IS_RPI:
        GPIO.output(LED_YELLOW, GPIO.LOW)
    logger.info("Kalibrasi selesai. Sensor mencapai resistansi stabil.")


# ============================================================
# PENGAMBILAN GAMBAR BUKTI
# ============================================================

def capture_image() -> str | None:
    """
    Mengambil foto dari RPi Camera OV5647 menggunakan OpenCV.

    Returns:
        Path ke file foto yang tersimpan, atau None jika kamera tidak tersedia.
    """
    logger.info("Mengambil gambar TKP via RPi Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Gagal mengakses kamera (/dev/video0).")
        return None

    ret, frame = cap.read()
    filename = None
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/siparta_danger_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Foto tersimpan: {filename}")
    else:
        logger.error("Gagal membaca frame dari kamera.")

    cap.release()
    return filename


# ============================================================
# INFERENSI AI (ANN / TFLite)
# ============================================================

# (Logika ini sekarang ditangani oleh ai_models/inference.py)


# ============================================================
# KIRIM LAPORAN KE FASTAPI BACKEND
# ============================================================

def send_report_to_backend(
    sensor_data: list[float],
    status: str,
    image_path: str | None = None,
) -> bool:
    """
    Mengirim laporan insiden ke FastAPI Backend via HTTP POST multipart.

    Backend yang akan meneruskan ke:
      - Supabase PostgreSQL (penyimpanan)
      - Google Gemini AI (analisis gambar)
      - Polygon Amoy via relay.ts (blockchain anchoring)

    Args:
        sensor_data  : [mics5524_v, tgs2600_v, mq2_v, mq135_v]
        status       : 'AMAN', 'WASPADA', atau 'BAHAYA'
        image_path   : Path gambar bukti (opsional)

    Returns:
        True jika backend mengonfirmasi terima, False jika gagal.
    """
    form_data = {
        "status": status,
        "sensor_mics5524": str(sensor_data[0]),
        "sensor_tgs2600": str(sensor_data[1]),
        "sensor_mq2": str(sensor_data[2]),
        "sensor_mq135": str(sensor_data[3]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": DEVICE_ID,
    }

    headers = {}
    if DEVICE_API_KEY:
        headers["X-API-Key"] = DEVICE_API_KEY

    files = {}
    opened_file = None
    if image_path and os.path.exists(image_path):
        opened_file = open(image_path, "rb")
        files["file"] = (os.path.basename(image_path), opened_file, "image/jpeg")

    try:
        logger.info(f"Mengirim laporan {status} ke backend ({API_URL})...")
        response = requests.post(
            API_URL,
            data=form_data,
            files=files,
            headers=headers,
            timeout=15,
        )
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Backend OK: incident_id={result.get('incident_id')}")
            return True
        else:
            logger.error(f"Backend error HTTP {response.status_code}: {response.text[:200]}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error(f"Gagal terhubung ke backend: {API_URL}. Pastikan server running dan IP benar.")
        return False
    except requests.exceptions.Timeout:
        logger.error("Request timeout (15s). Backend tidak merespons.")
        return False
    except Exception as e:
        logger.error(f"Error saat mengirim ke backend: {e}")
        return False
    finally:
        if opened_file:
            opened_file.close()


# ============================================================
# KONTROL AKTUATOR
# ============================================================

def _set_leds_and_buzzer(status: str):
    """Nyalakan LED dan buzzer sesuai status. No-op di mode simulasi."""
    if not IS_RPI:
        return
    GPIO.output(LED_GREEN, GPIO.LOW)
    GPIO.output(LED_YELLOW, GPIO.LOW)
    GPIO.output(LED_RED, GPIO.LOW)
    GPIO.output(BUZZER, GPIO.LOW)

    if status == "AMAN":
        GPIO.output(LED_GREEN, GPIO.HIGH)
    elif status == "WASPADA":
        GPIO.output(LED_YELLOW, GPIO.HIGH)
    elif status == "BAHAYA":
        GPIO.output(LED_RED, GPIO.HIGH)
        GPIO.output(BUZZER, GPIO.HIGH)

# ============================================================
# HEARTBEAT BACKGROUND THREAD
# ============================================================
import threading

def _heartbeat_worker():
    """Background thread untuk mengirim heartbeat ke backend setiap 30 detik."""
    heartbeat_url = f"{_API_BASE}/api/v1/devices/heartbeat"
    headers = {}
    if DEVICE_API_KEY:
        headers["X-API-Key"] = DEVICE_API_KEY
    payload = {"device_id": DEVICE_ID}
    
    while True:
        try:
            res = requests.post(heartbeat_url, json=payload, headers=headers, timeout=5)
            if res.status_code != 200:
                logger.error(f"[HEARTBEAT] Backend mengembalikan status {res.status_code}")
        except Exception as e:
            logger.error(f"[HEARTBEAT] Gagal mengirim heartbeat: {e}")
        time.sleep(30.0)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    if not IS_RPI:
        logger.error("FATAL: Perangkat keras RPi (GPIO/I2C) tidak terdeteksi.")
        logger.error("Mode produksi tidak mengizinkan pengiriman data simulasi.")
        logger.error("Aplikasi dihentikan untuk mencegah polusi data di production.")
        return

    setup_gpio()

    try:
        ch0, ch1, ch2, ch3 = setup_adc()
    except Exception as e:
        logger.error(f"Gagal inisialisasi I2C / ADC ADS1115: {e}")
        return

    logger.info("Sistem SIPARTA Aktif! (Production Mode)")
    logger.info(f"Backend URL  : {API_URL}")
    logger.info(f"Device ID    : {DEVICE_ID or '(tidak diset)'}")
    
    # Start heartbeat thread
    hb_thread = threading.Thread(target=_heartbeat_worker, daemon=True)
    hb_thread.start()
    logger.info("Heartbeat thread dimulai.")

    warm_up_routine(duration=60)

    # Cooldown tracker (hindari spam laporan berulang)
    _last_report_time = 0.0
    _report_cooldown_s = 30.0  # detik antar laporan

    try:
        while True:
            # ── Pembacaan Sensor Fisik ──────────────────────────────────────────────
            try:
                sensor_data = [ch0.voltage, ch1.voltage, ch2.voltage, ch3.voltage]
            except Exception as e:
                logger.error(f"Gagal membaca sensor fisik: {e}")
                time.sleep(2.0)
                continue

            v0, v1, v2, v3 = sensor_data
            logger.info(f"Sensors [MICS={v0:.2f}V, TGS={v1:.2f}V, MQ2={v2:.2f}V, MQ135={v3:.2f}V]")

            # ── Inferensi ANN ─────────────────────────────────────────────────
            status = run_inference(sensor_data)

            # ── Kontrol Aktuator ──────────────────────────────────────────────
            _set_leds_and_buzzer(status)

            # ── Kirim Laporan ke Backend (dengan cooldown) ────────────────────
            now = time.time()
            should_report = (
                status in ("BAHAYA", "WASPADA")
                and (now - _last_report_time) >= _report_cooldown_s
            )

            if should_report:
                if status == "BAHAYA":
                    logger.warning("!!! GAS BERACUN TERDETEKSI — STATUS BAHAYA !!!")
                    img_path = capture_image()
                else:
                    img_path = None

                sent = send_report_to_backend(sensor_data, status, img_path)
                if sent:
                    _last_report_time = now
                    logger.info(f"Cooldown aktif selama {_report_cooldown_s}s...")

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Sistem dihentikan (Ctrl+C).")
    except Exception as e:
        logger.error(f"Sistem crash (Fatal Error): {e}")
        raise
    finally:
        if IS_RPI:
            GPIO.cleanup()
            logger.info("GPIO port diamankan.")


if __name__ == "__main__":
    main()

