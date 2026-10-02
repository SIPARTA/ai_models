import os
import requests
import logging

logger = logging.getLogger("siparta.rpi")

# Backend Prediction endpoint
API_URL = os.getenv("SIPARTA_BACKEND_URL", "https://siparta-backend.onrender.com")
PREDICT_URL = f"{API_URL}/api/v1/predict"

def run_inference(sensor_data: list[float]) -> str:
    """
    Kirim raw sensor data ke Backend untuk inferensi JST.
    Ini menghilangkan dependensi TensorFlow/Keras di edge device.
    """
    try:
        # Jika nilai eksak 0.0, itu berarti sensor belum tersambung (offline/unavailable)
        if all(v == 0.0 for v in sensor_data):
            return "DATA_UNAVAILABLE"
            
        payload = {
            "mics5524": float(sensor_data[0]),
            "tgs2600": float(sensor_data[1]),
            "mq2": float(sensor_data[2]),
            "mq135": float(sensor_data[3])
        }
        
        response = requests.post(PREDICT_URL, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return result.get("ai_analysis", {}).get("status", "MODEL_ERROR")
        else:
            logger.error(f"[INFERENCE] Gagal inferensi di backend: HTTP {response.status_code}")
            return "MODEL_ERROR"
            
    except requests.exceptions.ConnectionError:
        logger.error("[INFERENCE] Koneksi backend terputus.")
        return "MODEL_ERROR"
    except requests.exceptions.Timeout:
        logger.error("[INFERENCE] Request inferensi timeout.")
        return "MODEL_ERROR"
    except Exception as e:
        logger.error(f"[INFERENCE] Error: {e}")
        return "MODEL_ERROR"
