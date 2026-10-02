import os
import logging
import pickle
import numpy as np

logger = logging.getLogger("siparta.ai_models")

# Attempt to load TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        logger.warning("TensorFlow Lite is not installed. Using fallback logic.")

class InferenceEngine:
    def __init__(self, model_path: str = "model/siparta_ann.tflite"):
        self.model_path = model_path
        self.scaler_path = os.path.join(os.path.dirname(model_path), "siparta_scaler.pkl")
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.scaler = None
        self.use_fallback = not TFLITE_AVAILABLE
        
        self._load_model()
        self._load_scaler()

    def _load_model(self):
        if self.use_fallback:
            return
            
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Using fallback logic.")
            self.use_fallback = True
            return
            
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"TFLite model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            self.use_fallback = True

    def _load_scaler(self):
        if not self.use_fallback and os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded successfully from {self.scaler_path}")
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}")

    def preprocess(self, sensor_values: list[float]) -> np.ndarray:
        """
        Preprocesses sensor values before inference.
        Ensures all values are clamped to typical voltage ranges (0.0 - 5.0)
        and normalized using the trained scaler.
        """
        # Clamp values between 0.0 and 5.0
        clamped = [max(0.0, min(5.0, v)) for v in sensor_values]
        
        input_data = np.array([clamped], dtype=np.float32)
        if self.scaler:
            input_data = self.scaler.transform(input_data).astype(np.float32)
            
        return input_data

    def predict(self, sensor_values: list[float]) -> str:
        """
        Runs TFLite inference or fallback logic.
        Args:
            sensor_values: [mics5524_v, tgs2600_v, mq2_v, mq135_v]
        """
        if self.use_fallback:
            return self._fallback_logic(sensor_values)
            
        try:
            input_data = self.preprocess(sensor_values)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            classes = ["AMAN", "WASPADA", "BAHAYA"]
            idx = np.argmax(output)
            return classes[idx]
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._fallback_logic(sensor_values)

    def _fallback_logic(self, sensor_values: list[float]) -> str:
        avg = sum(sensor_values) / len(sensor_values)
        if avg > 3.0:
            return "BAHAYA"
        elif avg > 2.0:
            return "WASPADA"
        else:
            return "AMAN"

# Global instance
_engine = None

def run_inference(sensor_values: list[float], model_path: str = None) -> str:
    global _engine
    if _engine is None:
        # Default model path relative to this script
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "model", "siparta_ann.tflite")
        _engine = InferenceEngine(model_path)
    
    return _engine.predict(sensor_values)

