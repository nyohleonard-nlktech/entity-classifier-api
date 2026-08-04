import fasttext
import os

class ProductionEntityClassifier:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        # Load FastText supervised model
        self.model = fasttext.load_model(model_path)

    def predict(self, text: str):
        # 1. Rigorous Input Sanitization: Remove newlines, tabs, and extra spaces
        clean_text = text.replace("\n", " ").replace("\r", " ").strip()
        
        if not clean_text:
            return {
                "input": text,
                "label": "unknown",
                "confidence": 0.0,
                "source": "validation_error"
            }

        # 2. FastText Inference
        try:
            labels, probs = self.model.predict(clean_text, k=1)
            raw_label = labels[0].replace("__label__", "")
            confidence = float(probs[0])
            
            return {
                "input": text,
                "label": raw_label,
                "confidence": confidence,
                "source": "fasttext_model"
            }
        except Exception as e:
            raise RuntimeError(f"Inference execution failed: {str(e)}")
