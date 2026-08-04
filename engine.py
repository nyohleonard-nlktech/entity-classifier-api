import os
import fasttext
from config import EXACT_RULES, CONFIDENCE_THRESHOLD

class ProductionEntityClassifier:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Binary model file not found at: {model_path}")
        self.model = fasttext.load_model(model_path)
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.exact_rules = EXACT_RULES

    def predict(self, text: str) -> dict:
        clean_text = text.strip().lower()

        if not clean_text.isalpha():
            return {
                "input": text,
                "label": "Ambiguous / Other",
                "confidence": 1.00,
                "source": "Alphanumeric Guard"
            }

        if clean_text in self.exact_rules:
            label, conf = self.exact_rules[clean_text]
            return {
                "input": text,
                "label": label,
                "confidence": conf,
                "source": "Rule Engine"
            }

        labels, probs = self.model.predict(clean_text)
        pred_label = labels[0].replace("__label__", "")
        conf = float(probs[0])

        if conf < self.confidence_threshold:
            return {
                "input": text,
                "label": "Ambiguous / Other",
                "confidence": conf,
                "source": "Threshold Rejection"
            }

        return {
            "input": text,
            "label": pred_label,
            "confidence": conf,
            "source": "FastText Core"
        }
