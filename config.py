import os

MODEL_PATH = os.getenv("MODEL_PATH", "classifier_model.ftz")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

EXACT_RULES = {
    "neba": ("Male", 1.00),
    "sasha": ("Ambiguous / Other", 1.00),
    "jordan": ("Ambiguous / Other", 1.00),
    "moroco": ("Country", 0.98),
}
