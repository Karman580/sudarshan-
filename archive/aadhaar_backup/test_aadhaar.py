from src.inference import run_aadhaar_pipeline
from src.aadhaar_kyc.pipeline import analyze as analyze_kyc
import sys

image_path = "assets/logo_left.png"

print("--- TESTING PIPELINE.PY ---")
try:
    kyc_res = analyze_kyc(image_path)
    print("Class:", kyc_res.get('classification'))
    print("Score:", kyc_res.get('confidence_score'))
    print("Explanation:", kyc_res.get('explanation'))
except Exception as e:
    print("PIPELINE ERROR:", e)

print("\n--- TESTING INFERENCE.PY ---")
try:
    inf_res = run_aadhaar_pipeline(image_path)
    print("Label:", inf_res.get('label'))
    print("Confidence:", inf_res.get('confidence'))
    print("Explanation:", inf_res.get('explanation'))
    print("Heatmap Frames:", list(inf_res.get('heatmap_frames', {}).keys()))
except Exception as e:
    print("INFERENCE ERROR:", e)
