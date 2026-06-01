import json

class ConsistencyEngine:
    def __init__(self):
        pass

    def evaluate(self, ocr_data, qr_data, layout_data):
        """
        Cross-validates data points between available modalities.
        Specifically handles logic mapping OCR text outputs to structured QR JSONs.
        """
        issues = []
        risk = 0.0
        
        # 1. OCR vs QR Mapping
        if qr_data.get("found") and qr_data.get("data"):
            # Usually Aadhaar QR contains XML or JSON. Let's do basic text inclusion
            qr_raw = qr_data["data"].lower()
            ocr_text = ocr_data.get("raw_text", "").lower()
            
            # If we have QR, but OCR can't find basic overlap
            # Simplified check: extracting 12 digit UID from OCR to match QR contents
            import re
            uids = re.findall(r'\d{4}\s?\d{4}\s?\d{4}', ocr_text)
            clean_uids = [u.replace(" ", "") for u in uids]
            
            if clean_uids:
                uid_in_qr = False
                for uid in clean_uids:
                    if uid in qr_raw:
                        uid_in_qr = True
                        break
                if not uid_in_qr:
                    risk += 0.8
                    issues.append({
                        "region": "global",
                        "risk": 0.8,
                        "box": qr_data.get("box", [0,0,0,0]),
                        "reason": "Severe Mismatch: UID generated from text OCR not found inside QR signature."
                    })
            else:
                 # If no UID matched but QR exists, indicates poor scan or manipulated front layout masking real text.
                 risk += 0.3
                 issues.append({
                     "region": "global",
                     "risk": 0.3,
                     "box": qr_data.get("box", [0,0,0,0]),
                     "reason": "UID text parsing failed while QR structure remains intact."
                 })

        return {
            "consistency_risk": min(1.0, risk),
            "issues": issues
        }
