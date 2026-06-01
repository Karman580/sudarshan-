import cv2
import json

class QRAnalyzer:
    def __init__(self):
        self.pyzbar_available = False
        try:
            from pyzbar.pyzbar import decode
            self.decode = decode
            self.pyzbar_available = True
        except ImportError:
            pass

    def analyze(self, image):
        """
        Locates and decodes digital QR payloads on the Aadhaar card.
        """
        if not self.pyzbar_available:
            return {"found": False, "data": None, "box": None, "error": "Pyzbar not installed"}
            
        try:
            barcodes = self.decode(image)
            if not barcodes:
                return {"found": False, "data": None, "box": None, "error": "No QR found"}
                
            qr_data = barcodes[0]
            rect = qr_data.rect
            box = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height]
            
            raw_text = qr_data.data.decode("utf-8", errors="ignore")
            
            return {
                "found": True,
                "data": raw_text,
                "box": box,
                "error": None
            }
        except Exception as e:
             return {"found": False, "data": None, "box": None, "error": str(e)}
