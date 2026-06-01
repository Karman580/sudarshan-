import cv2
import concurrent.futures

class OCREngine:
    def __init__(self):
        self.tesseract_available = False
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.tesseract_available = True
        except ImportError:
            pass
            
        self.easyocr_available = False
        try:
            import easyocr
            self.reader = easyocr.Reader(['en', 'hi'], gpu=False)
            self.easyocr_available = True
        except ImportError:
            pass

    def extract_text(self, image):
        """
        Executes parallel OCR processing using available engines.
        Returns a fusion mapping containing texts, bounds, and token confidence.
        """
        results = {
            "tesseract": [],
            "easyocr": []
        }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_tess = executor.submit(self._run_tesseract, image) if self.tesseract_available else None
            fut_easy = executor.submit(self._run_easyocr, image) if self.easyocr_available else None
            
            if fut_tess:
                results["tesseract"] = fut_tess.result()
            if fut_easy:
                results["easyocr"] = fut_easy.result()
                
        # Simple consensus mapping. We will default to easyocr if valid, else tesseract.
        consensus_tokens = self._build_consensus(results)
        
        full_text = " ".join([t['text'] for t in consensus_tokens])
        
        return {
            "raw_text": full_text,
            "tokens": consensus_tokens,
            "disagreement_score": self._calculate_disagreement(results)
        }
        
    def _run_tesseract(self, image):
        try:
            data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
            tokens = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 10:
                    text_val = str(data['text'][i]).strip()
                    if text_val:
                        tokens.append({
                            "text": text_val,
                            "box": [data['left'][i], data['top'][i], data['left'][i]+data['width'][i], data['top'][i]+data['height'][i]],
                            "conf": int(data['conf'][i]) / 100.0,
                            "source": "tesseract"
                        })
            return tokens
        except Exception as e:
            return []

    def _run_easyocr(self, image):
        try:
            data = self.reader.readtext(image)
            tokens = []
            for (bbox, text, prob) in data:
                text_val = str(text).strip()
                if text_val:
                    tokens.append({
                        "text": text_val,
                        "box": [int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])],
                        "conf": float(prob),
                        "source": "easyocr"
                    })
            return tokens
        except Exception as e:
            return []

    def _build_consensus(self, results):
        if results["easyocr"]:
            return results["easyocr"]
        return results["tesseract"]

    def _calculate_disagreement(self, results):
        if not results["tesseract"] or not results["easyocr"]:
            return 0.0 # Can't calculate disagreement if one fails
        
        tess_text = " ".join([t["text"].lower() for t in results["tesseract"]])
        easy_text = " ".join([t["text"].lower() for t in results["easyocr"]])
        
        # Simple Jaccard similarity based disagreement
        tess_set = set(tess_text.split())
        easy_set = set(easy_text.split())
        
        intersection = tess_set.intersection(easy_set)
        union = tess_set.union(easy_set)
        
        if not union:
            return 0.0
            
        similarity = len(intersection) / len(union)
        return 1.0 - similarity
