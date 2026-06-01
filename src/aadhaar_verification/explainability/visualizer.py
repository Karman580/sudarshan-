import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def draw(self, orig_image, proc_image, visual_data):
        """
        Draws boxes based on regions given by the pipeline.
        Enforces strict resolution mapping back to the ORIGINAL image.
        
        orig_image: Unaltered cv2 matrix loaded directly from disk.
        proc_image: The cv2 matrix that was actually passed through the inference engines.
        visual_data: Strict array [{'id': int, 'region': str, 'box': [x1,y1,x2,y2], 'risk': float, 'reason': str, 'severity': str}]
        """
        assert orig_image is not None and proc_image is not None
        
        orig_h, orig_w = orig_image.shape[:2]
        proc_h, proc_w = proc_image.shape[:2]
        
        scale_x = orig_w / max(1, proc_w)
        scale_y = orig_h / max(1, proc_h)
        
        output_image = orig_image.copy()
        
        # Ensure our final output matches the original input image resolution exactly.
        assert output_image.shape == orig_image.shape
        
        for item in visual_data:
            box = item.get("box")
            if not box or len(box) != 4:
                continue
            
            # Map inference coordinates back to original image scale
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            severity = item.get("severity", "low").lower()
            
            # OpenCV is BGR
            if severity == "high":
                color = (0, 0, 255)      # Red
            elif severity == "medium":
                color = (0, 255, 255)    # Yellow
            else:
                color = (255, 0, 0)      # Blue (inconsistency/conflict zones)
                
            region_lbl = item.get("region", "Anomaly")
            risk_val = item.get("risk", 0.0)
            tag_id = item.get("id", "?")
            
            text_str = f"[{tag_id}] {region_lbl} - Risk: {risk_val:.2f}"
            
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw semi-transparent background for text legibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text_str, font, font_scale, thickness)
            
            text_x = x1
            text_y = max(0, y1 - 10)
            
            cv2.rectangle(output_image, (text_x, text_y - text_h - 2), (text_x + text_w, text_y + baseline), (0,0,0), -1)
            cv2.putText(output_image, text_str, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            
        return output_image
