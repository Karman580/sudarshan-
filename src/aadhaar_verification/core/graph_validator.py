class GraphValidator:
    def __init__(self):
        pass

    def validate(self, layout_data, ocr_data):
        """
        Validates spatial geometric relationship between field tokens (Graph Validation).
        For example: Does the photo sit to the right side of the main demographic stack?
        """
        issues = []
        risk_score = 0.0
        
        photo_box = layout_data.get("photo_box")
        
        # We need an anchor, let's try to find an Aadhar UID mapping block
        uid_boxes = []
        for token in ocr_data.get("tokens", []):
            if len(token["text"].replace(" ", "")) == 12 and token["text"].replace(" ", "").isdigit():
                uid_boxes.append(token["box"])

        if photo_box and uid_boxes:
            # Usually the photo is physically aligned above the UID, or to the left/right depending on format type.
            # E-Aadhaar format: photo is on the right of the demographic payload, UID is mostly on bottom.
            # Basic validation: Photo shouldn't physically overlap the UID bounding box.
            px1, py1, px2, py2 = photo_box
            for (ux1, uy1, ux2, uy2) in uid_boxes:
                # Check for overlap
                overlap_x = max(0, min(px2, ux2) - max(px1, ux1))
                overlap_y = max(0, min(py2, uy2) - max(py1, uy1))
                
                if overlap_x > 0 and overlap_y > 0:
                    risk_score += 0.5
                    issues.append({
                        "region": "geometric_anomaly",
                        "risk": 0.6,
                        "box": photo_box,
                        "reason": "Photo improperly bound overlapping explicit text fields."
                    })
        
        return {
            "graph_valid": len(issues) == 0,
            "graph_risk": min(1.0, risk_score),
            "issues": issues
        }
