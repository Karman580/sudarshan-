class RiskEngine:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def calculate_risk(self, ocr_data, tamper_data, checksum_data, graph_data, consistency_data):
        """
        Normalizes all inputs to [0,1] array structures, handles probabilistic weighting
        and incorporates an uncertainty factor if certain detectors fail independently.
        """
        # Normalize local inputs
        ocr_risk = min(1.0, ocr_data.get("disagreement_score", 0.0))
        tamp_risk = tamper_data.get("tamper_risk", 0.0)
        tamp_risk = max(0.0, min(1.0, tamp_risk))
        
        graph_risk = graph_data.get("graph_risk", 0.0)
        consist_risk = consistency_data.get("consistency_risk", 0.0)
        
        chk_risk = 0.0 if checksum_data.get("valid") else 0.4 # if we find checksums but they are invalid or if none are found.
        if checksum_data.get("uids_found", 0) > 0 and not checksum_data.get("valid"):
            chk_risk = 0.9 # Explicitly invalid Verhoeff string is a huge flag
            
        uncertainty = 0.0
        
        # Calculate uncertainty if specific modalities were totally missing
        base_components = 5
        available = base_components
        
        if ocr_data.get("raw_text", "") == "":
            available -= 1
            uncertainty += 0.2
        if not checksum_data.get("uids_found", 0):
             available -= 0.5
             uncertainty += 0.1
             
        # Normalized weighted aggregation
        w_ocr = 0.15
        w_tamp = 0.35
        w_consist = 0.40
        w_graph = 0.05
        w_chk = 0.05
        
        total_risk = (
            (ocr_risk * w_ocr) + 
            (tamp_risk * w_tamp) + 
            (consist_risk * w_consist) +
            (graph_risk * w_graph) +
            (chk_risk * w_chk)
        )
        
        # Integrate uncertainty directly amplifying risk
        final_risk = min(1.0, total_risk + (uncertainty * 0.3))
        
        # Overall confidence is inversely proportional to uncertainty and highly divergent metrics
        # If all components scream "FAKE", confidence is high. 
        # If components disagree strongly (high variance), confidence drops.
        variance = pow((total_risk - ocr_risk), 2) + pow((total_risk - tamp_risk), 2) + pow((total_risk - consist_risk), 2)
        variance = variance / 3.0
        
        confidence = 1.0 - uncertainty - min(0.5, variance)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "final_risk": final_risk,
            "confidence": confidence,
            "uncertainty": uncertainty
        }
