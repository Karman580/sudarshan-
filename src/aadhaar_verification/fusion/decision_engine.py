class DecisionEngine:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def extract_decision(self, risk_data):
        """
        Interprets probabilistic Risk maps to final labels.
        """
        score = risk_data.get("final_risk", 0.0)
        conf = risk_data.get("confidence", 0.0)
        
        if score > self.thresholds.get("high_risk_threshold", 0.8):
            label = "FAKE"
        elif score > self.thresholds.get("suspicious_threshold", 0.4):
             label = "SUSPICIOUS"
        else:
             label = "VALID"
             
        # Guard rails for extremely low confidence
        if conf < 0.3:
             label = "SUSPICIOUS" # Fallback to suspicious if we literally have no idea
             
        return label
