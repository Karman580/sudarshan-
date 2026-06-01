class ReportGenerator:
    def __init__(self):
        pass

    def generate(self, decision, risk_data, visual_data, checksum_data):
        """
        Synthesizes modular reports into a highly structured read-out.
        Output uses English, which will later be piped into the translation module as a full string block.
        """
        lines = []
        
        # Header
        lines.append("🔍 AADHAAR VERIFICATION REPORT")
        lines.append(f"📊 Risk Score: {risk_data.get('final_risk', 0.0) * 100:.1f}%")
        lines.append(f"📈 Confidence: {risk_data.get('confidence', 0.0) * 100:.1f}%")
        lines.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("🧠 DETAILED FINDINGS:\n")
        
        # Ensure strict 1:1 text block logic mapped to ID's provided by pipeline.
        if len(visual_data) == 0:
            lines.append("✅ No visual or structural anomalies detected on geometric surfaces.")
        else:
            for item in visual_data:
                tag_id = item.get("id", "?")
                region_str = item.get("region", "Anomaly").replace("_", " ").upper()
                reason_str = item.get("reason", "Detected constraint mismatch.")
                risk_val = item.get("risk", 0.0)
                
                # Format: [X] REGION - Reason (Risk: XX%)
                lines.append(f"[{tag_id}] {region_str}")
                lines.append(f"  └─ {reason_str}")
                lines.append(f"  └─ Risk Contribution: {risk_val * 100:.1f}%\n")
                
        # Checksum baseline (Global textual non-visual risk rule)
        lines.append("🔢 DIGIT INTEGRITY (VERHOEFF):")
        found = checksum_data.get('uids_found', 0)
        if found == 0:
            lines.append("  └─ ⚠️ No verifiable 12-digit payloads parsed.")
        else:
            if checksum_data.get('valid'):
                lines.append(f"  └─ ✅ {found} UID(s) parsed and cryptographically validated.")
            else:
                 lines.append(f"  └─ ❌ FAILED Verhoeff checksum validation.")
        lines.append("")
                 
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("📌 FINAL DECISION:")
        lines.append(f"{decision}")
        
        return "\n".join(lines)
