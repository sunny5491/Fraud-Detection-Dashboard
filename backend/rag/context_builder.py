"""
Context builder for converting user risk profiles and transaction timelines into structured text context.
"""
from typing import Dict, List, Any, Union

def build_user_context(profile: dict, timeline: list) -> str:
    """
    Build a comprehensive, structured Markdown text representation of a user case profile and timeline.
    
    Args:
        profile: Dictionary containing user risk scores, features, and SHAP explanations.
        timeline: List of dictionaries representing transaction and return history.
        
    Returns:
        Structured string containing sections for profile, SHAP analysis, and timeline.
    """
    if not profile:
        return ""
        
    user_id = profile.get("User ID", "Unknown")
    risk_score = profile.get("Risk Score", 0.0)
    risk_band = profile.get("Risk Band", "Low")
    total_returns = profile.get("Total Returns", 0)
    exposure = profile.get("Financial Exposure ($)", 0.0)
    days_active = profile.get("Days Active", 0)
    reason_diversity = profile.get("Reason Diversity", 0.0)
    
    lines = []
    lines.append(f"## Profile Overview for User {user_id}")
    lines.append(f"- Risk Score: {risk_score} / 100 ({risk_band} Risk)")
    lines.append(f"- Total Returns Count: {total_returns}")
    lines.append(f"- Total Financial Exposure: ${exposure:.2f}")
    lines.append(f"- Days Active: {days_active}")
    lines.append(f"- Return Reason Diversity Score: {reason_diversity:.2f}\n")
    
    lines.append("## SHAP Risk Analysis")
    shap = profile.get("SHAP", {})
    if isinstance(shap, dict) and "Feature" in shap:
        features = shap.get("Feature", [])
        contributions = shap.get("Contribution", [])
        directions = shap.get("Direction", [])
        for feat, contrib, dirn in zip(features, contributions, directions):
            effect = "increases risk" if dirn == "increases_risk" else "decreases risk"
            lines.append(f"- Feature '{feat}': SHAP contribution {contrib} ({effect})")
    else:
        lines.append("- No detailed SHAP feature contribution data available.")
    lines.append("")
    
    lines.append("## Transaction & Return Timeline")
    if timeline and isinstance(timeline, list):
        for idx, item in enumerate(timeline, 1):
            if isinstance(item, dict):
                date = item.get("Date", "N/A")
                event_type = item.get("Event Type", "N/A")
                amount = item.get("Amount", "$0.00")
                category = item.get("Item Category", "N/A")
                status = item.get("Status", "N/A")
                flag = item.get("Flag", "None")
                line = f"{idx}. [{date}] {event_type} | Amount: {amount} | Category: {category} | Status: {status}"
                if flag and flag != "None":
                    line += f" | Flag: {flag}"
                lines.append(line)
            else:
                lines.append(f"{idx}. {str(item)}")
    else:
        lines.append("No recorded return timeline events for this user.")
        
    return "\n".join(lines)


def chunk_context(context: Union[str, dict]) -> dict:
    """
    Chunk context text into standard named sections.
    
    Args:
        context: Either a string representation of context or a pre-structured dict.
        
    Returns:
        Dict mapping section names to text chunks (e.g. {"profile": "...", "shap": "...", "timeline": "..."}).
    """
    if isinstance(context, dict):
        return context
        
    if not isinstance(context, str) or not context.strip():
        return {}
        
    chunks = {}
    current_section = "general"
    current_lines = []
    
    for line in context.split("\n"):
        if line.startswith("## "):
            if current_lines:
                text_content = "\n".join(current_lines).strip()
                if text_content:
                    chunks[current_section] = text_content
            header_text = line.replace("## ", "").strip().lower()
            if "profile" in header_text:
                current_section = "profile"
            elif "shap" in header_text or "risk" in header_text:
                current_section = "shap"
            elif "timeline" in header_text or "transaction" in header_text:
                current_section = "timeline"
            else:
                current_section = header_text.replace(" ", "_")
            current_lines = [line]
        else:
            current_lines.append(line)
            
    if current_lines:
        text_content = "\n".join(current_lines).strip()
        if text_content:
            chunks[current_section] = text_content
            
    return chunks


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import config
    from backend.ml.pipeline import FraudPipeline

    pipeline = FraudPipeline(config.DATA_PATH)
    pipeline.run()
    if pipeline.user_features is not None and not pipeline.user_features.empty:
        user_id = str(pipeline.user_features['user_id'].iloc[0])
        profile = pipeline.get_user_profile(user_id)
        timeline = pipeline.get_user_timeline(user_id)

        print(f"=== Generated Case Context for Real User ({user_id}) ===")
        context_str = build_user_context(profile, timeline)
        print(context_str)

        print("\n=== Chunked Sections ===")
        chunks = chunk_context(context_str)
        for section, text in chunks.items():
            print(f"\n--- Section: {section} ---")
            print(text)

