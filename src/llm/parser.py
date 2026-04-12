from __future__ import annotations

import re


def parse_prediction_response(text: str) -> dict:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    prediction_match = re.search(
        r"^PREDICTION:\s*(YES|NO)\s*$",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )
    confidence_match = re.search(
        r"^CONFIDENCE:\s*(\d+\.\d+|\d+)\s*$",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )
    rationale_match = re.search(
        r"^RATIONALE:\s*(.+)$",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )
    evidence_match = re.search(
        r"^EVIDENCE:\s*(.+)$",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )

    confidence = 0.5
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            if confidence > 1.0:
                confidence = confidence / 100.0
        except ValueError:
            confidence = 0.5

    if not prediction_match:
        return {
            "prediction": "NO",
            "probability": 0.5,
            "rationale": "",
            "evidence": "",
            "parsed_ok": False,
            "parse_error": "Missing structured PREDICTION line.",
            "cleaned_output": cleaned,
        }

    return {
        "prediction": prediction_match.group(1).upper(),
        "probability": confidence,
        "rationale": rationale_match.group(1).strip() if rationale_match else "",
        "evidence": evidence_match.group(1).strip() if evidence_match else "",
        "parsed_ok": True,
        "parse_error": None,
        "cleaned_output": cleaned,
    }


def parse_yes_no(text: str) -> tuple[str, float]:
    parsed = parse_prediction_response(text)
    return parsed["prediction"], float(parsed["probability"])
