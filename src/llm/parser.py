from __future__ import annotations

import re

def parse_yes_no(text: str) -> tuple[str, float]:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    upper = cleaned.upper()
    confidence = 0.5

    conf_match = re.search(r"CONFIDENCE[:\s]+(\d+\.\d+|\d+)", cleaned, re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            if confidence > 1.0:
                confidence = confidence / 100.0
        except ValueError:
            confidence = 0.5

    if 'YES' in upper:
        return 'YES', confidence if confidence != 0.5 else 0.8
    if 'NO' in upper:
        return 'NO', confidence if confidence != 0.5 else 0.8
    return 'NO', 0.5
