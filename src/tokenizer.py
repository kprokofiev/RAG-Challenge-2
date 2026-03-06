"""
Pharma-aware tokenizer for BM25.

Handles:
- Hyphenated terms: COX-2, ACE-inhibitor, beta-blocker
- Dosage normalization: 500 mg → 500mg, 10 мг → 10мг
- Cyrillic: ибупрофен, парацетамол
- En/em dash normalization
- Unicode normalization (NFC)
"""

import re
import unicodedata
from typing import List

# Match word tokens including hyphenated compounds
TOKEN_RE = re.compile(
    r"[A-Za-zА-Яа-яёЁ0-9]+"
    r"(?:[-_][A-Za-zА-Яа-яёЁ0-9]+)*"
)

# Dosage units (EN + RU) — normalize "500 mg" → "500mg"
DOSE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|µg|g|ml|iu|мг|мкг|мл|г|ме|%)\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 indexing/querying.

    Produces lowercased tokens with pharma-aware normalization:
    - Dashes preserved within compounds (COX-2 → cox-2)
    - Dosage units joined to numbers (500 mg → 500mg)
    - Unicode NFC normalized
    """
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    # Normalize dashes
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    # Normalize dosages: "500 mg" → "500mg"
    text = DOSE_RE.sub(r"\1\2", text)
    # Extract tokens
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return tokens
