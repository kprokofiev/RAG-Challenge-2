"""
PDF Font Registration — Cyrillic support for reportlab renderers.

Tries DejaVuSans (full Cyrillic), falls back to system fonts,
then to reportlab's bundled Vera (limited Cyrillic).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel: have we already registered?
_REGISTERED = False

# Font family name used in styles
FONT_NORMAL = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"

_SEARCH_PATHS = [
    # Linux / Docker (Debian/Ubuntu)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # Windows
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
    # macOS (Homebrew)
    "/usr/local/share/fonts/DejaVuSans.ttf",
    "/opt/homebrew/share/fonts/DejaVuSans.ttf",
    # Bundled in repo (if copied)
    os.path.join(os.path.dirname(__file__), "..", "fonts", "DejaVuSans.ttf"),
    os.path.join(os.path.dirname(__file__), "..", "fonts", "DejaVuSans-Bold.ttf"),
]


def _find_font(name_fragment: str) -> Optional[str]:
    """Find a TTF file by name fragment in search paths."""
    for p in _SEARCH_PATHS:
        if name_fragment in os.path.basename(p) and os.path.isfile(p):
            return p
    return None


def register_cyrillic_fonts() -> str:
    """
    Register Cyrillic-capable fonts with reportlab.

    Returns the base font family name to use in ParagraphStyle.
    """
    global _REGISTERED, FONT_NORMAL, FONT_BOLD

    if _REGISTERED:
        return FONT_NORMAL

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return "Helvetica"

    normal_path = _find_font("DejaVuSans.ttf")
    bold_path = _find_font("DejaVuSans-Bold.ttf")

    if normal_path:
        try:
            pdfmetrics.registerFont(TTFont("DejaVuSans", normal_path))
            if bold_path:
                pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold_path))
            else:
                pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", normal_path))

            from reportlab.lib.fonts import addMapping
            addMapping("DejaVuSans", 0, 0, "DejaVuSans")
            addMapping("DejaVuSans", 1, 0, "DejaVuSans-Bold")
            addMapping("DejaVuSans", 0, 1, "DejaVuSans")
            addMapping("DejaVuSans", 1, 1, "DejaVuSans-Bold")

            FONT_NORMAL = "DejaVuSans"
            FONT_BOLD = "DejaVuSans-Bold"
            _REGISTERED = True
            logger.info("Registered DejaVuSans for Cyrillic PDF rendering")
            return FONT_NORMAL
        except Exception as e:
            logger.warning("Failed to register DejaVuSans: %s — falling back to Helvetica", e)

    # Fallback: Helvetica (no Cyrillic but won't crash)
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"
    _REGISTERED = True
    return FONT_NORMAL
