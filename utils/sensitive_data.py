import re
from typing import Dict, List

# ── PCI (Payment Card Industry) ───────────────────────────────────────────────
_CARD_RE = re.compile(r'\b(?:\d[ \-]?){13,19}\b')
_CVV_RE = re.compile(r'\b(?:cvv|cvc|security\s+code)[\s:]*\d{3,4}\b', re.I)
_EXPIRY_RE = re.compile(r'\b(?:expir(?:y|es|ation))[\s:]*\d{1,2}[/\-]\d{2,4}\b', re.I)
# Standalone MM/YY or MM-YY (bare card expiry response, e.g. "09-28" or "09/28")
# Negative lookahead prevents matching MM/DD inside a full MM/DD/YYYY date
_EXPIRY_BARE_RE = re.compile(r'\b(0[1-9]|1[0-2])[/\-]\d{2}(?![/\-\d])\b')

# ── PII (Personally Identifiable Information) ─────────────────────────────────
_SSN_RE = re.compile(r'\b\d{3}[- ]\d{2}[- ]\d{4}\b')
_EMAIL_RE = re.compile(r'\b[\w.%+\-]+@[\w.\-]+\.[a-z]{2,}\b', re.I)
_PHONE_RE = re.compile(r'\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b')
_ACCOUNT_RE = re.compile(r'\b(?:account|routing)\s*(?:number)?[\s:#]*\d{6,17}\b', re.I)
_PASSPORT_RE = re.compile(r'\b(?:passport)[\s:#]*[A-Z0-9]{6,9}\b', re.I)
_DL_RE = re.compile(r"\b(?:driver'?s?\s+license)[\s:#]*[A-Z0-9\-]{6,12}\b", re.I)

# ── PHI (Protected Health Information) ───────────────────────────────────────
_MRN_RE = re.compile(r'\b(?:medical\s+record|mrn|patient\s+id)[\s:#]*[A-Z0-9\-]{4,}\b', re.I)
_DOB_RE = re.compile(r'\b(?:date\s+of\s+birth|dob|born\s+on)[\s:]*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', re.I)
# Standalone month-name DOB, e.g. "March 14, 1971" or "March 14th 1971"
_DOB_MONTH_RE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', re.I
)
_PHI_RE = re.compile(
    r'\b(?:diagnosed\s+with|prescribed|medication\s+dosage|health\s+insurance\s+id|'
    r'insurance\s+member\s+id|patient\s+record|medical\s+history)\b', re.I
)

_CHECKS: List[tuple] = [
    (_CARD_RE,       "PCI — card number"),
    (_CVV_RE,        "PCI — CVV / security code"),
    (_EXPIRY_RE,     "PCI — card expiry"),
    (_EXPIRY_BARE_RE,"PCI — card expiry"),
    (_SSN_RE,        "PII — Social Security Number"),
    (_EMAIL_RE,      "PII — email address"),
    (_PHONE_RE,      "PII — phone number"),
    (_ACCOUNT_RE,    "PII — bank account / routing number"),
    (_PASSPORT_RE,   "PII — passport number"),
    (_DL_RE,         "PII — driver's license"),
    (_MRN_RE,        "PHI — medical record number"),
    (_DOB_RE,        "PHI — date of birth"),
    (_DOB_MONTH_RE,  "PHI — date of birth"),
    (_PHI_RE,        "PHI — medical / health information"),
]

_PROFANITY = {
    "fuck", "fucking", "fucked", "fucker", "shit", "shitty", "bullshit",
    "bitch", "bitching", "asshole", "ass", "bastard", "cunt", "damn",
    "motherfucker", "dickhead", "piss", "pissed", "crap", "wtf", "stfu",
}

# Agent question keywords that signal the next customer response is sensitive
_CVV_Q_RE    = re.compile(r'security\s+code|cvv|cvc|verification\s+code', re.I)
_EXPIRY_Q_RE = re.compile(r'expir', re.I)
_DOB_Q_RE    = re.compile(r'date\s+of\s+birth|dob|\bborn\b', re.I)

# Patterns used inside context-aware masking
_BARE_CVV_RE    = re.compile(r'\b\d{3,4}\b')
_BARE_EXPIRY_RE  = re.compile(r'\b\d{1,2}[/\-]\d{2,4}\b')
_BARE_DATE_RE    = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b'
    r'|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', re.I
)
# Extracts the year portion from a DOB match for partial masking
_YEAR_FROM_NUMERIC_RE = re.compile(r'\d{4}$|\d{2}$')


def _mask_dob_numeric(m: re.Match) -> str:
    """##/##/YYYY — keep year, mask month and day. Preserves any label prefix."""
    date_match = re.search(r'\d{1,2}[/\-]\d{1,2}[/\-](\d{2,4})', m.group(0))
    if date_match:
        year = date_match.group(1)
        return m.group(0)[:date_match.start()] + f"##/##/{year}"
    return "####"


def _mask_dob_month_name(m: re.Match) -> str:
    """[DOB: YYYY] — keep year, mask month and day for written dates."""
    year_match = re.search(r'\d{4}', m.group(0))
    year = year_match.group(0) if year_match else "####"
    return f"[DOB: {year}]"


def _mask_by_context(text: str) -> str:
    """
    Mask bare values (CVV, expiry, DOB) that appear alone in a customer turn
    when the preceding agent turn asked for that specific piece of data.
    """
    lines = text.splitlines()
    result = []
    pending_context = None  # 'cvv' | 'expiry' | 'dob'

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        if low.startswith("agent:"):
            if _CVV_Q_RE.search(stripped):
                pending_context = "cvv"
            elif _EXPIRY_Q_RE.search(stripped):
                pending_context = "expiry"
            elif _DOB_Q_RE.search(stripped):
                pending_context = "dob"
            else:
                pending_context = None
            result.append(line)

        elif low.startswith("customer:") and pending_context:
            if pending_context == "cvv":
                line = _BARE_CVV_RE.sub("####", stripped)
            elif pending_context == "expiry":
                line = _BARE_EXPIRY_RE.sub("####", stripped)
            elif pending_context == "dob":
                def _bare_dob_mask(m: re.Match) -> str:
                    year = re.search(r'\d{4}', m.group(0))
                    if year:
                        return f"[DOB: {year.group(0)}]" if re.search(r'[A-Za-z]', m.group(0)) else f"##/##/{year.group(0)}"
                    return "####"
                line = _BARE_DATE_RE.sub(_bare_dob_mask, stripped)
            pending_context = None
            result.append(line)

        else:
            result.append(line)

    return "\n".join(result)


def detect_sensitive(text: str) -> Dict:
    """Returns which PCI/PHI/PII categories were found in the text."""
    found: List[str] = []
    seen: set = set()
    for pattern, label in _CHECKS:
        if label not in seen and pattern.search(text):
            found.append(label)
            seen.add(label)
    return {"has_sensitive_data": bool(found), "sensitive_data_types": found}


def detect_profanity(text: str) -> bool:
    """Returns True if any profane words are found."""
    words = {w.lower().strip("*!?.,'\"") for w in text.split()}
    return bool(words & _PROFANITY)


def mask_sensitive(text: str) -> str:
    """Replace all PCI/PHI/PII matches with ####, including bare cross-line responses."""
    for pattern, label in _CHECKS:
        if label == "PII — Social Security Number":
            text = pattern.sub(
                lambda m: "###-##-" + m.group(0).replace("-", "").replace(" ", "")[-4:], text
            )
        elif label == "PHI — date of birth":
            if pattern is _DOB_MONTH_RE:
                text = pattern.sub(_mask_dob_month_name, text)
            else:
                text = pattern.sub(_mask_dob_numeric, text)
        else:
            text = pattern.sub("####", text)
    text = _mask_by_context(text)
    return text


def mask_profanity(text: str) -> str:
    """Replace profane words with ####, case-insensitive."""
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(w) for w in sorted(_PROFANITY, key=len, reverse=True)) + r')\b',
        re.I,
    )
    return pattern.sub("####", text)
