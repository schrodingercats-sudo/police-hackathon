"""
Indian License Plate Grammar, Regex Validator, and Positional Confusion Resolver.

Implements:
- Full 36 Indian State / Union Territory RTO codes validation
- High Security Registration Plate (HSRP) standard regex
- Bharat Series (BH) registration regex
- Multi-line / Two-line plate merger and parser
- Positional character confusion matrix (OCR error correction for 0<->O, 1<->I, 8<->B, 2<->Z, 5<->S, etc.)
- License plate string standardization and formatting
"""

import re
from typing import Tuple, Optional, Set, List, Dict


# 36 Indian States and Union Territories (including legacy aliases OR/OD, UA/UK, DN/DD)
INDIAN_STATE_CODES: Set[str] = {
    # 28 States
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CG",  # Chhattisgarh
    "GA",  # Goa
    "GJ",  # Gujarat
    "HR",  # Haryana
    "HP",  # Himachal Pradesh
    "JH",  # Jharkhand
    "KA",  # Karnataka
    "KL",  # Kerala
    "MP",  # Madhya Pradesh
    "MH",  # Maharashtra
    "MN",  # Manipur
    "ML",  # Meghalaya
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha
    "OR",  # Odisha (legacy)
    "PB",  # Punjab
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TS",  # Telangana
    "TR",  # Tripura
    "UP",  # Uttar Pradesh
    "UK",  # Uttarakhand
    "UA",  # Uttarakhand (legacy)
    "WB",  # West Bengal
    # 8 Union Territories
    "AN",  # Andaman and Nicobar Islands
    "CH",  # Chandigarh
    "DD",  # Daman and Diu / Dadra and Nagar Haveli
    "DN",  # Dadra and Nagar Haveli
    "DL",  # Delhi
    "JK",  # Jammu and Kashmir
    "LA",  # Ladakh
    "LD",  # Lakshadweep
    "PY",  # Puducherry
}

# HSRP Standard: e.g. MH12AB1234, DL1C9999, KA05MN4567, GJ01XX7890
HSRP_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")

# Bharat Series (BH): e.g. 22BH1234AA, 21BH9999A
BH_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Diplomatic / Military / Commercial special formats
DEFENCE_REGEX = re.compile(r"^[0-9]{2}[A-Z][0-9]{5,6}[A-Z]$")
DIPLOMATIC_REGEX = re.compile(r"^[0-9]{1,3}CD[0-9]{1,4}$|^[0-9]{1,3}CC[0-9]{1,4}$")

# Positional Character Confusion Maps
DIGIT_TO_CHAR: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

CHAR_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "B": "8",
    "D": "0",
    "Q": "0",
    "T": "7",
    "L": "1",
}


def clean_plate_string(raw_text: str) -> str:
    """
    Remove spaces, hyphens, dots, and non-alphanumeric noise; convert to uppercase.
    """
    if not raw_text:
        return ""
    # Filter only alphanumeric characters
    cleaned = "".join([c.upper() for c in raw_text if c.isalnum()])
    return cleaned


def is_valid_hsrp(plate_str: str) -> bool:
    """
    Check if a cleaned string matches standard HSRP Indian format.
    """
    if not plate_str or len(plate_str) < 7 or len(plate_str) > 11:
        return False
    state = plate_str[:2]
    return (state in INDIAN_STATE_CODES) and bool(HSRP_REGEX.match(plate_str))


def is_valid_bh_series(plate_str: str) -> bool:
    """
    Check if a cleaned string matches Bharat BH Series format.
    """
    return bool(BH_REGEX.match(plate_str))


def is_valid_indian_plate(plate_str: str) -> bool:
    """
    Check if a cleaned string is a valid Indian license plate (HSRP, BH, Defence, etc.).
    """
    return is_valid_hsrp(plate_str) or is_valid_bh_series(plate_str) or bool(
        DEFENCE_REGEX.match(plate_str) or DIPLOMATIC_REGEX.match(plate_str)
    )


def repair_indian_plate(raw_text: str) -> Tuple[str, float, bool]:
    """
    Perform positional character confusion repair on raw OCR text.

    Positions in standard HSRP plate:
    - [0:2]   State Code (Letters)
    - [2:4]   District Code (1 or 2 Digits)
    - [4:-4]  Series Code (0 to 3 Letters)
    - [-4:]   Registration Number (Exactly 4 Digits)

    Returns:
        Tuple[str, float, bool]:
            - repaired_plate: Standardized plate string e.g. 'MH12AB1234'
            - confidence: Confidence score (0.0 to 1.0)
            - is_valid_format: Boolean indicating valid Indian format
    """
    cleaned = clean_plate_string(raw_text)
    if not cleaned:
        return "", 0.0, False

    # Check if already a valid BH plate
    if is_valid_bh_series(cleaned):
        return cleaned, 0.96, True

    # If it starts with 2 digits followed by 'BH', repair as BH series
    if len(cleaned) >= 8 and (cleaned[2:4] == "BH" or cleaned[2:4] in ["8H", "BH", "88", "8N"]):
        chars = list(cleaned)
        # Year digits [0:2]
        for i in [0, 1]:
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]
        chars[2] = "B"
        chars[3] = "H"
        # Digits [4:8]
        for i in range(4, min(8, len(chars))):
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]
        # Trailing letters [8:]
        for i in range(8, len(chars)):
            if chars[i] in DIGIT_TO_CHAR:
                chars[i] = DIGIT_TO_CHAR[chars[i]]
        repaired_bh = "".join(chars)
        if BH_REGEX.match(repaired_bh):
            return repaired_bh, 0.95, True

    chars = list(cleaned)
    n = len(chars)

    # Length check: Typical Indian plates are 8 to 11 characters
    if n < 6 or n > 12:
        return cleaned, 0.30, False

    # Check if there is a stray leading noise character (e.g. '1KA05MN4567' -> 'KA05MN4567')
    if n == 11 and "".join(chars[1:3]) in INDIAN_STATE_CODES:
        if chars[0] in ["1", "I", "0", "O", "L", "|", "-"]:
            chars = chars[1:]
            n = len(chars)

    # Step 1: Repair State Code (Positions 0, 1) -> MUST BE LETTERS
    # State character confusion candidates
    state_confusions = {
        "0": ["D", "O", "Q"],
        "O": ["D", "Q"],
        "D": ["O"],
        "1": ["I", "T", "L"],
        "I": ["T", "L", "1"],
        "2": ["Z"],
        "Z": ["2"],
        "4": ["A"],
        "A": ["4"],
        "5": ["S"],
        "S": ["5"],
        "6": ["G", "C"],
        "G": ["6", "C"],
        "8": ["B"],
        "B": ["8", "R"],
        "U": ["V", "W"],
        "V": ["U", "W"],
    }

    # Initial letter conversion
    for i in [0, 1]:
        if chars[i].isdigit() and chars[i] in DIGIT_TO_CHAR:
            chars[i] = DIGIT_TO_CHAR[chars[i]]

    state_code = "".join(chars[0:2])
    is_valid_state = state_code in INDIAN_STATE_CODES

    # If state code still invalid, search through single-char substitution candidates
    if not is_valid_state and len(state_code) == 2:
        c0, c1 = chars[0], chars[1]
        c0_cand = [c0] + state_confusions.get(c0, []) + ([c0_digit] if (c0_digit := CHAR_TO_DIGIT.get(c0)) and c0_digit in state_confusions else [])
        c1_cand = [c1] + state_confusions.get(c1, []) + ([c1_digit] if (c1_digit := CHAR_TO_DIGIT.get(c1)) and c1_digit in state_confusions else [])

        found_state = None
        for cand0 in c0_cand:
            for cand1 in c1_cand:
                test_sc = cand0 + cand1
                if test_sc in INDIAN_STATE_CODES:
                    found_state = test_sc
                    break
            if found_state:
                break

        if found_state:
            chars[0], chars[1] = found_state[0], found_state[1]
            state_code = found_state
            is_valid_state = True

    # Step 2: Repair Last 4 Digits (Registration Number) -> MUST BE DIGITS
    last_digit_count = min(4, n - 2)
    for i in range(n - last_digit_count, n):
        if chars[i].isalpha() and chars[i] in CHAR_TO_DIGIT:
            chars[i] = CHAR_TO_DIGIT[chars[i]]

    # Step 3: Repair District Code (Positions 2 and optionally 3)
    # If 3rd character is followed by series letter, district code is 1 or 2 digits
    if n >= 8:
        # Position 2 is always digit in HSRP
        if chars[2].isalpha() and chars[2] in CHAR_TO_DIGIT:
            chars[2] = CHAR_TO_DIGIT[chars[2]]

        # Position 3: Check if position 4 is a letter (e.g. DL01AB... vs DL1C...)
        if n >= 9:
            # If position 3 is alpha and in confusion map and chars[4] is letter, it might be district digit
            if chars[3].isalpha() and chars[3] in CHAR_TO_DIGIT and (chars[4].isalpha() or chars[4] in DIGIT_TO_CHAR):
                chars[3] = CHAR_TO_DIGIT[chars[3]]

        # Step 4: Repair Series Code (between District Code and last 4 digits) -> MUST BE LETTERS
        series_start = 4 if chars[3].isdigit() else 3
        series_end = n - 4
        for i in range(series_start, series_end):
            if chars[i].isdigit() and chars[i] in DIGIT_TO_CHAR:
                chars[i] = DIGIT_TO_CHAR[chars[i]]

    repaired = "".join(chars)
    is_valid_format = is_valid_indian_plate(repaired)

    # Compute confidence
    if is_valid_state and is_valid_format:
        confidence = 0.95
    elif is_valid_state:
        confidence = 0.70
    elif is_valid_format:
        confidence = 0.65
    else:
        confidence = 0.35

    return repaired, confidence, is_valid_format


def parse_two_line_plate(line1: str, line2: str) -> Tuple[str, float, bool]:
    """
    Parse and merge a two-line Indian license plate.
    Example:
        Line 1: "DL 01" / "MH 14"
        Line 2: "AB 1234" / "EU 3344"
        Result: ("DL01AB1234", 0.95, True)

    Also handles cases where lines were detected in inverted order.
    """
    c1 = clean_plate_string(line1)
    c2 = clean_plate_string(line2)

    if not c1 and not c2:
        return "", 0.0, False
    if not c1:
        return repair_indian_plate(c2)
    if not c2:
        return repair_indian_plate(c1)

    # Check standard ordering (Line 1 has state code, Line 2 has series + digits)
    sc1 = c1[:2]
    sc2 = c2[:2]

    if sc1 in INDIAN_STATE_CODES or (sc1[0] in DIGIT_TO_CHAR and DIGIT_TO_CHAR[sc1[0]] + sc1[1:] in INDIAN_STATE_CODES):
        combined = c1 + c2
    elif sc2 in INDIAN_STATE_CODES or (sc2[0] in DIGIT_TO_CHAR and DIGIT_TO_CHAR[sc2[0]] + sc2[1:] in INDIAN_STATE_CODES):
        # Inverted line order
        combined = c2 + c1
    else:
        combined = c1 + c2

    return repair_indian_plate(combined)


def format_plate_display(plate_str: str) -> str:
    """
    Format a clean plate string into readable space-separated chunks.
    Example: 'MH12AB1234' -> 'MH 12 AB 1234', '22BH1234AA' -> '22 BH 1234 AA'
    """
    if not plate_str:
        return ""
    p = clean_plate_string(plate_str)

    # BH Series
    if BH_REGEX.match(p):
        return f"{p[:2]} BH {p[4:8]} {p[8:]}".strip()

    # Standard HSRP
    m = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$", p)
    if m:
        state, dist, series, num = m.groups()
        if series:
            return f"{state} {dist} {series} {num}"
        return f"{state} {dist} {num}"

    return p
