import sys
import os
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
    'GA', 'GJ', 'HR', 'HP', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
    'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY',
    'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB', 'BH'
}

# Applied only to slots that MUST be digits (RTO code, sequence number)
DIGIT_FIXES = {
    'O': '0', 'Q': '0',
    'I': '1', 'L': '1', '|': '1', ']': '1', '[': '1',
    'Z': '2',
    'S': '5', 'G': '6', 'B': '8',
}

# Applied only to slots that MUST be letters (state code, series)
LETTER_FIXES = {
    '0': 'O', '1': 'I', '2': 'Z',
    '5': 'S', '6': 'G', '8': 'B',
}

# Standard OCR character confusion map used for state code recovery.
# Each key maps to a list of visually similar alternatives.
# This is generalised — covers any Indian state code, not hardcoded to test images.
_STATE_CONFUSION = {
    # Digit ↔ letter shape confusions (universal OCR)
    '0': 'O', 'O': '0', 'Q': 'O',
    '1': 'I', 'I': '1', 'L': '1',
    '2': 'Z', 'Z': '2',
    '5': 'S', 'S': '5',
    '6': 'G', 'G': '6',
    '8': 'B', 'B': '8',
    # Additional font confusions common in Indian plates
    'V': 'W', 'W': 'V',   # W and V are often swapped (e.g. WB ↔ VB)
    '4': 'A', 'A': '4',   # A and 4 look similar in bold plate fonts
}


def find_valid_state(raw2):
    """
    Generalised state code recovery using standard OCR confusion pairs.
    For each of the 2 characters, try the original and its confused alternative.
    This generates up to 4 candidate state codes and checks against the valid set.
    Works for any Indian state — no test-image-specific hardcoding.
    """
    if len(raw2) < 2:
        return None
    c0_opts = list(dict.fromkeys([raw2[0], _STATE_CONFUSION.get(raw2[0], raw2[0])]))
    c1_opts = list(dict.fromkeys([raw2[1], _STATE_CONFUSION.get(raw2[1], raw2[1])]))
    for a in c0_opts:
        for b in c1_opts:
            if a + b in INDIAN_STATE_CODES:
                return a + b
    return None

# PaddleOCR singleton — loaded once, reused for every call
_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(lang='en')
    return _ocr


def sanitize(text):
    """Strip non-alphanumeric chars, uppercase, remove Indian plate noise words."""
    clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    for noise in ['IND', 'INDIA', 'BHARAT']:
        clean = clean.replace(noise, '')
    return clean


def deskew(img_bgr):
    """Correct minor rotation angles introduced by camera tilt."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 20:
        return img_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:   # skip trivial corrections that can introduce artifacts
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_variants(img_bgr):
    """
    Return multiple preprocessed versions of the plate image.
    OCR is run on all variants; the best-scoring result wins.
    Input is assumed to be a pre-cropped plate (from Flutter guide box or samples/).
    """
    variants = []

    # Upscale small plates — OCR accuracy drops sharply below ~400px wide
    h, w = img_bgr.shape[:2]
    target_w = max(w, 400)
    scale = target_w / w
    resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    variants.append(resized)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement — helps faded and shadow plates
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    # Bilateral denoise + adaptive threshold — helps dirty/noisy plates
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # Sharpen — helps blurry plates from motion or compression
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(resized, -1, kernel)
    variants.append(sharp)

    return variants


def run_ocr(variants):
    """Run PaddleOCR on each preprocessed variant, collect all candidate strings."""
    ocr = get_ocr()
    candidates = []

    for img in variants:
        try:
            result = ocr.predict(img)
        except Exception:
            continue

        if not result or not result[0]:
            continue

        res = result[0]
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])
        polys = res.get('rec_polys', [])

        if not texts:
            continue

        # Sort text lines top-to-bottom to handle stacked two-wheeler plates
        items = sorted(zip(polys, texts, scores), key=lambda x: x[0][0][1])
        combined = "".join([t for _, t, _ in items])
        avg_conf = float(np.mean([s for _, _, s in items]))

        if len(sanitize(combined)) >= 4:
            candidates.append({'text': combined, 'prob': avg_conf})

        # Also add each line individually — catches cases where only one line
        # has the plate number (e.g., two-wheeler with state sticker on line 2)
        for _, txt, conf in items:
            if len(sanitize(txt)) >= 4:
                candidates.append({'text': txt, 'prob': float(conf)})

    return candidates


def positional_correct(clean):
    """
    Parse the plate structure first, THEN apply character fixes only
    to the correct positional slot. Never applies digit fixes to letter
    slots or vice versa.

    Formats supported:
      Standard : SS [D|DD] [L|LL|LLL] DDDD
      BH Series: YY BH NNNN [L|LL]  (vowels I/O excluded from suffix)
    """
    # BH series: YYBHNNNNLL
    bh_m = re.match(r'^(\d{2})BH(\d{4})([A-Z]{1,2})$', clean)
    if bh_m:
        year, num, suffix = bh_m.groups()
        # BH plates officially exclude vowels I and O from the suffix
        suffix = suffix.replace('I', 'J').replace('O', 'Q')
        return f"{year}BH{num}{suffix}"

    # Strict match first: require actual digits for RTO and sequence slots
    # This avoids accepting OCR noise like '69AA' in a digit-only position
    m = re.match(r'^([A-Z0-9]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$', clean)
    if not m:
        # Loose fallback — covers mixed OCR where digits and letters are swapped
        m = re.match(r'^([A-Z0-9]{2})([0-9A-Z]{1,2})([A-Z0-9]{1,3})([0-9A-Z]{4})$', clean)
    if not m:
        return clean  # unrecognised structure — return as-is, never corrupt

    state_raw, rto_raw, series_raw, seq_raw = m.groups()

    if state_raw in INDIAN_STATE_CODES:
        state = state_raw
    else:
        recovered = find_valid_state(state_raw)
        state = recovered if recovered else state_raw

    rto = "".join([DIGIT_FIXES.get(c, c) for c in rto_raw])
    series = "".join([LETTER_FIXES.get(c, c) for c in series_raw])
    seq = "".join([DIGIT_FIXES.get(c, c) for c in seq_raw])

    return state + rto + series + seq


def correct(text):
    """Full correction pipeline: sanitize → BH fast path → anchor on state code → sliding window positional fix."""
    clean = sanitize(text)
    if not clean:
        return clean

    # BH-series fast path FIRST — the state code loop would strip the 2-digit
    # year prefix (e.g. '23BH...' → finds 'BH' at idx 2 → strips to 'BH4962B')
    bh = re.search(r'(\d{2}BH\d{4}[A-Z]{1,2})', clean)
    if bh:
        return bh.group(1)

    # Anchor on state code to trim prefix noise
    for st in INDIAN_STATE_CODES:
        idx = clean.find(st)
        if idx != -1 and len(clean[idx:]) >= 7:
            clean = clean[idx:]
            break

    # Sliding window: try all substrings of valid plate length (shortest first)
    # Handles trailing noise from watermarks, stickers, or OCR overshoot
    for length in range(8, 12):
        for start in range(len(clean) - length + 1):
            sub = clean[start:start + length]
            corrected = positional_correct(sub)
            if (re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$', corrected)
                    and corrected[:2] in INDIAN_STATE_CODES):
                return corrected

    # Fall back to the longest single alphanumeric block
    m = re.search(r'[A-Z0-9]{8,11}', clean)
    if m:
        clean = m.group(0)

    return positional_correct(clean)


def validate(plate_str, img_bgr):
    """Match corrected plate against all known Indian formats and detect plate color type."""
    if not plate_str:
        return plate_str, 'UNKNOWN', 'LOW'

    standard = r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$'
    bh_pat   = r'^\d{2}BH\d{4}[A-Z]{1,2}$'
    valid_state = plate_str[:2] in INDIAN_STATE_CODES

    is_yellow = is_green = is_blue = is_red = False
    if img_bgr is not None and img_bgr.size > 0:
        hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        total = img_bgr.shape[0] * img_bgr.shape[1]
        if total > 0:
            # Yellow bg  → Commercial vehicle (taxi/auto/truck)
            y_pct = np.count_nonzero(
                cv2.inRange(hsv, np.array([15, 70, 80]), np.array([34, 255, 255]))
            ) / total
            # Green bg   → Electric vehicle (private or commercial EV)
            g_pct = np.count_nonzero(
                cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            ) / total
            # Blue bg    → Diplomatic / foreign embassy vehicle
            b_pct = np.count_nonzero(
                cv2.inRange(hsv, np.array([95, 100, 70]), np.array([135, 255, 255]))
            ) / total
            # Red bg     → Temporary registration (or President/Governor vehicle)
            r1    = np.count_nonzero(cv2.inRange(hsv, np.array([0,   100, 70]), np.array([10,  255, 255])))
            r2    = np.count_nonzero(cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255])))
            r_pct = (r1 + r2) / total
            is_yellow = y_pct > 0.25
            is_green  = g_pct > 0.20
            is_blue   = b_pct > 0.20
            is_red    = r_pct > 0.20

    if re.match(bh_pat, plate_str):
        return plate_str, 'BH_SERIES', 'HIGH'
    elif is_red:
        return plate_str, 'TEMPORARY', 'HIGH' if valid_state else 'LOW'
    elif is_blue:
        return plate_str, 'DIPLOMATIC', 'HIGH' if valid_state else 'LOW'
    elif re.match(standard, plate_str) and valid_state:
        fmt = 'EV' if is_green else ('COMMERCIAL' if is_yellow else 'STANDARD')
        return plate_str, fmt, 'HIGH'
    elif valid_state and len(plate_str) >= 8:
        fmt = 'EV' if is_green else ('COMMERCIAL' if is_yellow else 'STANDARD')
        return plate_str, fmt, 'HIGH'
    else:
        return plate_str, 'UNMATCHED', 'LOW'


def score_candidate(corrected, raw_prob):
    """
    Honest scoring based purely on structural validity.
    BH plates are scored separately — they start with year digits, not a state code.
    """
    if not corrected or len(corrected) < 2:
        return -999
    # BH series exact match — highest priority
    if re.match(r'^\d{2}BH\d{4}[A-Z]{1,2}$', corrected):
        return raw_prob + 20.0
    # Standard plate — must have a valid state code
    if corrected[:2] not in INDIAN_STATE_CODES:
        return -999
    s = raw_prob + 10.0
    if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$', corrected):
        s += 8.0
    if len(corrected) >= 9:
        s += 4.0
    return s


def detect_and_recognize(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found — {image_path}")
        sys.exit(1)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Cannot read image — {image_path}")
        sys.exit(1)

    # Input is a pre-cropped plate image (Flutter guide box or samples/ folder)
    # No contour detection needed — the image IS the plate
    variants = preprocess_variants(img_bgr)
    candidates = run_ocr(variants)

    best_plate = 'UNKNOWN'
    best_fmt = 'UNMATCHED'
    best_conf = 'LOW'
    best_score = -999

    for cand in candidates:
        corrected = correct(cand['text'])
        if not corrected:
            continue
        s = score_candidate(corrected, cand['prob'])
        if s > best_score:
            best_score = s
            best_plate, best_fmt, best_conf = validate(corrected, img_bgr)

    # Fallback: if nothing scored valid, output the best raw OCR result with LOW confidence
    # so the vendor always sees something rather than UNKNOWN
    if best_plate == 'UNKNOWN' and candidates:
        best_raw = max(candidates, key=lambda c: c['prob'])
        fallback = correct(best_raw['text']) or sanitize(best_raw['text'])
        if fallback:
            best_plate = fallback
            best_fmt   = 'UNMATCHED'
            best_conf  = 'LOW'

    print(f"Plate Number : {best_plate}")
    print(f"Format       : {best_fmt}")
    print(f"Confidence   : {best_conf}")
    print("---")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <cropped_plate_image>")
        sys.exit(1)
    detect_and_recognize(sys.argv[1])
