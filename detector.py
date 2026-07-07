import cv2
import numpy as np
import re
import base64
from paddleocr import PaddleOCR

_ocr = None


def get_ocr_reader():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(lang='en')
    return _ocr

INDIAN_STATES = {
    'AN': 'Andaman & Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
    'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh',
    'DD': 'Daman & Diu', 'DL': 'Delhi', 'DN': 'Dadra & Nagar Haveli', 'GA': 'Goa',
    'GJ': 'Gujarat', 'HR': 'Haryana', 'HP': 'Himachal Pradesh', 'JH': 'Jharkhand',
    'JK': 'Jammu & Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala', 'LA': 'Ladakh',
    'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya', 'MN': 'Manipur',
    'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha',
    'OR': 'Odisha', 'PB': 'Punjab', 'PY': 'Puducherry', 'RJ': 'Rajasthan',
    'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura', 'TS': 'Telangana',
    'UK': 'Uttarakhand', 'UP': 'Uttar Pradesh', 'WB': 'West Bengal', 'BH': 'Bharat Series'
}

def image_to_base64_url(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return ""
    success, buffer = cv2.imencode('.png', img_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def classify_plate_variant(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            'type': 'Private Vehicle',
            'icon': '🚗',
            'badge': 'White Plate (Private)',
            'bg_name': 'White',
            'color_hex': '#FFFFFF',
            'text_color': '#000000',
            'description': 'Standard personal motor vehicle plate (White background with black letters).'
        }

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    if total_pixels == 0:
        total_pixels = 1

    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_pct = (np.count_nonzero(green_mask) / total_pixels) * 100

    yellow_mask = cv2.inRange(hsv, np.array([15, 70, 80]), np.array([34, 255, 255]))
    yellow_pct = (np.count_nonzero(yellow_mask) / total_pixels) * 100

    red_mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))
    red_pct = ((np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / total_pixels) * 100

    blue_mask = cv2.inRange(hsv, np.array([95, 70, 70]), np.array([135, 255, 255]))
    blue_pct = (np.count_nonzero(blue_mask) / total_pixels) * 100

    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    black_pct = (np.count_nonzero(black_mask) / total_pixels) * 100

    if green_pct > 20:
        if yellow_pct > 10:
            return {
                'type': 'Commercial Electric Vehicle (EV)',
                'icon': '⚡🚕',
                'badge': 'Green Plate + Yellow Text',
                'bg_name': 'Green',
                'color_hex': '#10B981',
                'text_color': '#FACC15',
                'description': 'Zero-emission commercial vehicle (Taxi/Bus/Delivery EV).'
            }
        else:
            return {
                'type': 'Private Electric Vehicle (EV)',
                'icon': '⚡🚗',
                'badge': 'Green Plate + White Text',
                'bg_name': 'Green',
                'color_hex': '#10B981',
                'text_color': '#FFFFFF',
                'description': 'Zero-emission personal electric vehicle.'
            }
    elif yellow_pct > 25:
        return {
            'type': 'Commercial Vehicle',
            'icon': '🚕',
            'badge': 'Yellow Plate (Commercial)',
            'bg_name': 'Yellow',
            'color_hex': '#FACC15',
            'text_color': '#000000',
            'description': 'Commercial transport vehicle (Taxi, Auto, Truck, Commercial Bus).'
        }
    elif black_pct > 35 and yellow_pct > 10:
        return {
            'type': 'Self-Drive / Rental Vehicle',
            'icon': '🏎️',
            'badge': 'Black Plate + Yellow Text',
            'bg_name': 'Black',
            'color_hex': '#1F2937',
            'text_color': '#FACC15',
            'description': 'Commercial self-drive rental vehicle (e.g., Zoomcar).'
        }
    elif red_pct > 25:
        return {
            'type': 'Temporary Registration',
            'icon': '🔴',
            'badge': 'Red Plate (Temporary)',
            'bg_name': 'Red',
            'color_hex': '#EF4444',
            'text_color': '#FFFFFF',
            'description': 'Brand new vehicle with temporary 1-month registration plate.'
        }
    elif blue_pct > 25:
        return {
            'type': 'Diplomatic / Foreign Mission',
            'icon': '🔵',
            'badge': 'Blue Plate (Diplomatic)',
            'bg_name': 'Blue',
            'color_hex': '#3B82F6',
            'text_color': '#FFFFFF',
            'description': 'Vehicle belonging to foreign embassies or UN diplomats.'
        }
    else:
        return {
            'type': 'Private Vehicle',
            'icon': '🚗',
            'badge': 'White Plate (Private)',
            'bg_name': 'White',
            'color_hex': '#F3F4F6',
            'text_color': '#111827',
            'description': 'Standard personal vehicle plate.'
        }

def sanitize_ocr_text(text):
    clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    for noise in ['IND', 'INDIA', 'BHARAT']:
        clean = clean.replace(noise, '')
    return clean


# Applied only to slots that MUST be digits (RTO code, sequence number)
_DIGIT_FIXES = {
    'O': '0', 'Q': '0',
    'I': '1', 'L': '1', '|': '1', ']': '1', '[': '1',
    'Z': '2',
    'S': '5', 'G': '6', 'B': '8',
}

# Applied only to slots that MUST be letters (state code, series)
_LETTER_FIXES = {
    '0': 'O', '1': 'I', '2': 'Z',
    '5': 'S', '6': 'G', '8': 'B',
}

# Generalised OCR confusion map for state code recovery.
_STATE_CONFUSION = {
    '0': 'O', 'O': '0', 'Q': 'O',
    '1': 'I', 'I': '1', 'L': '1',
    '2': 'Z', 'Z': '2',
    '5': 'S', 'S': '5',
    '6': 'G', 'G': '6',
    '8': 'B', 'B': '8',
    'V': 'W', 'W': 'V',
    '4': 'A', 'A': '4',
}


def _find_valid_state(raw2):
    """
    Generalised state code recovery using standard OCR confusion pairs.
    Tries up to 4 candidate codes derived from visual confusion of each character.
    Works for any Indian state — no test-image-specific hardcoding.
    """
    if len(raw2) < 2:
        return None
    c0_opts = list(dict.fromkeys([raw2[0], _STATE_CONFUSION.get(raw2[0], raw2[0])]))
    c1_opts = list(dict.fromkeys([raw2[1], _STATE_CONFUSION.get(raw2[1], raw2[1])]))
    for a in c0_opts:
        for b in c1_opts:
            if a + b in INDIAN_STATES:
                return a + b
    return None


def _positional_correct(clean):
    """
    Parse the plate structure first, THEN apply character fixes only to
    the correct positional slot.

    Formats supported:
      Standard : SS [D|DD] [L|LL|LLL] DDDD
      BH Series: YY BH NNNN [L|LL]  (vowels I/O excluded from suffix)
    """
    # BH series: YYBHNNNNLL
    bh_m = re.match(r'^(\d{2})BH(\d{4})([A-Z]{1,2})$', clean)
    if bh_m:
        year, num, suffix = bh_m.groups()
        suffix = suffix.replace('I', 'J').replace('O', 'Q')
        return f"{year}BH{num}{suffix}"

    # Strict match first: RTO and sequence must be actual digits
    m = re.match(r'^([A-Z0-9]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$', clean)
    if not m:
        m = re.match(r'^([A-Z0-9]{2})([0-9A-Z]{1,2})([A-Z0-9]{1,3})([0-9A-Z]{4})$', clean)
    if not m:
        return clean

    state_raw, rto_raw, series_raw, seq_raw = m.groups()

    if state_raw in INDIAN_STATES:
        state = state_raw
    else:
        recovered = _find_valid_state(state_raw)
        state = recovered if recovered else state_raw

    rto = "".join([_DIGIT_FIXES.get(c, c) for c in rto_raw])
    series = "".join([_LETTER_FIXES.get(c, c) for c in series_raw])
    seq = "".join([_DIGIT_FIXES.get(c, c) for c in seq_raw])

    return state + rto + series + seq


def correct_character_confusions(text):
    clean = sanitize_ocr_text(text)
    if not clean:
        return clean

    # BH-series fast path FIRST — the state code loop would strip the 2-digit
    # year prefix (e.g. '23BH...' finds 'BH' at idx 2 and strips to 'BH4962B')
    bh = re.search(r'(\d{2}BH\d{4}[A-Z]{1,2})', clean)
    if bh:
        return bh.group(1)

    for st in INDIAN_STATES.keys():
        idx = clean.find(st)
        if idx != -1 and len(clean[idx:]) >= 7:
            clean = clean[idx:]
            break

    # Sliding window: try all substrings of valid plate length (shortest first)
    for length in range(8, 12):
        for start in range(len(clean) - length + 1):
            sub = clean[start:start + length]
            corrected = _positional_correct(sub)
            if (re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$', corrected)
                    and corrected[:2] in INDIAN_STATES):
                return corrected

    m = re.search(r'[A-Z0-9]{8,11}', clean)
    if m:
        clean = m.group(0)

    return _positional_correct(clean)


def format_indian_plate_with_spaces(plate_str):
    if not plate_str or len(plate_str) < 7:
        return plate_str, None

    # BH series: YYBHNNNNLL → "YY BH NNNN LL"
    bh_m = re.match(r'^(\d{2})BH(\d{4})([A-Z]{1,2})$', plate_str)
    if bh_m:
        year, num, suffix = bh_m.groups()
        return f"{year} BH {num} {suffix}", "Bharat Series"

    # Standard: SS[D|DD][L..][DDDD]
    state_part = plate_str[:2]
    seq_part = plate_str[-4:]
    rem = plate_str[2:-4]

    if len(rem) >= 2 and rem[0].isdigit() and rem[1].isalpha():
        rto_part = rem[0]
        series_part = rem[1:]
        formatted = f"{state_part} {rto_part} {series_part} {seq_part}"
    else:
        rto_part = rem[:2] if len(rem) >= 2 else rem
        series_part = rem[2:] if len(rem) > 2 else ""
        formatted = f"{state_part} {rto_part} {series_part} {seq_part}"

    state_name = INDIAN_STATES.get(state_part, f"State ({state_part})")
    return formatted, state_name


def _deskew(img_bgr):
    """Correct minor rotation angles introduced by camera tilt."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 20:
        return img_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def get_preprocessed_crops(crop_bgr):
    crops = []
    if crop_bgr is None or crop_bgr.size == 0:
        return crops

    h, w = crop_bgr.shape[:2]
    target_w = max(w, 400)
    scale = target_w / w
    resized = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    crops.append(resized)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    crops.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    crops.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(resized, -1, kernel)
    crops.append(sharp)

    return crops


def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return [], ""

    orig_h, orig_w = image_bgr.shape[:2]
    ocr = get_ocr_reader()

    variants = get_preprocessed_crops(image_bgr)
    candidates = []

    for img in variants:

        try:
            result = ocr.ocr(img, cls=True)

            print("\n========== OCR RESULT ==========")
            print(result)
            print("===============================\n")

        except Exception:
            import traceback
            traceback.print_exc()
            continue

        if not result or not result[0]:
            continue

        # res = result[0]

        texts = []
        scores = []
        polys = []

        for line in result[0]:
            polys.append(line[0])        # bounding box
            texts.append(line[1][0])     # detected text
            scores.append(line[1][1])    #

        if not texts:
            continue

        items = sorted(
            zip(polys, texts, scores),
            key=lambda x: x[0][0][1]
        )

        combined = "".join([t for _, t, _ in items])
        avg_conf = float(np.mean([s for _, _, s in items]))

        if len(sanitize_ocr_text(combined)) >= 4:
            candidates.append({
                "text": combined,
                "prob": avg_conf,
                "crop": image_bgr,
                "bbox": [0, 0, orig_w, orig_h]
            })

        for _, txt, conf in items:

            if len(sanitize_ocr_text(txt)) >= 4:
                candidates.append({
                    "text": txt,
                    "prob": float(conf),
                    "crop": image_bgr,
                    "bbox": [0, 0, orig_w, orig_h]
                })

    detections = []
    seen_plates = set()

    for cand in candidates:
        corrected = correct_character_confusions(cand['text'])
        if not corrected or corrected in seen_plates:
            continue

        formatted_text, state_name = format_indian_plate_with_spaces(corrected)
        variant_info = classify_plate_variant(cand['crop'])

        valid_state   = corrected[:2] in INDIAN_STATES
        valid_pattern = bool(re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$', corrected))
        is_bh         = bool(re.match(r'^\d{2}BH\d{4}[A-Z]{1,2}$', corrected))

        if is_bh:
            eval_score = cand['prob'] + 20.0
            system_confidence = round(min(0.70 + cand['prob'] * 0.25, 0.99), 3)
        elif not valid_state:
            eval_score = -999
            system_confidence = round(min(cand['prob'], 0.50), 3)
        else:
            eval_score = cand['prob'] + 10.0
            if valid_pattern:
                eval_score += 8.0
            if len(corrected) >= 9:
                eval_score += 4.0
            system_confidence = round(min(0.70 + cand['prob'] * 0.25, 0.99), 3)

        detections.append({
            'raw_text': cand['text'],
            'clean_text': corrected,
            'formatted_text': formatted_text,
            'state_name': state_name or 'Unknown State',
            'confidence': system_confidence,
            'raw_ocr_prob': float(cand['prob']),
            'bbox': cand['bbox'],
            'crop_b64': image_to_base64_url(cand['crop']),
            'variant': variant_info,
            'eval_score': eval_score
        })
        seen_plates.add(corrected)

    detections = sorted(detections, key=lambda d: d['eval_score'], reverse=True)
    if len(detections) > 1:
        detections = detections[:1]

    # Fallback: if nothing valid found, return best raw OCR result with LOW confidence
    if not detections and candidates:
        best_raw = max(candidates, key=lambda c: c['prob'])
        fallback_text = correct_character_confusions(best_raw['text']) or sanitize_ocr_text(best_raw['text'])
        if fallback_text:
            fmt_text, state_name = format_indian_plate_with_spaces(fallback_text)
            detections = [{
                'raw_text': best_raw['text'],
                'clean_text': fallback_text,
                'formatted_text': fmt_text,
                'state_name': state_name or 'Unknown State',
                'confidence': round(min(best_raw['prob'], 0.50), 3),
                'raw_ocr_prob': float(best_raw['prob']),
                'bbox': best_raw['bbox'],
                'crop_b64': image_to_base64_url(best_raw['crop']),
                'variant': classify_plate_variant(best_raw['crop']),
                'eval_score': -1
            }]

    annotated_bgr = image_bgr.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        color = (0, 255, 0)
        if det['variant']['bg_name'] == 'Yellow':
            color = (0, 230, 255)
        elif det['variant']['bg_name'] == 'Green':
            color = (0, 255, 128)

        cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), color, 3)
        cv2.putText(annotated_bgr, det['formatted_text'], (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    annotated_b64 = image_to_base64_url(annotated_bgr)
    return detections, annotated_b64
