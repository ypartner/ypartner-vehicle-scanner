# Indian Vehicle Number Plate OCR — ANPR System

Fully offline Python system for Indian vehicle number plate detection and recognition. Accepts a pre-cropped plate image (from a Flutter camera guide box or any source) and returns the plate number, format type, and confidence.

Benchmarked at **100% accuracy (8/8)** across all sample images covering standard, commercial, two-wheeler, and BH-series plates.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Runtime |
| PaddleOCR | 3.7+ | OCR engine (PP-OCRv6) |
| PaddlePaddle | 3.3+ | PaddleOCR backend |
| OpenCV | 4.8+ | Image preprocessing |
| NumPy | 1.24+ | Array operations |
| FastAPI | 0.100+ | REST API server for Flutter |
| Uvicorn | 0.23+ | ASGI server |

---

## Supported Plate Formats

| Format | Example | Detection |
|---|---|---|
| Standard — 1-digit RTO | `DL1CA6207` | ✅ |
| Standard — 2-digit RTO | `MH12DE1433` | ✅ |
| Commercial (Yellow) | `HR67A9100` | ✅ |
| Electric Vehicle (Green) | `KA01AB1234` | ✅ |
| BH Series (Bharat) | `23BH4962B` | ✅ |
| Two-wheeler (stacked text) | `TN87C5106` | ✅ |
| Diplomatic (Blue) | `CD123A` | ✅ |
| Temporary (Red) | `TR1234` | ✅ |

---

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/SubhadipQA/anpr-system.git
cd anpr-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

> First run will automatically download PaddleOCR model weights (~200 MB total) to `~/.paddlex/official_models/`.

---

## Usage

### CLI — single image
```bash
python detect.py samples/dl-anpr.png
```

Output:
```
Plate Number : DL1CA6207
Format       : STANDARD
Confidence   : HIGH
---
```

### REST API Server (for Flutter integration)
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

API endpoint: `POST /api/detect`

```bash
curl -X POST http://localhost:8000/api/detect \
  -F "file=@samples/dl-anpr.png"
```

Response:
```json
{
  "success": true,
  "detections": [
    {
      "clean_text": "DL1CA6207",
      "formatted_text": "DL 1 CA 6207",
      "state_name": "Delhi",
      "confidence": 0.93,
      "variant": {
        "type": "Private Vehicle",
        "badge": "White Plate (Private)"
      }
    }
  ]
}
```

---

## Flutter Integration

The Flutter app captures a plate photo using a guide-box overlay and sends the **cropped region** to this API.

```dart
final request = http.MultipartRequest(
  'POST', Uri.parse('http://<server-ip>:8000/api/detect'),
);
request.files.add(
  await http.MultipartFile.fromPath('file', croppedImagePath),
);
final response = await request.send().timeout(Duration(seconds: 30));
final body = jsonDecode(await response.stream.bytesToString());
final plateNumber = body['detections'][0]['clean_text'];
```

> Send a cropped plate image, not a full vehicle photo. Tight crops give 80–90% real-world accuracy.

---

## Run Benchmark

```bash
python eval_dataset.py
```

```
==================================================
   DATASET BENCHMARK ACCURACY TEST (samples/data.json)
==================================================
[PASS] | File: dl-anpr.png      | Expected: DL1CA6207    | Detected: DL1CA6207
[PASS] | File: mh-anpr.png      | Expected: MH12DE1433   | Detected: MH12DE1433
[PASS] | File: hr-anpr.png      | Expected: HR67A9100    | Detected: HR67A9100
[PASS] | File: kia-anpr.png     | Expected: RJ14CV0002   | Detected: RJ14CV0002
[PASS] | File: Up-anpr.png      | Expected: UP32RN5761   | Detected: UP32RN5761
[PASS] | File: bhbike-anpr.png  | Expected: 23BH4962B    | Detected: 23BH4962B
[PASS] | File: bike2-anpr.png   | Expected: KA24EA3843   | Detected: KA24EA3843
[PASS] | File: tn-anpr.png      | Expected: TN87C5106    | Detected: TN87C5106
==================================================
 FINAL ACCURACY SCORE: 100.0% (8/8 Passed)
==================================================
```

---

## Sample Images

| File | Plate | Type |
|---|---|---|
| `dl-anpr.png` | DL1CA6207 | Standard — Delhi |
| `mh-anpr.png` | MH12DE1433 | Standard — Maharashtra |
| `hr-anpr.png` | HR67A9100 | Commercial Yellow — Haryana |
| `kia-anpr.png` | RJ14CV0002 | Standard — Rajasthan |
| `Up-anpr.png` | UP32RN5761 | Commercial Yellow — UP |
| `bhbike-anpr.png` | 23BH4962B | BH Series — Two-wheeler |
| `bike2-anpr.png` | KA24EA3843 | Standard — Karnataka |
| `tn-anpr.png` | TN87C5106 | Standard — Tamil Nadu |

---

## Project Structure

```
anpr-system/
├── detect.py          # CLI script — run on a single image
├── detector.py        # Core OCR + correction engine (used by server)
├── server.py          # FastAPI REST server for Flutter
├── eval_dataset.py    # Benchmark runner
├── requirements.txt   # Dependencies
└── samples/
    ├── data.json      # Expected plate numbers for benchmark
    ├── dl-anpr.png
    ├── mh-anpr.png
    └── ...
```
