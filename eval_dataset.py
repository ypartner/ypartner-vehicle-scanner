import json
import os
from detector import process_image

def test_dataset():
    with open("samples/data.json", "r") as f:
        data = json.load(f)

    total = len(data)
    correct = 0

    print("==================================================")
    print("   DATASET BENCHMARK ACCURACY TEST (samples/data.json) ")
    print("==================================================")

    for item in data:
        filename = item["file"]
        expected = item["number"]
        filepath = os.path.join("samples", filename)

        if not os.path.exists(filepath):
            print(f"[!] File missing: {filepath}")
            continue

        with open(filepath, "rb") as img_f:
            bytes_data = img_f.read()

        detections, _ = process_image(bytes_data)
        detected = detections[0]["clean_text"] if detections else "NONE"

        is_match = (detected == expected)
        if is_match:
            correct += 1

        status = "[PASS]" if is_match else "[FAIL]"
        print(f"{status} | File: {filename:<10} | Expected: {expected:<12} | Detected: {detected}")

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print("==================================================")
    print(f" FINAL ACCURACY SCORE: {accuracy:.1f}% ({correct}/{total} Passed)")
    print("==================================================")

if __name__ == "__main__":
    test_dataset()
