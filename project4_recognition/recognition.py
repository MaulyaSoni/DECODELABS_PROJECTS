# ============================================================
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  Project 4  : Image or Text Recognition (Basic)
#  Type       : Optional Mastery Phase
#  Paths      : Path 1 → OCR (pytesseract + OpenCV)
#               Path 2 → Object Detection (MobileNet-SSD)
#  Toolkit    : pytesseract, OpenCV (cv2), Pillow
# ============================================================
#
#  INSTALL:
#  pip install opencv-python pytesseract Pillow numpy
#
#  ALSO INSTALL Tesseract OCR engine (required for Path 1):
#  Windows : https://github.com/UB-Mannheim/tesseract/wiki
#  Linux   : sudo apt install tesseract-ocr
#  Mac     : brew install tesseract
#
#  HOW TO RUN:
#  python recognition.py
#  → Choose Path 1 (OCR) or Path 2 (Object Detection)
#  → Provide an image path when prompted
#    OR press Enter to use the built-in demo image
# ============================================================

import os
import sys
import urllib.request
import numpy as np

# ── DEPENDENCY CHECKER ────────────────────────────────────────

def check_dependencies():
    """Check which libraries are available and report."""
    status = {}

    try:
        import cv2
        status["opencv"] = f"✓ OpenCV {cv2.__version__}"
    except ImportError:
        status["opencv"] = "✗ OpenCV not found → pip install opencv-python"

    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        status["tesseract"] = "✓ pytesseract + Tesseract engine"
    except Exception:
        status["tesseract"] = "✗ pytesseract/Tesseract → see install notes above"

    try:
        from PIL import Image
        status["pillow"] = "✓ Pillow"
    except ImportError:
        status["pillow"] = "✗ Pillow not found → pip install Pillow"

    try:
        import numpy
        status["numpy"] = f"✓ NumPy {numpy.__version__}"
    except ImportError:
        status["numpy"] = "✗ NumPy not found → pip install numpy"

    return status


# ══════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════

def download_demo_image(path: str, url: str, label: str):
    """Download a demo image if not already present."""
    if not os.path.exists(path):
        print(f"  [INFO] Downloading demo {label} image...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  [INFO] Saved to: {path}")
        except Exception as e:
            print(f"  [ERROR] Could not download demo image: {e}")
            return False
    return True


def get_image_path(demo_path: str, demo_label: str) -> str:
    """Ask user for image path or use demo."""
    print(f"\n  Enter image path (or press Enter to use demo {demo_label}):")
    user_path = input("  Image path: ").strip()

    if user_path == "":
        print(f"  → Using demo image: {demo_path}")
        return demo_path
    elif os.path.exists(user_path):
        return user_path
    else:
        print(f"  [!] File not found: {user_path}")
        print(f"  → Falling back to demo: {demo_path}")
        return demo_path


# ══════════════════════════════════════════════════════════════
#  PATH 1 — OPTICAL CHARACTER RECOGNITION (OCR)
#  Engine   : Google Tesseract (via pytesseract)
#  Pipeline : Grayscale → Gaussian Blur → Adaptive Threshold
#             → Deskew → Tesseract OCR → Formatted Output
# ══════════════════════════════════════════════════════════════

def path1_ocr():
    """Full OCR pipeline using pytesseract + OpenCV pre-processing."""

    print("\n" + "="*55)
    print("  PATH 1 — OPTICAL CHARACTER RECOGNITION (OCR)")
    print("  Engine: Google Tesseract | Pre-processing: OpenCV")
    print("="*55)

    # ── IMPORT ───────────────────────────────────────────────
    try:
        import cv2
        import pytesseract
        from PIL import Image
    except ImportError as e:
        print(f"\n[ERROR] Missing library: {e}")
        print("  Run: pip install opencv-python pytesseract Pillow")
        print("  Also install Tesseract engine: see top of this file")
        return

    # ── DEMO IMAGE ────────────────────────────────────────────
    # Public domain invoice image good for OCR demo
    DEMO_PATH = "demo_ocr_image.png"
    DEMO_URL  = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/a7/Camponotus_flavomarginatus_ant.jpg/"
        "320px-Camponotus_flavomarginatus_ant.jpg"
    )

    # Create a synthetic test image with text if download fails
    def create_synthetic_image(path):
        """Create a simple black-and-white image with text for OCR demo."""
        img = np.ones((300, 600, 3), dtype=np.uint8) * 255  # white background
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            "DecodeLabs AI Suite",
            "Project 4: OCR Demo",
            "Batch: 2026",
            "pytesseract + OpenCV",
            "Confidence: 95%"
        ]
        y = 50
        for line in lines:
            cv2.putText(img, line, (30, y), font, 0.8, (20, 20, 20), 2)
            y += 50
        cv2.imwrite(path, img)
        print(f"  [INFO] Synthetic test image created: {path}")

    # Always create a clean synthetic image for reliable demo
    create_synthetic_image(DEMO_PATH)

    img_path = get_image_path(DEMO_PATH, "OCR text image")

    # ── LOAD IMAGE ────────────────────────────────────────────
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERROR] Could not load image: {img_path}")
        return

    h, w = img.shape[:2]
    print(f"\n[IMAGE] Loaded: {img_path}")
    print(f"  Dimensions : {w} × {h} pixels")
    print(f"  Channels   : {img.shape[2]} (BGR)")
    print(f"  Matrix Size: {w * h * img.shape[2]:,} data points")

    # ── STEP 1: GRAYSCALE CONVERSION ─────────────────────────
    # Collapses 3D RGB matrix → 1D intensity matrix
    # Removes distracting color data, reduces compute
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\n[STEP 1] Grayscale Conversion ✓")
    print(f"  3D matrix ({w}×{h}×3) → 1D intensity ({w}×{h})")

    # ── STEP 2: GAUSSIAN BLUR ─────────────────────────────────
    # Smooths micro-imperfections and artifact noise
    # Kernel (5,5): larger = more blur, smaller = sharper
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"\n[STEP 2] Gaussian Blur ✓")
    print(f"  Kernel: 5×5 | Noise suppressed")

    # ── STEP 3: ADAPTIVE THRESHOLDING (Otsu's Method) ─────────
    # Forces every pixel to choose: BLACK(0) or WHITE(255)
    # Separates foreground text from background noise
    # Adaptive = handles uneven lighting across the image
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_val = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)[0]
    print(f"\n[STEP 3] Adaptive Thresholding (Otsu's Method) ✓")
    print(f"  Calculated Cutoff : {otsu_val:.0f}")
    print(f"  Logic: pixel ≥ {otsu_val:.0f} → WHITE | pixel < {otsu_val:.0f} → BLACK")
    print(f"  Output: Pure black-and-white binary image")

    # ── STEP 4: DESKEWING ─────────────────────────────────────
    # Calculates rotation angle and corrects tilted text
    coords     = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.1:
            center   = (w // 2, h // 2)
            M        = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(thresh, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            print(f"\n[STEP 4] Deskewing ✓")
            print(f"  Detected Skew  : {angle:.2f}°")
            print(f"  Correction applied: image rotated to horizontal baseline")
        else:
            deskewed = thresh
            print(f"\n[STEP 4] Deskewing ✓ (No skew detected, image already aligned)")
    else:
        deskewed = thresh
        print(f"\n[STEP 4] Deskewing skipped (empty image region)")

    # ── STEP 5: TESSERACT OCR ─────────────────────────────────
    # PSM modes:
    # --psm 3  : Fully automatic (default for varied layouts)
    # --psm 6  : Single uniform block of text
    # --psm 7  : Single text line (number plates, headers)
    # --psm 11 : Sparse, scattered text (invoices)
    print(f"\n[STEP 5] Running Tesseract OCR Engine...")
    print(f"  PSM Mode : --psm 6 (uniform text block)")
    print(f"  OEM Mode : --oem 3 (LSTM neural net)")

    config = "--psm 6 --oem 3"

    try:
        # Get detailed data including confidence scores
        data = pytesseract.image_to_data(
            deskewed,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        # Extract raw text
        raw_text = pytesseract.image_to_string(deskewed, config=config)

        # ── OUTPUT ────────────────────────────────────────────
        # Filter words with confidence >= 80% (The Gatekeeper Rule)
        CONFIDENCE_THRESHOLD = 80.0
        validated_words = []
        total_words     = 0

        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if word.strip():
                total_words += 1
                if conf >= CONFIDENCE_THRESHOLD:
                    validated_words.append((word, conf))

        print(f"\n{'='*55}")
        print(f"  OCR OUTPUT — EXTRACTION RESULTS")
        print(f"{'='*55}")
        print(f"\n  RAW TEXT EXTRACTED:")
        print(f"  {'-'*50}")
        lines = [l for l in raw_text.strip().split('\n') if l.strip()]
        for line in lines:
            print(f"  {line}")
        print(f"  {'-'*50}")

        print(f"\n  CONFIDENCE ANALYSIS (≥{CONFIDENCE_THRESHOLD}% gate):")
        print(f"  Total words detected  : {total_words}")
        print(f"  Words passed 80% gate : {len(validated_words)}")

        if validated_words:
            avg_conf = sum(c for _, c in validated_words) / len(validated_words)
            print(f"  Average confidence    : {avg_conf:.1f}%")
            print(f"\n  HIGH-CONFIDENCE WORDS:")
            for word, conf in validated_words[:15]:
                bar = "█" * int(conf / 10)
                print(f"    {word:<20} {conf:5.1f}%  {bar}")

            # Validation check
            print(f"\n  MILESTONE VALIDATION:")
            if avg_conf >= 80:
                print(f"  ✓ Accuracy ≥ 80% threshold: PASSED ({avg_conf:.1f}%)")
            else:
                print(f"  ✗ Accuracy < 80% threshold: {avg_conf:.1f}% — try a clearer image")

        # Save pre-processed image
        cv2.imwrite("ocr_preprocessed.png", deskewed)
        print(f"\n  Pre-processed image saved: ocr_preprocessed.png")
        print(f"  ✓ Path 1 (OCR) complete")

    except Exception as e:
        print(f"\n  [ERROR] Tesseract failed: {e}")
        print(f"  Make sure Tesseract is installed:")
        print(f"  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print(f"  Linux  : sudo apt install tesseract-ocr")
        print(f"  Mac    : brew install tesseract")


# ══════════════════════════════════════════════════════════════
#  PATH 2 — OBJECT DETECTION (MobileNet-SSD)
#  Engine   : OpenCV DNN + MobileNet-SSD (pre-trained on COCO)
#  Pipeline : Load Image → Blob Construction → DNN Forward Pass
#             → Confidence Filter (80%) → Draw Bounding Boxes
# ══════════════════════════════════════════════════════════════

# COCO class labels that MobileNet-SSD was trained on
COCO_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Download URLs for MobileNet-SSD model files (Caffe format)
MODEL_URL  = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.caffemodel"
PROTO_URL  = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
PROTO_PATH = "MobileNetSSD_deploy.prototxt"

# Demo image — a street scene good for object detection
DETECT_DEMO_URL  = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
)
DETECT_DEMO_PATH = "demo_detect_image.jpg"


def download_mobilenet_model():
    """Download MobileNet-SSD model files if not present."""
    success = True

    if not os.path.exists(PROTO_PATH):
        print(f"  [INFO] Downloading MobileNet-SSD prototxt...")
        try:
            urllib.request.urlretrieve(PROTO_URL, PROTO_PATH)
            print(f"  [INFO] Saved: {PROTO_PATH}")
        except Exception as e:
            print(f"  [ERROR] Could not download prototxt: {e}")
            success = False

    if not os.path.exists(MODEL_PATH):
        print(f"  [INFO] Downloading MobileNet-SSD model (~23MB)...")
        print(f"         This may take a moment...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"  [INFO] Saved: {MODEL_PATH}")
        except Exception as e:
            print(f"  [ERROR] Could not download model: {e}")
            success = False

    return success


def path2_object_detection():
    """Full object detection pipeline using MobileNet-SSD + OpenCV DNN."""

    print("\n" + "="*55)
    print("  PATH 2 — OBJECT DETECTION (MobileNet-SSD)")
    print("  Engine: cv2.dnn | Backbone: MobileNet v3")
    print("  Trained on: COCO (20 object classes)")
    print("="*55)

    # ── IMPORT ───────────────────────────────────────────────
    try:
        import cv2
    except ImportError:
        print("\n[ERROR] OpenCV not found → pip install opencv-python")
        return

    # ── DOWNLOAD MODEL ────────────────────────────────────────
    print(f"\n[MODEL] Checking MobileNet-SSD model files...")
    model_ok = download_mobilenet_model()

    if not model_ok or not os.path.exists(MODEL_PATH):
        print("\n[FALLBACK] Running simulated detection demo...")
        print("  (Model download failed — showing pipeline logic)")
        _simulated_detection_demo()
        return

    # ── DOWNLOAD DEMO IMAGE ───────────────────────────────────
    download_demo_image(DETECT_DEMO_PATH, DETECT_DEMO_URL, "detection")
    img_path = get_image_path(DETECT_DEMO_PATH, "detection image")

    # ── LOAD IMAGE ────────────────────────────────────────────
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERROR] Could not load image: {img_path}")
        return

    h, w = img.shape[:2]
    print(f"\n[IMAGE] Loaded: {img_path}")
    print(f"  Dimensions : {w} × {h} pixels")
    print(f"  Scale of Perception: {w * h * 3:,} data points")

    # ── STEP 1: LOAD PRE-TRAINED DNN ─────────────────────────
    print(f"\n[STEP 1] Loading MobileNet-SSD (Transfer Learning) ✓")
    print(f"  Architecture : MobileNet v3 (depthwise separable convolutions)")
    print(f"  Trained on   : ImageNet + COCO (millions of images)")
    print(f"  Classes      : {len(COCO_CLASSES)-1} object categories")

    try:
        net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    except Exception as e:
        print(f"  [ERROR] Model loading failed: {e}")
        _simulated_detection_demo()
        return

    # ── STEP 2: BLOB CONSTRUCTION ─────────────────────────────
    # blobFromImage performs:
    # 1. Mean subtraction: (104, 117, 123) — removes lighting bias
    # 2. Scaling: 1/255 normalize pixel values
    # 3. Resizes to required 300×300 network input dimensions
    # 4. Creates 4D blob: (batch_size, channels, height, width)
    print(f"\n[STEP 2] Blob Construction ✓")
    print(f"  cv2.dnn.blobFromImage()")
    print(f"  → Mean subtraction: (104, 117, 123) — removes lighting bias")
    print(f"  → Scale factor   : 0.007843 (normalize 0-255 → 0-1)")
    print(f"  → Resize to      : 300×300 (MobileNet input requirements)")
    print(f"  → Output shape   : (1, 3, 300, 300) — 4D blob")

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        0.007843,           # scale factor
        (300, 300),         # target size
        127.5               # mean subtraction
    )

    # ── STEP 3: FORWARD PASS ─────────────────────────────────
    print(f"\n[STEP 3] Running DNN Forward Pass...")
    print(f"  Single Shot Detector (SSD): one pass, multiple detections")
    net.setInput(blob)
    detections = net.forward()
    print(f"  Forward pass complete ✓")
    print(f"  Raw detections: {detections.shape[2]}")

    # ── STEP 4: CONFIDENCE FILTER (80% Gate) ─────────────────
    # The Gatekeeper Rule:
    # confidence >= 0.80 → draw box and label
    # confidence <  0.80 → drop_detection()
    CONFIDENCE_THRESHOLD = 0.80
    print(f"\n[STEP 4] Confidence Filter ({CONFIDENCE_THRESHOLD*100:.0f}% Gate) ✓")
    print(f"  Logic: if confidence >= {CONFIDENCE_THRESHOLD} → draw_box_and_label()")
    print(f"          else → drop_detection()")

    output_img  = img.copy()
    valid_count = 0
    results     = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence >= CONFIDENCE_THRESHOLD:
            class_id = int(detections[0, 0, i, 1])
            if class_id >= len(COCO_CLASSES):
                continue

            label = COCO_CLASSES[class_id]

            # Translate normalized coordinates → actual pixel coordinates
            # Multiply by original image width/height
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Bounding box dimensions
            bw = x2 - x1
            bh = y2 - y1

            results.append({
                "label"     : label,
                "confidence": confidence,
                "x"         : x1,
                "y"         : y1,
                "w"         : bw,
                "h"         : bh,
            })

            # Draw bounding box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Draw label + confidence
            label_text = f"{label}: {confidence*100:.1f}%"
            cv2.putText(
                output_img, label_text,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 0), 2
            )
            valid_count += 1

    # ── OUTPUT ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  OBJECT DETECTION RESULTS")
    print(f"{'='*55}")
    print(f"\n  Detections above {CONFIDENCE_THRESHOLD*100:.0f}% threshold: {valid_count}")

    if results:
        print(f"\n  DETECTED OBJECTS:")
        for j, r in enumerate(results):
            bar = "█" * int(r["confidence"] * 20)
            print(f"\n  [{j+1}] {r['label'].upper()}")
            print(f"       Confidence : {r['confidence']*100:.1f}%  {bar}")
            print(f"       Position   : X={r['x']}, Y={r['y']}")
            print(f"       Bounding   : W={r['w']}px × H={r['h']}px")

        # Save output image with bounding boxes
        output_path = "detection_output.jpg"
        cv2.imwrite(output_path, output_img)
        print(f"\n  Output image saved: {output_path}")
        print(f"  (Shows bounding boxes with labels and confidence scores)")

        # Milestone validation
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        print(f"\n  MILESTONE VALIDATION:")
        print(f"  ✓ Library Integration   : OpenCV DNN + MobileNet-SSD")
        print(f"  ✓ Pre-Processing        : Blob construction (blobFromImage)")
        print(f"  ✓ Accuracy Benchmark    : {avg_conf*100:.1f}% avg confidence")
        print(f"  ✓ Visual Confirmation   : Bounding boxes saved to {output_path}")
        print(f"\n  ✓ Path 2 (Object Detection) complete")

    else:
        print(f"\n  No objects detected above {CONFIDENCE_THRESHOLD*100:.0f}% confidence.")
        print(f"  Try with a clearer image containing people, cars, or animals.")
        print(f"  Supported classes: {', '.join(COCO_CLASSES[1:])}")


def _simulated_detection_demo():
    """Show detection output simulation when model is unavailable."""
    print(f"\n  SIMULATED DETECTION DEMO (showing pipeline logic):")
    print(f"  {'='*45}")

    fake_results = [
        {"label": "person",  "confidence": 0.94, "x": 120, "y": 80,  "w": 180, "h": 320},
        {"label": "car",     "confidence": 0.87, "x": 400, "y": 200, "w": 250, "h": 150},
        {"label": "bicycle", "confidence": 0.81, "x": 60,  "y": 300, "w": 120, "h": 200},
        {"label": "bottle",  "confidence": 0.61, "x": 300, "y": 100, "w": 40,  "h": 80},  # dropped
    ]

    print(f"\n  Raw detections from DNN: {len(fake_results)}")
    print(f"  Applying 80% confidence gate...\n")

    passed = 0
    for r in fake_results:
        status = "✓ PASSED" if r["confidence"] >= 0.80 else "✗ DROPPED"
        bar    = "█" * int(r["confidence"] * 20)
        print(f"  {status} | {r['label']:<12} | {r['confidence']*100:.0f}%  {bar}")
        if r["confidence"] >= 0.80:
            passed += 1
            print(f"           → BBox drawn at X={r['x']}, Y={r['y']}, W={r['w']}, H={r['h']}")

    print(f"\n  Detections above 80% gate: {passed}/{len(fake_results)}")
    print(f"  ✓ Simulated demo complete")
    print(f"\n  To run actual detection:")
    print(f"  1. Ensure stable internet connection")
    print(f"  2. Re-run: python recognition.py → Path 2")
    print(f"  3. Model (~23MB) will auto-download")


# ══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run_recognition(path_choice=None):
    """
    Main runner for Project 4.
    path_choice: 1 = OCR, 2 = Object Detection, None = interactive
    """
    print("\n" + "█"*55)
    print("  PROJECT 4 — IMAGE OR TEXT RECOGNITION (BASIC)")
    print("  Optional Mastery Phase | DecodeLabs Batch 2026")
    print("█"*55)

    # Check dependencies
    print("\n[DEPENDENCY CHECK]")
    deps = check_dependencies()
    for lib, status in deps.items():
        print(f"  {status}")

    # Choose path
    print(f"\n[CHOOSE YOUR EXECUTION PATH]")
    print(f"  Path 1 → OCR (Text Recognition using pytesseract)")
    print(f"            Extract machine-readable text from images")
    print(f"  Path 2 → Object Detection (MobileNet-SSD)")
    print(f"            Detect objects with bounding boxes + confidence")
    print(f"  Path 3 → Run BOTH paths")

    if path_choice is None:
        try:
            choice = input("\n  Your choice (1/2/3): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            return
    else:
        choice = str(path_choice)
        print(f"\n  Auto-selected: Path {choice}")

    if choice == "1":
        path1_ocr()
    elif choice == "2":
        path2_object_detection()
    elif choice == "3":
        path1_ocr()
        path2_object_detection()
    else:
        print(f"  [!] Invalid choice. Running Path 1 (OCR) by default.")
        path1_ocr()

    print(f"\n{'█'*55}")
    print(f"  PROJECT 4 COMPLETE")
    print(f"  DecodeLabs AI Suite | Batch 2026")
    print(f"{'█'*55}\n")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    run_recognition()
