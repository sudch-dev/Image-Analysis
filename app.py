import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---- Config ----
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024   # 10 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _ensure_odd(val: int, minv=3):
    val = max(val, minv)
    return val if val % 2 == 1 else val + 1


def count_nodules_and_mask(image_path, min_area=50, max_area=5000, block_size=11, C=2, blur_ksize=5):
    """
    Returns: nodule_count, overlay_bgr, nodules_mask (uint8 0/255)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the uploaded image. Unsupported format or corrupt file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    block_size = _ensure_odd(block_size, 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = img.copy()
    nodule_count = 0
    nod_mask = np.zeros(gray.shape, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            nodule_count += 1
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            cv2.circle(overlay, center, max(3, int(radius)), (0, 255, 0), 2)
            cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 1)
            cv2.drawContours(nod_mask, [cnt], -1, 255, -1)

    return nodule_count, overlay, nod_mask


def estimate_pearlite_percent(img_bgr, nodules_mask,
                              method='otsu', block_size=31, C=5,
                              open_ksize=3):
    """
    Estimate pearlite area % (dark matrix). Excludes graphite nodules area.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast normalize (CLAHE) to stabilize thresholding across etch/lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    if method == 'adaptive':
        block_size = _ensure_odd(block_size, 3)
        pearlite_bin = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
    else:
        # Otsu: darker region -> pearlite
        _, pearlite_bin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove nodule pixels from pearlite mask
    pearlite_bin = cv2.bitwise_and(pearlite_bin, cv2.bitwise_not(nodules_mask))

    # Clean small specks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    pearlite_bin = cv2.morphologyEx(pearlite_bin, cv2.MORPH_OPEN, k, iterations=1)

    # Matrix area excludes nodules
    matrix_mask = cv2.bitwise_not(nodules_mask)

    matrix_area = int(np.count_nonzero(matrix_mask))
    pearlite_area = int(np.count_nonzero(cv2.bitwise_and(pearlite_bin, matrix_mask)))
    if matrix_area == 0:
        return 0.0, pearlite_bin, matrix_mask

    return 100.0 * pearlite_area / matrix_area, pearlite_bin, matrix_mask


def estimate_carbide_percent(img_bgr, matrix_mask,
                             bright_percentile=99.3,
                             min_area=12, max_area=2000,
                             min_aspect_ratio=2.0, max_solidity=0.95):
    """
    Detect bright/white carbide features (needles/angular).
    Heuristics:
      - Take top intensity percentile as 'bright'
      - Keep connected components in [min_area, max_area]
      - Prefer thin/elongated (aspect ratio) or angular (lower solidity) shapes
    Returns: carbide_percent, carbide_mask
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast normalize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # Threshold by percentile on the matrix-only pixels
    matrix_pixels = g[matrix_mask > 0]
    if matrix_pixels.size == 0:
        return 0.0, np.zeros_like(g)

    thr = np.percentile(matrix_pixels, bright_percentile)
    bright = (g >= thr).astype(np.uint8) * 255

    # Restrict to matrix only
    bright = cv2.bitwise_and(bright, matrix_mask)

    # Clean noise, thin lines retained
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                              iterations=1)

    # Connected components / contours
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    carb_mask = np.zeros_like(bright)
    kept_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = (max(w, h) / max(1, min(w, h)))  # >=1
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else 1.0

        # Keep thin needles or low-solidity angular bits
        if aspect >= min_aspect_ratio or solidity <= max_solidity:
            cv2.drawContours(carb_mask, [cnt], -1, 255, -1)
            kept_area += int(area)

    matrix_area = int(np.count_nonzero(matrix_mask))
    if matrix_area == 0:
        return 0.0, carb_mask

    carbide_percent = 100.0 * kept_area / matrix_area
    return carbide_percent, carb_mask


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400

    # Tunables
    try:
        # Nodule params
        min_area = int(request.form.get('min_area', 50))
        max_area = int(request.form.get('max_area', 5000))
        block_size = int(request.form.get('block_size', 11))
        C = int(request.form.get('C', 2))
        blur_ksize = int(request.form.get('blur_ksize', 5))
        fov_area_mm2 = float(request.form.get('fov_area_mm2', 2.0))

        # Pearlite params
        pearl_method = request.form.get('pearl_method', 'otsu').lower()  # 'otsu' or 'adaptive'
        pearl_block = int(request.form.get('pearl_block', 31))
        pearl_C = int(request.form.get('pearl_C', 5))
        pearl_open = int(request.form.get('pearl_open', 3))

        # Carbide params
        carb_percentile = float(request.form.get('carb_percentile', 99.3))
        carb_min_area = int(request.form.get('carb_min_area', 12))
        carb_max_area = int(request.form.get('carb_max_area', 2000))
        carb_min_ar = float(request.form.get('carb_min_ar', 2.0))
        carb_max_sol = float(request.form.get('carb_max_sol', 0.95))
    except ValueError:
        return jsonify({'error': 'Invalid numeric parameter(s).'}), 400

    # Save file
    filename = secure_filename(file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(in_path)

    try:
        # Nodules + mask
        count, overlay, nod_mask = count_nodules_and_mask(
            in_path,
            min_area=min_area,
            max_area=max_area,
            block_size=block_size,
            C=C,
            blur_ksize=blur_ksize
        )

        # Pearlite %
        img = cv2.imread(in_path)
        pearl_pct, pearl_mask, matrix_mask = estimate_pearlite_percent(
            img, nod_mask,
            method=pearl_method, block_size=pearl_block, C=pearl_C, open_ksize=pearl_open
        )

        # Carbide %
        carb_pct, carb_mask = estimate_carbide_percent(
            img, matrix_mask,
            bright_percentile=carb_percentile,
            min_area=carb_min_area, max_area=carb_max_area,
            min_aspect_ratio=carb_min_ar, max_solidity=carb_max_sol
        )

        # Compose diagnostic overlay: green nodules, red pearlite, white carbides (alpha blend)
        diag = img.copy()
        # Pearlite mask in red overlay
        red = np.zeros_like(img)
        red[:, :, 2] = pearl_mask
        diag = cv2.addWeighted(diag, 1.0, red, 0.25, 0)

        # Carbide mask in white overlay
        white = cv2.merge([carb_mask, carb_mask, carb_mask])
        diag = cv2.addWeighted(diag, 1.0, white, 0.35, 0)

        # Nodule outlines already in 'overlay'; also draw onto diag for single preview
        contours, _ = cv2.findContours(nod_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(diag, contours, -1, (0, 255, 0), 1)

        # Save outputs
        base, ext = os.path.splitext(filename)
        out_overlay = f"{base}_nodules{ext if ext.lower() in ['.png', '.jpg', '.jpeg'] else '.png'}"
        out_diag = f"{base}_diag{ext if ext.lower() in ['.png', '.jpg', '.jpeg'] else '.png'}"

        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], out_overlay), overlay)
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], out_diag), diag)

        nod_per_mm2 = count / fov_area_mm2 if fov_area_mm2 > 0 else None

        return jsonify({
            'nodule_count': int(count),
            'nodules_per_mm2': round(nod_per_mm2, 2) if nod_per_mm2 is not None else None,
            'pearlite_percent': round(float(pearl_pct), 1),
            'carbide_percent': round(float(carb_pct), 2),
            'nodules_overlay_url': url_for('static', filename=f"results/{out_overlay}", _external=False),
            'diagnostic_overlay_url': url_for('static', filename=f"results/{out_diag}", _external=False)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
