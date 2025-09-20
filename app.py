import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -----------------------------
# ENV CONFIG (with safe defaults)
# -----------------------------
def env_bool(name, default=False):
    v = os.getenv(name, str(default))
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def env_int(name, default):
    try: return int(os.getenv(name, str(default)))
    except: return default

def env_float(name, default):
    try: return float(os.getenv(name, str(default)))
    except: return default

# Folders / uploads
UPLOAD_FOLDER       = os.getenv("UPLOAD_FOLDER", "static/uploads")
RESULTS_FOLDER      = os.getenv("RESULTS_FOLDER", "static/results")
ALLOWED_EXTENSIONS  = set(os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,tif,tiff,bmp").split(","))
MAX_CONTENT_MB      = env_int("MAX_CONTENT_MB", 10)

# Autotune master switch
ENABLE_AUTOTUNE     = env_bool("ENABLE_AUTOTUNE", True)

# Illumination decision
ILLUM_STD_THRESHOLD = env_float("ILLUM_STD_THRESHOLD", 6.0)  # if blurred std > this -> adaptive

# Adaptive block sizing (block = clamp(odd(min_dim / denom), min,max))
ADAPTIVE_BLOCK_DENOM= env_int("ADAPTIVE_BLOCK_DENOM", 30)
ADAPTIVE_BLOCK_MIN  = env_int("ADAPTIVE_BLOCK_MIN", 11)
ADAPTIVE_BLOCK_MAX  = env_int("ADAPTIVE_BLOCK_MAX", 51)
ADAPTIVE_C_DEFAULT  = env_int("ADAPTIVE_C_DEFAULT", 2)

# Nodule area (px^2) guard rails + percentile trimming for re-pass
NOD_AREA_MIN_PX     = env_int("NOD_AREA_MIN_PX", 30)
NOD_AREA_MAX_PX     = env_int("NOD_AREA_MAX_PX", 12000)
NOD_AREA_PCL_LOW    = env_float("NOD_AREA_PCL_LOW", 10.0)    # percentile low after first pass
NOD_AREA_PCL_HIGH   = env_float("NOD_AREA_PCL_HIGH", 90.0)   # percentile high after first pass
NOD_REFINE_FACTOR   = env_float("NOD_REFINE_FACTOR", 1.3)    # loosen upper bound a bit

# Carbide threshold tail
CARB_BASE_PCTL      = env_float("CARB_BASE_PCTL", 99.0)
CARB_PCTL_RANGE     = env_float("CARB_PCTL_RANGE", 0.6)      # +/- range based on highlights fraction
CARB_MIN_AREA       = env_int("CARB_MIN_AREA", 12)
CARB_MAX_AREA       = env_int("CARB_MAX_AREA", 2000)
CARB_MIN_AR         = env_float("CARB_MIN_AR", 2.0)
CARB_MAX_SOLIDITY   = env_float("CARB_MAX_SOLIDITY", 0.95)

# Matrix/FOV
DEFAULT_FOV_MM2     = env_float("DEFAULT_FOV_MM2", 2.0)      # used if form value missing and no px/um
PX_PER_UM           = env_float("PX_PER_UM", 0.0)            # optional: if provided, area in µm² → mm²

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _ensure_odd(x: int, minv=3, maxv=999):
    x = max(minv, min(x, maxv))
    return x if x % 2 == 1 else x + 1

def _clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _pick_threshold_mode(gray):
    """
    Decide Otsu vs Adaptive by measuring low-frequency illumination variation.
    """
    # heavy blur to approximate background
    k = max(31, _ensure_odd(min(gray.shape) // 15))
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    illum_std = float(np.std(bg))
    use_adaptive = illum_std > ILLUM_STD_THRESHOLD
    return ("adaptive" if use_adaptive else "otsu", illum_std)

def _auto_block_size(h, w):
    b = max(ADAPTIVE_BLOCK_MIN, min(ADAPTIVE_BLOCK_MAX, (min(h, w) // max(3, ADAPTIVE_BLOCK_DENOM))))
    return _ensure_odd(b, ADAPTIVE_BLOCK_MIN, ADAPTIVE_BLOCK_MAX)

def _threshold(gray, mode, block_size=None, C=None):
    if mode == "adaptive":
        block_size = _ensure_odd(block_size or ADAPTIVE_BLOCK_MIN, ADAPTIVE_BLOCK_MIN, ADAPTIVE_BLOCK_MAX)
        C = ADAPTIVE_C_DEFAULT if C is None else int(C)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, C)
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return th

def _mask_cleanup(bin_img, k=3, iterations=1):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, ker, iterations=iterations)

def _area_percentiles(contours):
    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    if areas.size == 0:
        return None
    return np.percentile(areas, [NOD_AREA_PCL_LOW, NOD_AREA_PCL_HIGH])

def _calc_fov_area_mm2(h, w, fov_from_form: float):
    # Prefer pixel scale if provided
    if PX_PER_UM > 0:
        # field area in µm² → mm²
        # NOTE: we don't know microns across the field unless you input real scale;
        # PX_PER_UM allows later conversion of *region* areas; for FOV we keep fallback.
        pass
    return fov_from_form if fov_from_form and fov_from_form > 0 else DEFAULT_FOV_MM2

# -----------------------------
# Core analysis (AUTO-TUNE)
# -----------------------------
def detect_nodules_autotune(img_bgr, params_used):
    gray0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe(gray0)

    mode, illum_std = _pick_threshold_mode(gray)
    params_used["illum_std"] = round(illum_std, 2)
    params_used["threshold_mode"] = mode

    h, w = gray.shape
    block = _auto_block_size(h, w) if mode == "adaptive" else None
    params_used["adaptive_block_size"] = int(block) if block else None
    params_used["adaptive_C"] = ADAPTIVE_C_DEFAULT

    # First pass threshold, wide area guard
    th1 = _threshold(gray, mode, block, ADAPTIVE_C_DEFAULT)
    th1 = _mask_cleanup(th1, k=3, iterations=1)
    cnts1, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area trimming to estimate nodule size band
    pcts = _area_percentiles(cnts1)
    if pcts is not None:
        low_p, high_p = pcts
        min_area = max(NOD_AREA_MIN_PX, int(low_p))
        max_area = min(NOD_AREA_MAX_PX, int(high_p * NOD_REFINE_FACTOR))
    else:
        min_area, max_area = NOD_AREA_MIN_PX, NOD_AREA_MAX_PX

    params_used["nodule_min_area_px2"] = int(min_area)
    params_used["nodule_max_area_px2"] = int(max_area)

    # Second pass: keep only within area band
    nod_mask = np.zeros_like(gray)
    nodule_count = 0
    overlay = img_bgr.copy()
    for c in cnts1:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        nodule_count += 1
        cv2.drawContours(nod_mask, [c], -1, 255, -1)
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(overlay, (int(x), int(y)), max(3, int(r)), (0, 255, 0), 2)
        cv2.drawContours(overlay, [c], -1, (255, 0, 0), 1)

    return nodule_count, overlay, nod_mask

def estimate_pearlite(img_bgr, nod_mask, params_used):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = _clahe(gray)
    # Choose method again for matrix area (may differ slightly)
    mode, _ = _pick_threshold_mode(g)
    block = _auto_block_size(*g.shape) if mode == "adaptive" else None
    params_used["pearl_mode"] = mode
    params_used["pearl_block_size"] = int(block) if block else None
    params_used["pearl_C"] = ADAPTIVE_C_DEFAULT

    pearl_bin = _threshold(g, mode, block, ADAPTIVE_C_DEFAULT)
    pearl_bin = _mask_cleanup(pearl_bin, k=3, iterations=1)

    # Exclude nodules
    pearl_bin = cv2.bitwise_and(pearl_bin, cv2.bitwise_not(nod_mask))
    matrix_mask = cv2.bitwise_not(nod_mask)

    matrix_area = int(np.count_nonzero(matrix_mask))
    pearl_area  = int(np.count_nonzero(cv2.bitwise_and(pearl_bin, matrix_mask)))
    pearl_pct   = (100.0 * pearl_area / matrix_area) if matrix_area > 0 else 0.0

    return pearl_pct, pearl_bin, matrix_mask

def _auto_carbide_percentile(matrix_pixels):
    """
    Pick a carbide bright percentile based on highlight tail.
    We start from CARB_BASE_PCTL and nudge by CARB_PCTL_RANGE depending on how "hot" the tail is.
    """
    if matrix_pixels.size == 0:
        return CARB_BASE_PCTL
    p98 = np.percentile(matrix_pixels, 98.0)
    p999 = np.percentile(matrix_pixels, 99.9)
    tail_contrast = float(p999 - p98)  # how strong is the extreme tail
    # Normalize by typical 8-bit range ~50
    score = np.clip(tail_contrast / 50.0, 0.0, 1.0)
    # More tail -> higher percentile
    return np.clip(CARB_BASE_PCTL + score * CARB_PCTL_RANGE, 98.5, 99.9)

def estimate_carbides(img_bgr, matrix_mask, params_used):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = _clahe(gray)
    mpx = g[matrix_mask > 0]
    bright_pct = _auto_carbide_percentile(mpx) if ENABLE_AUTOTUNE else CARB_BASE_PCTL
    params_used["carb_bright_percentile"] = round(float(bright_pct), 2)

    thr = np.percentile(mpx, bright_pct) if mpx.size > 0 else 255
    bright = (g >= thr).astype(np.uint8) * 255
    bright = cv2.bitwise_and(bright, matrix_mask)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    carb_mask = np.zeros_like(bright)
    kept_area = 0

    min_area = CARB_MIN_AREA
    max_area = CARB_MAX_AREA
    min_ar   = CARB_MIN_AR
    max_sol  = CARB_MAX_SOLIDITY

    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = (max(w, h) / max(1, min(w, h)))
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        solidity = (a / ha) if ha > 0 else 1.0
        if aspect >= min_ar or solidity <= max_sol:
            cv2.drawContours(carb_mask, [c], -1, 255, -1)
            kept_area += int(a)

    matrix_area = int(np.count_nonzero(matrix_mask))
    carb_pct = (100.0 * kept_area / matrix_area) if matrix_area > 0 else 0.0
    return carb_pct, carb_mask

# -----------------------------
# Flask routes
# -----------------------------
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

    # Optional overrides from form; with autotune, they’re rarely needed
    fov_area_mm2 = request.form.get('fov_area_mm2', '')
    try:
        fov_area_mm2 = float(fov_area_mm2) if fov_area_mm2 else None
    except:
        fov_area_mm2 = None

    # Save file
    filename = secure_filename(file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(in_path)

    try:
        img = cv2.imread(in_path)
        if img is None:
            return jsonify({'error': 'Failed to read image (corrupt or unsupported).'}), 400

        params_used = {
            "autotune": ENABLE_AUTOTUNE,
            "env": {
                "ILLUM_STD_THRESHOLD": ILLUM_STD_THRESHOLD,
                "ADAPTIVE_BLOCK_DENOM": ADAPTIVE_BLOCK_DENOM,
                "NOD_AREA_MIN_PX": NOD_AREA_MIN_PX,
                "NOD_AREA_MAX_PX": NOD_AREA_MAX_PX,
                "NOD_AREA_PCL_LOW": NOD_AREA_PCL_LOW,
                "NOD_AREA_PCL_HIGH": NOD_AREA_PCL_HIGH,
                "NOD_REFINE_FACTOR": NOD_REFINE_FACTOR,
                "CARB_BASE_PCTL": CARB_BASE_PCTL,
                "CARB_PCTL_RANGE": CARB_PCTL_RANGE,
                "CARB_MIN_AREA": CARB_MIN_AREA,
                "CARB_MAX_AREA": CARB_MAX_AREA,
                "CARB_MIN_AR": CARB_MIN_AR,
                "CARB_MAX_SOLIDITY": CARB_MAX_SOLIDITY,
                "DEFAULT_FOV_MM2": DEFAULT_FOV_MM2,
                "PX_PER_UM": PX_PER_UM
            }
        }

        # --- Nodules (auto) ---
        nodule_count, nod_overlay, nod_mask = detect_nodules_autotune(img, params_used)

        # --- Pearlite % (auto) ---
        pearl_pct, pearl_mask, matrix_mask = estimate_pearlite(img, nod_mask, params_used)

        # --- Carbide % (auto) ---
        carb_pct, carb_mask = estimate_carbides(img, matrix_mask, params_used)

        # --- Diagnostic overlay ---
        diag = img.copy()
        red = np.zeros_like(img); red[:, :, 2] = pearl_mask
        diag = cv2.addWeighted(diag, 1.0, red, 0.25, 0)
        white = cv2.merge([carb_mask, carb_mask, carb_mask])
        diag = cv2.addWeighted(diag, 1.0, white, 0.35, 0)
        cnts, _ = cv2.findContours(nod_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(diag, cnts, -1, (0, 255, 0), 1)

        # Save visuals
        base, ext = os.path.splitext(filename)
        out_overlay = f"{base}_nodules{ext if ext.lower() in ['.png', '.jpg', '.jpeg'] else '.png'}"
        out_diag    = f"{base}_diag{ext if ext.lower() in ['.png', '.jpg', '.jpeg'] else '.png'}"
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], out_overlay), nod_overlay)
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], out_diag),    diag)

        # Nodule density
        fov = _calc_fov_area_mm2(img.shape[0], img.shape[1], fov_area_mm2)
        nod_per_mm2 = (nodule_count / fov) if fov and fov > 0 else None

        return jsonify({
            'nodule_count': int(nodule_count),
            'nodules_per_mm2': round(nod_per_mm2, 2) if nod_per_mm2 is not None else None,
            'pearlite_percent': round(float(pearl_pct), 1),
            'carbide_percent': round(float(carb_pct), 2),
            'nodules_overlay_url': url_for('static', filename=f"results/{out_overlay}", _external=False),
            'diagnostic_overlay_url': url_for('static', filename=f"results/{out_diag}", _external=False),
            'params_used': params_used
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
