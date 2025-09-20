import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
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


def count_nodules(image_path, min_area=50, max_area=5000, block_size=11, C=2, blur_ksize=5):
    """
    Counts (and marks) nodules using simple CV steps:
    1) Gray -> Gaussian blur
    2) Adaptive threshold (binary INV)
    3) Contours -> area filter
    Returns: count, overlay_bgr
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the uploaded image. Unsupported format or corrupt file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # avoid even kernel for gaussian blur
    blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # keep block_size odd and >=3
    if block_size < 3: block_size = 3
    if block_size % 2 == 0: block_size += 1

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    # optional morphology to split/clean blobs (tweak if needed)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = img.copy()
    nodule_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            nodule_count += 1
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            cv2.circle(overlay, center, max(3, int(radius)), (0, 255, 0), 2)
            cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 1)

    return nodule_count, overlay


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

    # optional params from UI
    try:
        min_area = int(request.form.get('min_area', 50))
        max_area = int(request.form.get('max_area', 5000))
        block_size = int(request.form.get('block_size', 11))
        C = int(request.form.get('C', 2))
        blur_ksize = int(request.form.get('blur_ksize', 5))
        fov_area_mm2 = float(request.form.get('fov_area_mm2', 2.0))  # ~2 mm^2 at 100× (≈1.6 mm dia)
    except ValueError:
        return jsonify({'error': 'Invalid numeric parameter(s).'}), 400

    # save upload
    filename = secure_filename(file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(in_path)

    try:
        count, overlay = count_nodules(
            in_path,
            min_area=min_area,
            max_area=max_area,
            block_size=block_size,
            C=C,
            blur_ksize=blur_ksize
        )

        # nodules per mm^2 (matches your lab method reference)
        nodules_per_mm2 = count / fov_area_mm2 if fov_area_mm2 > 0 else None

        # save overlay image with suffix
        base, ext = os.path.splitext(filename)
        out_name = f"{base}_overlay{ext if ext.lower() in ['.png', '.jpg', '.jpeg'] else '.png'}"
        out_path = os.path.join(app.config['RESULTS_FOLDER'], out_name)
        ok = cv2.imwrite(out_path, overlay)
        if not ok:
            raise RuntimeError("Failed to save the processed overlay image.")

        return jsonify({
            'nodule_count': int(count),
            'nodules_per_mm2': round(nodules_per_mm2, 2) if nodules_per_mm2 is not None else None,
            'overlay_url': url_for('static', filename=f"results/{out_name}", _external=False)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # For production, use a real WSGI server (gunicorn/uwsgi) and set debug=False
    app.run(host='0.0.0.0', port=5000, debug=True)
