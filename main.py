import base64
import math
import io
from PIL import Image
import numpy as np
from flask import Flask, render_template, request

from utils import (
    add_gaussian_noise,
    apply_svd_matrix,
    calculate_all_metrics,
    ensure_even_dimensions,
    generate_synthetic_mri,
    haar_dwt_2d,
    haar_idwt_2d,
    soft_threshold,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

NOISE_OPTIONS = [0.02, 0.05, 0.09, 0.15, 0.30]
DEFAULTS = {
    "noise_var": 0.05,
    "wt_multiplier": 1.0,
    "k_svd": 20,
    "k_wtsvd": 40,
}


def image_to_data_uri(img: np.ndarray) -> str:
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _read_uploaded_image(file_bytes=None):
    if file_bytes is None:
        upload_file = request.files.get("image")
        if upload_file and upload_file.filename:
            file_bytes = upload_file.read()
        else:
            return generate_synthetic_mri(), None

    try:
        pil = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception:
        return None, "Format file tidak valid"

    image = ensure_even_dimensions(np.array(pil))
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return None, "Gambar tidak valid atau kosong"

    return image, None


def _serialize_image_png(img: np.ndarray) -> str:
    return image_to_data_uri(img)


def _make_image_props(original: np.ndarray, file_size: int):
    h, w = original.shape
    return {
        "resolution": f"{w} x {h} px",
        "aspect_ratio": f"{w/h:.2f}:1",
        "file_size": f"{file_size/1024:.2f} KB",
        "total_pixels": f"{original.size:,}",
        "min_pixel": int(np.min(original)),
        "max_pixel": int(np.max(original)),
        "mean_pixel": f"{np.mean(original):.2f}",
    }


def _compute_method(original: np.ndarray, method: str, noise_var: float, wt_multiplier: float, k_svd: int, k_wtsvd: int):
    noisy_img, sigma = add_gaussian_noise(original, noise_var)

    if method == "noise":
        rmse_n, psnr_n, snr_n = calculate_all_metrics(original, noisy_img)
        return {
            "result_data": _serialize_image_png(noisy_img),
            "title": "Citra Rusak",
            "metrics": {
                "rmse": f"{rmse_n:.2f}",
                "psnr": f"{psnr_n:.2f}",
                "snr": f"{snr_n:.2f}",
            },
            "params": {
                "noise_var": noise_var,
                "sigma": f"{sigma:.2f}",
            },
        }

    if method == "wt":
        LL, HL, LH, HH = haar_dwt_2d(noisy_img)
        threshold = sigma * math.sqrt(2 * math.log(original.size)) * wt_multiplier
        HL_t = soft_threshold(HL, threshold)
        LH_t = soft_threshold(LH, threshold)
        HH_t = soft_threshold(HH, threshold)
        result_img = haar_idwt_2d(LL, HL_t, LH_t, HH_t).astype(np.uint8)
        rmse, psnr, snr = calculate_all_metrics(original, result_img)
        return {
            "result_data": _serialize_image_png(result_img),
            "title": "WT Murni",
            "metrics": {
                "rmse": f"{rmse:.4f}",
                "psnr": f"{psnr:.4f}",
                "snr": f"{snr:.4f}",
            },
            "params": {
                "noise_var": noise_var,
                "wt_multiplier": wt_multiplier,
                "threshold": f"{threshold:.4f}",
            },
        }

    if method == "svd":
        img_svd_float, energy_svd = apply_svd_matrix(noisy_img, k_svd)
        result_img = np.clip(np.rint(img_svd_float), 0, 255).astype(np.uint8)
        rmse, psnr, snr = calculate_all_metrics(original, result_img)
        return {
            "result_data": _serialize_image_png(result_img),
            "title": "SVD Murni",
            "metrics": {
                "rmse": f"{rmse:.4f}",
                "psnr": f"{psnr:.4f}",
                "snr": f"{snr:.4f}",
            },
            "params": {
                "noise_var": noise_var,
                "k_svd": k_svd,
                "energy_retained": f"{energy_svd:.2f}%",
            },
        }

    if method == "wtsvd":
        LL, HL, LH, HH = haar_dwt_2d(noisy_img)
        LL_denoised, energy_wtsvd = apply_svd_matrix(LL, k_wtsvd)
        threshold = sigma * math.sqrt(2 * math.log(original.size)) * wt_multiplier
        HL_t = soft_threshold(HL, threshold)
        LH_t = soft_threshold(LH, threshold)
        HH_t = soft_threshold(HH, threshold)
        result_img = haar_idwt_2d(LL_denoised, HL_t, LH_t, HH_t).astype(np.uint8)
        rmse, psnr, snr = calculate_all_metrics(original, result_img)
        return {
            "result_data": _serialize_image_png(result_img),
            "title": "WT-SVD",
            "metrics": {
                "rmse": f"{rmse:.4f}",
                "psnr": f"{psnr:.4f}",
                "snr": f"{snr:.4f}",
            },
            "params": {
                "noise_var": noise_var,
                "wt_multiplier": wt_multiplier,
                "k_wtsvd": k_wtsvd,
                "energy_retained": f"{energy_wtsvd:.2f}%",
            },
        }

    return None


def _prepare_response(original, image_data, props=None, metrics=None, title=None, params=None):
    response = {
        "original_data": _serialize_image_png(original),
        "original_props": props or {},
    }
    if image_data:
        response.update({
            "result_data": image_data,
            "title": title,
            "metrics": metrics or {},
            "params": params or {},
        })
    return response


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    upload_file = request.files.get("image")
    if not upload_file or not upload_file.filename:
        return {"error": "Tidak ada file yang diunggah"}, 400

    file_bytes = upload_file.read()
    original, error = _read_uploaded_image(file_bytes)
    if error:
        return {"error": error}, 400

    props = _make_image_props(original, len(file_bytes))
    return {
        "original_data": _serialize_image_png(original),
        "original_props": props,
    }


@app.route("/process_noise", methods=["POST"])
def process_noise():
    upload_file = request.files.get("image")
    if not upload_file or not upload_file.filename:
        return {"error": "Tidak ada file yang diunggah"}, 400

    file_bytes = upload_file.read()
    original, error = _read_uploaded_image(file_bytes)
    if error:
        return {"error": error}, 400

    try:
        noise_var = float(request.form.get("noise_var", DEFAULTS["noise_var"]))
    except ValueError:
        noise_var = DEFAULTS["noise_var"]

    result = _compute_method(original, "noise", noise_var, 1.0, 1, 1)
    return result


@app.route("/process_method", methods=["POST"])
def process_method():
    upload_file = request.files.get("image")
    if not upload_file or not upload_file.filename:
        return {"error": "Tidak ada file yang diunggah"}, 400

    file_bytes = upload_file.read()
    original, error = _read_uploaded_image(file_bytes)
    if error:
        return {"error": error}, 400

    method = request.form.get("method", "wt")
    try:
        noise_var = float(request.form.get("noise_var", DEFAULTS["noise_var"]))
    except ValueError:
        noise_var = DEFAULTS["noise_var"]

    try:
        wt_multiplier = float(request.form.get("wt_multiplier", DEFAULTS["wt_multiplier"]))
    except ValueError:
        wt_multiplier = DEFAULTS["wt_multiplier"]

    try:
        k_svd = int(request.form.get("k_svd", DEFAULTS["k_svd"]))
    except ValueError:
        k_svd = DEFAULTS["k_svd"]

    try:
        k_wtsvd = int(request.form.get("k_wtsvd", DEFAULTS["k_wtsvd"]))
    except ValueError:
        k_wtsvd = DEFAULTS["k_wtsvd"]

    result = _compute_method(original, method, noise_var, wt_multiplier, k_svd, k_wtsvd)
    if result is None:
        return {"error": "Metode tidak dikenali"}, 400
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
