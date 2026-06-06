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


def make_metrics_table(original, noisy, img_wt, img_svd, img_wtsvd, energy_svd, energy_wtsvd):
    rmse_n, psnr_n, snr_n = calculate_all_metrics(original, noisy)
    rmse_wt, psnr_wt, snr_wt = calculate_all_metrics(original, img_wt)
    rmse_svd, psnr_svd, snr_svd = calculate_all_metrics(original, img_svd)
    rmse_wtsvd, psnr_wtsvd, snr_wtsvd = calculate_all_metrics(original, img_wtsvd)

    return [
        {
            "label": "RMSE",
            "original": f"{rmse_n:.4f}",
            "wt": f"{rmse_wt:.4f}",
            "svd": f"{rmse_svd:.4f}",
            "wtsvd": f"{rmse_wtsvd:.4f}",
        },
        {
            "label": "PSNR (dB)",
            "original": f"{psnr_n:.4f}",
            "wt": f"{psnr_wt:.4f}",
            "svd": f"{psnr_svd:.4f}",
            "wtsvd": f"{psnr_wtsvd:.4f}",
        },
        {
            "label": "SNR (dB)",
            "original": f"{snr_n:.4f}",
            "wt": f"{snr_wt:.4f}",
            "svd": f"{snr_svd:.4f}",
            "wtsvd": f"{snr_wtsvd:.4f}",
        },
        {
            "label": "Energy Retained (%)",
            "original": "-",
            "wt": "-",
            "svd": f"{energy_svd:.2f}",
            "wtsvd": f"{energy_wtsvd:.2f}",
        },
    ]


def process_image(file_bytes, noise_var, wt_multiplier, k_svd, k_wtsvd):
    if file_bytes is None:
        original = generate_synthetic_mri()
        file_size = 0
    else:
        try:
            pil = Image.open(io.BytesIO(file_bytes)).convert("L")
            original = np.array(pil)
            file_size = len(file_bytes)
        except Exception:
            original = generate_synthetic_mri()
            file_size = 0

    original = ensure_even_dimensions(original)
    h, w = original.shape
    max_k_svd = min(h, w)

    noisy_img, sigma = add_gaussian_noise(original, noise_var)
    rmse_n, psnr_n, snr_n = calculate_all_metrics(original, noisy_img)

    LL, HL, LH, HH = haar_dwt_2d(noisy_img)
    max_k_wtsvd = min(LL.shape)

    k_svd = max(1, min(k_svd, max_k_svd))
    k_wtsvd = max(1, min(k_wtsvd, max_k_wtsvd))

    threshold = sigma * math.sqrt(2 * math.log(original.size)) * wt_multiplier
    HL_t = soft_threshold(HL, threshold)
    LH_t = soft_threshold(LH, threshold)
    HH_t = soft_threshold(HH, threshold)
    img_wt = haar_idwt_2d(LL, HL_t, LH_t, HH_t).astype(np.uint8)

    img_svd_float, energy_svd = apply_svd_matrix(noisy_img, k_svd)
    img_svd = np.clip(img_svd_float, 0, 255).astype(np.uint8)

    LL_denoised, energy_wtsvd = apply_svd_matrix(LL, k_wtsvd)
    img_wtsvd = haar_idwt_2d(LL_denoised, HL_t, LH_t, HH_t).astype(np.uint8)

    rmse_wt, psnr_wt, snr_wt = calculate_all_metrics(original, img_wt)
    rmse_svd, psnr_svd, snr_svd = calculate_all_metrics(original, img_svd)
    rmse_wtsvd, psnr_wtsvd, snr_wtsvd = calculate_all_metrics(original, img_wtsvd)

    return {
        "original": original,
        "noisy": noisy_img,
        "img_wt": img_wt,
        "img_svd": img_svd,
        "img_wtsvd": img_wtsvd,
        "sigma": sigma,
        "energy_svd": energy_svd,
        "energy_wtsvd": energy_wtsvd,
        "original_props": {
            "resolution": f"{w} x {h} px",
            "aspect_ratio": f"{w/h:.2f}:1",
            "file_size": f"{file_size/1024:.2f} KB",
            "total_pixels": f"{original.size:,}",
            "min_pixel": int(np.min(original)),
            "max_pixel": int(np.max(original)),
            "mean_pixel": f"{np.mean(original):.2f}",
        },
        "noise_metrics": {
            "rmse": f"{rmse_n:.2f}",
            "psnr": f"{psnr_n:.2f}",
            "snr": f"{snr_n:.2f}",
        },
        "wt_metrics": {
            "rmse": f"{rmse_wt:.4f}",
            "psnr": f"{psnr_wt:.4f}",
            "snr": f"{snr_wt:.4f}",
        },
        "svd_metrics": {
            "rmse": f"{rmse_svd:.4f}",
            "psnr": f"{psnr_svd:.4f}",
            "snr": f"{snr_svd:.4f}",
        },
        "wtsvd_metrics": {
            "rmse": f"{rmse_wtsvd:.4f}",
            "psnr": f"{psnr_wtsvd:.4f}",
            "snr": f"{snr_wtsvd:.4f}",
        },
        "max_k_svd": max_k_svd,
        "max_k_wtsvd": max_k_wtsvd,
        "noise_var": noise_var,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {
        "noise_var": DEFAULTS["noise_var"],
        "wt_multiplier": DEFAULTS["wt_multiplier"],
        "k_svd": DEFAULTS["k_svd"],
        "k_wtsvd": DEFAULTS["k_wtsvd"],
    }

    if request.method == "POST":
        try:
            form_data["noise_var"] = float(request.form.get("noise_var", DEFAULTS["noise_var"]))
        except ValueError:
            form_data["noise_var"] = DEFAULTS["noise_var"]

        try:
            form_data["wt_multiplier"] = float(request.form.get("wt_multiplier", DEFAULTS["wt_multiplier"]))
        except ValueError:
            form_data["wt_multiplier"] = DEFAULTS["wt_multiplier"]

        try:
            form_data["k_svd"] = int(request.form.get("k_svd", DEFAULTS["k_svd"]))
        except ValueError:
            form_data["k_svd"] = DEFAULTS["k_svd"]

        try:
            form_data["k_wtsvd"] = int(request.form.get("k_wtsvd", DEFAULTS["k_wtsvd"]))
        except ValueError:
            form_data["k_wtsvd"] = DEFAULTS["k_wtsvd"]

        upload_file = request.files.get("image")
        file_bytes = upload_file.read() if upload_file and upload_file.filename else None

        result = process_image(
            file_bytes,
            form_data["noise_var"],
            form_data["wt_multiplier"],
            form_data["k_svd"],
            form_data["k_wtsvd"],
        )

        result["original_data"] = image_to_data_uri(result["original"])
        result["noisy_data"] = image_to_data_uri(result["noisy"])
        result["wt_data"] = image_to_data_uri(result["img_wt"])
        result["svd_data"] = image_to_data_uri(result["img_svd"])
        result["wtsvd_data"] = image_to_data_uri(result["img_wtsvd"])
        result["metrics_table"] = make_metrics_table(
            result["original"],
            result["noisy"],
            result["img_wt"],
            result["img_svd"],
            result["img_wtsvd"],
            result["energy_svd"],
            result["energy_wtsvd"],
        )

    return render_template(
        "index.html",
        result=result,
        form_data=form_data,
        noise_options=NOISE_OPTIONS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
