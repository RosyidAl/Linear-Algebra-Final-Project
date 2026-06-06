import numpy as np
from PIL import Image
import base64
import io


def ensure_even_dimensions(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    return img[: h - (h % 2), : w - (w % 2)] if (h % 2 or w % 2) else img


def calculate_noise_sigma(noise_var: float, max_pixel: float = 255.0) -> float:
    # Interpret noise_var as normalized variance over [0,1], then scale to 8-bit pixel range.
    return np.sqrt(noise_var) * max_pixel


def add_gaussian_noise(img: np.ndarray, noise_var: float, seed: int = 42, max_pixel: float = 255.0):
    sigma = calculate_noise_sigma(noise_var, max_pixel)
    rng = np.random.default_rng(seed)
    noisy_img = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy_img, 0, max_pixel).astype(np.uint8), sigma


def haar_dwt_2d(img: np.ndarray):
    img = img.astype(np.float64)
    h, w = img.shape
    if h % 2 != 0:
        img = img[:-1, :]
        h -= 1
    if w % 2 != 0:
        img = img[:, :-1]
        w -= 1

    s2 = np.sqrt(2.0)

    low_rows = (img[:, 0::2] + img[:, 1::2]) / s2
    high_rows = (img[:, 0::2] - img[:, 1::2]) / s2

    LL = (low_rows[0::2, :] + low_rows[1::2, :]) / s2
    LH = (low_rows[0::2, :] - low_rows[1::2, :]) / s2
    HL = (high_rows[0::2, :] + high_rows[1::2, :]) / s2
    HH = (high_rows[0::2, :] - high_rows[1::2, :]) / s2

    return LL, HL, LH, HH


def haar_idwt_2d(LL: np.ndarray, HL: np.ndarray, LH: np.ndarray, HH: np.ndarray):
    s2 = np.sqrt(2.0)
    h2, w2 = LL.shape
    h = h2 * 2
    w = w2 * 2

    low_rows = np.empty((h, w2), dtype=np.float64)
    high_rows = np.empty((h, w2), dtype=np.float64)
    low_rows[0::2, :] = (LL + LH) / s2
    low_rows[1::2, :] = (LL - LH) / s2
    high_rows[0::2, :] = (HL + HH) / s2
    high_rows[1::2, :] = (HL - HH) / s2

    recon = np.empty((h, w), dtype=np.float64)
    recon[:, 0::2] = (low_rows + high_rows) / s2
    recon[:, 1::2] = (low_rows - high_rows) / s2

    return np.clip(np.rint(recon), 0, 255)


def soft_threshold(data: np.ndarray, threshold: float):
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)


def _randomized_svd(matrix: np.ndarray, k: int, n_oversamples: int = 10, n_iter: int = 2, random_state: int = 42):
    m, n = matrix.shape
    rng = np.random.default_rng(random_state)
    l = min(k + n_oversamples, min(m, n))

    Omega = rng.standard_normal(size=(n, l))
    Y = matrix @ Omega
    for _ in range(n_iter):
        Y = matrix @ (matrix.T @ Y)

    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ matrix
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :k], S[:k], Vt[:k, :]


def apply_svd_matrix(matrix: np.ndarray, k: int):
    img_data = matrix.astype(np.float64)
    k = min(k, min(img_data.shape))

    if k < min(img_data.shape) // 4:
        U_k, S_k, Vt_k = _randomized_svd(img_data, k)
    else:
        U, S, Vt = np.linalg.svd(img_data, full_matrices=False)
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

    denoised_matrix = (U_k * S_k) @ Vt_k
    total_energy = np.linalg.norm(img_data, 'fro') ** 2
    energy_retained = (np.sum(S_k ** 2) / total_energy) * 100 if total_energy > 0 else 0.0

    return denoised_matrix, energy_retained


def calculate_all_metrics(original: np.ndarray, processed: np.ndarray):
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    mse = np.mean((orig - proc) ** 2)

    if mse == 0:
        return 0.0, float("inf"), float("inf")

    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255.0 / rmse)
    signal_power = np.mean(orig**2)
    snr = 10 * np.log10(signal_power / mse)

    return rmse, psnr, snr


def image_to_base64(img_pil: Image.Image) -> str:
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_synthetic_mri():
    x, y = np.ogrid[-128:128, -128:128]
    skull_mask = (x**2 / 100**2) + (y**2 / 120**2) <= 1
    brain_mask = (x**2 / 90**2) + (y**2 / 110**2) <= 1
    v1 = ((x-30)**2 / 15**2) + ((y-20)**2 / 30**2) <= 1
    v2 = ((x+30)**2 / 15**2) + ((y-20)**2 / 30**2) <= 1
    v3 = (x**2 / 10**2) + ((y+40)**2 / 15**2) <= 1

    img = np.zeros((256, 256))
    img[skull_mask] = 80
    img[brain_mask] = 160
    img[v1] = 40
    img[v2] = 40
    img[v3] = 40

    for i in range(6):
        angle = i * np.pi / 6
        fold = np.sin((x * np.cos(angle) + y * np.sin(angle)) / 6) * 18
        img[brain_mask] += fold[brain_mask]

    noise = np.random.normal(0, 32, (256, 256))
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def calculate_metrics(img_true, img_pred):
    mse = np.mean((img_true - img_pred) ** 2)
    if mse == 0:
        return 0, float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr


def hybrid_wavelet_svd_reconstruct(A, wavelet_name, level, k):
    try:
        import pywt
    except Exception:
        raise ImportError("hybrid_wavelet_svd_reconstruct requires PyWavelets; install PyWavelets to use this function.")

    coeffs = pywt.wavedec2(A, wavelet=wavelet_name, level=level)
    coeffs_list = list(coeffs)
    LL = coeffs_list[0].copy()

    U_ll, s_ll, Vt_ll = np.linalg.svd(LL, full_matrices=False)
    rank_ll = len(s_ll)
    actual_k = min(k, rank_ll)

    U_k = U_ll[:, :actual_k]
    s_k = s_ll[:actual_k]
    Vt_k = Vt_ll[:actual_k, :]

    LL_denoised = U_k @ np.diag(s_k) @ Vt_k
    coeffs_list[0] = LL_denoised

    sigma = np.median(np.abs(coeffs_list[-1][0])) / 0.6745 if len(coeffs_list) > 1 else 0
    threshold = sigma * np.sqrt(2 * np.log(A.size))

    for i in range(1, len(coeffs_list)):
        coeffs_list[i] = tuple(pywt.threshold(detail, threshold, mode='soft') for detail in coeffs_list[i])

    A_mod = pywt.waverec2(coeffs_list, wavelet=wavelet_name)
    return np.clip(A_mod, 0, 255).astype(np.uint8), U_k, s_k, Vt_k, actual_k
