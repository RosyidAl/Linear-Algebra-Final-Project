import streamlit as st
import numpy as np
import cv2
from scipy.linalg import svd

# ==========================================
# 1. FUNGSI TRANSFORMASI WAVELET (HAAR DWT)
# ==========================================

def haar_dwt_2d(img):
    img = img.astype(np.float64)
    h, w = img.shape
    if h % 2 != 0: img = img[:-1, :]
    if w % 2 != 0: img = img[:, :-1]
    h, w = img.shape

    W_row = np.zeros_like(img, dtype=np.float64)
    for i in range(h):
        a = img[i, 0::2]
        b = img[i, 1::2]
        W_row[i, :w//2] = (a + b) / np.sqrt(2.0)
        W_row[i, w//2:] = (a - b) / np.sqrt(2.0)

    W_total = np.zeros_like(W_row, dtype=np.float64)
    for j in range(w):
        a = W_row[0::2, j]
        b = W_row[1::2, j]
        W_total[:h//2, j] = (a + b) / np.sqrt(2.0)
        W_total[h//2:, j] = (a - b) / np.sqrt(2.0)

    LL = W_total[:h//2, :w//2]
    HL = W_total[:h//2, w//2:]
    LH = W_total[h//2:, :w//2]
    HH = W_total[h//2:, w//2:]
    return LL, HL, LH, HH


def haar_idwt_2d(LL, HL, LH, HH):
    h_sub, w_sub = LL.shape
    h, w = h_sub * 2, w_sub * 2

    W_total = np.zeros((h, w), dtype=np.float64)
    W_total[:h_sub, :w_sub] = LL
    W_total[:h_sub, w_sub:] = HL
    W_total[h_sub:, :w_sub] = LH
    W_total[h_sub:, w_sub:] = HH

    W_row = np.zeros_like(W_total, dtype=np.float64)
    for j in range(w):
        L = W_total[:h_sub, j]
        H = W_total[h_sub:, j]
        W_row[0::2, j] = (L + H) / np.sqrt(2.0)
        W_row[1::2, j] = (L - H) / np.sqrt(2.0)

    img_rec = np.zeros_like(W_row, dtype=np.float64)
    for i in range(h):
        L = W_row[i, :w_sub]
        H = W_row[i, w_sub:]
        img_rec[i, 0::2] = (L + H) / np.sqrt(2.0)
        img_rec[i, 1::2] = (L - H) / np.sqrt(2.0)

    return np.clip(img_rec, 0, 255)

# ==========================================
# 2. SVD PADA SUB-BAND LL
# ==========================================

def apply_svd_ll(ll_matrix, k):
    img_data = ll_matrix.astype(np.float64)
    
    # I_mn = U_mm * S_mn * V^T_nn
    U, S, Vt = svd(img_data, full_matrices=False)
    
    k = min(k, len(S))
    S_filtered = S.copy()
    S_filtered[k:] = 0.0
    
    denoised_matrix = np.dot(U, np.dot(np.diag(S_filtered), Vt))
    return denoised_matrix

# ==========================================
# 3. METRIK EVALUASI (PSNR)
# ==========================================

def calculate_psnr_only(original, processed):
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    mse = np.mean((orig - proc) ** 2)
    if mse == 0:
        return float('inf')
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255.0 / rmse)
    return psnr

# ==========================================
# 4. ANTARMUKA STREAMLIT UI
# ==========================================

st.set_page_config(page_title="WT-SVD Denoising", layout="wide")
st.title("🩺 Pemrosesan Denoising Citra Medis (Kombinasi WT-SVD)")

uploaded_file = st.file_uploader(
    "Unggah Citra Medis (PNG / JPG / BMP)",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    h, w = original_img.shape
    original_img = original_img[:h - (h % 2), :w - (w % 2)]
    h, w = original_img.shape

    st.header("📊 Properti Citra Asli")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("Resolusi", f"{w} x {h} px")
    with col_p2:
        st.metric("Aspek Rasio", f"{w/h:.2f}:1")
    with col_p3:
        st.metric("Ukuran File", f"{len(file_bytes)/1024:.2f} KB")
    with col_p4:
        st.metric("Total Piksel", f"{original_img.size:,}")

    col_p5, col_p6, col_p7, col_p8 = st.columns(4)
    with col_p5:
        st.metric("Tipe Data", str(original_img.dtype))
    with col_p6:
        st.metric("Piksel Min", f"{np.min(original_img)}")
    with col_p7:
        st.metric("Piksel Max", f"{np.max(original_img)}")
    with col_p8:
        st.metric("Intensitas Rata-rata", f"{np.mean(original_img):.2f}")

    st.divider()
    st.image(original_img, caption="Citra Asli", width=320)

    st.header("1. Pemodelan Noise Gaussian")
    noise_var = st.select_slider(
        "Pilih Varians Noise (σ²):",
        options=[0.02, 0.05, 0.09, 0.15, 0.30],
        value=0.05
    )
    
    sigma = np.sqrt(noise_var) * 255
    np.random.seed(42)
    noisy_img = np.clip(
        original_img.astype(np.float64) + np.random.normal(0, sigma, original_img.shape),
        0, 255
    ).astype(np.uint8)

    psnr_n = calculate_psnr_only(original_img, noisy_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(noisy_img, caption=f"Citra Rusak Noise (var={noise_var})", use_container_width=True)
    with col2:
        st.subheader("Metrik Kerusakan")
        st.metric("PSNR Noisy", f"{psnr_n:.4f} dB")

    st.header("2. Pipeline WT-SVD")

    # Dekomposisi WT
    LL, HL, LH, HH = haar_dwt_2d(noisy_img)

    max_k = LL.shape[0]
    default_k = max(1, int(max_k * 0.40))   

    k_keep = st.slider(
        "Jumlah Singular Values LL Dipertahankan (k):",
        min_value=1,
        max_value=max_k,
        value=default_k
    )

    # Reduksi SVD pada LL
    LL_denoised = apply_svd_ll(LL, k_keep)

    # Rekonstruksi IDWT
    denoised_img = haar_idwt_2d(LL_denoised, HL, LH, HH).astype(np.uint8)

    st.header("3. Evaluasi Kinerja")
    psnr_d = calculate_psnr_only(original_img, denoised_img)

    col3, col4 = st.columns(2)
    with col3:
        st.image(denoised_img, caption="Hasil Kombinasi WT-SVD", use_container_width=True)
    with col4:
        st.subheader("Metrik Hasil Pembersihan")
        st.metric("PSNR", f"{psnr_d:.4f} dB", delta=f"{psnr_d - psnr_n:+.4f} dB")

        st.divider()
        if psnr_d > psnr_n:
            st.success("Terdapat peningkatan kualitas PSNR.")
        else:
            st.warning("Penyesuaian nilai k diperlukan karena PSNR menurun atau stagnan.")

else:
    st.info("Unggah citra medis terlebih dahulu.")