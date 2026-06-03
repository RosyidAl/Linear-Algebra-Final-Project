import streamlit as st
import numpy as np
import cv2
import pandas as pd
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

def soft_threshold(data, threshold):
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

# ==========================================
# 2. FUNGSI SVD
# ==========================================

def apply_svd_matrix(matrix, k):
    img_data = matrix.astype(np.float64)
    U, S, Vt = svd(img_data, full_matrices=False)
    k = min(k, len(S))
    S_filtered = S.copy()
    S_filtered[k:] = 0.0
    denoised_matrix = np.dot(U, np.dot(np.diag(S_filtered), Vt))
    
    # Kalkulasi Energy Retained
    energy_retained = (np.sum(S[:k]**2) / np.sum(S**2)) * 100 if np.sum(S**2) > 0 else 0.0
    return denoised_matrix, energy_retained

# ==========================================
# 3. METRIK EVALUASI
# ==========================================

def calculate_all_metrics(original, processed):
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    mse = np.mean((orig - proc) ** 2)
    
    if mse == 0:
        return 0.0, float('inf'), float('inf')
        
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255.0 / rmse)
    
    signal_power = np.mean(orig ** 2)
    snr = 10 * np.log10(signal_power / mse)
    
    return rmse, psnr, snr

# ==========================================
# 4. ANTARMUKA STREAMLIT UI
# ==========================================

st.set_page_config(page_title="WT-SVD Denoising Evaluator", layout="wide")
st.title("🩺 Evaluasi Metode Denoising Citra Medis")

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

    # --- TAHAP 1: PEMUATAN CITRA DAN PROPERTI ---
    with st.container(border=True):
        st.header("1. Citra Asli & Properti")
        col_img, col_prop = st.columns([1.2, 1])
        
        with col_img:
            st.image(original_img, caption="Citra Asli", use_container_width=True)
            
        with col_prop:
            st.subheader("Informasi Matriks Piksel")
            p1, p2 = st.columns(2)
            p1.metric("Resolusi Spasial", f"{w} x {h} px")
            p2.metric("Aspek Rasio", f"{w/h:.2f}:1")
            
            p3, p4 = st.columns(2)
            p3.metric("Ukuran File", f"{len(file_bytes)/1024:.2f} KB")
            p4.metric("Total Piksel", f"{original_img.size:,}")
            
            p5, p6 = st.columns(2)
            p5.metric("Piksel Terendah", f"{np.min(original_img)}")
            p6.metric("Piksel Tertinggi", f"{np.max(original_img)}")
            
            st.metric("Intensitas Piksel Rata-rata", f"{np.mean(original_img):.2f}")

    # --- TAHAP 2: PEMODELAN NOISE GAUSSIAN ---
    with st.container(border=True):
        st.header("2. Pemodelan Noise Gaussian")
        col_noise_ctrl, col_noise_img = st.columns([1, 1.5])
        
        with col_noise_ctrl:
            st.write("**Pengaturan Varians Noise**")
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

            rmse_n, psnr_n, snr_n = calculate_all_metrics(original_img, noisy_img)
            
            st.write("---")
            st.write("**Metrik Degradasi Awal**")
            n1, n2, n3 = st.columns(3)
            n1.metric("RMSE", f"{rmse_n:.2f}")
            n2.metric("PSNR", f"{psnr_n:.2f} dB")
            n3.metric("SNR", f"{snr_n:.2f} dB")
            
        with col_noise_img:
            st.image(noisy_img, caption=f"Citra Rusak (Varians: {noise_var})", use_container_width=True)

    # --- TAHAP 3: PEMROSESAN DENOISING ---
    with st.container(border=True):
        st.header("3. Pemrosesan Denoising")
        tab1, tab2, tab3 = st.tabs(["Metode WT Murni", "Metode SVD Murni", "Kombinasi WT-SVD"])

        img_wt = img_svd = img_wtsvd = None
        energy_wt = energy_svd = energy_wtsvd = 0.0

        # Tab 1: WT MURNI
        with tab1:
            st.subheader("Wavelet Transform Denoising (Soft Thresholding)")
            thresh_multiplier = st.slider("Multiplier Threshold WT", 0.1, 5.0, 1.0, step=0.1)
            
            LL, HL, LH, HH = haar_dwt_2d(noisy_img)
            threshold = sigma * np.sqrt(2 * np.log(h * w)) * thresh_multiplier
            
            HL_t = soft_threshold(HL, threshold)
            LH_t = soft_threshold(LH, threshold)
            HH_t = soft_threshold(HH, threshold)
            
            img_wt = haar_idwt_2d(LL, HL_t, LH_t, HH_t).astype(np.uint8)
            rmse_wt, psnr_wt, snr_wt = calculate_all_metrics(original_img, img_wt)
            
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.image(img_wt, caption="Hasil WT Murni", use_container_width=True)
            with c2:
                st.write("**Peningkatan Metrik**")
                st.metric("RMSE", f"{rmse_wt:.4f}", delta=f"{rmse_wt - rmse_n:.4f}", delta_color="inverse")
                st.metric("PSNR", f"{psnr_wt:.4f} dB", delta=f"{psnr_wt - psnr_n:+.4f} dB")
                st.metric("SNR", f"{snr_wt:.4f} dB", delta=f"{snr_wt - snr_n:+.4f} dB")
                st.metric("Energy Retained", "N/A", help="Pemotongan energi tidak relevan pada hard/soft thresholding.")

        # Tab 2: SVD MURNI
        with tab2:
            st.subheader("Singular Value Decomposition (Full Image)")
            max_k_full = min(h, w)
            k_svd = st.slider("Jumlah Rank (k) SVD Murni", 1, max_k_full, int(max_k_full * 0.15))
            
            img_svd_float, energy_svd = apply_svd_matrix(noisy_img, k_svd)
            img_svd = np.clip(img_svd_float, 0, 255).astype(np.uint8)
            rmse_svd, psnr_svd, snr_svd = calculate_all_metrics(original_img, img_svd)
            
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.image(img_svd, caption="Hasil SVD Murni", use_container_width=True)
            with c2:
                st.write("**Peningkatan Metrik**")
                st.metric("RMSE", f"{rmse_svd:.4f}", delta=f"{rmse_svd - rmse_n:.4f}", delta_color="inverse")
                st.metric("PSNR", f"{psnr_svd:.4f} dB", delta=f"{psnr_svd - psnr_n:+.4f} dB")
                st.metric("SNR", f"{snr_svd:.4f} dB", delta=f"{snr_svd - snr_n:+.4f} dB")
                st.metric("Energy Retained", f"{energy_svd:.2f} %")

        # Tab 3: WT-SVD KOMBINASI
        with tab3:
            st.subheader("Kombinasi SVD pada Sub-band LL")
            LL_komb, HL_komb, LH_komb, HH_komb = haar_dwt_2d(noisy_img)
            max_k_komb = min(LL_komb.shape)
            
            k_wtsvd = st.slider("Jumlah Rank (k) Sub-band LL", 1, max_k_komb, int(max_k_komb * 0.40))
            
            LL_denoised, energy_wtsvd = apply_svd_matrix(LL_komb, k_wtsvd)
            img_wtsvd = haar_idwt_2d(LL_denoised, HL_komb, LH_komb, HH_komb).astype(np.uint8)
            
            rmse_wtsvd, psnr_wtsvd, snr_wtsvd = calculate_all_metrics(original_img, img_wtsvd)
            
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.image(img_wtsvd, caption="Hasil Kombinasi WT-SVD", use_container_width=True)
            with c2:
                st.write("**Peningkatan Metrik**")
                st.metric("RMSE", f"{rmse_wtsvd:.4f}", delta=f"{rmse_wtsvd - rmse_n:.4f}", delta_color="inverse")
                st.metric("PSNR", f"{psnr_wtsvd:.4f} dB", delta=f"{psnr_wtsvd - psnr_n:+.4f} dB")
                st.metric("SNR", f"{snr_wtsvd:.4f} dB", delta=f"{snr_wtsvd - snr_n:+.4f} dB")
                st.metric("Energy Retained (LL)", f"{energy_wtsvd:.2f} %")

    # --- TAHAP 4: TABEL KOMPARASI ---
    with st.container(border=True):
        st.header("4. Tabel Komparasi Metrik Evaluasi Akhir")
        st.write(f"**Kondisi Pengujian:** Varians Noise = {noise_var}")
        
        df_metrics = pd.DataFrame({
            "Metrik Evaluasi": ["RMSE", "PSNR (dB)", "SNR (dB)", "Energy Retained (%)"],
            "Citra Noisy": [f"{rmse_n:.4f}", f"{psnr_n:.4f}", f"{snr_n:.4f}", "-"],
            "Metode WT Murni": [f"{rmse_wt:.4f}", f"{psnr_wt:.4f}", f"{snr_wt:.4f}", "-"],
            f"Metode SVD Murni (k={k_svd})": [f"{rmse_svd:.4f}", f"{psnr_svd:.4f}", f"{snr_svd:.4f}", f"{energy_svd:.2f}"],
            f"Kombinasi WT-SVD (k={k_wtsvd})": [f"{rmse_wtsvd:.4f}", f"{psnr_wtsvd:.4f}", f"{snr_wtsvd:.4f}", f"{energy_wtsvd:.2f}"]
        })
        
        st.table(df_metrics)

else:
    st.info("Unggah citra medis terlebih dahulu.")