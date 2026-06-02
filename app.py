import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import svd

# ==========================================
# FUNGSI UTAMA PIPELINE WT-SVD
# ==========================================

def haar_dwt_2d(img):
    """
    Tahap 3: Forward Haar Wavelet Transform (Dekomposisi manual 1-Level)
    Memisahkan gambar menjadi: LL (Inti), HL (Garis Tegak), LH (Garis Datar), HH (Bintik Detail/Noise)
    """
    h, w = img.shape
    # Memastikan dimensi genap
    if h % 2 != 0: img = img[:-1, :]
    if w % 2 != 0: img = img[:, :-1]
    h, w = img.shape

    # Transformasi Horizontal (Per Baris)
    W_row = np.zeros_like(img, dtype=np.float64)
    for i in range(h):
        a = img[i, 0::2]
        b = img[i, 1::2]
        W_row[i, :w//2] = (a + b) / 2.0  # Low
        W_row[i, w//2:] = (a - b) / 2.0  # High

    # Transformasi Vertikal (Per Kolom)
    W_total = np.zeros_like(W_row, dtype=np.float64)
    for j in range(w):
        a = W_row[0::2, j]
        b = W_row[1::2, j]
        W_total[:h//2, j] = (a + b) / 2.0  # Low
        W_total[h//2:, j] = (a - b) / 2.0  # High

    # Memecah matriks menjadi 4 sub-band (kuadran)
    LL = W_total[:h//2, :w//2]
    HL = W_total[:h//2, w//2:]
    LH = W_total[h//2:, :w//2]
    HH = W_total[h//2:, w//2:]
    return LL, HL, LH, HH

def haar_idwt_2d(LL, HL, LH, HH):
    """
    Tahap 5: Inverse Haar Wavelet Transform (Rekonstruksi manual 1-Level)
    """
    h_sub, w_sub = LL.shape
    h, w = h_sub * 2, w_sub * 2
    
    # Rekombinasi menjadi matriks total frekuensi
    W_total = np.zeros((h, w), dtype=np.float64)
    W_total[:h_sub, :w_sub] = LL
    W_total[:h_sub, w_sub:] = HL
    W_total[h_sub:, :w_sub] = LH
    W_total[h_sub:, w_sub:] = HH

    # Rekonstruksi Vertikal (Per Kolom)
    W_row = np.zeros_like(W_total, dtype=np.float64)
    for j in range(w):
        L = W_total[:h_sub, j]
        H = W_total[h_sub:, j]
        W_row[0::2, j] = L + H
        W_row[1::2, j] = L - H

    # Rekonstruksi Horizontal (Per Baris)
    img_reconstructed = np.zeros_like(W_row, dtype=np.float64)
    for i in range(h):
        L = W_row[i, :w_sub]
        H = W_row[i, w_sub:]
        img_reconstructed[i, 0::2] = L + H
        img_reconstructed[i, 1::2] = L - H

    return np.clip(img_reconstructed, 0, 255)

def apply_svd_denoising(LL_matrix, k_values_to_keep):
    """
    Tahap 4: SVD Denoising pada Sub-band LL (Rangka Utama)
    Membuang nilai singular terkecil yang didominasi noise
    """
    U, S_diag, Vt = svd(LL_matrix, full_matrices=False)
    
    # Membuat matriks diagonal S baru
    S = np.zeros((S_diag.shape[0], S_diag.shape[0]))
    # Hanya mempertahankan k nilai singular terbesar (Filter deviasi rendah/noise)
    S[:k_values_to_keep, :k_values_to_keep] = np.diag(S_diag[:k_values_to_keep])
    
    # Rekonstruksi sub-band LL yang telah dibersihkan
    LL_denoised = np.dot(U, np.dot(S, Vt))
    return LL_denoised

# ==========================================
# METRIKS EVALUASI (TAHAP 6)
# ==========================================

def calculate_metrics(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 0, 0, float('inf'), float('inf')
    rmse = np.sqrt(mse)
    
    # PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / rmse)
    
    # SNR Kontras Sederhana
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    
    return mse, rmse, psnr, snr

# ==========================================
# INTERFACE STREAMLIT (DASHBOARD MVP)
# ==========================================

st.title("MVP Pemrosesan Denoising Citra Medis (WT-SVD)")
st.write("Aplikasi simulasi pengujian restorasi gambar berbasis pipeline dekomposisi Wavelet dan reduksi dimensi SVD.")

# Langkah 1: Input Citra Medis Asli (Menggunakan file uploader)
uploaded_file = st.file_uploader("Unggah Citra Medis Asli (Format: PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # Konversi file ke array opencv grayscale
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Normalisasi dimensi agar genap
    h, w = original_img.shape
    if h % 2 != 0 or w % 2 != 0:
        original_img = original_img[:h-(h%2), :w-(w%2)]
    
    st.image(original_img, caption="Citra Medis Asli (Referensi)", width=300)

    # ------------------------------------------------
    # TAHAP 2: PEMODELAN SIMULASI NOISE GAUSSIAN
    # ------------------------------------------------
    st.header("1. Pemodelan & Variasi Tingkat Noise")
    
    # Parameter varians sesuai spesifikasi riset (skala diadaptasi ke rentang piksel 0-255)
    noise_variance_selection = st.select_slider(
        "Pilih Varians Gaussian Noise (Sesuai parameter uji):",
        options=[0.02, 0.05, 0.09],
        value=0.02
    )
    
    # Konversi nilai varians relatif ke standar deviasi intensitas piksel 8-bit
    sigma = np.sqrt(noise_variance_selection) * 255
    
    # Generate Gaussian Noise
    np.random.seed(42)
    gaussian_noise = np.random.normal(0, sigma, original_img.shape)
    noisy_img = np.clip(original_img + gaussian_noise, 0, 255).astype(np.uint8)
    
    # Hitung metrik kerusakan citra awal
    mse_n, rmse_n, psnr_n, snr_n = calculate_metrics(original_img, noisy_img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(noisy_img, caption=f"Citra Rusak (Noise Var: {noise_variance_selection})", use_container_width=True)
    with col2:
        st.subheader("Metrik Kerusakan Awal")
        st.text(f"MSE  : {mse_n:.4f}")
        st.text(f"RMSE : {rmse_n:.4f}")
        st.text(f"PSNR : {psnr_n:.4f} dB")
        st.text(f"SNR  : {snr_n:.4f} dB")

    # ------------------------------------------------
    # TAHAP 3, 4, 5: ALGORITMA WT-SVD & REKONSTRUKSI
    # ------------------------------------------------
    st.header("2. Pemrosesan Algoritma & Sinergi")
    
    # Terapkan Tahap 3: Haar Wavelet Transform
    LL, HL, LH, HH = haar_dwt_2d(noisy_img)
    
    # Terapkan Parameter Batas Potong Nilai Singular (SVD Thresholding)
    max_singular_values = LL.shape[0]
    # Default: mempertahankan 60% dari total komponen variasi utama untuk memisahkan noise
    default_keep = int(max_singular_values * 0.6)
    
    k_keep = st.slider(
        "Jumlah Nilai Singular Matriks LL yang Dipertahankan (SVD):", 
        min_value=1, 
        max_value=max_singular_values, 
        value=default_keep
    )
    
    # Terapkan Tahap 4: Reduksi Noise via SVD pada Sub-band LL
    LL_denoised = apply_svd_denoising(LL, k_keep)
    
    # Terapkan Tahap 5: Sinergi dan Rekonstruksi Akhir (IWT)
    # Sub-band detail (HL, LH, HH) dipertahankan untuk menjaga ketajaman tepi garis organ
    denoised_img = haar_idwt_2d(LL_denoised, HL, LH, HH)
    denoised_img_uint8 = denoised_img.astype(np.uint8)

    # ------------------------------------------------
    # TAHAP 6: EVALUASI KINERJA METRIK OBJEKTIF
    # ------------------------------------------------
    st.header("3. Evaluasi Kinerja (Performance Assessment)")
    
    mse_d, rmse_d, psnr_d, snr_d = calculate_metrics(original_img, denoised_img_uint8)
    
    col3, col4 = st.columns(2)
    with col3:
        st.image(denoised_img_uint8, caption="Hasil Akhir Rekonstruksi WT-SVD", use_container_width=True)
    with col4:
        st.subheader("Metrik Hasil Pemurnian")
        st.text(f"MSE  : {mse_d:.4f}")
        st.text(f"RMSE : {rmse_d:.4f}")
        st.text(f"PSNR : {psnr_d:.4f} dB")
        st.text(f"SNR  : {snr_d:.4f} dB")
        
        # Validasi Indikator Keberhasilan Pipeline
        st.subheader("Validasi Indikator Keberhasilan:")
        if mse_d < mse_n and psnr_d > psnr_n:
            st.success("✓ BERHASIL: Nilai MSE menurun dan nilai PSNR meningkat secara signifikan.")
        else:
            st.warning("! OPTIMALISASI: Parameter nilai SVD yang dipertahankan terlalu rendah/tinggi sehingga struktur ikut tereduksi.")

    # Tampilan visualisasi visual perbandingan sub-band untuk analisis besok
    with st.expander("Lihat Visualisasi Komponen Sub-band Frekuensi Haar WT"):
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        axes[0, 0].imshow(LL, cmap='gray')
        axes[0, 0].set_title('LL (Rangka Utama)')
        axes[0, 1].imshow(HL, cmap='gray')
        axes[0, 1].set_title('HL (Garis Tegak)')
        axes[1, 0].imshow(LH, cmap='gray')
        axes[1, 0].set_title('LH (Garis Datar)')
        axes[1, 1].imshow(HH, cmap='gray')
        axes[1, 1].set_title('HH (Bintik Noise)')
        for ax in axes.flat:
            ax.axis('off')
        st.pyplot(fig)

else:
    st.info("Silakan unggah gambar sampel citra medis terlebih dahulu untuk memulai pengujian.")