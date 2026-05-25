import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

# --- SETUP HALAMAN WEB ---
st.set_page_config(page_title="SVD Medical Denoising", layout="wide")
st.title("🩺 Aplikasi Denoising Citra Medis dengan SVD")
st.write("Projek Aljabar Linear - Penerapan & Detail Komputasi Matriks")
st.write("---")

# --- FUNGSI MEMBUAT GAMBAR SIMULASI (JIKA BELUM ADA UPLOAD) ---
@st.cache_data
def create_mock_medical_image():
    # Membuat gambar ukuran kecil (20x20) agar angka matriksnya mudah dibaca di layar
    img = Image.new("L", (20, 20), color=30)
    draw = ImageDraw.Draw(img)
    # Gambar bentuk kotak putih di tengah (simulasi objek medis/tulang)
    draw.rectangle([5, 5, 14, 14], fill=180)
    
    # Tambahkan noise acak
    np.random.seed(42)
    img_array = np.array(img).astype(float)
    noise = np.random.normal(0, 35, img_array.shape)
    noisy_img = img_array + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# --- SIDEBAR INPUT ---
st.sidebar.header("⚙️ Konfigurasi")
uploaded_file = st.sidebar.file_uploader("Unggah Foto Medis (Disarankan gambar kecil atau grayscale)", type=["png", "jpg", "jpeg"])

# Load gambar
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    # Resize jika terlalu besar agar angka matriksnya muat di layar
    if img.size[0] > 100 or img.size[1] > 100:
        img = img.resize((50, 50))
        st.sidebar.warning("Gambar di-resize ke 50x50 agar angka matriksnya tidak terlalu padat.")
    A = np.array(img)
else:
    A = create_mock_medical_image()
    st.sidebar.info("Menggunakan gambar simulasi grid 20x20 piksel agar matriks mudah dipelajari.")

m, n = A.shape

# --- PROSES KOMPUTASI SVD ---
U, s, Vt = np.linalg.svd(A.astype(float), full_matrices=False)
max_k = len(s)

# Slider Nilai K
k = st.sidebar.slider(
    "Pilih Jumlah Nilai Singular (k):", 
    min_value=1, 
    max_value=max_k, 
    value=int(max_k * 0.3) if max_k > 3 else 1
)

# Ambil komponen k pertama (Truncated SVD)
U_k = U[:, :k]
s_k = s[:k]
Sigma_k = np.diag(s_k)
Vt_k = Vt[:k, :]

# Jalankan perkalian matriks untuk rekonstruksi: A_clean = U_k * Sigma_k * Vt_k
A_reconstructed = np.dot(U_k, np.dot(Sigma_k, Vt_k))
A_reconstructed_clipped = np.clip(A_reconstructed, 0, 255).astype(np.uint8)

# --- TAMPILKAN GAMBAR (BEFORE VS AFTER) ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("❌ Citra Asli (Banyak Noise)")
    st.image(A, width=300)
    st.caption(f"Dimensi Matriks Gambar Asli ($A$): **{m} x {n}**")

with col2:
    st.subheader(f"✅ Hasil Denoising (k = {k})")
    st.image(A_reconstructed_clipped, width=300)
    st.caption(f"Direkonstruksi menggunakan **k = {k}** nilai singular.")

# --- SECTION: DETAIL HITUNG-HITUNGAN MATRIKS ---
st.write("---")
st.subheader("🧮 Detail Langkah Komputasi Aljabar Linear")
st.write("Di bawah ini adalah angka asli dari matriks gambar kamu dan bagaimana SVD memotongnya secara matematis:")

tab1, tab2, tab3 = st.tabs(["1. Matriks Gambar Asli (A)", "2. Komponen SVD Terpotong (Truncated)", "3. Hasil Perkalian U × Σ × Vᵀ"])

with tab1:
    st.write(f"Matriks $A$ berukuran **{m} x {n}** (Setiap angka merepresentasikan kecerahan piksel 0-255):")
    st.dataframe(A)

with tab2:
    st.write(f"Berdasarkan nilai **k = {k}** yang kamu pilih, matriks dipecah menjadi:")
    
    col_u, col_s, col_v = st.columns(3)
    with col_u:
        st.write(f"**Matriks $U_k$** (Ukuran: {m} x {k})")
        st.dataframe(U_k)
    with col_s:
        st.write(f"**Matriks $\Sigma_k$** (Diagonal Nilai Singular, Ukuran: {k} x {k})")
        st.dataframe(Sigma_k)
    with col_v:
        st.write(f"**Matriks $V_k^T$** (Ukuran: {k} x {n})")
        st.dataframe(Vt_k)

with tab3:
    st.write("Proses rekonstruksi dilakukan dengan mengalikan ketiga matriks di atas:")
    st.latex(r"A_{clean} = U_k \times \Sigma_k \times V_k^T")
    
    st.write("**Hasil perkalian matriks sebelum pembulatan (Float):**")
    st.dataframe(A_reconstructed)
    
    st.write("**Hasil akhir setelah pembulatan nilai piksel (0 - 255):**")
    st.dataframe(A_reconstructed_clipped)

st.info(
    "💡 **Tips Presentasi:** Tunjukkan menu tab di atas kepada dosen. "
    "Jelaskan bahwa dengan mengubah nilai *slider K*, ukuran kolom pada $U_k$ dan baris pada $V_k^T$ akan berubah secara dinamis. "
    "Semakin kecil nilai $k$, data yang diproses semakin sedikit, sehingga detail kecil (*noise*) otomatis hilang."
)