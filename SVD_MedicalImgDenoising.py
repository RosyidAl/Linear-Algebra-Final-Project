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
    # Resize jika terlalu besar agar angka matriksnya tetap bisa ditampilkan tanpa membuat tabel terlalu padat
    max_dim = 256
    if img.size[0] > max_dim or img.size[1] > max_dim:
        scale = max_dim / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size)
        st.sidebar.warning(
            f"Gambar di-resize ke {img.size[0]}x{img.size[1]} agar angka matriksnya masih nyaman ditampilkan. "
            "Nilai k maksimal akan mengikuti ukuran gambar ini."
        )
    A = np.array(img)
else:
    A = create_mock_medical_image()
    st.sidebar.info("Menggunakan gambar simulasi grid 20x20 piksel agar matriks mudah dipelajari.")

m, n = A.shape

# --- PROSES KOMPUTASI SVD ---
U, s, Vt = np.linalg.svd(A.astype(float), full_matrices=False)
max_k = len(s)  # Rank maksimum (r) dari matriks

# Tampilkan informasi rank matriks
st.sidebar.info(
    f"📊 **Informasi Rank Matriks:**\n\n"
    f"Rank maksimum (r): **{max_k}**\n\n"
    f"Dimensi matriks: {m} × {n}"
)

# Slider Jumlah Komponen K
st.sidebar.subheader("🎚️ Pilih Jumlah Komponen SVD")
st.sidebar.write("Tentukan berapa banyak komponen SVD (k) yang digunakan untuk rekonstruksi gambar.")
k = st.sidebar.slider(
    label="Jumlah Komponen (k): ", 
    min_value=1, 
    max_value=max_k,
    value=min(int(max_k * 0.3), max_k) if max_k > 3 else 1,
    help=f"Pilih k dari 1 hingga {max_k}. Semakin besar k, semakin detail gambar (lebih mirip asli)."
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
    
    # Tampilkan informasi kompresi
    compression_ratio = (k * (m + n + 1)) / (m * n) * 100
    energy_retained = (np.sum(s_k**2) / np.sum(s**2)) * 100
    
    st.caption(f"Direkonstruksi menggunakan **k = {k}** dari **r = {max_k}** komponen ({(k/max_k)*100:.1f}%)")
    st.caption(f"📈 Energi yang dipertahankan: **{energy_retained:.2f}%**")
    st.caption(f"📦 Rasio kompresi: **{compression_ratio:.2f}%**")

# --- SECTION: DETAIL HITUNG-HITUNGAN MATRIKS ---
st.write("---")
st.subheader("🧮 Detail Langkah Komputasi Aljabar Linear")

# Box informasi parameter k
param_col1, param_col2, param_col3 = st.columns(3)
with param_col1:
    st.metric("Jumlah Komponen (k)", f"{k}", f"dari {max_k}")
with param_col2:
    st.metric("Persentase Komponen", f"{(k/max_k)*100:.1f}%", f"({k}/{max_k})")
with param_col3:
    st.metric("Energi Dipertahankan", f"{energy_retained:.2f}%", f"dari total energi")

st.write("Catatan: Semakin besar nilai **k**, semakin banyak detail gambar yang dipertahankan. "
         "Namun, noise juga akan terikut semakin banyak. Sebaliknya, **k yang kecil** menghasilkan denoising "
         "yang lebih agresif tapi detail gambar akan hilang.")
st.write("---")

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