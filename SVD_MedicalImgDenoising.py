import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# --- SETUP HALAMAN WEB ---
st.set_page_config(page_title="SVD Medical Denoising", layout="wide")
st.title("🩺 Aplikasi Denoising Citra Medis dengan SVD")
st.write("Projek Aljabar Linear Sederhana - Demonstrasi Dekomposisi Nilai Singular")
st.write("---")

# --- FUNGSI MEMBUAT GAMBAR SIMULASI (JIKA BELUM ADA UPLOAD) ---
def create_mock_medical_image():
    # Membuat gambar hitam ukuran 300x300
    img = Image.new("L", (300, 300), color=30)
    draw = ImageDraw.Draw(img)
    # Menggambar bentuk menyerupai tulang (simulasi X-Ray)
    draw.ellipse([120, 30, 180, 270], fill=200)
    draw.ellipse([90, 30, 210, 80], fill=220)
    draw.ellipse([90, 220, 210, 270], fill=220)
    
    # Menambahkan "Noise/Derau" acak (Gausian Noise)
    img_array = np.array(img).astype(float)
    noise = np.random.normal(0, 40, img_array.shape)
    noisy_img = img_array + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# --- SIDEBAR INPUT ---
st.sidebar.header("⚙️ Konfigurasi")
uploaded_file = st.sidebar.file_uploader("Unggah Foto Medis (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Load gambar (dari upload atau simulasi)
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    A = np.array(img)
    st.sidebar.success("Gambar berhasil diunggah!")
else:
    A = create_mock_medical_image()
    st.sidebar.info("Menggunakan gambar simulasi X-Ray (Belum ada gambar diunggah).")

# --- PROSES ALGEBRA: SVD ---
# Memecah matriks gambar menjadi U, s, dan Vt
U, s, Vt = np.linalg.svd(A.astype(float), full_matrices=False)
max_k = len(s)

# Slider untuk menentukan nilai k (Rank)
k = st.sidebar.slider(
    "Pilih Jumlah Nilai Singular (k):", 
    min_value=1, 
    max_value=max_k, 
    value=int(max_k * 0.15) # Default mengambil 15% nilai singular teratas
)

# Rekonstruksi gambar hanya menggunakan k komponen pertama
S_k = np.diag(s[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]
A_reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))
A_reconstructed = np.clip(A_reconstructed, 0, 255).astype(np.uint8)

# --- TAMPILKAN GAMBAR (BEFORE VS AFTER) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("❌ Citra Asli (Banyak Noise)")
    st.image(A, use_container_width=True)
    st.caption(f"Ukuran Matriks Asli: {A.shape[0]} x {A.shape[1]} (Rank Maksimal: {max_k})")

with col2:
    st.subheader(f"✅ Hasil Denoising (k = {k})")
    st.image(A_reconstructed, use_container_width=True)
    st.caption(f"Direkonstruksi hanya dengan {k} komponen utama.")

# --- VISUALISASI PROSES KOMPUTASI ---
st.write("---")
st.subheader("📊 Proses Komputasi Aljabar Linear (Analisis Nilai Singular)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(s, color='gray', linewidth=2, label="Nilai Singular ($\Sigma$)")
ax.plot(range(k), s[:k], color='red', linewidth=3, label=f"{k} Nilai yang Dipertahankan (Sinyal)")
ax.axvline(x=k, color='red', linestyle='--', alpha=0.7, label=f'Batas Potong (k={k})')

ax.set_title("Grafik Penurunan Magnitudo Nilai Singular")
ax.set_xlabel("Indeks Nilai Singular (Dari Terbesar ke Terkecil)")
ax.set_ylabel("Nilai Magnitudo")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Penjelasan Edukatif untuk Presentasi
st.info(
    f"💡 **Penjelasan Matematis untuk Dosen:**\n\n"
    f"Nilai singular di sebelah kiri grafik (berwarna merah) memiliki magnitudo besar karena menyimpan informasi struktur utama gambar. "
    f"Nilai singular di sebelah kanan grafik cenderung kecil dan mendatar, yang merepresentasikan **Noise/Derau**.\n\n"
    f"Dengan memotong komponen pada **k = {k}**, kita berhasil membuang **{max_k - k}** komponen noise, sehingga gambar terlihat lebih bersih!"
)