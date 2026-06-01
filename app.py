import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image

from utils import image_to_base64, generate_synthetic_mri, reconstruct
from styles import CUSTOM_CSS, get_header_html, get_direct_zoom_html

# ─── KONFIGURASI HALAMAN UTAMA ────────────────────────────────────────────────
st.set_page_config(
    page_title="SVD Medical Image Denoising",
    layout="wide"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(get_header_html(), unsafe_allow_html=True)

if 'app_launched' not in st.session_state:
    st.session_state.app_launched = False

# ─── HALAMAN AWAL (LANDING PAGE) ─────────────────────────────────────────────
if not st.session_state.app_launched:
    col_spacer_left, col_left, col_right, col_spacer_right = st.columns([0.3, 1.3, 1.1, 0.3])
    
    with col_left:
        st.markdown("<p style='color:#0284C7; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-top:2.5rem; margin-bottom:0.5rem;'>SVD WORKSPACE</p>", unsafe_allow_html=True)
        st.markdown("""
        <h1>
            See SVD <br> Denoising <br> Come to Life
        </h1>
        <p style="color: #475569; font-size: 1.1rem; line-height: 1.6; max-width: 480px; margin-bottom: 2.5rem;">
            Dekomposisi nilai singular untuk mereduksi noise pada citra diagnostik secara real-time. 
            Kontrol aproksimasi low-rank secara interaktif untuk mempertahankan detail struktural anatomis medis.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="launch-container">', unsafe_allow_html=True)
        if st.button("Launch Visualizer →"):
            st.session_state.app_launched = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_right:
        svg_graphic = """
        <svg viewBox="0 0 400 400" width="100%" height="100%" style="background: transparent; max-height: 400px; margin-top: 2rem;">
            <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#E2E8F0" stroke-width="1"/>
                </pattern>
            </defs>
            <rect width="360" height="360" x="20" y="20" fill="url(#grid)" rx="16" stroke="#E2E8F0" stroke-width="1.5" />
            <line x1="20" y1="200" x2="380" y2="200" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4 4" />
            <line x1="200" y1="20" x2="200" y2="380" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4 4" />
            <ellipse cx="200" cy="200" rx="110" ry="60" transform="rotate(-30, 200, 200)" fill="none" stroke="#0284C7" stroke-width="2.5" opacity="0.8"/>
            <line x1="200" y1="200" x2="295" y2="145" stroke="#0369A1" stroke-width="3.5" />
            <circle cx="295" cy="145" r="5" fill="#0369A1" />
            <text x="310" y="145" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" font-size="14" fill="#0369A1">σ₁ u₁</text>
            <line x1="200" y1="200" x2="170" y2="148" stroke="#38BDF8" stroke-width="3" />
            <circle cx="170" cy="148" r="4" fill="#38BDF8" />
            <text x="145" y="135" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" font-size="14" fill="#38BDF8">σ₂ u₂</text>
            <rect x="35" y="35" width="130" height="40" rx="8" fill="white" stroke="#E2E8F0" stroke-width="1" />
            <text x="47" y="60" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="600" fill="#0F172A">A = U Σ Vᵀ</text>
            <rect x="235" y="325" width="130" height="40" rx="8" fill="white" stroke="#E2E8F0" stroke-width="1" />
            <text x="247" y="350" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="600" fill="#0284C7">Rank-k Approx</text>
        </svg>
        """
        components.html(svg_graphic, height=430)
    st.stop()

# ─── HALAMAN WORKSPACE UTAMA CITRA MEDIS ──────────────────────────────────────
col_ws_ws_l, col_ws_content, col_ws_ws_r = st.columns([0.05, 0.9, 0.05])

with col_ws_content:
    st.title("SVD Medical Image Denoising")
    uploaded_file = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
    else:
        img_array = generate_synthetic_mri()
        img = Image.fromarray(img_array)

    if img.size[0] > 512 or img.size[1] > 512:
        scale = 512 / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))

    A = np.array(img).astype(float)
    m, n = A.shape

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank_matrix = len(s)

    # ─── SECTION 1: MATRIKS ORIGINAL A ───
    col_text_left, col_img_right = st.columns([0.8, 1.2])
    with col_text_left:
        st.markdown("""
            <div style='padding-top: 1rem;'>
                <h1 style='font-size: 3.5rem !important; margin-bottom: 1rem; line-height: 1.15;'>Analisis Karakteristik Matriks Citra Medis</h1>
            </div>
        """, unsafe_allow_html=True)
    with col_img_right:
        st.image(img, use_container_width=True)

    st.write("**Representasi Nilai Matriks $A$:**")
    st.dataframe(A.astype(np.uint8), use_container_width=True)

    st.write("---")

    # ─── SECTION 2: DEKOMPOSISI AWAL ───
    st.header("Dekomposisi SVD Awal")
    st.write("Hasil faktorisasi linear penuh sebelum reduksi: $A = U \\Sigma V^T$")
    col_u, col_s, col_v = st.columns(3)
    with col_u:
        st.write(f"**Matriks Kiri $U$** ({m} × {rank_matrix})")
        st.dataframe(U, use_container_width=True)
    with col_s:
        st.write(f"**Matriks Diagonal $\\Sigma$** ({rank_matrix} × {rank_matrix})")
        st.dataframe(np.diag(s), use_container_width=True)
    with col_v:
        st.write(f"**Matriks Kanan Transpose $V^T$** ({rank_matrix} × {n})")
        st.dataframe(Vt, use_container_width=True)

    st.write("---")

    # ─── SECTION 3: MENGATUR SINGULAR VALUE ───
    st.header("Pengaturan Nilai Singular ($k$)")
    st.write("Konfigurasikan nilai pemotongan rank k untuk mengeliminasi komponen berfrekuensi tinggi (noise).")

    k = st.number_input(
        f"Aproksimasi Rank k (Maks {rank_matrix})",
        min_value=1, max_value=rank_matrix,
        value=max(1, min(int(rank_matrix * 0.15), rank_matrix)),
        key="k"
    )

    A_mod, U_k, s_k, Vt_k = reconstruct(U, s, Vt, k, m, n)
    energy = (np.sum(s_k**2) / np.sum(s**2)) * 100

    st.write("---")

    # ─── SECTION 4: HASIL AKHIR DARI MATRIKS A ───
    st.header("Hasil Akhir dari Matriks A")

    m1, m2, m3 = st.columns(3)
    m1.metric("Original Matrix Rank", rank_matrix)
    m2.metric("Matrix Size", f"{m}×{n}")
    m3.metric("Energy Retention", f"{energy:.2f}%")

    st.write("")
    st.write("**Komparasi Hasil Akhir (Gunakan Tombol Kontrol Pada Setiap Gambar Untuk Zoom & Drag):**")
    
    # Memproses data gambar ke Base64 sebelum dimasukkan ke objek visualizer kustom
    img_orig_b64 = image_to_base64(img)
    img_mod_b64 = image_to_base64(Image.fromarray(A_mod))

    # Memanggil Komponen HTML Interaksi Zoom Langsung dari styles.py
    comparison_html = get_direct_zoom_html(img_orig_b64, img_mod_b64, k)
    components.html(comparison_html, height=520)

    st.write("")
    st.write("**Log Data Matriks Hasil Akhir Rekonstruksi:**")
    st.write(f"**Matriks Rekonstruksi $A_{{mod}}$ ($k={k}$)**")
    st.dataframe(A_mod, use_container_width=True)