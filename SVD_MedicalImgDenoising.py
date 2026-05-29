import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import base64
import io

st.set_page_config(page_title="SVD Medical Denoising", layout="wide")
st.title("SVD Medical Image Denoising")


def image_to_base64(img_pil):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# Upload gambar
uploaded_file = st.file_uploader("Upload medical image", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.stop()

img = Image.open(uploaded_file).convert("L")
if img.size[0] > 256 or img.size[1] > 256:
    scale = 256 / max(img.size)
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))

A = np.array(img)
m, n = A.shape

# SVD decomposition
U, s, Vt = np.linalg.svd(A.astype(float), full_matrices=False)
rank_matrix = len(s)
max_k_slider = max(m, n)

# Input k
k = st.number_input(f"Jumlah Komponen (k) - Max: {max_k_slider}", min_value=1, max_value=max_k_slider, value=min(int(rank_matrix * 0.3), rank_matrix) or 1)

# Reconstruct image
if k <= rank_matrix:
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
else:
    U_k = np.column_stack([U, np.zeros((m, k - rank_matrix))])
    s_k = np.concatenate([s, np.zeros(k - rank_matrix)])
    Vt_k = np.vstack([Vt, np.zeros((k - rank_matrix, n))])

Sigma_k = np.diag(s_k)
A_reconstructed = np.dot(U_k, np.dot(Sigma_k, Vt_k))
A_reconstructed_clipped = np.clip(A_reconstructed, 0, 255).astype(np.uint8)

# Display
st.write("---")

original_image = Image.fromarray(A)
denosed_image = Image.fromarray(A_reconstructed_clipped)
original_base64 = image_to_base64(original_image)
denoised_base64 = image_to_base64(denosed_image)
energy_retained = (np.sum(s_k**2) / np.sum(s**2)) * 100

comparison_html = f"""
<div style="max-width: 900px; margin: auto;">
  <h3 style="text-align: center;">Comparison: Original vs Denoised</h3>
  <div class="img-comp-container" style="position: relative; width: 100%; max-width: 900px; overflow: hidden;">
    <div class="img-comp-img" style="position: relative; width: 100%;">
      <img src="data:image/png;base64,{original_base64}" style="display: block; width: 100%; height: auto;" />
    </div>
    <div class="img-comp-img img-comp-overlay" style="position: absolute; top: 0; left: 0; width: 50%; overflow: hidden;">
      <img src="data:image/png;base64,{denoised_base64}" style="display: block; width: 100%; height: auto;" />
    </div>
    <div class="img-comp-slider" style="position: absolute; z-index: 9; top: 0; bottom: 0; left: 50%; width: 4px; background: rgba(255,255,255,0.8); cursor: ew-resize;" id="slider"></div>
  </div>
  <div style="display: flex; justify-content: space-between; margin-top: 0.75rem; font-size: 0.95rem; color: #ddd;">
    <span>Size: {m} x {n}</span>
    <span>Energy retained: {energy_retained:.2f}%</span>
  </div>
</div>
<style>
.img-comp-container img {{
  vertical-align: middle;
}}
</style>
<script>
(function() {{
  const container = document.querySelector(".img-comp-container");
  const overlay = container.querySelector(".img-comp-overlay");
  const slider = document.getElementById("slider");
  let clicked = false;
  const slideReady = () => clicked = true;
  const slideFinish = () => clicked = false;
  const slideMove = (x) => {{
    const rect = container.getBoundingClientRect();
    let pos = x - rect.left;
    if (pos < 0) pos = 0;
    if (pos > rect.width) pos = rect.width;
    overlay.style.width = pos + "px";
    slider.style.left = pos + "px";
  }};
  slider.addEventListener("mousedown", slideReady);
  window.addEventListener("mouseup", slideFinish);
  window.addEventListener("mousemove", (event) => {{
    if (!clicked) return;
    slideMove(event.pageX);
  }});
  slider.addEventListener("touchstart", slideReady);
  window.addEventListener("touchend", slideFinish);
  window.addEventListener("touchmove", (event) => {{
    if (!clicked) return;
    slideMove(event.touches[0].pageX);
  }});
}})();
</script>
"""

components.html(comparison_html, height=520)
st.write("---")

# Info
st.write(f"**Rank: {rank_matrix} | Dimension: {m} x {n}**")

# Tables
st.subheader("Computation Details")
tab1, tab2, tab3 = st.tabs(["Matrix A", "SVD Components", "Result"])

with tab1:
    st.dataframe(A)

with tab2:
    col_u, col_s, col_v = st.columns(3)
    with col_u:
        st.write(f"U_k ({m} x {k})")
        st.dataframe(U_k)
    with col_s:
        st.write(f"Sigma_k ({k} x {k})")
        st.dataframe(Sigma_k)
    with col_v:
        st.write(f"V_k^T ({k} x {n})")
        st.dataframe(Vt_k)

with tab3:
    st.dataframe(A_reconstructed_clipped)