CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --primary: #0284C7;
  --primary-hover: #0369A1;
  --primary-light: #E0F2FE;
  --bg-card: #FFFFFF;
  --text-title: #0F172A;
  --text-body: #334155;
  --text-muted: #64748B;
  --border-color: #E2E8F0;
}

/* Menghilangkan Bar Native Streamlit (Deploy & Menu) dan Footer */
header[data-testid="stHeader"], 
div[data-testid="stHeader"],
footer {
  display: none !important;
  visibility: hidden !important;
}

/* Latar Belakang Luar Berdimensi */
.stApp, 
div[data-testid="stAppViewContainer"] {
  background-color: #F8FAFC !important;
  background-image: 
    radial-gradient(at 0% 0%, rgba(2, 132, 199, 0.05) 0px, transparent 50%), 
    radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.03) 0px, transparent 50%),
    linear-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px), 
    linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px) !important;
  background-size: 100% 100%, 100% 100%, 24px 24px, 24px 24px !important;
  margin: 0 !important;
  padding: 0 !important;
}

html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

/* Kanvas Utama */
.main .block-container {
  background-color: var(--bg-card) !important;
  padding: 3rem 5rem 4rem 5rem !important; 
  border-radius: 24px !important;
  box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.05) !important;
  margin-top: 6.5rem !important; 
  margin-bottom: 4rem !important;
  max-width: 1350px !important;
  border: 1px solid var(--border-color) !important;
}

/* Typography Hierarchy */
h1 {
  font-family: 'Cabinet Grotesk', sans-serif !important;
  font-weight: 800 !important;
  font-size: 3.5rem !important;
  line-height: 1.1 !important;
  color: var(--text-title) !important;
  letter-spacing: -0.04em !important;
  margin-bottom: 1.5rem !important;
}

h2, h3, h4 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  color: var(--text-title) !important;
  letter-spacing: -0.02em !important;
  margin-top: 1.5rem !important;
  margin-bottom: 1rem !important;
}

/* Shading Frame Kontainer Komponen */
[data-testid="stImage"], 
[data-testid="stDataFrame"], 
.stNumberInput, 
[data-testid="stMetric"],
[data-testid="stTable"] {
  position: relative !important;
  background-color: #FFFFFF !important;
  padding: 1.25rem !important;
  border-radius: 16px !important;
  box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04), 0 1px 3px 0 rgba(15, 23, 42, 0.02) !important;
  border: 1px solid var(--border-color) !important;
  margin-bottom: 1.5rem !important;
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
  padding: 0 !important;
  overflow: hidden !important;
}

/* Navigasi Kustom Menempel Sempurna ke Atas Layar */
.custom-header {
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  width: 100% !important;
  height: 70px !important;
  z-index: 99999 !important;
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  padding: 0 6rem !important; 
  background-color: #FFFFFF !important; 
  border-bottom: 1px solid var(--border-color) !important;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03) !important;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.logo-icon {
  background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
  color: white !important;
  padding: 5px 12px;
  border-radius: 8px;
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 0.9rem;
  letter-spacing: 0.03em;
}

.brand-name {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--text-title);
  letter-spacing: -0.02em;
}

.header-menu {
  display: flex;
  gap: 2.5rem;
  font-size: 0.85rem;
  font-weight: 700;
}

.menu-item-active {
  color: var(--primary) !important;
  position: relative;
  padding-bottom: 6px;
}

.menu-item-active::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg, #0284C7, #38BDF8);
  border-radius: 2px;
}

.menu-item-disabled {
  color: var(--text-muted) !important;
  opacity: 0.6;
}

.header-right .status-badge {
  font-size: 0.8rem;
  font-weight: 700;
  color: #0369A1;
  background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
  padding: 4px 14px;
  border-radius: 20px;
  box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
}

.stNumberInput input {
  border: none !important;
  background-color: transparent !important;
  box-shadow: none !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 600 !important;
}

[data-testid="stFileUploader"] {
  border: 2px dashed #0284C7 !important;
  border-radius: 20px !important;
  background: linear-gradient(180deg, #F0F9FF 0%, #E0F2FE 100%) !important;
  padding: 2.5rem !important;
  box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stMetric"] {
  border: none !important;
  box-shadow: none !important;
  padding: 0.5rem !important;
}

[data-testid="stMetricValue"], 
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] span {
  color: #0284C7 !important;
  -webkit-text-fill-color: #0284C7 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 800 !important;
  font-size: 1.8rem !important;
}

.launch-container .stButton button {
  background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
  color: #FFFFFF !important;
  border-radius: 14px !important;
  padding: 0.9rem 3rem !important;
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  border: none !important;
  box-shadow: 0 8px 24px -6px rgba(2, 132, 199, 0.5) !important;
}
</style>
"""

def get_header_html():
    return """
    <div class="custom-header">
        <div class="header-left">
            <div class="logo-icon">WT-SVD</div>
            <div class="brand-name">HYBRID STUDIO</div>
        </div>
        <div class="header-menu">
            <span class="menu-item-active">DENOISING VISUALIZER</span>
            <span class="menu-item-disabled">MATRIX ANALYSIS</span>
            <span class="menu-item-disabled">DOCUMENTATION</span>
        </div>
        <div class="header-right">
            <span class="status-badge">v2.0 Hybrid</span>
        </div>
    </div>
    """

def get_direct_zoom_html(img_orig_b64, img_mod_b64, k):
    return f"""
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ background: transparent; font-family: 'Plus Jakarta Sans', sans-serif; overflow: hidden; }}
      .wrapper {{ display: flex; gap: 28px; width: 100%; align-items: flex-start; }}
      
      .panel {{ 
        flex: 1; 
        border-radius: 20px; 
        overflow: hidden; 
        background: white; 
        box-shadow: 0 20px 40px -10px rgba(15,23,42,0.06); 
        border: 1px solid #E2E8F0; 
        position: relative;
      }}
      .panel-header {{ 
        background: linear-gradient(90deg, #0284C7, #0369A1); 
        color: white; 
        padding: 16px 24px; 
        font-size: 0.9rem; 
        font-weight: 700; 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        position: relative;
        z-index: 10;
      }}
      .panel-header .badge {{ background: rgba(255,255,255,0.2); border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
      
      .view-container {{
        width: 100%;
        height: 420px;
        overflow: hidden;
        position: relative;
        background: #000;
        display: flex;
        align-items: center;
        justify-content: center;
      }}
      .view-container img {{
        display: block;
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        transform-origin: center center;
        cursor: grab;
      }}
      .view-container img:active {{ cursor: grabbing; }}
      
      .control-panel {{
        position: absolute;
        top: 70px;
        right: 16px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        z-index: 20;
      }}
      .control-btn {{
        width: 36px;
        height: 36px;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(4px);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s, transform 0.1s;
        user-select: none;
      }}
      .control-btn:hover {{ background: rgba(2, 132, 199, 0.9); }}
      .control-btn:active {{ transform: scale(0.95); }}
      .control-btn.reset-btn {{ font-size: 0.9rem; }}

      .panel-footer {{ padding: 14px 24px; font-size: 0.85rem; color: #64748B; background: #F8FAFC; border-top: 1px solid #E2E8F0; position: relative; z-index: 10; }}
    </style>

    <div class="wrapper">
      <div class="panel" id="panel-left">
        <div class="panel-header">Gambar Input (Noisy)</div>
        <div class="control-panel">
            <button class="control-btn" onclick="zoom('left', 0.2)">+</button>
            <button class="control-btn" onclick="zoom('left', -0.2)">-</button>
            <button class="control-btn reset-btn" onclick="resetZoom('left')">⟲</button>
        </div>
        <div class="view-container" id="view-left">
            <img src="data:image/png;base64,{img_orig_b64}" id="img-left" />
        </div>
        <div class="panel-footer"><span>Spasial Domain Awal</span></div>
      </div>

      <div class="panel" id="panel-right">
        <div class="panel-header">Hasil Rekonstruksi Hibrida <span class="badge">k = {k}</span></div>
        <div class="control-panel">
            <button class="control-btn" onclick="zoom('right', 0.2)">+</button>
            <button class="control-btn" onclick="zoom('right', -0.2)">-</button>
            <button class="control-btn reset-btn" onclick="resetZoom('right')">⟲</button>
        </div>
        <div class="view-container" id="view-right">
            <img src="data:image/png;base64,{img_mod_b64}" id="img-right" />
        </div>
        <div class="panel-footer"><span>Domain Inverse-DWT (LL Rank-{k} + Detail Filtered)</span></div>
      </div>
    </div>

    <script>
      const savedState = localStorage.getItem('svd_zoom_state');
      const state = savedState ? JSON.parse(savedState) : {{
          left: {{ scale: 1, tx: 0, ty: 0, isDragging: false, startX: 0, startY: 0 }},
          right: {{ scale: 1, tx: 0, ty: 0, isDragging: false, startX: 0, startY: 0 }}
      }};

      state.left.isDragging = false;
      state.right.isDragging = false;

      function saveState() {{
          localStorage.setItem('svd_zoom_state', JSON.stringify(state));
      }}

      function updateTransform(side) {{
          const img = document.getElementById('img-' + side);
          const s = state[side];
          img.style.transform = `translate(${{s.tx}}px, ${{s.ty}}px) scale(${{s.scale}})`;
          saveState();
      }}

      ['left', 'right'].forEach(side => updateTransform(side));

      function zoom(side, amount) {{
          state[side].scale = Math.max(0.5, Math.min(state[side].scale + amount, 6));
          updateTransform(side);
      }}

      function resetZoom(side) {{
          state[side].scale = 1;
          state[side].tx = 0;
          state[side].ty = 0;
          updateTransform(side);
      }}

      ['left', 'right'].forEach(side => {{
          const img = document.getElementById('img-' + side);
          const view = document.getElementById('view-' + side);

          img.addEventListener('mousedown', function(e) {{
              e.preventDefault();
              state[side].isDragging = true;
              state[side].startX = e.clientX - state[side].tx;
              state[side].startY = e.clientY - state[side].ty;
          }});

          window.addEventListener('mousemove', function(e) {{
              if (!state[side].isDragging) return;
              state[side].tx = e.clientX - state[side].startX;
              state[side].ty = e.clientY - state[side].startY;
              updateTransform(side);
          }});

          window.addEventListener('mouseup', () => {{ 
              if(state[side].isDragging) {{
                  state[side].isDragging = false;
                  saveState();
              }}
          }});
          
          view.addEventListener('wheel', function(e) {{
              e.preventDefault();
              const speed = e.deltaY < 0 ? 0.15 : -0.15;
              state[side].scale = Math.max(0.5, Math.min(state[side].scale + speed, 6));
              updateTransform(side);
          }}, {{ passive: false }});
      }});
    </script>
    """