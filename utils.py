import numpy as np
from PIL import Image
import base64
import io

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

def reconstruct(U, s, Vt, k, m, n):
    rank = len(s)
    if k <= rank:
        U_k  = U[:, :k]
        s_k  = s[:k]
        Vt_k = Vt[:k, :]
    else:
        pad  = k - rank
        U_k  = np.column_stack([U, np.zeros((m, pad))])
        s_k  = np.concatenate([s, np.zeros(pad)])
        Vt_k = np.vstack([Vt, np.zeros((pad, n))])
    A_rec = U_k @ np.diag(s_k) @ Vt_k
    return np.clip(A_rec, 0, 255).astype(np.uint8), U_k, s_k, Vt_k