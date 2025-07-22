import os
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------
# 0. Clonage dynamique du dépôt ELiC_ReImplemetation s’il n’existe pas
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SRC_DIR  = BASE_DIR / "ELiC_ReImplemetation"

if not SRC_DIR.exists():
    # Clone en depth=1 pour gagner du temps
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/VincentChandelier/ELiC-ReImplemetation.git",
        str(SRC_DIR)
    ], check=True)

# Ajout au PYTHONPATH pour pouvoir importer Network.py
sys.path.insert(0, str(SRC_DIR))

# À présent l’import ne lèvera plus d’erreur
from Network import TestModel
from compressai.zoo import bmshj2018_factorized

import streamlit as st
import torch
import tifffile
import numpy as np
import tempfile
import time
import subprocess
# —————————————————————————————
# 1. Caching des modèles
# —————————————————————————————
@st.cache_resource
def load_elic_model(model_path, device):
    model = TestModel(N=192, M=320, num_slices=5).to(device)
    ck = torch.load(model_path, map_location=device)
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    model.load_state_dict(ck)
    model.eval(); model.update(force=True)
    return model

@st.cache_resource
def load_bmshj2018_model(quality, device):
    model = bmshj2018_factorized(quality=quality, metric='mse', pretrained=True).to(device)
    model.eval()
    return model

# —————————————————————————————
# 2. Normalisation
# —————————————————————————————
def dynamic_normalize(img):
    max_vals = img.max(axis=(0, 1))
    max_vals[max_vals == 0] = 1.0
    return img.astype(np.float32) / max_vals, max_vals

def dynamic_denormalize(img, max_vals):
    return img * max_vals

# —————————————————————————————
# 3. Compression / Décompression
# —————————————————————————————
def compress_tile(tile, model, device):
    tile_n, maxv = dynamic_normalize(tile)
    tensor = torch.from_numpy(tile_n).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.compress(tensor)
    return {"strings": out["strings"], "shape": out["shape"], "max": maxv.tolist()}

def decompress_tile(pkg, model, device):
    with torch.no_grad():
        out = model.decompress(pkg["strings"], pkg["shape"])
    tile = out["x_hat"].squeeze(0).permute(1,2,0).cpu().numpy()
    maxv = np.array(pkg["max"], dtype=np.float32)
    return dynamic_denormalize(tile, maxv)

# —————————————————————————————
# 4. Dashboard
# —————————————————————————————
st.title("Tile Compression Dashboard")
device = 'cpu'  # vous avez demandé CPU-only

# Sélection du modèle
type_model = st.sidebar.selectbox("Choisir le modèle", ["ELIC", "BMSHJ2018"])
if type_model == "ELIC":
    default_model = BASE_DIR / "compute-resources" / "ELIC_0150_ft_3980_Plateau.pth.tar"
    elic_path = st.sidebar.text_input("Chemin du modèle ELIC", value=str(default_model))
    if not os.path.isfile(elic_path):
        st.sidebar.error(f"Fichier introuvable: {elic_path}")
        st.stop()
    model = load_elic_model(elic_path, device)
else:
    quality = st.sidebar.selectbox("Qualité BMSHJ2018", [6, 4, 2])
    model = load_bmshj2018_model(quality, device)

# Upload des fichiers
tile_file = st.sidebar.file_uploader("Tuile pré-traitée (TIFF 2 bandes)", type=["tif","tiff"])
mask_file = st.sidebar.file_uploader("Masque de fond (TIFF)", type=["tif","tiff"])
orig_file = st.sidebar.file_uploader("Image originale (TIFF 2 bandes)", type=["tif","tiff"])
jar_file  = st.sidebar.file_uploader("Jar ComputeMetrics (.jar)", type=["jar"])

if st.sidebar.button("Lancer compression/décompression"):
    if not (tile_file and mask_file and orig_file and jar_file):
        st.error("Veuillez téléverser tuile, masque, image originale et JAR.")
        st.stop()

    # Création de temporaires
    tmp_tile = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff"); tmp_tile.write(tile_file.read()); tmp_tile.flush()
    tmp_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff"); tmp_mask.write(mask_file.read()); tmp_mask.flush()
    tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff"); tmp_orig.write(orig_file.read()); tmp_orig.flush()
    tmp_jar  = tempfile.NamedTemporaryFile(delete=False, suffix=".jar");  tmp_jar.write(jar_file.read());  tmp_jar.flush()

    # Lecture & mise en forme
    img2  = tifffile.imread(tmp_tile.name)
    orig2 = tifffile.imread(tmp_orig.name)
    mask  = tifffile.imread(tmp_mask.name)
    # reconstruire en 3 canaux pour le modèle
    img = np.stack([img2[...,0], img2[...,0], img2[...,1]], axis=2)

    # Compression / décompression torch
    t0 = time.time()
    pkg = compress_tile(img, model, device)
    t1 = time.time()
    rec = decompress_tile(pkg, model, device)
    t2 = time.time()

    # Calcul taille compressée
    def _sum_bytes(obj):
        if isinstance(obj, (bytes, bytearray)): return len(obj)
        if isinstance(obj, (list, tuple)): return sum(_sum_bytes(x) for x in obj)
        return 0
    csize = _sum_bytes(pkg["strings"])

    # Sauvegarde 2-bandes pour JAR
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff").name
    tifffile.imwrite(
        tmp_out,
        rec[..., [0,2]].astype(np.uint16),
        planarconfig="CONTIG",
        photometric="MINISBLACK",
        tile=(512,512)
    )

    # Appel JAR ComputeMetrics
    cmd = ["java", "-jar", tmp_jar.name,
           "-a", tmp_out,
           "-b", tmp_mask.name,
           "-i", tmp_orig.name]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = [l for l in res.stdout.splitlines() if l and l[0].isdigit()]

    # Extraction & affichage métriques
    cols = ["band","non_bg_px","bg_px","total_px","mse","rmse","psnr","ssim",
            "ssim_win","ssim_k1","ssim_k2","dyn_range"]
    df = [dict(zip(cols, l.split(","))) for l in lines]
    comp_factor = (orig2.size * orig2.dtype.itemsize * orig2.shape[2]) / csize

    filtered = [{
        "RMSE": row["rmse"],
        "PSNR": row["psnr"],
        "SSIM": row["ssim"],
        "Facteur de compression": comp_factor
    } for row in df]

    st.subheader("Métriques sélectionnées")
    st.dataframe(filtered)

    st.write({
        "Taille compressée (octets)": csize,
        "Temps compression (s)": round(t1-t0,3),
        "Temps décompression (s)": round(t2-t1,3)
    })

    # Visualisation
    def normalize_for_display(img):
        p2, p98 = np.percentile(img, (2, 98))
        return np.clip((img.astype(np.float32)-p2)/(p98-p2),0,1)
    st.subheader("Visualisations")
    st.image(normalize_for_display(orig2[...,0]), caption="Original bande 0")
    st.image(normalize_for_display(orig2[...,1]), caption="Original bande 1")
    st.image(normalize_for_display(rec[...,0]), caption="Reconstruit bande 0")
    st.image(normalize_for_display(rec[...,1]), caption="Reconstruit bande 1")

    # Cleanup
    for f in [tmp_tile.name, tmp_mask.name, tmp_orig.name, tmp_out, tmp_jar.name]:
        try: os.remove(f)
        except: pass
