import streamlit as st

# Top‑level minimal pour UI instantanée
st.title("Tile Compression Dashboard")
st.write("✅ L’interface est prête ! Cliquez sur ‘Init models’ pour charger le modèle.")

# Forcer le modèle ELIC (plus de BMSHJ2018)
elic_path = st.sidebar.text_input(
    "Chemin du modèle ELIC",
    "/home/appuser/models/ELIC_0150_ft_3980_Plateau.pth.tar"
)

# Upload des fichiers
st.sidebar.markdown("### Fichiers requis")
tile_file = st.sidebar.file_uploader("Tuile (TIFF 2 bandes)", type=["tif","tiff"])
mask_file = st.sidebar.file_uploader("Masque (TIFF)", type=["tif","tiff"])
orig_file = st.sidebar.file_uploader("Original (TIFF 2 bandes)", type=["tif","tiff"])
jar_file  = st.sidebar.file_uploader("Jar ComputeMetrics (.jar)", type=["jar"])

# Étape 1 : init models
if st.sidebar.button("Init models"):
    st.write("⏳ Import des bibliothèques lourdes…")
    import os, time, numpy as np, tempfile, subprocess
    import torch
    import tifffile
    from ELiC_ReImplemetation.Network import TestModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write("🖥️ Environnement :", device.upper())

    # Chargement du modèle ELIC
    st.write("⏳ Chargement du modèle ELIC…")
    t0 = time.time()
    model = TestModel(N=192, M=320, num_slices=5).to(device)
    ck = torch.load(elic_path, map_location=device)
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    model.load_state_dict(ck)
    model.eval(); model.update(force=True)
    st.write(f"✅ ELIC chargé en {time.time()-t0:.1f}s")

    # Étape 2 : exécution du pipeline
    if st.sidebar.button("Lancer compression/décompression"):
        st.write("▶️ Début du pipeline…")

        # Vérification des uploads
        if not all([tile_file, mask_file, orig_file, jar_file]):
            st.error("❗ Merci de téléverser tous les fichiers.")
            st.stop()

        # Sauvegarde temporaire
        st.write("💾 Sauvegarde des fichiers envoyés")
        tmp_tile = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff"); tmp_tile.write(tile_file.read()); tmp_tile.flush()
        tmp_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff"); tmp_mask.write(mask_file.read()); tmp_mask.flush()
        tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff"); tmp_orig.write(orig_file.read()); tmp_orig.flush()
        tmp_jar  = tempfile.NamedTemporaryFile(delete=False, suffix=".jar");  tmp_jar.write(jar_file.read());  tmp_jar.flush()

        # Lecture des images
        st.write("📖 Lecture des images")
        img2  = tifffile.imread(tmp_tile.name)
        orig2 = tifffile.imread(tmp_orig.name)
        mask  = tifffile.imread(tmp_mask.name)
        img   = np.stack([img2[...,0], img2[...,0], img2[...,1]], axis=2)

        # Compression & décompression
        st.write("🔧 Compression de la tuile…")
        t0 = time.time()
        tile_n, maxv = dynamic_normalize(img)
        tensor = torch.from_numpy(tile_n).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.compress(tensor)
        pkg = {"strings": out["strings"], "shape": out["shape"], "max": maxv.tolist()}
        st.write(f"✅ Compression en {time.time()-t0:.2f}s")

        st.write("🔧 Décompression de la tuile…")
        t1 = time.time()
        with torch.no_grad():
            out2 = model.decompress(pkg["strings"], pkg["shape"])
        rec = out2["x_hat"].squeeze(0).permute(1,2,0).cpu().numpy()
        rec = dynamic_denormalize(rec, np.array(pkg["max"], dtype=np.float32))
        st.write(f"✅ Décompression en {time.time()-t1:.2f}s")

        # Calcul taille compressée
        st.write("📦 Calcul de la taille compressée")
        def _sum_bytes(obj):
            if isinstance(obj,(bytes,bytearray)): return len(obj)
            if isinstance(obj,(list,tuple)): return sum(_sum_bytes(x) for x in obj)
            return 0
        csize = _sum_bytes(pkg["strings"])
        st.write(f"Taille compressée : {csize} octets")

        # Préparer pour JAR
        st.write("💾 Écriture TIFF 2-canaux pour JAR")
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff").name
        tifffile.imwrite(tmp_out, rec[..., [0,2]].astype(np.uint16),
                         planarconfig="CONTIG", photometric="MINISBLACK", tile=(512,512))

        # Exécuter le JAR
        st.write("☕ Exécution du JAR ComputeMetrics")
        cmd = ["java","-jar", tmp_jar.name, "-a", tmp_out, "-b", tmp_mask.name, "-i", tmp_orig.name]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        st.text_area("Sortie JAR", res.stdout, height=200)

        # Extraction et affichage des métriques
        st.write("📊 Extraction des métriques")
        lines = [l for l in res.stdout.splitlines() if l and l[0].isdigit()]
        cols = ["band","non_bg_px","bg_px","total_px","mse","rmse","psnr","ssim","ssim_win","ssim_k1","ssim_k2","dyn_range"]
        df = [dict(zip(cols, l.split(","))) for l in lines]
        st.subheader("Métriques")
        st.table(df)

        st.success("✅ Pipeline terminé")

# Fonctions utilitaires
def dynamic_normalize(img):
    max_vals = img.max(axis=(0,1))
    max_vals[max_vals==0] = 1.0
    return img.astype(np.float32)/max_vals, max_vals

def dynamic_denormalize(img, max_vals):
    return img*max_vals
