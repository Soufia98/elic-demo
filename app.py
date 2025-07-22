import streamlit as st
from pathlib import Path
import subprocess

# —————————————————————————————
# 1. Chemins relatifs vers vos ressources
# —————————————————————————————
BASE = Path(__file__).parent / "compute-resources"
MODEL_PATH_ELIC = BASE / "ELIC_0150_ft_3980_Plateau.pth.tar"
JAR_PATH         = BASE / "ComputeMetrics.jar"

# —————————————————————————————
# 2. Interface utilisateur
# —————————————————————————————
st.title("Tile Compression Dashboard")
st.write("✅ L’interface est prête ! Cliquez sur ‘Init models’ pour charger le modèle.")

# Sélecteur de modèle
type_model = st.sidebar.selectbox("Choisir le modèle", ["ELIC", "BMSHJ2018"])
elic_path  = None

if type_model == "ELIC":
    elic_path = st.sidebar.text_input(
        "Chemin du modèle ELIC",
        str(MODEL_PATH_ELIC)
    )
else:
    elic_path = st.sidebar.text_input(
        "Chemin du modèle BMSHJ2018",
        ""  # adaptez si vous avez un checkpoint BMSHJ2018
    )

# Bouton de chargement
if st.button("Init models"):
    # Exemple de chargement PyTorch CPU
    import torch
    model = torch.load(elic_path, map_location="cpu")
    st.success(f"Modèle chargé depuis : {elic_path}")

# Téléversement et appel du JAR
uploaded_img  = st.file_uploader("Téléversez une tuile", type=["png","jpg","tif"])
uploaded_mask = st.file_uploader("Téléversez un masque", type=["png","jpg","tif"])
if uploaded_img and uploaded_mask:
    img_path  = "./temp_input.png"
    mask_path = "./temp_mask.png"
    # Sauvegarde locale
    with open(img_path,  "wb") as f: f.write(uploaded_img.getbuffer())
    with open(mask_path, "wb") as f: f.write(uploaded_mask.getbuffer())

    # Exécution du .jar ComputeMetrics
    try:
        out = subprocess.check_output([
            "java", "-jar", str(JAR_PATH),
            "--model", str(MODEL_PATH_ELIC),
            "--input", img_path,
            "--mask",  mask_path
        ], stderr=subprocess.STDOUT).decode("utf-8")
        st.text("Résultat Java :\n" + out)
    except subprocess.CalledProcessError as e:
        st.error("Erreur lors de l’exécution Java :\n" + e.output.decode("utf-8"))

# —————————————————————————————
# 3. Fonctions utilitaires (inchangées)
# —————————————————————————————
def dynamic_normalize(img):
    max_vals = img.max(axis=(0,1))
    max_vals[max_vals==0] = 1.0
    return img.astype("float32")/max_vals, max_vals

def dynamic_denormalize(img, max_vals):
    return img * max_vals
