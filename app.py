import os
import base64
from datetime import datetime

import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

# =========================
# CONFIG GÉNÉRALE
# =========================
st.set_page_config(
    page_title="Modélisation des accidents mortels (Open data BAAC)",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "Input_Site_Web")

# ---------- FICHIERS D'ENTRÉE ----------
EDA_PDF_PATH = os.path.join(INPUT_DIR, "EDA_double_axes_propre.pdf")

HEX_CORPO_PATH = os.path.join(INPUT_DIR, "hexbin_corporels_fond_all.png")
HEX_MORT_PATH = os.path.join(INPUT_DIR, "hexbin_mortels_fond_all.png")

CHORO_HTML_PATH = os.path.join(INPUT_DIR, "taux_mortels_departements_numDep.html")

TABLE_S0_PATH = os.path.join(INPUT_DIR, "table_S0_in_memory.png")
TABLE_S1_PATH = os.path.join(INPUT_DIR, "table_S1_in_memory.png")

# Graphiques de métriques & courbes PR/ROC
PERF_HTML = {
    "S0 – Barres (métriques @ t*)": os.path.join(INPUT_DIR, "BAR_S0_baseline.html"),
    "S1 – Barres (métriques @ t*)": os.path.join(INPUT_DIR, "BAR_S1_spatial.html"),
}

PERF_PNG = {
    "S0 – Courbe PR (Precision–Recall)": os.path.join(INPUT_DIR, "PR_S0_baseline.png"),
    "S0 – Courbe ROC":                   os.path.join(INPUT_DIR, "ROC_S0_baseline.png"),
    "S1 – Courbe PR (Precision–Recall)": os.path.join(INPUT_DIR, "PR_S1_spatial.png"),
    "S1 – Courbe ROC":                   os.path.join(INPUT_DIR, "ROC_S1_spatial.png"),
}


GAINS_HTML_PATH = os.path.join(INPUT_DIR, "mini_dashboard_gains.html")
BEST_MODELS_HTML_PATH = os.path.join(INPUT_DIR, "best_models_report_in_memory.html")

SHAP_IMAGES = {
    "S0 – Baseline": {
        "beeswarm": os.path.join(INPUT_DIR, "S0_lgbm_shap_beeswarm.png"),
        "bar":       os.path.join(INPUT_DIR, "S0_lgbm_shap_bar.png"),
    },
    "S1 – Géographique": {
        "beeswarm": os.path.join(INPUT_DIR, "S1_lgbm_shap_beeswarm.png"),
        "bar":       os.path.join(INPUT_DIR, "S1_lgbm_shap_bar.png"),
    }
}

DIST_MORT_PATH = os.path.join(INPUT_DIR, "dist_is_mortel_all.png")

# =========================
# HELPERS
# =========================

def load_img(path):
    if os.path.exists(path):
        return Image.open(path)
    return None


def show_html(path, height=600, label_if_missing=None):
    """Affiche un HTML local (Plotly, choroplèthe, métriques…)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            html_str = f.read()
        components.html(html_str, height=height, scrolling=True)
    else:
        label = label_if_missing or os.path.basename(path)
        st.warning(f"Fichier HTML non trouvé : `{label}`\n\nChemin attendu : `{path}`")


def show_pdf(path, height=900):
    """Affiche un PDF + bouton de téléchargement."""
    if not os.path.exists(path):
        st.warning(f"PDF non trouvé : `{path}`")
        return

    with open(path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="📥 Télécharger le rapport EDA complet (PDF)",
        data=pdf_bytes,
        file_name=os.path.basename(path),
        mime="application/pdf",
    )

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
                width="100%" height="{height}" type="application/pdf">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à :",
    [
        "🏠 Accueil",
        "⚖️ Déséquilibre de la cible",
        "📊 EDA – variables explicatives",
        "🗺️ Cartographie",
        "🤖 Modélisation & SHAP"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Projet pour Certificat Data Science - CEPE ENSAE-ENSAI**")
st.sidebar.markdown("*Auteur : Tonakpon Karl ATTAKPA kattakpa@yahoo.fr*")
st.sidebar.markdown(f"*Dernière MAJ affichée :* {datetime.now():%d/%m/%Y}")

# =========================
# PAGE 1 – ACCUEIL
# =========================
if page == "🏠 Accueil":
    st.title("Modélisation des accidents mortels (Open data BAAC)")
    st.write("")
    st.write("")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### 🎯 Objectif

            Prédire la **probabilité qu’un accident corporel soit mortel** à partir des données BAAC.

            - Cible binaire : `is_mortel` (accident mortel vs non mortel)  
            - Construction de variables explicatives :
              contexte de l'accident (type de route, luminosité, type de collision, …),
              profil des usagers (âge moyen / min/max, proportion d’hommes, nombre d’usagers, conducteurs, piétons, …),
              caractéristiques géographiques (latitude,longitude, commune, département, …)
            - Analyse exploratoire (EDA)  
            - Classification supervisée (plusieurs familles de modèles)  
            - Interprétabilité via **SHAP values**
            """
        )

    with col2:
        st.info(
            """
            **Contenu du mini-site :**
            - Déséquilibre de la cible  
            - EDA (double-axes)  
            - Cartographie (choroplèthe + hexbins)  
            - Résultats de modélisation & SHAP  
            """
        )

    st.markdown("---")
    st.markdown(
        """
        #### Données

        - Source : Bases de données annuelles des accidents corporels de la circulation routière – BAAC  
          (fichiers Caractéristiques, Lieux, Usagers)  
        - Unité : **accident corporel**  
        - Période : **2015–2023**
        """
    )

# =========================
# PAGE 2 – DÉSÉQUILIBRE CIBLE
# =========================
elif page == "⚖️ Déséquilibre de la cible":
    st.title("⚖️ Déséquilibre de la variable cible `is_mortel`")
    st.write("")

    st.markdown(
        """
        La cible `is_mortel` est fortement déséquilibrée :  
        les accidents mortels représentent une **très faible proportion** de l’ensemble des accidents corporels.
        """
    )

    st.write("")  # petit espace avant le graphe

    img = load_img(DIST_MORT_PATH)
    if img is not None:
        # On centre et on réduit visuellement à ~60% de la largeur via des colonnes
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.image(
                img,
                caption="Distribution des accidents corporels (mortels vs non mortels)",
                use_container_width=True,
            )
    else:
        st.warning(f"Image non trouvée : `{DIST_MORT_PATH}`.")

    st.write("")  # petit espace après le graphe

    st.markdown(
        """
        Conséquences pratiques :

        - on privilégie des métriques adaptées aux classes rares (AUC, **AP**/PR, F1, Brier),  
        - on surveille particulièrement le **rappel** sur la classe minoritaire (`is_mortel = 1`),  
        - on teste des variantes ré-équilibrées (pondération, SMOTE, etc.).
        """
    )

# =========================
# PAGE 3 – EDA
# =========================
elif page == "📊 EDA – variables explicatives":
    st.title("📊 Analyse exploratoire (EDA) – Variables explicatives")
    st.write("")

    st.markdown(
        """
        Les graphes EDA (double-axes) sont regroupés dans un **rapport unique** :  

        - barres : nombre d’accidents  
        - courbe : taux d’accidents mortels (proportion d’`is_mortel = 1`)  
        """
    )

    st.subheader("Rapport EDA complet")
    show_pdf(EDA_PDF_PATH, height=900)

    st.markdown(
        """
        Ces figures permettent d’identifier les **contextes les plus accidentogènes**
        et ceux où la **gravité (mortalité)** est particulièrement forte :
        type de route, luminosité, type de collision, profils d’âge, etc.
        """
    )

# =========================
# PAGE 4 – CARTOGRAPHIE
# =========================
elif page == "🗺️ Cartographie":
    st.title("🗺️ Cartographie des accidents")
    st.write("")

    st.subheader("4.1 Choroplèthe – taux d’accidents mortels par département")

    st.markdown(
        """
        Une **carte choroplèthe** colore chaque département en fonction d’une **valeur numérique** :
        ici, le **taux d’accidents mortels** observé sur la période.

        - les teintes les plus foncées correspondent aux départements où la part d’accidents mortels
          est la plus élevée,  
        - les teintes plus claires indiquent des taux plus faibles.
        """
    )

    show_html(CHORO_HTML_PATH, height=650,
              label_if_missing="taux_mortels_departements_numDep.html")

    st.markdown(
        """
        On observe notamment :

        - des départements avec une **concentration plus forte** d’accidents mortels
          dans certaines zones du territoire (par ex. certains départements du nord-est,
          du centre ou du sud-ouest),  
        - des contrastes entre départements voisins qui suggèrent un rôle de la **structure du réseau routier**,
          des vitesses pratiquées ou d’autres facteurs locaux.
        """
    )

    st.markdown("---")
    st.subheader("4.2 Densité géographique (hexbin)")

    st.markdown(
        """
        Les cartes **hexbin** représentent la **densité d’accidents** dans l’espace :

        - chaque hexagone agrège les accidents tombant dans la cellule,  
        - la couleur reflète le **logarithme du nombre d’accidents** (`log(N)`),
          ce qui permet de visualiser à la fois les zones très denses et les zones plus diffuses.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        img_hex_corp = load_img(HEX_CORPO_PATH)
        if img_hex_corp is not None:
            st.image(
                img_hex_corp,
                caption="Densité d’accidents corporels (log N)",
                use_container_width=True,
            )
        else:
            st.warning(f"Image non trouvée : `{HEX_CORPO_PATH}`.")

    with col2:
        img_hex_mort = load_img(HEX_MORT_PATH)
        if img_hex_mort is not None:
            st.image(
                img_hex_mort,
                caption="Densité d’accidents mortels (log N)",
                use_container_width=True,
            )
        else:
            st.warning(f"Image non trouvée : `{HEX_MORT_PATH}`.")

    st.markdown(
        """
        Lecture croisée :

        - la carte des **accidents corporels** fait ressortir les zones de trafic intense
          (grandes agglomérations, axes structurants),  
        - la carte des **accidents mortels** met en avant certaines zones périurbaines ou rurales,
          où la vitesse pratiquée et la configuration des infrastructures peuvent conduire
          à une mortalité plus élevée.

        L’enjeu de la modélisation sera d’exploiter cette information **géographique**
        en complément des variables locales (type de route, luminosité, profils d’usagers, etc.).
        """
    )

# =========================
# PAGE 5 – MODÉLISATION & SHAP
# =========================
elif page == "🤖 Modélisation & SHAP":
    st.title("🤖 Modélisation & interprétabilité (SHAP)")
    st.write("")

    st.markdown(
        """
        Deux scénarios de modélisation sont comparés :  

        - **S0_baseline** : sans variable géographique agrégée,  
        - **S1_géographique** : avec la variable synthétique `taux_mortels_dep_feature`
          (taux d’accidents mortels par département).

        Pour chaque scénario, plusieurs familles de modèles de classification sont évaluées
        (régression logistique, variantes pondérées/SVOTE, Random Forest, **LGBM**, XGBoost…),
        avec recherche d’hyperparamètres et calcul d’un seuil optimal `t*` par validation croisée.
        """
    )

    # --- 5.1 Tables de métriques (toutes variantes) ---
    st.markdown("### 5.1 Tables de métriques – toutes variantes")

    col1, col2 = st.columns(2)

    with col1:
        img_s0 = load_img(TABLE_S0_PATH)
        if img_s0 is not None:
            st.image(img_s0, caption="Tableau métriques – S0_baseline", use_container_width=True)
        else:
            st.warning(f"Tableau S0 non trouvé : `{TABLE_S0_PATH}`.")

    with col2:
        img_s1 = load_img(TABLE_S1_PATH)
        if img_s1 is not None:
            st.image(img_s1, caption="Tableau métriques – S1_géographique", use_container_width=True)
        else:
            st.warning(f"Tableau S1 non trouvé : `{TABLE_S1_PATH}`.")

    st.markdown(
        """
        Les lignes avec `seuil = t*` correspondent aux **seuils optimaux** déterminés en OOF
        (maximisation du F1) et servent de base à la comparaison des variantes
        sur les métriques AUC, **AP** (Average Precision / aire sous la courbe PR),
        F1, Precision, Recall et Brier.
        """
    )

    # --- 5.2 Best modèles S0 / S1 – barres & courbes PR / ROC ---
    st.markdown(
        """
        <h3 style="margin-top:20px; margin-bottom:5px;">
            5.2 Best modèles S0 / S1 – barres & courbes PR / ROC
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Les graphiques ci-dessous présentent les **meilleurs modèles** de chaque scénario  
        (ici : LGBM pour S0_baseline et S1_géographique) :

        - **Graphique barres** : comparaison des métriques globales (AP, AUC, F1, Precision, Recall) au seuil `t*`,  
        - **Courbes PR / ROC** : analyse fine de la capacité de discrimination sur la classe `is_mortel = 1`.
        """
    )

    # ========================================================
    # 🔹 5.2.1 — BARRES (métriques globales)
    # ========================================================
    st.subheader("Graphiques barres (métriques @ t*)")

    choix_barres = st.selectbox(
        "Sélectionner un scénario pour les métriques globales :",
        ["S0 – Barres (métriques @ t*)", "S1 – Barres (métriques @ t*)"],
        key="barres_selector"
    )

    barres_path = PERF_HTML[choix_barres]
    show_html(barres_path, height=560, label_if_missing=os.path.basename(barres_path))

    st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    # ========================================================
    # 🔹 5.2.2 — COURBES PR / ROC (PNG)
    # ========================================================
    st.subheader("Courbes PR / ROC")

    choix_courbes = st.selectbox(
        "Sélectionner une courbe PR / ROC :",
        list(PERF_PNG.keys()),
        key="courbes_selector"
    )

    courbe_path = PERF_PNG[choix_courbes]

    courbe_img = load_img(courbe_path)
    if courbe_img is not None:
        st.image(courbe_img, caption=choix_courbes, width="stretch")
    else:
        st.warning(f"Image non trouvée : `{courbe_path}`")

    st.markdown(
        """
        Ces courbes montrent que les modèles **LGBM** sont les plus performants dans les deux scénarios,
        avec un meilleur rappel des accidents mortels et une discrimination plus stable aux différents seuils.
        """,
        unsafe_allow_html=True
    )



    # --- 5.3 Hyperparamètres des best modèles ---
    st.markdown("### 5.3 Hyperparamètres des best modèles")

    show_html(BEST_MODELS_HTML_PATH, height=500, label_if_missing="best_models_report_in_memory.html")

    st.markdown(
        """
        Pour les deux scénarios, le best modèle retenu est un **LGBMClassifier** avec :

        - profondeur modérée et nombre de feuilles suffisant pour modéliser des interactions
          (route × contexte de l’accident × profils des usagers),  
        - taux d’apprentissage relativement faible (`learning_rate`) compensé par un nombre
          d’arbres plus élevé (`n_estimators`),  
        - régularisation et sous-échantillonnage de features (`feature_fraction`) permettant
          de limiter la variance et d’éviter un sur-apprentissage excessif.

        Ces réglages sont cohérents avec un contexte de **classification déséquilibrée**
        où l’on souhaite capturer des signaux fins sans surexploiter le bruit.
        """
    )

    # --- 5.4 Gains relatifs S1 vs S0 ---
    st.markdown("### 5.4 Gains relatifs S1 vs S0")

    show_html(GAINS_HTML_PATH, height=560, label_if_missing="mini_dashboard_gains.html")

    st.markdown(
        """
        Le mini-dashboard met en évidence les **gains relatifs (%)** du scénario S1_géographique
        par rapport à S0_baseline sur les principales métriques :

        - amélioration du **rappel** et du **F1** sur la classe mortelle,  
        - léger gain en **AP** et **AUC**,  
        - baisse du **Brier score** (meilleure calibration des probabilités).

        Concrètement, l’ajout de `taux_mortels_dep_feature` permet au modèle de mieux
        discriminer les situations à **risque mortel élevé**, tout en restant bien calibré.
        """
    )

    # --- 5.5 SHAP – importance globale des variables ---
    st.markdown("### 5.5 SHAP – importance globale des variables")

    st.markdown(
        """
        Les graphiques ci-dessous montrent, pour chaque scénario :

        - un **beeswarm SHAP** : dispersion des impacts individuels de chaque variable
          (un point = un accident),  
        - un **SHAP bar** : importance globale des variables via la moyenne de |SHAP|
          (impact moyen sur la log-odds de l’issue mortelle).
        """
    )

    choix_shap = st.selectbox("Scénario SHAP :", list(SHAP_IMAGES.keys()))
    paths = SHAP_IMAGES[choix_shap]

    col_bsw, col_bar = st.columns(2)

    with col_bsw:
        img_bsw = load_img(paths["beeswarm"])
        if img_bsw is not None:
            st.image(img_bsw, caption=f"{choix_shap} – SHAP beeswarm", use_container_width=True)
        else:
            st.warning(f"Image beeswarm non trouvée : `{paths['beeswarm']}`.")

    with col_bar:
        img_bar = load_img(paths["bar"])
        if img_bar is not None:
            st.image(img_bar, caption=f"{choix_shap} – SHAP bar (|SHAP| moyen)", use_container_width=True)
        else:
            st.warning(f"Image bar non trouvée : `{paths['bar']}`.")

    st.markdown(
        """
        #### Comment lire le beeswarm SHAP ?

        - chaque **point** représente un accident,  
        - la **position horizontale** indique l’impact SHAP de la variable sur la probabilité
          d’accident mortel (à droite → contribution positive, à gauche → contribution négative),  
        - la **couleur** encode la valeur de la variable : bleu = valeur faible, rouge = valeur élevée.

        En combinant couleur et position, on voit par exemple si des valeurs élevées d’une variable
        poussent la probabilité vers le haut ou vers le bas.
        """
    )

    st.markdown(
        """
        #### Exemple d’interprétation – scénario S0 (baseline)

        - **`agg` (en / hors agglomération)**  
          Les modalités `agg=Hors_agglomération` apparaissent surtout avec des SHAP
          positifs, alors que `agg=En_agglomération` est plus proche de 0 voire négatif.
          Le modèle apprend donc que, toutes choses égales par ailleurs, un accident
          **hors agglomération** a plus de chances d’être mortel.

        - **`pct_hommes` (proportion d’hommes impliqués)**  
          Dans le beeswarm S0, les points rouges (forte proportion d’hommes) se situent
          préférentiellement à droite de l’axe 0, tandis que les valeurs faibles
          sont plutôt neutres ou négatives.  
          Le modèle associe donc une forte proportion d’hommes à une **augmentation
          de la probabilité d’accident mortel**, ce qui est cohérent avec la littérature
          en accidentologie (vitesse, comportements à risque, etc.).
        """
    )

    st.markdown(
        """
        #### Exemple d’interprétation – scénario S1 (géographique)

        - **`taux_mortels_dep_feature`**  
          Dans S1, cette variable arrive clairement en tête du graphique SHAP bar.
          Sur le beeswarm, les accidents situés dans des départements à
          **taux historique de mortalité élevé** (points rouges) ont des SHAP
          nettement positifs, alors que ceux issus de départements à taux faible
          (points bleus) ont des impacts proches de 0 ou négatifs.

          Le modèle utilise donc ce taux départemental comme un **a priori géographique de risque** :
          à exposition individuelle comparable, un accident survenant dans un
          département historiquement plus “mortel” reçoit une probabilité prédite
          plus élevée d’être mortel.

          Les autres variables structurelles (type de route `catr`, type de collision `col`,
          luminosité `lum`, structure d’âge, etc.) restent contributives dans les deux scénarios,
          mais l’ajout de `taux_mortels_dep_feature` dans S1 renforce clairement la capacité
          du modèle à discriminer les situations les plus à risque, ce qui est cohérent
          avec les gains observés entre S0 et S1.
        """
    )
