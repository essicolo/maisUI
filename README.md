# 🌽 maisUI - Interface Web de Prédiction du Rendement du Maïs

Application web interactive en français pour prédire le rendement du maïs en fonction de la dose d'azote et des caractéristiques agronomiques.

## 🎯 Fonctionnalités

- ✅ **Interface dynamique** : Génération automatique des contrôles à partir du schéma du modèle
- ✅ **Multilangue** : Interface complète en français
- ✅ **Visualisation interactive** : Courbes de réponse dose-rendement avec Plotly
- ✅ **Upload CSV** : Support des séries temporelles météo via drag-and-drop
- ✅ **Optimisation économique** : Calcul automatique de la dose optimale
- ✅ **Export HTML** : Rapports auto-contenus sans dépendances CDN
- ✅ **Déploiement Fly.io** : Configuration prête pour production

## 📋 Prérequis

- Python 3.11+
- Modèle entraîné (`.keras`) et préprocesseur (`.pkl`) depuis `mais-npk`

## 🚀 Installation rapide

### 1. Cloner et installer les dépendances

```bash
cd maisUI
pip install -r requirements.txt
```

### 2. Copier les fichiers du modèle

Depuis le dépôt `mais-npk` :

```bash
# Copier le modèle et préprocesseur les plus récents
cp ../mais-npk/data/models/baseline_model_*.keras ./models/
cp ../mais-npk/data/models/baseline_preprocessor_*.pkl ./models/
```

### 3. Lancer l'application

```bash
python app.py --model models/baseline_model_*.keras \
              --preprocessor models/baseline_preprocessor_*.pkl \
              --port 7860
```

L'application sera disponible sur `http://localhost:7860`

## 🔧 Utilisation

### Interface web

1. **Onglet "Caractéristiques statiques"**
   - Latitude, longitude
   - pH du sol, matière organique
   - Texture du sol (ILR1, ILR2)
   - Type de travail du sol
   - Culture précédente
   - Densité de semis

2. **Onglet "Données météo" (optionnel)**
   - Upload CSV pré-semis (30 jours)
   - Upload CSV saison de croissance (200 jours)
   - Format attendu : colonnes = variables météo, lignes = jours

3. **Onglet "Azote et économie"**
   - Plage de doses à tester (0-300 kg N/ha par défaut)
   - Prix de l'azote ($/kg N)
   - Prix du grain ($/kg)

4. **Résultats**
   - Courbe de réponse interactive (Plotly)
   - Dose optimale recommandée
   - Revenu net estimé
   - Tableau récapitulatif des entrées
   - Export HTML complet

### API programmatique

```python
from model_wrapper import MaizeYieldPredictor
import pandas as pd

# Charger le modèle
predictor = MaizeYieldPredictor(
    model_path="models/baseline_model.keras",
    preprocessor_path="models/baseline_preprocessor.pkl"
)

# Inspecter le schéma des features
schema = predictor.describe_features()
print(schema)

# Prédire une courbe de réponse
features = {
    "latitude": 46.5,
    "longitude": -71.5,
    "ph_eau": 6.5,
    "som_log_ratio": -3.4,
    "soil_ilr1": 0.0,
    "soil_ilr2": 0.0,
    "tillage_ord": 1,
    "prev_crop_n": 1,
    "density_norm": 0.0
}

nitrogen_range = list(range(0, 301, 25))
dose_response = predictor.predict_response_curve(
    features=features,
    nitrogen_range=nitrogen_range
)

print(dose_response)

# Trouver la dose optimale
optimal = predictor.find_optimal_nitrogen(
    dose_response,
    n_price=1.5,
    grain_price=0.20
)

print(f"Dose optimale: {optimal['optimal_n_kg_ha']} kg N/ha")
print(f"Rendement: {optimal['predicted_yield_t_ha']} t/ha")
```

## 🐳 Déploiement Docker

### Build local

```bash
docker build -t maisui .
docker run -p 8080:8080 \
    -v $(pwd)/models:/app/models \
    maisui
```

### Déploiement Fly.io

1. **Installer Fly CLI**
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login**
```bash
flyctl auth login
```

3. **Créer l'application**
```bash
flyctl launch
# Suivez les instructions (région: Montreal/yul recommandée)
```

4. **Copier les fichiers du modèle**

Option A: Volume persistant
```bash
flyctl volumes create model_data --region yul --size 10
flyctl ssh console
# Dans le container:
# Copiez vos fichiers .keras et .pkl dans /app/models/
```

Option B: Secrets d'environnement
```bash
flyctl secrets set MODEL_PATH=/app/models/model.keras
flyctl secrets set PREPROCESSOR_PATH=/app/models/preprocessor.pkl
```

5. **Déployer**
```bash
flyctl deploy
```

6. **Ouvrir l'application**
```bash
flyctl open
```

## 📊 Format des fichiers CSV météo

### Pré-semis (30 jours avant semis)

| date       | temperature_2m_mean | precipitation_sum | soil_moisture_0_to_7cm_mean | ... |
|------------|--------------------:|------------------:|----------------------------:|-----|
| 2023-04-01 | 8.5                | 2.3               | 0.25                        | ... |
| 2023-04-02 | 10.2               | 0.0               | 0.24                        | ... |
| ...        | ...                | ...               | ...                         | ... |

### Saison de croissance (jusqu'à 200 jours)

Même format, avec les jours suivant le semis.

**Colonnes requises** (correspondant au préprocesseur) :
- `temperature_2m_mean`, `temperature_2m_min`, `temperature_2m_max`
- `precipitation_sum`
- `soil_temperature_0_to_7cm_mean`
- `soil_moisture_0_to_7cm_mean`
- `surface_solar_radiation_downwards_sum`
- `wind_speed_10m_mean`
- `potential_evaporation_sum`

## 📁 Structure du projet

```
maisUI/
├── app.py                    # Application principale (FastAPI + Gradio)
├── model_wrapper.py          # Wrapper du modèle ML
├── requirements.txt          # Dépendances Python
├── Dockerfile               # Image Docker
├── fly.toml                 # Configuration Fly.io
├── .dockerignore           # Exclusions Docker
├── README.md               # Cette documentation
└── models/                 # Fichiers du modèle (non versionnés)
    ├── baseline_model.keras
    └── baseline_preprocessor.pkl
```

## 🎨 Personnalisation

### Modifier le thème Gradio

Dans `app.py`, ligne ~459 :

```python
theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink")
```

Options : `Base`, `Default`, `Glass`, `Monochrome`, `Soft`

### Ajouter des features

1. Mettre à jour `_get_static_feature_metadata()` dans `model_wrapper.py`
2. L'interface se génère automatiquement depuis le schéma

### Changer les labels/descriptions

Modifier les dictionnaires dans `model_wrapper.py` :
- `feature_metadata` : Features statiques
- `weather_labels` : Variables météo

## 🐛 Dépannage

### Erreur : "Aucun modèle trouvé"

Solution : Spécifiez les chemins explicitement
```bash
python app.py --model path/to/model.keras --preprocessor path/to/preprocessor.pkl
```

### Erreur : "Shape mismatch"

Les fichiers CSV météo doivent avoir les **mêmes colonnes** que le préprocesseur utilisé à l'entraînement.

### Application lente

- Réduire la plage d'azote testée (moins de doses)
- Utiliser un modèle plus léger
- Augmenter les ressources VM sur Fly.io

## 📝 Export HTML

Les fichiers HTML exportés sont **complètement auto-contenus** :

- ✅ Plotly.js embarqué (pas de CDN)
- ✅ Données JSON dans `<script type="application/json">`
- ✅ Visualisable offline
- ✅ Schéma du modèle inclus

Structure des données embarquées :

```html
<script type="application/json" id="embedded-data">
{
  "timestamp": "20240112_143022",
  "features": { ... },
  "nitrogen_range": [0, 25, 50, ...],
  "dose_response": [ ... ],
  "optimal": { ... },
  "model_version": "1.0"
}
</script>

<script type="application/json" id="feature-schema">
{
  "model_version": "1.0",
  "description": "...",
  "features": { "static": [...], "time_series": [...] },
  ...
}
</script>
```

## 🤝 Contribution

Ce projet est lié au dépôt `mais-npk` pour l'entraînement du modèle.

**Workflow recommandé** :
1. Entraîner/améliorer le modèle dans `mais-npk`
2. Exporter le modèle avec `scripts/utils/export_model_for_webapp.py`
3. Copier les fichiers dans `maisUI/models/`
4. Tester localement
5. Déployer sur Fly.io

## 📄 Licence

Voir fichier `LICENSE` à la racine du projet.

## 📧 Contact

Pour questions ou suggestions, ouvrir une issue sur le dépôt GitHub.

---

**Version** : 1.0
**Dernière mise à jour** : Janvier 2025