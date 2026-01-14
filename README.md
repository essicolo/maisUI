## installation

maisUI a besoin de

- Python 3.11+
- Modèle entraîné (`.keras`) et préprocesseur (`.pkl`) depuis `mais-npk`

```bash
cd maisUI
pip install -r requirements.txt
```

copier les fichiers du modèle

```bash
# Copier le modèle et préprocesseur les plus récents
cp ../mais-npk/data/models/baseline_model_*.keras ./models/
cp ../mais-npk/data/models/baseline_preprocessor_*.pkl ./models/
```

lancer l'application

```bash
python app.py --model models/baseline_model_*.keras \
              --preprocessor models/baseline_preprocessor_*.pkl \
              --port 7860
```

l'application sera disponible sur `http://localhost:7860`


## déploiement Docker

local

```bash
docker build -t maisui .
docker run -p 8080:8080 \
    -v $(pwd)/models:/app/models \
    maisui
```

Fly.io

Installer Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
```

Login
```bash
flyctl auth login
```

créer l'application
```bash
flyctl launch
# Suivez les instructions (région: Montreal/yul recommandée)
```

copier les fichiers du modèle

- Option A: Volume persistant

```bash
flyctl volumes create model_data --region yul --size 10
flyctl ssh console
# Dans le container:
# Copiez vos fichiers .keras et .pkl dans /app/models/
```

- Option B: Secrets d'environnement

```bash
flyctl secrets set MODEL_PATH=/app/models/model.keras
flyctl secrets set PREPROCESSOR_PATH=/app/models/preprocessor.pkl
```

déployer

```bash
flyctl deploy
```

ouvrir l'application

```bash
flyctl open
```

