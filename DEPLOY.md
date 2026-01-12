# 🚀 Guide de Déploiement - maisUI

Ce guide explique comment déployer l'application maisUI sur Fly.io en production.

## 📋 Prérequis

1. **Compte Fly.io** : [Créer un compte gratuit](https://fly.io/app/sign-up)
2. **Fly CLI installé** : Suivre les instructions ci-dessous
3. **Modèle entraîné** : Fichiers `.keras` et `.pkl` depuis `mais-npk`

## 🛠️ Installation de Fly CLI

### Linux / macOS
```bash
curl -L https://fly.io/install.sh | sh
```

### Windows (PowerShell)
```powershell
pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

Vérifier l'installation :
```bash
flyctl version
```

## 🔐 Authentification

```bash
flyctl auth login
```

Cela ouvrira votre navigateur pour la connexion.

## 📦 Préparation des fichiers du modèle

### Option 1 : Copier depuis mais-npk

```bash
# Créer le répertoire models
mkdir -p models

# Copier les fichiers les plus récents
cp ../mais-npk/data/models/baseline_model_*.keras models/model.keras
cp ../mais-npk/data/models/baseline_preprocessor_*.pkl models/preprocessor.pkl
```

### Option 2 : Télécharger depuis un stockage cloud

Si vos modèles sont volumineux (>100 MB), utilisez un volume Fly.io (voir plus bas).

## 🏗️ Déploiement initial

### 1. Lancer l'application

```bash
flyctl launch
```

Lors du lancement, répondez aux questions :

```
? Choose an app name (leave blank to generate one): maisui
? Choose a region for deployment: Montreal, Canada (yul)
? Would you like to set up a PostgreSQL database now? No
? Would you like to set up an Upstash Redis database now? No
? Create .dockerignore from 1 .gitignore files? Yes
```

### 2. Configurer les ressources

Modifier `fly.toml` si nécessaire :

```toml
[[vm]]
  cpu_kind = "shared"
  cpus = 2           # Ajuster selon la taille du modèle
  memory_mb = 2048   # Minimum 2GB pour modèles Keras + PyTorch
```

Pour les modèles plus volumineux :
```toml
[[vm]]
  cpu_kind = "shared"
  cpus = 4
  memory_mb = 4096
```

### 3. Déployer

```bash
flyctl deploy
```

Le processus va :
1. Builder l'image Docker
2. La pousser vers Fly.io
3. Déployer sur les machines virtuelles
4. Effectuer les health checks

### 4. Vérifier le déploiement

```bash
# Ouvrir l'application dans le navigateur
flyctl open

# Voir les logs
flyctl logs

# Vérifier le status
flyctl status
```

## 💾 Gestion des fichiers du modèle

### Option A : Inclure dans l'image Docker (< 100 MB)

**Avantages** : Simple, rapide à déployer
**Inconvénients** : Augmente la taille de l'image

1. Modifier `.dockerignore` pour **ne pas** exclure les modèles :
```
# .dockerignore
# Commentez ces lignes :
# *.keras
# *.pkl
```

2. Modifier le `Dockerfile` pour copier les modèles :
```dockerfile
# Après COPY app.py .
COPY models/ /app/models/

# Modifier CMD pour pointer vers les modèles
CMD ["python", "app.py", "--model", "/app/models/model.keras", "--preprocessor", "/app/models/preprocessor.pkl", "--host", "0.0.0.0", "--port", "8080"]
```

3. Redéployer :
```bash
flyctl deploy
```

### Option B : Volume persistant (> 100 MB)

**Avantages** : Image Docker légère, modèles mis à jour sans redéploiement
**Inconvénients** : Configuration plus complexe

#### 1. Créer un volume

```bash
flyctl volumes create model_data --region yul --size 10
```

#### 2. Modifier `fly.toml`

Décommenter la section `[[mounts]]` :
```toml
[[mounts]]
  source = "model_data"
  destination = "/app/models"
```

#### 3. Déployer l'application

```bash
flyctl deploy
```

#### 4. Copier les fichiers du modèle

```bash
# Se connecter au container
flyctl ssh console

# Dans le container :
cd /app/models

# Depuis votre machine locale (nouveau terminal) :
flyctl ssh sftp shell
put models/model.keras /app/models/
put models/preprocessor.pkl /app/models/
exit
```

Alternativement, utiliser `scp` ou un bucket S3/GCS.

#### 5. Redémarrer l'application

```bash
flyctl apps restart
```

### Option C : Téléchargement au démarrage

Pour les très gros modèles, téléchargez-les depuis S3/GCS au démarrage :

Modifier `app.py` :
```python
import os
import urllib.request

def download_model_if_needed():
    model_url = os.getenv("MODEL_URL")
    preprocessor_url = os.getenv("PREPROCESSOR_URL")

    if model_url and not os.path.exists("/app/models/model.keras"):
        print("Téléchargement du modèle...")
        urllib.request.urlretrieve(model_url, "/app/models/model.keras")

    if preprocessor_url and not os.path.exists("/app/models/preprocessor.pkl"):
        print("Téléchargement du préprocesseur...")
        urllib.request.urlretrieve(preprocessor_url, "/app/models/preprocessor.pkl")

# Avant initialize_model()
download_model_if_needed()
```

Définir les secrets :
```bash
flyctl secrets set MODEL_URL=https://your-bucket.s3.amazonaws.com/model.keras
flyctl secrets set PREPROCESSOR_URL=https://your-bucket.s3.amazonaws.com/preprocessor.pkl
```

## 🔒 Secrets et variables d'environnement

### Définir des secrets

```bash
# Prix par défaut
flyctl secrets set N_PRICE=1.5
flyctl secrets set GRAIN_PRICE=0.20

# Chemins des modèles (si option B ou C)
flyctl secrets set MODEL_PATH=/app/models/model.keras
flyctl secrets set PREPROCESSOR_PATH=/app/models/preprocessor.pkl
```

### Lister les secrets

```bash
flyctl secrets list
```

## 📊 Monitoring et logs

### Voir les logs en temps réel

```bash
flyctl logs -a maisui
```

### Logs des dernières 24h

```bash
flyctl logs -a maisui --since 24h
```

### Métriques de l'application

```bash
flyctl status -a maisui
flyctl vm status -a maisui
```

### Dashboard Fly.io

Ouvrir le dashboard :
```bash
flyctl dashboard
```

## 🔄 Mise à jour de l'application

### Mise à jour du code uniquement

```bash
# Après modification de app.py ou model_wrapper.py
flyctl deploy
```

### Mise à jour du modèle (Option A)

```bash
# Copier le nouveau modèle
cp ../mais-npk/data/models/new_model.keras models/model.keras
cp ../mais-npk/data/models/new_preprocessor.pkl models/preprocessor.pkl

# Redéployer
flyctl deploy
```

### Mise à jour du modèle (Option B - Volume)

```bash
# Se connecter et remplacer les fichiers
flyctl ssh console
cd /app/models
# Uploader les nouveaux fichiers via SFTP

# Redémarrer
flyctl apps restart
```

## 🛑 Scaling et arrêt

### Scaler horizontalement

```bash
# Augmenter le nombre d'instances
flyctl scale count 3

# Retour à 1 instance
flyctl scale count 1
```

### Scaler verticalement

```bash
# Augmenter les ressources
flyctl scale vm shared-cpu-4x --memory 4096
```

### Arrêter l'application

```bash
# Suspension (conserve la configuration)
flyctl scale count 0

# Destruction complète
flyctl apps destroy maisui
```

## 💰 Coûts estimés

### Niveau gratuit (Hobby)
- Jusqu'à 3 machines partagées 256MB
- 160GB de transfert sortant/mois
- **Coût** : Gratuit

### Configuration recommandée pour production
- 1 machine : 2 CPU, 2GB RAM
- Auto-start/stop activé
- **Coût** : ~$10-15/mois (facturé à l'heure d'utilisation)

### Avec volume persistant (10GB)
- **Coût additionnel** : ~$1.50/mois

Voir la [tarification complète](https://fly.io/docs/about/pricing/)

## 🔧 Dépannage

### L'application ne démarre pas

1. Vérifier les logs :
```bash
flyctl logs
```

2. Vérifier les health checks :
```bash
flyctl status
```

3. Se connecter au container :
```bash
flyctl ssh console
python app.py --model /app/models/model.keras --preprocessor /app/models/preprocessor.pkl
```

### Erreur "Out of memory"

Augmenter la RAM :
```bash
flyctl scale vm shared-cpu-2x --memory 4096
```

### Application lente

1. Augmenter les ressources CPU :
```bash
flyctl scale vm shared-cpu-4x
```

2. Vérifier les métriques :
```bash
flyctl metrics
```

### Modèle non trouvé

Vérifier que les fichiers existent :
```bash
flyctl ssh console
ls -lh /app/models/
```

## 🌐 Custom domain

### Ajouter un domaine personnalisé

```bash
# Ajouter le domaine
flyctl certs add maisui.votre-domaine.com

# Vérifier le certificat SSL
flyctl certs show maisui.votre-domaine.com
```

### Configurer le DNS

Ajouter un enregistrement `CNAME` ou `A` pointant vers :
- **CNAME** : `maisui.fly.dev`
- **A** : (Voir les adresses IP dans `flyctl ips list`)

## 📝 Checklist de déploiement

- [ ] Fly CLI installé et authentifié
- [ ] Modèle et préprocesseur copiés dans `models/`
- [ ] `fly.toml` configuré (région, ressources)
- [ ] Option de stockage choisie (image, volume, ou téléchargement)
- [ ] `flyctl launch` exécuté avec succès
- [ ] Application accessible via `flyctl open`
- [ ] Logs vérifiés sans erreurs
- [ ] Prédiction testée avec des données réelles
- [ ] Export HTML testé et fonctionnel
- [ ] (Optionnel) Domaine personnalisé configuré
- [ ] (Optionnel) Monitoring configuré

## 🆘 Support

- **Documentation Fly.io** : https://fly.io/docs/
- **Forum Fly.io** : https://community.fly.io/
- **Status Fly.io** : https://status.fly.io/

---

**Bon déploiement!** 🚀
