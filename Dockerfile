# Utilise une image Python officielle
FROM python:3.10-slim

# Installe les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Crée le dossier de l'app
WORKDIR /app

# Copie les fichiers dans le conteneur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose le port Streamlit par défaut
EXPOSE 8501

# Lance l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
