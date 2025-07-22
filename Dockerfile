# 1) Python 3.11 slim (wheels NumPy/Matplotlib compatibles)
FROM python:3.11-slim

# 2) Outils système : Java + compilation C
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      default-jre \
      build-essential \
      python3-dev \
      python3-distutils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3) Copier dépendances et code
COPY packages.txt requirements-cpu.txt ./
COPY ELiC_ReImplemetation/ ./ELiC_ReImplemetation/
COPY compute-resources/ ./compute-resources/
COPY app.py ./

# 4) Installer Python deps
RUN pip install --no-cache-dir -r requirements-cpu.txt

# 5) Exposer le port
EXPOSE 8080

# 6) Lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080"]
