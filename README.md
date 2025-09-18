# Anime Recommender System
![Doraemon](https://github.com/user-attachments/assets/c5b006ba-cebd-4249-ad9b-3f88234853fb)

## Overview
This project is an anime recommender system built with a hybrid approach that combines user-based collaborative filtering and content-based methods. Data versioning is managed with DVC, and experiments are tracked using Comet.ml for reproducibility. The system is containerized with Docker, integrated into a Jenkins CI/CD pipeline, and deployed on Google Kubernetes Engine (GKE) within Google Cloud Platform (GCP).

**Primary Goal**
- Deliver accurate, personalized anime recommendations
- Ensure reproducible experiments and clear data lineage
- Enable automated CI/CD and scalable cloud deployment

## Key Features
- Hybrid recommender system
- Offline training + online inference API
- Data versioning with DVC (data & features)
- Experiment tracking & metadata with COMET ML
- Containerized app + model serving
- CI/CD pipeline (Docker & Jenkins) building images, running tests, and deploying to GKE
- All artifacts and hosting pushed to GCP (Artifact Registry / GCR + GKE + Cloud Storage)

## Architecture
- **Data** - raw dataset(s) tracked with DVC (e.g., user history, anime metadata, ratings). Stored remotely (GCS) via DVC remotes.
- **Training** - training scripts produce versioned model artifacts and metrics. Comet logs experiments (hyperparams, metrics, visualizations).
- **Serving** - model packaged into a Docker image exposing a REST API for recommendations.
- **CI/CD** - Jenkins pipeline builds image, runs tests, tags, pushes to Container Registry / Artifact Registry, triggers deployment to GKE.
- **Deployment** - Kubernetes manifests (Deployment, Service, HPA) run the service in GKE. Optionally use a load balancer + Ingress.

## Techonologies Used
- Python (data processing, model training, API server — e.g., FastAPI / Flask)
- DVC for data/version control
- COMET ML for experiments
- Docker for containerization
- Jenkins for CI/CD pipeline
- Kubernetes (GKE) for production deployment
- Google Cloud Platform: GCS (DVC remote), Artifact Registry/GCR, GKE

## Project Structure
```
├── application.py              # Main entrypoint for running the app
├── artifacts/                  # DVC-tracked artifacts: raw, processed data, models, weights
│   ├── raw/                    # Original datasets (anime.csv, animelist.csv, etc.)
│   ├── processed/              # Cleaned & transformed datasets + encodings
│   ├── model/                  # Final trained model (.h5)
│   ├── model_checkpoint/       # Intermediate checkpoints
│   └── weights/                # Saved weight matrices for CF/CB models
├── config/                     # Configuration files (YAML, paths, etc.)
├── custom_jenkins/             # Custom Jenkins Dockerfile
├── deployment.yaml             # Kubernetes deployment manifest
├── Dockerfile                  # App Dockerfile
├── Jenkinsfile                 # CI/CD pipeline definition
├── logs/                       # Training / app logs
├── notebook/                   # Jupyter notebooks for experimentation
├── pipeline/                   # Training and prediction pipelines
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Packaging configuration
├── src/                        # Core source code
│   ├── base_model.py           # Base model class
│   ├── data_ingestion.py       # Data ingestion logic
│   ├── data_processing.py      # Preprocessing and feature engineering
│   ├── model_training.py       # Model training scripts
│   ├── custom_exception.py     # Custom exception handling
│   └── logger.py               # Logging utility
├── static/                     # Static assets (CSS, etc.)
├── templates/                  # HTML templates (Flask frontend)
└── utils/                      # Helper functions and common utilities
```
## Local Set Up Instruction
### 1. Clone the repository
```
git clone https://github.com/tranminhanh1512/AnimeRecommenderSystem.git
cd AnimeRecommenderSystem
```
### 2. Create a Virtual Environment & Install dependencies
```
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```
### 3. Run the application in your local machine
```
python application.py
```
## CI/CD Pipeline Instruction
### 1. Set Up Jenkins Container
- Build and run Jenkins Container
```
cd custom_jenkins
docker build -t jenkins-dind .
docker run -d --name jenkins-dind \
  --privileged \
  -p 8080:8080 -p 50000:50000 \
  -v //var/run/docker.sock:/var/run/docker.sock \
  -v jenkins_home:/var/jenkins_home \
  jenkins-dind
```
- Verify container
```
docker ps
docker logs jenkins-dind
```
- Copy the initial password from logs and open Jenkins at http://localhost:8080. Install suggested plugins and create your admin user.
### 2. Install Python in Jenkins Container
- Come to custom_jenkins folder terminal again and write the following commands

```
docker exec -u root -it jenkins-dind bash
apt update -y
apt install -y python3
python3 --version
ln -s /usr/bin/python3 /usr/bin/python
python --version
apt install -y python3-pip
apt install -y python3-venv
exit

docker restart jenkins-dind
```
- Go to Jenkins Dashboard and Sign in again
### 3. Project Dockerfile
Docker file is provided in the folder
### 4. Install Google Cloud & Kubernetes CLI in Jenkins
- Come to Terminal (custom_jenkins terminal) and run the following commands
```
docker exec -u root -it jenkins-dind bash
apt-get update
apt-get install -y curl apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update && apt-get install -y google-cloud-sdk
apt-get update && apt-get install -y kubectl
apt-get install -y google-cloud-sdk-gke-gcloud-auth-plugin
kubectl version --client
gcloud --version
exit
```
### 5. Grant Docker Permissions to Jenkins User
```
docker exec -u root -it jenkins-dind bash
groupadd docker
usermod -aG docker jenkins
usermod -aG root jenkins
exit
docker restart jenkins-dind
```
### 6. Configure GCP API Permissions & Jenkins Secrets
Before deploying to Kubernetes, ensure Jenkins has the proper credentials and permissions to access GCP resources:
#### a. Enable GCP APIs
- Artifact Registry API (for Docker images)
- Compute Engine API (for GKE access)
- Kubernetes Engine API
- Cloud Storage API (optional, for DVC remote)
#### b. Create GCP Service Account
- Create a service account with Editor or custom permissions covering the above APIs.
- Generate a JSON key file.
#### c. Add Secrets to Jenkins
- Go to Jenkins Dashboard → Manage Jenkins → Credentials → System → Global credentials.
- Add a Secret File: upload the JSON key file (ID: gcp-service-account).
- Add GitHub token as Secret Text (ID: github-token).
### 7. Kubernetes Deployment
- Create a Kubernetes cluster
- Push your repo on Github 
- Choose Build Now on Jenkins to deploy your application to Kubernetes

