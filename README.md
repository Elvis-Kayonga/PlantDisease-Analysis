# 🌱 Plant Disease Classification - End-to-End MLOps Pipeline

**African Leadership University (ALU) - Year 3, Term 1**  
**Course:** Machine Learning (MLOps Summative Assignment)  
**Student Name:** Kayonga Elvis  
**Email:** e.kayonga@alustudent.com  

---

## 🔗 Assignment Submission Links
- **Live Frontend (Vercel):** Deploy to Vercel using instructions in [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Live Backend API (Render):** [API](https://plant-disease-api-5fzb.onrender.com)
- **Video Demonstration:** [Youtube Link](https://www.youtube.com/watch?v=k-yrTpBvWWTDcY)

---

## 📖 Project Description
This project is a production-ready, end-to-end Machine Learning pipeline designed to classify healthy and diseased plant leaves. It builds upon transfer learning using a custom-tuned **MobileNetV2** architecture (processing 224x224 images), served via a **FastAPI** backend, and consumed by a modern **React (Vite)** User Interface.

The system fulfills the complete ML Lifecycle: Data Acquisition, Preprocessing, Model Creation, Automated Retraining, API Deployment, UI consumption, and Containerized Load Balancing.

## ✨ Rubric Features & Functionality

### 1. Model Prediction & Evaluation
- **Prediction Process:** Users can upload a leaf image via the React UI and receive real-time classification.
- **Evaluation Metrics:** The Jupyter Notebook and API demonstrate the model's robustness using 4 key metrics: **Accuracy, Precision, Recall, and F1-Score**. Data augmentation and early stopping are utilized to prevent overfitting.

### 2. Data Visualizations & Interpretations
The UI includes dynamic interpretations of the dataset:
- **Class Distribution & Imbalance:** Identifies the skew towards healthy crops and how augmentation counters this.
- **RGB Color Channel Intensity:** Visualizes how necrotic lesions shift pixel concentrations from green to red.
- **Image Brightness Bimodality:** Analyzes the contrast between the dark background and foreground leaf tissue.

### 3. Data Intake & Automated Retraining
- **Upload Data:** The UI allows bulk uploading of labeled images, saving them securely to the backend payload directories.
- **Trigger Retraining:** Users can press "Start Retraining" in the UI to trigger a fresh ML training cycle. The newly trained model is seamlessly saved as a `.h5` file and re-mounted into memory without server downtime.

### 4. Telemetry & Continuous Monitoring
- **SQLite Database:** Tracks up-time, prediction logs, data upload sizes, and historic training runs.

---

## ![Locust Load Testing](https://locust.io/assets/img/logo.png)
 Locust Load Testing (Flood Request Simulation)
To demonstrate scalability and monitor latency, a flood of concurrent user requests was simulated using Locust. 

**Simulated Load Setup:** 100 Concurrent Users, 10 Spawn Rate.

| Infrastructure Setup | Average Latency (Response Time) | Requests Per Second (RPS) |
|----------------------|--------------------------------|---------------------------|
| **1 Docker Container** | `2700ms` | `2.7/s` |
| **2 Docker Containers** (Scaled) | `140` | `35` |

*Analysis:* Scaling the Docker backend significantly improved the response times of the prediction API under heavy load.

---

## 🚀 Setup & Execution Instructions

### 🌐 Production Deployment (Recommended)
**Separate deployments for optimal resource usage:**
- **Frontend → Vercel** (Static hosting with CDN)
- **Backend → Render** (ML model + API)

📖 **See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment guide**

This approach:
- ✅ Avoids Render free tier size limits (512MB)
- ✅ Leverages Vercel's optimized static hosting
- ✅ Enables independent frontend/backend updates
- ✅ Better performance with global CDN

### 💻 Running Locally via Docker
The entire application (FastAPI + React) is bundled into a unified Docker container.
```bash
# 1. Build the Docker image
docker-compose build

# 2. Start the unified server
docker-compose up
```
Access the Web UI at: `http://localhost:8000`

### 🔧 Running Locally for Development
```bash
# Terminal 1: Backend API
python main.py
# Runs on http://localhost:8000

# Terminal 2: Frontend Dev Server
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

### Running Load Tests
With the Docker container fully running, open a new terminal:
```bash
# Run the Locust load tester
locust -f locustfile.py --host=http://localhost:8000
```
Open `http://localhost:8089` to start the swarm. 
To test scaled containers, stop Docker and run: `docker-compose up --scale api=2`

---

## 📂 Repository Structure
```text
/PlantDisease-Analysis
├── Dockerfile & docker-compose.yml   # Multi-stage container orchestration
├── frontend/                         # React Vite Single-Page Application
├── main.py                           # FastAPI Backend & UI Router
├── src/                              # ML source code (model.py, preprocessing.py)
├── locustfile.py                     # Load testing script
├── data/ & uploaded_data/            # Datasets and user-uploaded batches
└── README.md                         # Documentation
```
