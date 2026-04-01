#!/bin/bash
# Quick start script for Plant Disease Detection System

set -e

echo "=" 
echo "Plant Disease Detection System - Setup & Run"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# Create virtual environment
echo ""
echo -e "${BLUE}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo -e "${BLUE}Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo ""
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p data/train data/validation data/test models uploaded_data logs
echo -e "${GREEN}✓ Directories created${NC}"

# Check GPU
echo ""
echo -e "${BLUE}Checking GPU availability...${NC}"
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✓ GPU detected: {len(gpus)} GPU(s) available')
    for gpu in gpus:
        print(f'  - {gpu}')
else:
    print('⚠ No GPU detected. CPU mode will be used (slower training)')
" || echo "⚠ Could not check GPU"

# Menu
echo ""
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1) Run Jupyter Notebook (Training)"
echo "2) Start API Backend (FastAPI)"
echo "3) Start Web UI (React)"
echo "4) Run Both API & UI"
echo "5) Run Load Tests (Locust)"
echo "6) Docker Deployment"
echo "7) Exit"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo -e "${BLUE}Starting Jupyter Notebook...${NC}"
        echo "Open: http://localhost:8888"
        jupyter notebook notebook/plant_disease_detection.ipynb
        ;;
    2)
        echo -e "${BLUE}Starting FastAPI Backend...${NC}"
        echo "API running at: http://localhost:8000"
        echo "API docs at: http://localhost:8000/docs"
        echo ""
        python3 main.py
        ;;
    3)
        echo -e "${BLUE}Starting React UI...${NC}"
        echo "UI running at: http://localhost:5173"
        echo ""
        echo -e "${YELLOW}Note: Make sure API is running on another terminal!${NC}"
        echo ""
        cd frontend
        npm install
        npm run dev -- --host 0.0.0.0 --port 5173
        ;;
    4)
        echo -e "${BLUE}Starting API and UI...${NC}"
        echo ""
        echo -e "${YELLOW}Starting API in background...${NC}"
        python3 main.py > logs/api.log 2>&1 &
        API_PID=$!
        echo -e "${GREEN}API started (PID: $API_PID)${NC}"
        
        # Wait for API to start
        echo "Waiting for API to start..."
        sleep 3
        
        echo ""
        echo -e "${BLUE}Starting React UI...${NC}"
        echo "UI running at: http://localhost:5173"
        cd frontend
        npm install
        npm run dev -- --host 0.0.0.0 --port 5173
        
        # Clean up on exit
        trap "kill $API_PID" EXIT
        ;;
    5)
        echo -e "${BLUE}Starting Locust Load Testing...${NC}"
        echo ""
        echo -e "${YELLOW}Make sure API is running on http://localhost:8000${NC}"
        echo ""
        echo "Locust UI running at: http://localhost:8089"
        locust -f locustfile.py --host=http://localhost:8000 --web --port=8089
        ;;
    6)
        echo -e "${BLUE}Docker Deployment Options:${NC}"
        echo ""
        echo "1) Build Docker image"
        echo "2) Run single container"
        echo "3) Run with docker-compose"
        echo ""
        read -p "Enter docker choice (1-3): " docker_choice
        
        case $docker_choice in
            1)
                echo -e "${BLUE}Building Docker image...${NC}"
                docker build -t plant-disease-detector:latest .
                echo -e "${GREEN}✓ Docker image built${NC}"
                ;;
            2)
                echo -e "${BLUE}Running Docker container...${NC}"
                docker run -p 8000:8000 -p 5173:5173 \
                    -v $(pwd)/models:/app/models \
                    -v $(pwd)/uploaded_data:/app/uploaded_data \
                    plant-disease-detector:latest
                ;;
            3)
                echo -e "${BLUE}Starting docker-compose...${NC}"
                docker-compose up
                ;;
            *)
                echo "Invalid choice"
                ;;
        esac
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
