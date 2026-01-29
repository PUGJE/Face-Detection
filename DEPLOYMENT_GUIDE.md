# Face Recognition Attendance System - Deployment Guide

Complete guide to deploy the Face Recognition Based Smart Attendance System with Anti-Spoofing on a new system.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Production Deployment](#production-deployment)

---

## 🖥️ System Requirements

### Hardware Requirements
- **CPU**: Intel i5 or equivalent (i7+ recommended)
- **RAM**: Minimum 8GB (16GB recommended)
- **Webcam**: Any USB or built-in webcam
- **Storage**: At least 5GB free space
- **GPU**: Optional (NVIDIA GPU with CUDA for faster processing)

### Software Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: Version 3.8 to 3.11 (3.9 or 3.10 recommended)
- **Git**: For cloning the repository (optional)

---

## 📦 Installation Steps

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"**
3. Verify installation:
   ```bash
   python --version
   ```

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# macOS (using Homebrew)
brew install python@3.10
```

### Step 2: Copy Project Files

**Option A: Using Git**
```bash
git clone <repository-url>
cd "Face Recognition System"
```

**Option B: Manual Copy**
1. Copy the entire project folder to your system
2. Navigate to the project directory:
   ```bash
   cd "Face Recognition System"
   ```

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Note:** This may take 10-20 minutes depending on your internet speed.

### Step 5: Verify Installation

```bash
python -c "import cv2, tensorflow, deepface; print('All packages installed successfully!')"
```

---

## ⚙️ Configuration

### Step 1: Environment Variables

Create a `.env` file in the project root:

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

Edit `.env` file with your settings:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite:///./data/attendance.db

# Security
SECRET_KEY=your-secret-key-here-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Face Recognition
FACE_DETECTION_CONFIDENCE=0.5
FACE_RECOGNITION_THRESHOLD=0.4

# Attendance Window
ATTENDANCE_WINDOW_START=09:00
ATTENDANCE_WINDOW_END=17:00

# File Paths
STUDENT_IMAGES_PATH=./data/student_images
EMBEDDINGS_PATH=./data/embeddings

# Camera
CAMERA_INDEX=0
```

### Step 2: Create Required Directories

The application will create these automatically, but you can create them manually:

```bash
# Windows
mkdir data
mkdir data\student_images
mkdir data\embeddings
mkdir logs

# Linux/macOS
mkdir -p data/student_images data/embeddings logs
```

---

## 🚀 Running the Application

### Step 1: Start the Backend Server

```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application started on 0.0.0.0:8000
INFO:     Frontend available at: http://127.0.0.1:8000/app
```

### Step 2: Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000/app
```

Or from another device on the same network:
```
http://<your-ip-address>:8000/app
```

### Step 3: First-Time Setup

1. **Dashboard** - View system statistics
2. **Register Student** - Add your first student
   - Enter student details
   - Capture face image
   - Click "Register Student"
3. **Mark Attendance** - Test attendance marking
   - Click "Start Camera"
   - Face the camera
   - Click "Mark Attendance"

---

## 🧪 Testing

### Test 1: API Health Check

```bash
# In a new terminal (keep server running)
python test_api.py
```

### Test 2: System Validation

```bash
python test_system.py
```

### Test 3: Anti-Spoofing

```bash
python test_simple_antispoofing.py
```

**Test scenarios:**
1. Your real face → Should PASS
2. No face → Should show "No face detected"
3. Photo on phone screen → Should FAIL (anti-spoofing)

---

## 🔧 Troubleshooting

### Issue 1: Camera Not Working

**Error:** `Could not open camera`

**Solution:**
1. Check if camera is being used by another application
2. Try different camera index in `.env`:
   ```env
   CAMERA_INDEX=1  # or 2, 3, etc.
   ```
3. On Linux, check permissions:
   ```bash
   sudo usermod -a -G video $USER
   ```

### Issue 2: Module Not Found

**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
# Ensure virtual environment is activated
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 3: TensorFlow Warnings

**Warning:** `oneDNN custom operations...`

**Solution:** These are informational warnings and can be ignored. To suppress:
```python
# Add to main.py at the top
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### Issue 4: Port Already in Use

**Error:** `Address already in use`

**Solution:**
1. Change port in `.env`:
   ```env
   PORT=8001
   ```
2. Or kill the process using port 8000:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   
   # Linux/macOS
   lsof -ti:8000 | xargs kill -9
   ```

### Issue 5: Database Locked

**Error:** `database is locked`

**Solution:**
```bash
# Stop the server
# Delete the database file
rm data/attendance.db  # Linux/macOS
del data\attendance.db  # Windows

# Restart the server (will recreate database)
python main.py
```

---

## 🌐 Production Deployment

### Option 1: Local Network Deployment

1. **Update `.env`:**
   ```env
   HOST=0.0.0.0
   PORT=8000
   ```

2. **Configure Firewall:**
   ```bash
   # Windows: Allow port 8000 in Windows Firewall
   # Linux:
   sudo ufw allow 8000
   ```

3. **Find Your IP:**
   ```bash
   # Windows
   ipconfig
   
   # Linux/macOS
   ifconfig
   ```

4. **Access from other devices:**
   ```
   http://<your-ip>:8000/app
   ```

### Option 2: Cloud Deployment (AWS/Azure/GCP)

1. **Prepare for production:**
   - Use PostgreSQL instead of SQLite
   - Set strong `SECRET_KEY`
   - Enable HTTPS
   - Configure CORS properly

2. **Update `main.py` CORS:**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Specific domain
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

3. **Use production server:**
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

### Option 3: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t face-attendance .
docker run -p 8000:8000 -v $(pwd)/data:/app/data face-attendance
```

---

## 📊 System Architecture

```
Face Recognition Attendance System
│
├── Backend (FastAPI)
│   ├── API Endpoints (main.py)
│   ├── Database (SQLAlchemy)
│   ├── Services (Student, Attendance)
│   └── ML Models
│       ├── Face Detection (MediaPipe)
│       ├── Face Recognition (DeepFace)
│       └── Anti-Spoofing (Texture Analysis)
│
├── Frontend (HTML/CSS/JS)
│   ├── Dashboard
│   ├── Student Registration
│   ├── Attendance Marking
│   ├── Student List
│   └── Reports
│
└── Data Storage
    ├── SQLite Database
    ├── Student Images
    └── Face Embeddings
```

---

## 📝 Quick Reference

### Common Commands

```bash
# Activate virtual environment
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Start server
python main.py

# Run tests
python test_system.py

# Check API
curl http://localhost:8000/health

# View logs
tail -f logs/app.log  # Linux/macOS
type logs\app.log     # Windows
```

### Important Files

| File | Purpose |
|------|---------|
| `main.py` | Main application entry point |
| `requirements.txt` | Python dependencies |
| `.env` | Configuration settings |
| `data/attendance.db` | SQLite database |
| `backend/config.py` | Application configuration |
| `API_DOCUMENTATION.md` | API reference |
| `QUICK_START.md` | Quick start guide |

---

## 🔐 Security Checklist

- [ ] Change default `SECRET_KEY` in `.env`
- [ ] Use strong passwords for any authentication
- [ ] Configure CORS for specific domains in production
- [ ] Enable HTTPS in production
- [ ] Regularly backup database
- [ ] Keep dependencies updated
- [ ] Review and limit API access
- [ ] Monitor logs for suspicious activity

---

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review `API_DOCUMENTATION.md`
3. Check `SYSTEM_VALIDATION_REPORT.md` for known issues
4. Review logs in `logs/` directory

---

## 📄 License

This project is for educational and internal use.

---

**Last Updated:** January 2026  
**Version:** 1.0.0  
**Status:** Production Ready
