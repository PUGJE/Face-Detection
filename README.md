# Face Recognition Based Smart Attendance System with Anti-Spoofing

## 🎯 Project Overview

A secure, automated attendance system using facial recognition technology combined with anti-spoofing mechanisms to prevent fraudulent attendance using photographs, videos, or masks.

## ✨ Key Features

- **Real-time Face Detection**: Using MediaPipe for fast and accurate face detection
- **Face Recognition**: FaceNet-based identity verification
- **Anti-Spoofing**: CNN-based liveness detection to prevent fraud
- **Automated Attendance**: Seamless attendance marking with duplicate prevention
- **Admin Dashboard**: Comprehensive student and attendance management
- **RESTful API**: FastAPI-based backend with automatic documentation

## 🏗️ Architecture

```
Client Layer (HTML/CSS/JS)
         ↓
Backend Layer (FastAPI)
         ↓
ML Layer (Detection → Recognition → Anti-Spoofing)
         ↓
Data Layer (SQLite + Embeddings)
```

## 📁 Project Structure

```
face-recognition-attendance/
├── backend/              # FastAPI backend
│   ├── api/             # API routes
│   ├── core/            # Business logic
│   ├── ml/              # ML modules
│   ├── models/          # Database models
│   ├── database/        # Database layer
│   └── schemas/         # Pydantic schemas
├── frontend/            # Web interface
│   ├── static/          # CSS, JS, images
│   └── templates/       # HTML templates
├── data/                # Data storage
│   ├── database/        # SQLite database
│   ├── embeddings/      # Face embeddings
│   └── student_images/  # Registration photos
├── models/              # Pre-trained ML models
└── tests/               # Test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- 8GB RAM (recommended)

### Installation

1. **Clone the repository** (or navigate to project directory)
   ```bash
   cd "d:\Projects\Face Recognition System"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   ```bash
   copy .env.example .env
   ```
   Edit `.env` and update the `SECRET_KEY` for production use.

6. **Run the application**
   ```bash
   uvicorn backend.main:app --reload
   ```

7. **Access the application**
   - Frontend: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework
- **SQLAlchemy**: ORM for database operations
- **Pydantic**: Data validation
- **Python-Jose**: JWT authentication

### Machine Learning
- **MediaPipe**: Face detection
- **TensorFlow/Keras**: Deep learning framework
- **DeepFace**: Face recognition
- **OpenCV**: Image processing

### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript**: Client-side logic
- **WebRTC**: Camera access

### Database
- **SQLite**: Lightweight, serverless database

## 📊 System Workflow

### Student Registration
1. Student enters details (name, ID, email)
2. Camera captures face images
3. Face detection validates face presence
4. Face embeddings generated and stored
5. Student record saved to database

### Attendance Marking
1. Student opens attendance page
2. Camera captures live video
3. Face detection locates face
4. Anti-spoofing verifies liveness
5. Face recognition matches identity
6. Attendance marked (if within time window)
7. Duplicate prevention applied

## 🔒 Security Features

- **JWT Authentication**: Secure API access
- **Password Hashing**: Bcrypt encryption
- **Anti-Spoofing**: Liveness detection
- **Role-Based Access**: Admin vs Student permissions
- **Secure Storage**: Encrypted embeddings

## 📈 Performance Metrics

- **Face Detection**: ~30-60 FPS (depending on hardware)
- **Face Recognition**: <100ms per face
- **Anti-Spoofing**: <200ms per frame
- **Database Queries**: <10ms average

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_face_detection.py -v
```

## 📝 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `POST /api/students/register` - Register new student
- `POST /api/attendance/mark` - Mark attendance
- `GET /api/attendance/history/{student_id}` - Get attendance history
- `GET /api/admin/students` - List all students (admin only)

## 🎓 Academic Context

This project demonstrates:
- **Machine Learning**: Face detection, recognition, and classification
- **Deep Learning**: CNN-based models
- **Software Engineering**: Clean architecture, modular design
- **API Development**: RESTful API design
- **Security**: Authentication, authorization, anti-spoofing
- **Database Design**: Relational database modeling

## 🔮 Future Enhancements

- [ ] Mobile app (React Native)
- [ ] Cloud deployment (AWS/Azure)
- [ ] Multi-camera support
- [ ] Advanced 3D liveness detection
- [ ] Real-time analytics dashboard
- [ ] Email/SMS notifications
- [ ] Export attendance reports (PDF/Excel)

## 👥 Contributors

- Face Recognition Team
- Version 1.0.0
- January 2026

## 📄 License

This project is for academic purposes.

## 📞 Support

For questions or issues, please refer to the documentation in the `docs/` folder.

---

**Built with ❤️ for automated, secure attendance management**
