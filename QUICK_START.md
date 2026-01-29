# Quick Start Guide

## Face Recognition Attendance System

### Prerequisites
- Python 3.8+ installed
- Webcam connected
- Virtual environment activated

### Installation

1. **Clone/Navigate to project**
   ```bash
   cd "d:\Projects\Face Recognition System"
   ```

2. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies** (if not already done)
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the server**
   ```bash
   python main.py
   ```

   You should see:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   INFO:     Frontend available at: http://127.0.0.1:8000/app
   ```

2. **Open the web interface**
   
   Open your browser and go to:
   ```
   http://127.0.0.1:8000/app
   ```

### Using the System

#### 1. Dashboard
- View overall statistics
- See today's attendance
- Monitor system status

#### 2. Register Students

1. Click **"Register"** in the navigation
2. Fill in student information:
   - Student ID (required)
   - Full Name (required)
   - Email, Department, Year (optional)
3. Click **"Start Camera"**
4. Click **"Capture"** to take photo
5. Click **"Register Student"**

#### 3. Mark Attendance

1. Click **"Mark Attendance"** in navigation
2. Click **"Start Camera"**
3. Position face in front of camera
4. Click **"Mark Attendance"**
5. System will recognize and mark attendance

#### 4. View Students

- Click **"Students"** to see all registered students
- Use search box to find specific students
- See who has face registered

#### 5. Reports

- Click **"Reports"** in navigation
- Select date to view attendance
- See student statistics and attendance percentages

### API Documentation

Interactive API docs available at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Troubleshooting

**Camera not working:**
- Check browser permissions
- Ensure webcam is connected
- Try different browser (Chrome recommended)

**Server not starting:**
- Check if port 8000 is available
- Ensure virtual environment is activated
- Check for error messages in console

**Face not detected:**
- Ensure good lighting
- Face camera directly
- Move closer to camera

**Student not recognized:**
- Re-register face with better image
- Check face detection confidence
- Ensure face is clearly visible

### Features

✅ **Student Management**
- Register students with personal info
- Upload and register face images
- Search and filter students

✅ **Attendance Tracking**
- Real-time face recognition
- Automatic attendance marking
- Duplicate prevention (one per day)
- Late arrival detection

✅ **Reports & Analytics**
- Daily attendance reports
- Student-wise statistics
- Attendance percentages
- Absent student tracking

✅ **Modern Web Interface**
- Responsive design
- Real-time webcam integration
- Beautiful UI with animations
- Easy navigation

### System Architecture

```
Frontend (Browser)
    ↓
FastAPI Server (Port 8000)
    ↓
Face Recognition ML
    ↓
SQLite Database
```

### File Structure

```
Face Recognition System/
├── frontend/
│   ├── index.html          # Main web interface
│   └── static/
│       ├── css/
│       │   └── style.css   # Styling
│       └── js/
│           └── app.js      # JavaScript logic
├── backend/
│   ├── ml/                 # Face detection & recognition
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   └── database/           # Database connection
├── data/
│   ├── database/           # SQLite database
│   ├── embeddings/         # Face embeddings
│   └── student_images/     # Student photos
├── main.py                 # FastAPI application
└── requirements.txt        # Dependencies
```

### Next Steps

1. **Register some students** using the web interface
2. **Test attendance marking** with webcam
3. **View reports** and statistics
4. **Customize** settings in `.env` file
5. **Deploy** to production server (optional)

### Production Deployment

For production:

1. **Update CORS settings** in `main.py`
2. **Set strong SECRET_KEY** in `.env`
3. **Use production database** (PostgreSQL recommended)
4. **Enable HTTPS**
5. **Set `reload=False`** in uvicorn
6. **Use process manager** (PM2, systemd)

### Support

For issues or questions:
- Check logs in console
- Review API documentation
- Check database with `setup_database.py`
- Test API with `test_api.py`

---

**Enjoy your Face Recognition Attendance System!** 🎉
