// Face Recognition Attendance System - Frontend JavaScript

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

// Global variables
let currentStream = null;
let capturedImage = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    loadDashboard();

    // Set today's date for reports
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('report-date').value = today;

    // Add event listeners
    document.getElementById('report-date').addEventListener('change', loadDateAttendance);
    document.getElementById('search-input').addEventListener('input', searchStudents);
});

// Navigation
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Show corresponding page
            const pageName = link.getAttribute('data-page');
            showPage(pageName);
        });
    });
}

function showPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });

    // Show selected page
    const page = document.getElementById(`${pageName}-page`);
    if (page) {
        page.classList.add('active');

        // Load page-specific data
        switch (pageName) {
            case 'dashboard':
                loadDashboard();
                break;
            case 'students':
                loadStudents();
                break;
            case 'reports':
                loadReports();
                break;
        }
    }
}

// Dashboard Functions
async function loadDashboard() {
    try {
        // Load statistics
        const statsResponse = await fetch(`${API_BASE_URL}/api/statistics/overall`);
        const statsData = await statsResponse.json();

        if (statsData.success) {
            const stats = statsData.data;
            document.getElementById('total-students').textContent = stats.total_students || 0;
            document.getElementById('today-present').textContent = stats.today_attendance || 0;
            document.getElementById('today-absent').textContent =
                (stats.total_students || 0) - (stats.today_attendance || 0);
            document.getElementById('attendance-rate').textContent =
                `${stats.today_percentage ? stats.today_percentage.toFixed(1) : 0}%`;
        }

        // Load today's attendance
        const attendanceResponse = await fetch(`${API_BASE_URL}/api/attendance/today`);
        const attendanceData = await attendanceResponse.json();

        if (attendanceData.success) {
            displayAttendanceList(attendanceData.data, 'today-attendance-list');
        }
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showToast('Error loading dashboard data', 'error');
    }
}

async function refreshDashboard() {
    showLoading();
    await loadDashboard();
    hideLoading();
    showToast('Dashboard refreshed', 'success');
}

// Student Registration Functions
async function startCamera(mode) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });

        currentStream = stream;
        const video = document.getElementById(`${mode}-video`);
        video.srcObject = stream;

        // Enable capture button
        if (mode === 'register') {
            document.getElementById('capture-btn').disabled = false;
            document.getElementById('start-camera-btn').disabled = true;
        } else {
            document.getElementById('mark-attendance-btn').disabled = false;
            document.getElementById('start-attendance-camera-btn').disabled = true;
        }

        showToast('Camera started', 'success');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showToast('Error accessing camera. Please check permissions.', 'error');
    }
}

function captureImage() {
    const video = document.getElementById('register-video');
    const canvas = document.getElementById('register-canvas');
    const preview = document.getElementById('register-preview');
    const previewImg = document.getElementById('register-preview-img');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Convert to blob
    canvas.toBlob((blob) => {
        capturedImage = blob;

        // Show preview
        const url = URL.createObjectURL(blob);
        previewImg.src = url;
        preview.style.display = 'block';
        video.style.display = 'none';

        // Stop camera
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }

        // Enable register button
        document.getElementById('register-btn').disabled = false;
        document.getElementById('capture-btn').disabled = true;

        showToast('Image captured', 'success');
    }, 'image/jpeg');
}

async function registerStudent() {
    // Validate form
    const studentId = document.getElementById('student-id').value.trim();
    const studentName = document.getElementById('student-name').value.trim();

    if (!studentId || !studentName) {
        showToast('Please fill in required fields', 'error');
        return;
    }

    if (!capturedImage) {
        showToast('Please capture an image first', 'error');
        return;
    }

    showLoading();

    try {
        // Step 1: Create student
        const studentData = {
            student_id: studentId,
            name: studentName,
            email: document.getElementById('student-email').value.trim() || null,
            enrollment_number: document.getElementById('student-enrollment').value.trim() || null,
            department: document.getElementById('student-department').value.trim() || null,
            year: document.getElementById('student-year').value.trim() || null
        };

        const params = new URLSearchParams(studentData);
        const createResponse = await fetch(`${API_BASE_URL}/api/students?${params}`, {
            method: 'POST'
        });

        const createData = await createResponse.json();

        if (!createData.success) {
            throw new Error(createData.detail || 'Failed to create student');
        }

        // Step 2: Register face
        const formData = new FormData();
        formData.append('file', capturedImage, 'face.jpg');

        const faceResponse = await fetch(
            `${API_BASE_URL}/api/students/${studentId}/register-face`,
            {
                method: 'POST',
                body: formData
            }
        );

        const faceData = await faceResponse.json();

        if (!faceData.success) {
            throw new Error(faceData.detail || 'Failed to register face');
        }

        hideLoading();
        showToast(`Student ${studentName} registered successfully!`, 'success');

        // Reset form
        document.getElementById('student-form').reset();
        document.getElementById('register-preview').style.display = 'none';
        document.getElementById('register-video').style.display = 'block';
        document.getElementById('register-btn').disabled = true;
        document.getElementById('start-camera-btn').disabled = false;
        capturedImage = null;

    } catch (error) {
        hideLoading();
        console.error('Error registering student:', error);
        showToast(error.message, 'error');
    }
}

// Attendance Functions
async function markAttendance() {
    const video = document.getElementById('attendance-video');
    const canvas = document.getElementById('attendance-canvas');

    // Set canvas size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Capture frame
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    showLoading();

    // Convert to blob and upload
    canvas.toBlob(async (blob) => {
        try {
            const formData = new FormData();
            formData.append('file', blob, 'attendance.jpg');

            const response = await fetch(`${API_BASE_URL}/api/attendance/mark`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            hideLoading();

            if (data.success) {
                showToast(
                    `Attendance marked for ${data.data.student_name}!`,
                    'success'
                );
                loadRecentAttendance();
            } else if (data.duplicate) {
                showToast(
                    `Attendance already marked for ${data.data.student_name}`,
                    'info'
                );
            } else {
                showToast(data.detail || 'Student not recognized', 'error');
            }
        } catch (error) {
            hideLoading();
            console.error('Error marking attendance:', error);
            showToast('Error marking attendance', 'error');
        }
    }, 'image/jpeg');
}

async function loadRecentAttendance() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/attendance/today`);
        const data = await response.json();

        if (data.success) {
            displayAttendanceList(data.data, 'recent-attendance-list');
        }
    } catch (error) {
        console.error('Error loading recent attendance:', error);
    }
}

// Students Functions
async function loadStudents() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/students`);
        const data = await response.json();

        if (data.success) {
            displayStudentsList(data.data);
        }
    } catch (error) {
        console.error('Error loading students:', error);
        showToast('Error loading students', 'error');
    }
}

function displayStudentsList(students) {
    const container = document.getElementById('students-list');

    if (students.length === 0) {
        container.innerHTML = '<p class="text-center text-muted">No students registered yet</p>';
        return;
    }

    container.innerHTML = students.map(student => `
        <div class="student-card">
            <h3>${student.name}</h3>
            <p><strong>ID:</strong> ${student.student_id}</p>
            ${student.email ? `<p><strong>Email:</strong> ${student.email}</p>` : ''}
            ${student.department ? `<p><strong>Department:</strong> ${student.department}</p>` : ''}
            ${student.year ? `<p><strong>Year:</strong> ${student.year}</p>` : ''}
            <span class="badge ${student.face_registered ? 'badge-registered' : 'badge-not-registered'}">
                ${student.face_registered ? '✓ Face Registered' : '✗ No Face'}
            </span>
        </div>
    `).join('');
}

function searchStudents() {
    const query = document.getElementById('search-input').value.toLowerCase();
    const cards = document.querySelectorAll('.student-card');

    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(query) ? 'block' : 'none';
    });
}

// Reports Functions
async function loadReports() {
    await loadDateAttendance();
    await loadStudentStats();
}

async function loadDateAttendance() {
    const date = document.getElementById('report-date').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/attendance/date/${date}`);
        const data = await response.json();

        if (data.success) {
            displayAttendanceList(data.data, 'date-attendance-list');
        }
    } catch (error) {
        console.error('Error loading date attendance:', error);
    }
}

async function loadStudentStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/students`);
        const data = await response.json();

        if (!data.success) return;

        const container = document.getElementById('student-stats-list');

        if (data.data.length === 0) {
            container.innerHTML = '<p class="text-center text-muted">No students found</p>';
            return;
        }

        // Get stats for each student
        const statsPromises = data.data.map(async student => {
            try {
                const statsResponse = await fetch(
                    `${API_BASE_URL}/api/statistics/student/${student.student_id}`
                );
                const statsData = await statsResponse.json();
                return {
                    student,
                    stats: statsData.success ? statsData.data : null
                };
            } catch {
                return { student, stats: null };
            }
        });

        const results = await Promise.all(statsPromises);

        container.innerHTML = results.map(({ student, stats }) => `
            <div class="attendance-item">
                <div class="attendance-info">
                    <h4>${student.name}</h4>
                    <p>${student.student_id} • ${student.department || 'N/A'}</p>
                </div>
                <div>
                    ${stats ? `
                        <p><strong>${stats.attendance_percentage.toFixed(1)}%</strong></p>
                        <p class="text-muted">${stats.present_days}/${stats.total_days} days</p>
                    ` : '<p class="text-muted">No data</p>'}
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading student stats:', error);
    }
}

// Helper Functions
function displayAttendanceList(records, containerId) {
    const container = document.getElementById(containerId);

    if (records.length === 0) {
        container.innerHTML = '<p class="text-center text-muted">No attendance records</p>';
        return;
    }

    container.innerHTML = records.map(record => `
        <div class="attendance-item">
            <div class="attendance-info">
                <h4>${record.student_name || 'Unknown'}</h4>
                <p>${record.time} • Confidence: ${record.recognition_confidence
            ? (record.recognition_confidence * 100).toFixed(1) + '%'
            : 'N/A'
        }</p>
            </div>
            <span class="attendance-badge badge-${record.status}">
                ${record.status.toUpperCase()}
            </span>
        </div>
    `).join('');
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show status-${type}`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
});
