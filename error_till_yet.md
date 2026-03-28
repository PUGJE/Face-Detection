# Codebase Error and Issue Report

This report lists identified errors, potential bugs, and architectural issues found in the Face Recognition Attendance System codebase.

## 1. Critical Dependency Issues
*   **Missing `scikit-image`**: The `backend/ml/anti_spoofing.py` module depends on `skimage.feature.local_binary_pattern`. If this package is not installed in the environment, the anti-spoofing module and any pipeline using it will fail to import.
*   **Missing `scipy`**: `backend/ml/anti_spoofing.py` imports `scipy.ndimage`, which is required for its frequency and texture analysis functions.
*   **DeepFace Model Downloads**: `backend/ml/face_recognition.py` uses `DeepFace`. By default, DeepFace downloads large model weights (e.g., Facenet, VGG-Face) to the user's home directory (`~/.deepface/weights`). This can cause issues in environments with restricted internet access or limited disk space if not pre-cached.

## 2. Identified Logic & Implementation Issues
*   **Anti-Spoofing "Skipped" Status**: While `backend/ml/anti_spoofing.py` is implemented, the `SYSTEM_VALIDATION_REPORT.md` indicates it was skipped in Step 5. This suggests it may not be fully validated or may be disabled in certain configurations.
*   **Motion Detection Inconsistency**: In `backend/ml/anti_spoofing.py`, `check_liveness` has `use_motion=False` by default. The `AttendanceSystem.process_webcam_attendance` method does not explicitly leverage the motion detection capabilities of the `AntiSpoofingDetector`, relying primarily on static texture analysis.
*   **Database Initialization Timing**: `AttendanceSystem.__init__` calls `init_database()`, which creates tables. If the database file is locked or the path is unwritable, the system will fail at initialization.
*   **Pickle Security**: `FaceRecognizer` uses the `pickle` module to save and load embeddings (`save_embeddings`/`load_embeddings`). This is a security risk if the `.pkl` file is tampered with, as it can execute arbitrary code during loading.
*   **Soft Error Handling**: Several methods in `anti_spoofing.py` catch all exceptions and return a default score of `0.5`. This may mask library failures (like missing `skimage`) and lead to unreliable liveness scores without clear error signals.

## 3. Configuration & Environment Issues
*   **Hardcoded Database URL**: `backend/config.py` defaults to `sqlite:///./data/database/attendance.db`. This relative path depends on the current working directory, which can cause "Database not found" errors if the application is started from a different folder.
*   **Debug Mode Enabled**: `backend/config.py` defaults `debug` to `True`, and `main.py` runs uvicorn with `reload=True`. These should be changed for production environments.
*   **Camera Index**: Default `CAMERA_INDEX` is `0`. On systems with multiple cameras or no camera, this will cause a `ConnectionRefusedError` or "Could not open camera" error as noted in the `DEPLOYMENT_GUIDE.md`.

## 4. Architectural & Design Gaps
*   **Lack of Authentication**: The API (in `main.py`) lacks any authentication or authorization mechanisms. All endpoints, including student creation and attendance marking, are publicly accessible.
*   **Route Consolidation**: All API routes are currently implemented directly in `main.py`. The `backend/api/routes` directory is empty (only `__init__.py`), which deviates from standard FastAPI modular structure for larger projects.
*   **Face Database Warning**: The reported "No faces registered yet" warning in `SYSTEM_VALIDATION_REPORT.md` indicates a discrepancy between the test environment and the runtime environment regarding how face embeddings are persisted and checked.

## 5. Potential Performance Bottlenecks
*   **Synchronous Image Processing**: Many image processing tasks (detection, recognition) are CPU/GPU intensive but are called within the same process as the FastAPI request handlers. Under high load, this could block the event loop if not handled with thread pools or separate workers.
