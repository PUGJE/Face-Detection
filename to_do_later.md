# University-Scale Deployment: To-Do List

This document outlines the critical upgrades and tasks required to transition the Face Recognition Attendance System from a local prototype to a university-scale production environment (5,000+ students).

## 1. Core Architecture Upgrades
- [ ] **Database Migration:** Replace SQLite with **PostgreSQL** or **MySQL** to handle concurrent writes and high traffic.
- [ ] **Vector Database Integration:** Replace the `.pkl` (Pickle) linear search with a dedicated vector database like **Milvus**, **Pinecone**, or **pgvector** (HNSW indexing) for $O(\log N)$ search speeds.
- [ ] **API Security:** Implement **OAuth2 with JWT** authentication for all endpoints (currently missing).
- [ ] **Containerization:** Create **Docker** images for the backend, frontend, and database services.
- [ ] **Orchestration:** Set up **Kubernetes** (K8s) for auto-scaling recognition workers during peak attendance hours (e.g., 8:00 AM - 10:00 AM).

## 2. Machine Learning & Performance
- [ ] **GPU Acceleration:** Deploy on servers with **NVIDIA GPUs** and configure `tf-keras` with CUDA/cuDNN for <50ms recognition.
- [ ] **Edge Processing:** Update the Frontend JS to perform **Face Detection and Alignment** in the browser to reduce server CPU load and bandwidth.
- [ ] **Anti-Spoofing Hardening:** Fully validate and enable the `AntiSpoofingDetector` in the main pipeline.
- [ ] **Face Quality Filtering:** Implement a check to reject blurry or poorly lit faces before they hit the recognition engine.

## 3. Infrastructure & Hardware
- [ ] **Load Balancing:** Use **Nginx** or a cloud load balancer (AWS ALB/GCP) to distribute requests across multiple FastAPI instances.
- [ ] **Fixed Terminals:** Research and test **Jetson Nano** or **Raspberry Pi 5** as eye-level attendance kiosks.
- [ ] **Mobile Support:** Develop a lightweight mobile app for professors to take attendance in smaller or remote classrooms.

## 4. Privacy & Compliance
- [ ] **Encryption at Rest:** Ensure all student embeddings are encrypted in the database.
- [ ] **Data Retention Policy:** Implement automated scripts to purge attendance logs according to university privacy regulations.
- [ ] **Anonymization:** Never store raw face images; store only the 128-dimensional embeddings.

## 5. Deployment Readiness
- [ ] **CI/CD Pipeline:** Set up GitHub Actions or GitLab CI for automated testing and deployment.
- [ ] **Monitoring & Logging:** Integrate **Prometheus** and **Grafana** to monitor API latency and recognition accuracy in real-time.
