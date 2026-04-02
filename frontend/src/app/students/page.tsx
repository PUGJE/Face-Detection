"use client";

/**
 * Students Page
 *
 * Manages student records and face registration.
 *
 * Features:
 *  - Create new student (form)
 *  - List all registered students
 *  - Open webcam and capture a face crop to register via ArcFace (browser-crop path)
 */

import { useState, useEffect, useRef } from "react";
import { UserPlus, Camera, UploadCloud, Loader2 } from "lucide-react";
import { useMediaPipeDetector } from "@/hooks/useMediaPipeDetector";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Student {
  student_id: string;
  name: string;
  email: string | null;
  enrollment_number: string | null;
  department: string | null;
  year: string | null;
  face_registered: boolean;
  is_active: boolean;
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function StudentsPage() {
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(true);

  // Registration form state
  const [studentId, setStudentId] = useState("");
  const [name, setName] = useState("");
  const [department, setDepartment] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Webcam state
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturingFor, setCapturingFor] = useState<string | null>(null); // student_id being registered
  const [registering, setRegistering] = useState(false);

  // Shared MediaPipe detector — loaded once on mount
  const { detectorRef, detectorStatus } = useMediaPipeDetector();

  useEffect(() => {
    fetchStudents();
    return stopWebcam;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ------------------------------------------------------------------
  // Data fetching
  // ------------------------------------------------------------------

  async function fetchStudents() {
    setLoading(true);
    try {
      const res = await fetch("/api/students");
      const data = await res.json();
      if (data.success) setStudents(data.data);
    } catch (err) {
      console.error("Failed to fetch students:", err);
    } finally {
      setLoading(false);
    }
  }

  // ------------------------------------------------------------------
  // Student creation
  // ------------------------------------------------------------------

  async function handleCreateStudent(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      const params = new URLSearchParams({ student_id: studentId, name, department });
      const res = await fetch(`/api/students?${params}`, { method: "POST" });
      if (res.ok) {
        setStudentId("");
        setName("");
        setDepartment("");
        await fetchStudents();
      } else {
        const data = await res.json();
        alert(`Error: ${data.detail}`);
      }
    } catch (err) {
      console.error("Failed to create student:", err);
    } finally {
      setSubmitting(false);
    }
  }

  // ------------------------------------------------------------------
  // Webcam management
  // ------------------------------------------------------------------

  async function startWebcam(id: string) {
    setCapturingFor(id);
    try {
      const ms = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(ms);
      if (videoRef.current) {
        videoRef.current.srcObject = ms;
      }
    } catch (err) {
      console.error("Webcam access denied:", err);
      alert("Could not access camera. Please allow camera permission.");
    }
  }

  function stopWebcam() {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    setCapturingFor(null);
  }

  // ------------------------------------------------------------------
  // Face capture and registration
  // ------------------------------------------------------------------

  async function captureAndRegister() {
    if (!videoRef.current || !capturingFor) return;

    if (detectorStatus !== "ready" || !detectorRef.current) {
      alert("Face detector is still loading. Please wait.");
      return;
    }

    setRegistering(true);

    // Snapshot the current video frame
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setRegistering(false);
      return;
    }
    ctx.drawImage(videoRef.current, 0, 0);

    try {
      // 1. Client-side face detection (browser, no server round-trip)
      const { detections } = detectorRef.current.detect(canvas);
      if (!detections?.length) {
        alert("No clear face detected! Please look straight into the camera.");
        return;
      }

      const face = detections[0].boundingBox;
      if (!face) {
        alert("Could not extract face bounds.");
        return;
      }

      // 2. Crop with 25% padding so ArcFace gets enough facial context
      const padX = face.width * 0.25;
      const padY = face.height * 0.25;
      const sx = Math.max(0, face.originX - padX);
      const sy = Math.max(0, face.originY - padY);
      const sw = Math.min(canvas.width - sx, face.width + 2 * padX);
      const sh = Math.min(canvas.height - sy, face.height + 2 * padY);

      const cropCanvas = document.createElement("canvas");
      cropCanvas.width = sw;
      cropCanvas.height = sh;
      cropCanvas.getContext("2d")?.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);

      // 3. Send the cropped face to the server — ArcFace only (~5 ms)
      cropCanvas.toBlob(async (blob) => {
        if (!blob) {
          setRegistering(false);
          return;
        }
        const form = new FormData();
        form.append("file", blob, "crop.jpg");

        try {
          const res = await fetch(
            `/api/students/${capturingFor}/register-face-crop`,
            { method: "POST", body: form }
          );
          if (res.ok) {
            alert("Face registered successfully!");
            stopWebcam();
            await fetchStudents();
          } else {
            const data = await res.json();
            alert(`Error: ${data.detail}`);
          }
        } catch (err) {
          console.error("Registration request failed:", err);
          alert("Network error while registering face.");
        } finally {
          setRegistering(false);
        }
      }, "image/jpeg", 0.9);
    } catch (err) {
      console.error("Error processing image:", err);
      alert("Error processing the image.");
      setRegistering(false);
    }
  }

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Students</h1>
          <p className="text-slate-400">Manage students and register Face ID models</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* ── Registration form ─────────────────────────────────────── */}
        <div className="glass-panel rounded-2xl p-6 h-fit border-purple-500/30 border-2">
          <div className="flex items-center space-x-3 mb-6">
            <UserPlus className="text-purple-400 w-6 h-6" />
            <h2 className="text-xl font-bold text-white">Add New Student</h2>
          </div>

          <form onSubmit={handleCreateStudent} className="space-y-4">
            <div>
              <label className="text-slate-400 text-sm">Student ID</label>
              <input
                required
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
                type="text"
                className="w-full bg-slate-800/50 mt-1 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:border-purple-500"
                placeholder="e.g. CS-2026-001"
              />
            </div>
            <div>
              <label className="text-slate-400 text-sm">Full Name</label>
              <input
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                type="text"
                className="w-full bg-slate-800/50 mt-1 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:border-purple-500"
                placeholder="John Doe"
              />
            </div>
            <div>
              <label className="text-slate-400 text-sm">Department</label>
              <input
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                type="text"
                className="w-full bg-slate-800/50 mt-1 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:border-purple-500"
                placeholder="Computer Science"
              />
            </div>
            <button
              disabled={submitting}
              type="submit"
              className="w-full mt-4 bg-purple-600 hover:bg-purple-700 transition-colors text-white font-bold py-2 rounded-lg flex justify-center"
            >
              {submitting ? <Loader2 className="animate-spin w-5 h-5" /> : "Create Student"}
            </button>
          </form>
        </div>

        {/* ── Student list & face registration ─────────────────────── */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-white">Registered Database</h2>
            {detectorStatus === "loading" && (
              <span className="flex items-center gap-2 text-sm text-amber-400 bg-amber-400/10 px-3 py-1.5 rounded-full">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading ML Engine…
              </span>
            )}
          </div>

          {/* Webcam panel — shown when a student is being registered */}
          {capturingFor && (
            <div className="mb-6 p-4 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-emerald-400 font-bold flex items-center gap-2">
                  <Camera className="w-5 h-5" /> Registering: {capturingFor}
                </h3>
                <button onClick={stopWebcam} className="text-slate-400 hover:text-white">
                  Cancel
                </button>
              </div>
              <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-slate-700">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
              </div>
              <button
                disabled={registering || detectorStatus !== "ready"}
                onClick={captureAndRegister}
                className="mt-4 w-full bg-emerald-600 hover:bg-emerald-700 transition text-white font-bold py-3 rounded-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {registering ? <Loader2 className="animate-spin" /> : <UploadCloud />}
                Capture &amp; Register
              </button>
            </div>
          )}

          {/* Student table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-slate-300">
              <thead className="bg-slate-800/50 text-slate-400 uppercase text-xs">
                <tr>
                  <th className="px-4 py-3 rounded-tl-lg">ID</th>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Dept</th>
                  <th className="px-4 py-3">Face ID</th>
                  <th className="px-4 py-3 rounded-tr-lg">Action</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={5} className="text-center py-8">
                      <Loader2 className="animate-spin mx-auto w-6 h-6 text-slate-500" />
                    </td>
                  </tr>
                ) : students.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="text-center py-8 text-slate-500">
                      No students found.
                    </td>
                  </tr>
                ) : (
                  students.map((s) => (
                    <tr
                      key={s.student_id}
                      className="border-b border-slate-700/50 hover:bg-slate-800/30"
                    >
                      <td className="px-4 py-3 font-medium text-white">{s.student_id}</td>
                      <td className="px-4 py-3">{s.name}</td>
                      <td className="px-4 py-3">{s.department ?? "—"}</td>
                      <td className="px-4 py-3">
                        {s.face_registered ? (
                          <span className="text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded text-xs font-bold">
                            Registered
                          </span>
                        ) : (
                          <span className="text-rose-400 bg-rose-400/10 px-2 py-1 rounded text-xs font-bold">
                            Pending
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => startWebcam(s.student_id)}
                          disabled={detectorStatus !== "ready"}
                          className="text-blue-400 hover:text-blue-300 text-sm font-medium disabled:opacity-50"
                        >
                          Add Face
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
