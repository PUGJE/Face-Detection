"use client";

/**
 * Students Page — manage students, register Face ID, delete records
 * Unified dark glass UI matching the rest of the app.
 */

import { useState, useEffect, useRef } from "react";
import { UserPlus, Camera, UploadCloud, Loader2, Trash2, RefreshCw, Lock, Eye, EyeOff } from "lucide-react";
import { useMediaPipeDetector } from "@/hooks/useMediaPipeDetector";

const ADMIN_PASSWORD = process.env.NEXT_PUBLIC_ADMIN_PASSWORD ?? "admin123";

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

export default function StudentsPage() {
  const [students,  setStudents]  = useState<Student[]>([]);
  const [loading,   setLoading]   = useState(true);
  const [studentId, setStudentId] = useState("");
  const [name,      setName]      = useState("");
  const [department, setDepartment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [deleteTarget,   setDeleteTarget]   = useState<Student | null>(null);
  const [deleting,       setDeleting]       = useState(false);
  const [deletePassword, setDeletePassword] = useState("");
  const [deletePassErr,  setDeletePassErr]  = useState("");
  const [showDelPass,    setShowDelPass]    = useState(false);

  function openDeleteModal(s: Student) {
    setDeleteTarget(s);
    setDeletePassword("");
    setDeletePassErr("");
    setShowDelPass(false);
  }

  function closeDeleteModal() {
    setDeleteTarget(null);
    setDeletePassword("");
    setDeletePassErr("");
  }

  const videoRef     = useRef<HTMLVideoElement>(null);
  const [stream,       setStream]       = useState<MediaStream | null>(null);
  const [capturingFor, setCapturingFor] = useState<string | null>(null);
  const [registering,  setRegistering]  = useState(false);

  const { detectorRef, detectorStatus } = useMediaPipeDetector();

  useEffect(() => {
    fetchStudents();
    return stopWebcam;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function fetchStudents() {
    setLoading(true);
    try {
      const res  = await fetch("/api/students");
      const data = await res.json();
      if (data.success) setStudents(data.data);
    } catch (err) {
      console.error("Failed to fetch students:", err);
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateStudent(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      const params = new URLSearchParams({ student_id: studentId, name, department });
      const res    = await fetch(`/api/students?${params}`, { method: "POST" });
      if (res.ok) {
        setStudentId(""); setName(""); setDepartment("");
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

  async function handleDelete() {
    if (!deleteTarget) return;

    // Verify admin password client-side before sending the request
    if (deletePassword !== ADMIN_PASSWORD) {
      setDeletePassErr("Incorrect password");
      setDeletePassword("");
      return;
    }

    setDeleting(true);
    try {
      const res = await fetch(`/api/students/${deleteTarget.student_id}`, { method: "DELETE" });
      if (res.ok) {
        closeDeleteModal();
        await fetchStudents();
      } else {
        const data = await res.json();
        alert(`Delete failed: ${data.detail}`);
      }
    } catch {
      alert("Network error while deleting student.");
    } finally {
      setDeleting(false);
    }
  }

  async function startWebcam(id: string) {
    setCapturingFor(id);
    try {
      const ms = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(ms);
      if (videoRef.current) videoRef.current.srcObject = ms;
    } catch {
      alert("Could not access camera. Please allow camera permission.");
    }
  }

  function stopWebcam() {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    setCapturingFor(null);
  }

  async function captureAndRegister() {
    if (!videoRef.current || !capturingFor) return;
    if (detectorStatus !== "ready" || !detectorRef.current) {
      alert("Face detector is still loading. Please wait.");
      return;
    }

    setRegistering(true);
    const canvas = document.createElement("canvas");
    canvas.width  = videoRef.current.videoWidth  || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) { setRegistering(false); return; }
    ctx.drawImage(videoRef.current, 0, 0);

    try {
      const { detections } = detectorRef.current.detect(canvas);
      if (!detections?.length) {
        alert("No clear face detected! Please look straight into the camera.");
        return;
      }
      const face = detections[0].boundingBox;
      if (!face) { alert("Could not extract face bounds."); return; }

      const padX = face.width  * 0.25;
      const padY = face.height * 0.25;
      const sx   = Math.max(0, face.originX - padX);
      const sy   = Math.max(0, face.originY - padY);
      const sw   = Math.min(canvas.width  - sx, face.width  + 2 * padX);
      const sh   = Math.min(canvas.height - sy, face.height + 2 * padY);

      const crop = document.createElement("canvas");
      crop.width  = 112; crop.height = 112;
      crop.getContext("2d")?.drawImage(canvas, sx, sy, sw, sh, 0, 0, 112, 112);

      crop.toBlob(async (blob) => {
        if (!blob) { setRegistering(false); return; }
        const form = new FormData();
        form.append("file", blob, "crop.jpg");

        try {
          const res = await fetch(`/api/students/${capturingFor}/register-face-crop`, {
            method: "POST", body: form,
          });
          if (res.ok) {
            alert("Face registered successfully!");
            stopWebcam();
            await fetchStudents();
          } else {
            const data = await res.json();
            alert(`Error: ${data.detail}`);
          }
        } catch {
          alert("Network error while registering face.");
        } finally {
          setRegistering(false);
        }
      }, "image/jpeg", 0.7);
    } catch {
      alert("Error processing the image.");
      setRegistering(false);
    }
  }

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Students</h1>
          <p className="text-slate-500 mt-1">Manage students and register Face ID</p>
        </div>
        <button
          onClick={fetchStudents}
          className="flex items-center gap-2 px-4 py-2 bg-slate-50 hover:bg-slate-700 rounded-2xl text-slate-600 text-sm font-medium transition"
        >
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {/* Delete confirmation modal — password protected */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel border border-slate-200 rounded-2xl p-8 max-w-sm w-full shadow-2xl">
            <Trash2 className="w-12 h-12 text-red-600 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-slate-800 text-center mb-1">Delete Student?</h3>
            <p className="text-slate-500 text-sm text-center mb-5">
              <span className="text-slate-800 font-medium">{deleteTarget.name}</span> ({deleteTarget.student_id})<br />
              This will soft-delete the record. Face data will be removed.
            </p>

            {/* Password gate */}
            <div className="mb-5">
              <label className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                <Lock className="w-3.5 h-3.5" /> Admin Password Required
              </label>
              <div className="relative">
                <input
                  id="delete-password-input"
                  autoFocus
                  type={showDelPass ? "text" : "password"}
                  value={deletePassword}
                  onChange={(e) => { setDeletePassword(e.target.value); setDeletePassErr(""); }}
                  onKeyDown={(e) => e.key === "Enter" && handleDelete()}
                  placeholder="Enter admin password"
                  className={`w-full px-4 py-2.5 pr-10 bg-slate-50/60 border rounded-2xl text-slate-800
                    placeholder-slate-500 focus:outline-none focus:ring-2 transition text-sm
                    ${deletePassErr
                      ? "border-red-200 focus:ring-red-500/30"
                      : "border-slate-200 focus:ring-red-500/30"
                    }`}
                />
                <button
                  type="button"
                  tabIndex={-1}
                  onClick={() => setShowDelPass(!showDelPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-600 transition"
                >
                  {showDelPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {deletePassErr && (
                <p className="text-red-600 text-xs font-medium mt-1.5">{deletePassErr}</p>
              )}
            </div>

            <div className="flex gap-3">
              <button
                onClick={closeDeleteModal}
                className="flex-1 py-2.5 rounded-2xl bg-slate-50 hover:bg-slate-700 text-slate-600 font-medium transition"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting || !deletePassword}
                className="flex-1 py-2.5 rounded-2xl bg-red-600 hover:bg-red-500 text-slate-800 font-semibold transition flex items-center justify-center gap-2 disabled:opacity-60"
              >
                {deleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Registration form */}
        <div className="glass-panel rounded-2xl p-6 h-fit border border-purple-200">
          <div className="flex items-center gap-3 mb-6">
            <UserPlus className="text-purple-600 w-6 h-6" />
            <h2 className="text-xl font-bold text-slate-800">Add New Student</h2>
          </div>

          <form onSubmit={handleCreateStudent} className="space-y-4">
            {[
              { label: "Student ID", value: studentId, setter: setStudentId, placeholder: "CS-2026-001", required: true },
              { label: "Full Name",  value: name,      setter: setName,      placeholder: "John Doe",      required: true },
              { label: "Department", value: department, setter: setDepartment, placeholder: "Computer Science", required: false },
            ].map(({ label, value, setter, placeholder, required }) => (
              <div key={label}>
                <label className="text-slate-500 text-sm block mb-1">{label}</label>
                <input
                  required={required}
                  value={value}
                  onChange={(e) => setter(e.target.value)}
                  placeholder={placeholder}
                  className="w-full bg-slate-50/50 border border-slate-200 rounded-2xl px-4 py-2.5 text-slate-800 placeholder-slate-500 outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 transition"
                />
              </div>
            ))}
            <button
              disabled={submitting}
              type="submit"
              className="w-full mt-4 bg-purple-600 hover:bg-purple-700 transition text-slate-800 font-bold py-2.5 rounded-2xl flex justify-center items-center gap-2"
            >
              {submitting ? <Loader2 className="animate-spin w-5 h-5" /> : <UserPlus className="w-5 h-5" />}
              Create Student
            </button>
          </form>
        </div>

        {/* Student list */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-slate-800">Registered Database</h2>
            {detectorStatus === "loading" && (
              <span className="flex items-center gap-2 text-sm text-amber-600 bg-amber-400/10 px-3 py-1.5 rounded-full border border-amber-400/20">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading ML Engine…
              </span>
            )}
          </div>

          {/* Webcam panel */}
          {capturingFor && (
            <div className="mb-6 p-4 rounded-2xl bg-white/50 border border-slate-200">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-[#10B981] font-bold flex items-center gap-2">
                  <Camera className="w-5 h-5" /> Registering: {capturingFor}
                </h3>
                <button onClick={stopWebcam} className="text-slate-500 hover:text-slate-800 transition text-sm">
                  Cancel
                </button>
              </div>
              <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden border border-slate-200">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
              </div>
              <button
                disabled={registering || detectorStatus !== "ready"}
                onClick={captureAndRegister}
                className="mt-4 w-full bg-emerald-600 hover:bg-emerald-700 transition text-slate-800 font-bold py-3 rounded-2xl flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {registering ? <Loader2 className="animate-spin" /> : <UploadCloud />}
                Capture & Register
              </button>
            </div>
          )}

          {/* Table */}
          <div className="overflow-x-auto rounded-2xl border border-slate-200/50">
            <table className="w-full text-left text-slate-600">
              <thead className="bg-slate-50/60 text-slate-500 text-xs font-mono uppercase">
                <tr>
                  <th className="px-6 py-4">ID</th>
                  <th className="px-6 py-4">Name</th>
                  <th className="px-6 py-4">Dept</th>
                  <th className="px-6 py-4">Face ID</th>
                  <th className="px-6 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/40">
                {loading ? (
                  <tr><td colSpan={5} className="text-center py-10">
                    <Loader2 className="animate-spin mx-auto w-6 h-6 text-slate-500" />
                  </td></tr>
                ) : students.length === 0 ? (
                  <tr><td colSpan={5} className="text-center py-10 text-slate-500">No students found.</td></tr>
                ) : (
                  students.map((s) => (
                    <tr key={s.student_id} className="hover:bg-slate-50/30 transition-colors group">
                      <td className="px-6 py-4 font-mono text-slate-800 font-medium text-sm">{s.student_id}</td>
                      <td className="px-6 py-4 font-medium">{s.name}</td>
                      <td className="px-6 py-4 text-slate-500">{s.department ?? "—"}</td>
                      <td className="px-6 py-4">
                        {s.face_registered ? (
                          <span className="text-[#10B981] bg-emerald-400/10 px-2.5 py-1 rounded-full text-xs font-bold border border-emerald-400/20">
                            ✓ Registered
                          </span>
                        ) : (
                          <span className="text-rose-400 bg-rose-400/10 px-2.5 py-1 rounded-full text-xs font-bold border border-rose-400/20">
                            Pending
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={() => startWebcam(s.student_id)}
                            disabled={detectorStatus !== "ready"}
                            className="px-3 py-1.5 rounded-2xl bg-blue-50 hover:bg-[#2563EB]/20 text-blue-600 text-xs font-semibold border border-blue-500/20 transition disabled:opacity-40"
                          >
                            Add Face
                          </button>
                          <button
                            onClick={() => openDeleteModal(s)}
                            className="p-1.5 rounded-2xl bg-red-50 hover:bg-red-50 text-red-600 border border-red-500/20 transition"
                            title="Delete student"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
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
