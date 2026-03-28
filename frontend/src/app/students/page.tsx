"use client";
import { useState, useEffect, useRef } from "react";
import { UserPlus, Camera, UploadCloud, Loader2 } from "lucide-react";

export default function StudentsPage() {
  const [students, setStudents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Registration Form State
  const [name, setName] = useState("");
  const [studentId, setStudentId] = useState("");
  const [department, setDepartment] = useState("");
  
  // Webcam State
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [registering, setRegistering] = useState(false);
  const [selectedStudentId, setSelectedStudentId] = useState<string | null>(null);

  // Detector State
  const detectorRef = useRef<any>(null);
  const [detectorStatus, setDetectorStatus] = useState<"loading" | "ready">("loading");

  useEffect(() => {
    fetchStudents();
    initMediaPipe();
    return () => {
      stopWebcam();
      if (detectorRef.current && typeof detectorRef.current.close === "function") {
        detectorRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const initMediaPipe = async () => {
    try {
      const { FaceDetector, FilesetResolver } = await import("@mediapipe/tasks-vision");
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const fd = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
          delegate: "GPU",
        },
        runningMode: "IMAGE",
      });
      detectorRef.current = fd;
      setDetectorStatus("ready");
    } catch (e) {
      console.error("MediaPipe failed to load:", e);
    }
  };

  const fetchStudents = async () => {
    try {
      const res = await fetch("/api/students");
      const data = await res.json();
      if (data.success) setStudents(data.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const createStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    setRegistering(true);
    try {
      const res = await fetch(`/api/students?student_id=${studentId}&name=${name}&department=${department}`, {
        method: "POST"
      });
      if (res.ok) {
        setName(""); setStudentId(""); setDepartment("");
        fetchStudents();
      }
    } catch (e) {
      console.error(e);
    } finally {
      setRegistering(false);
    }
  };

  const startWebcam = async (id: string) => {
    setSelectedStudentId(id);
    try {
      const ms = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(ms);
      if (videoRef.current) {
        videoRef.current.srcObject = ms;
      }
    } catch (e) {
      console.error("Webcam access denied", e);
    }
  };

  const captureAndRegister = async () => {
    if (!videoRef.current || !selectedStudentId) return;
    if (detectorStatus !== "ready" || !detectorRef.current) {
      alert("Face detector is still loading. Please wait.");
      return;
    }
    
    setRegistering(true);
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) { setRegistering(false); return; }
    ctx.drawImage(videoRef.current, 0, 0);

    try {
      // 1. Client-side face detection
      const { detections } = detectorRef.current.detect(canvas);
      
      if (!detections || detections.length === 0) {
        alert("No clear face detected! Please look straight into the camera.");
        setRegistering(false);
        return;
      }
      
      const face = detections[0].boundingBox;
      if (!face) {
        alert("Could not extract face bounds.");
        setRegistering(false);
        return;
      }

      // 2. Crop the face (with 25% padding so ArcFace sees full context)
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
      
      // 3. Send ONLY the cropped 112x112ish face to the backend ArcFace
      cropCanvas.toBlob(async (blob) => {
        if (!blob) { setRegistering(false); return; }
        const formData = new FormData();
        formData.append("file", blob, "crop.jpg");

        try {
          const res = await fetch(`/api/students/${selectedStudentId}/register-face-crop`, {
            method: "POST",
            body: formData
          });
          if (res.ok) {
            alert("Face Registered Successfully!");
            stopWebcam();
            fetchStudents();
          } else {
            const data = await res.json();
            alert("Error: " + data.detail);
          }
        } catch (e) {
          console.error(e);
        } finally {
          setRegistering(false);
        }
      }, "image/jpeg", 0.9);
    } catch (e) {
      console.error(e);
      alert("Error processing the image");
      setRegistering(false);
    }
  };

  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setSelectedStudentId(null);
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Students</h1>
          <p className="text-slate-400">Manage students and register Face ID models</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Registration Form */}
        <div className="glass-panel rounded-2xl p-6 h-fit border-purple-500/30 border-2">
          <div className="flex items-center space-x-3 mb-6">
            <UserPlus className="text-purple-400 w-6 h-6" />
            <h2 className="text-xl font-bold text-white">Add New Student</h2>
          </div>
          <form onSubmit={createStudent} className="space-y-4">
            <div>
              <label className="text-slate-400 text-sm">Student ID</label>
              <input required value={studentId} onChange={e => setStudentId(e.target.value)} type="text" className="w-full bg-slate-800/50 mt-1 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:border-purple-500" placeholder="e.g. CS-2026-001" />
            </div>
            <div>
              <label className="text-slate-400 text-sm">Full Name</label>
              <input required value={name} onChange={e => setName(e.target.value)} type="text" className="w-full bg-slate-800/50 mt-1 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:border-purple-500" placeholder="John Doe" />
            </div>
            <div>
              <label className="text-slate-400 text-sm">Department</label>
              <input value={department} onChange={e => setDepartment(e.target.value)} type="text" className="w-full bg-slate-800/50 mt-1 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:border-purple-500" placeholder="Computer Science" />
            </div>
            <button disabled={registering} type="submit" className="w-full mt-4 bg-purple-600 hover:bg-purple-700 transition-colors text-white font-bold py-2 rounded-lg flex justify-center">
               {registering ? <Loader2 className="animate-spin w-5 h-5" /> : "Create Student"}
            </button>
          </form>
        </div>

        {/* Student List & Face Reg */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-white">Registered Database</h2>
            {detectorStatus === "loading" && (
              <span className="flex items-center gap-2 text-sm text-amber-400 bg-amber-400/10 px-3 py-1.5 rounded-full">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading ML Engine...
              </span>
            )}
          </div>

          {selectedStudentId && (
            <div className="mb-6 p-4 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-emerald-400 font-bold flex items-center gap-2"><Camera className="w-5 h-5" /> Registering Face: {selectedStudentId}</h3>
                <button onClick={stopWebcam} className="text-slate-400 hover:text-white">Cancel</button>
              </div>
              <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-slate-700">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
              </div>
              <button disabled={registering || detectorStatus !== "ready"} onClick={captureAndRegister} className="mt-4 w-full bg-emerald-600 hover:bg-emerald-700 transition text-white font-bold py-3 rounded-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
                {registering ? <Loader2 className="animate-spin" /> : <UploadCloud />} Capture & Register
              </button>
            </div>
          )}

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
                    <tr><td colSpan={5} className="text-center py-8"><Loader2 className="animate-spin mx-auto w-6 h-6 text-slate-500" /></td></tr>
                  ) : students.length === 0 ? (
                    <tr><td colSpan={5} className="text-center py-8 text-slate-500">No students found.</td></tr>
                  ) : (
                    students.map(s => (
                      <tr key={s.student_id} className="border-b border-slate-700/50 hover:bg-slate-800/30">
                        <td className="px-4 py-3 font-medium text-white">{s.student_id}</td>
                        <td className="px-4 py-3">{s.name}</td>
                        <td className="px-4 py-3">{s.department || "-"}</td>
                        <td className="px-4 py-3">
                          {s.face_registered ? 
                            <span className="text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded text-xs font-bold">Registered</span> : 
                            <span className="text-rose-400 bg-rose-400/10 px-2 py-1 rounded text-xs font-bold">Pending</span>}
                        </td>
                        <td className="px-4 py-3">
                          <button onClick={() => startWebcam(s.student_id)} disabled={detectorStatus !== "ready"} className="text-blue-400 hover:text-blue-300 text-sm font-medium disabled:opacity-50">Add Face</button>
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
