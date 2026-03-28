"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Camera, ScanFace, CheckSquare, Loader2, RefreshCw } from "lucide-react";

interface LogEntry {
  id: number;
  student_id: string | number;
  student_name: string;
  timestamp: string;
  time: string;
  status: string;
  recognition_confidence?: number;
}

export default function AttendancePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isActiveRef = useRef(false);
  const detectorRef = useRef<any>(null);

  const [isActive, setIsActive] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [scanning, setScanning] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [lastScanResult, setLastScanResult] = useState<string | null>(null);
  const [detectorStatus, setDetectorStatus] = useState<"loading" | "ready">("loading");

  useEffect(() => {
    fetchTodayLogs();
    initMediaPipe();
    return () => {
      stopWebcam();
      if (pollRef.current) clearInterval(pollRef.current);
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

  const fetchTodayLogs = async () => {
    try {
      const res = await fetch("/api/attendance/today");
      const data = await res.json();
      if (data.success) setLogs(data.data ?? []);
    } catch (e) {
      console.error("Failed to fetch logs", e);
    }
  };

  const captureAndScan = useCallback(async () => {
    if (!videoRef.current || !isActiveRef.current || !detectorRef.current) return;
    if (videoRef.current.readyState < 2) return;

    setScanning(true);
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) { setScanning(false); return; }
    ctx.drawImage(videoRef.current, 0, 0);

    try {
      // 1. Client-side face detection
      const { detections } = detectorRef.current.detect(canvas);
      
      if (!detections || detections.length === 0) {
        setScanning(false);
        return;
      }
      
      const face = detections[0].boundingBox;
      if (!face) {
        setScanning(false);
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
        if (!blob) { setScanning(false); return; }
        const formData = new FormData();
        formData.append("file", blob, "crop.jpg");

        try {
          // HIT THE NEW CROP ENDPOINT
          const res = await fetch("/api/attendance/mark-crop", {
            method: "POST",
            body: formData,
          });
          const data = await res.json();

          if (data.success) {
            setLastScanResult(`✅ ${data.data?.student_name ?? data.data?.student_id} marked Present`);
            fetchTodayLogs();
            setTimeout(() => setLastScanResult(null), 4000);
          } else if (data.duplicate) {
            setLastScanResult(`ℹ️ Already marked today`);
            setTimeout(() => setLastScanResult(null), 2000);
          }
        } catch {
          // ignore rapid scan errors
        } finally {
          setScanning(false);
        }
      }, "image/jpeg", 0.9);
    } catch (e) {
      console.error(e);
      setScanning(false);
    }
  }, []);

  const startWebcam = async () => {
    setWebcamError(null);
    try {
      const ms = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      streamRef.current = ms;

      if (videoRef.current) {
        videoRef.current.srcObject = ms;
        await videoRef.current.play();
      }

      isActiveRef.current = true;
      setIsActive(true);

      timerRef.current = setInterval(captureAndScan, 3000);
      pollRef.current = setInterval(fetchTodayLogs, 10000);
    } catch (e: any) {
      setWebcamError(e?.message ?? "Camera permission denied or unavailable.");
    }
  };

  const stopWebcam = () => {
    isActiveRef.current = false;
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsActive(false);
    setScanning(false);
  };

  const statusColor = (status: string) => {
    if (status === "present") return "bg-emerald-500/10 text-emerald-400";
    if (status === "late") return "bg-amber-500/10 text-amber-400";
    return "bg-rose-500/10 text-rose-400";
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold text-white">Live Attendance</h1>
        <p className="text-slate-400">Continuous face tracking and recognition — scans every 3 seconds</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-3">
              <ScanFace className="w-6 h-6 text-blue-400" />
              <h2 className="text-xl font-bold text-white">Scanner Feed</h2>
            </div>
            <div>
              {isActive ? (
                <button id="btn-stop-camera" onClick={stopWebcam}
                  className="bg-red-500/20 text-red-400 px-4 py-2 rounded-lg font-medium hover:bg-red-500/30 transition">
                  Stop Camera
                </button>
              ) : (
                <button id="btn-start-camera" onClick={startWebcam} disabled={detectorStatus !== "ready"}
                  className="bg-emerald-500/20 text-emerald-400 px-4 py-2 rounded-lg font-medium hover:bg-emerald-500/30 transition flex items-center gap-2 disabled:opacity-50">
                  <Camera className="w-5 h-5" /> {detectorStatus === "loading" ? "Loading Engine..." : "Start Camera"}
                </button>
              )}
            </div>
          </div>

          {webcamError && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
              ⚠️ {webcamError}
            </div>
          )}

          {lastScanResult && (
            <div className="mb-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-300 text-sm font-medium">
              {lastScanResult}
            </div>
          )}

          <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden shadow-2xl border border-slate-700">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${isActive ? "block" : "hidden"}`}
            />
            {!isActive && (
              <div className="absolute inset-0 flex items-center justify-center flex-col text-slate-500">
                {detectorStatus === "loading" ? (
                  <>
                    <Loader2 className="w-16 h-16 mb-4 opacity-50 animate-spin" />
                    <p className="font-medium">Warming up ML engine...</p>
                  </>
                ) : (
                  <>
                    <Camera className="w-16 h-16 mb-4 opacity-50" />
                    <p className="font-medium">Camera Offline</p>
                    <p className="text-xs mt-1 text-slate-600">Click &quot;Start Camera&quot; to begin scanning</p>
                  </>
                )}
              </div>
            )}
            {isActive && scanning && (
              <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full flex items-center gap-2">
                <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                <span className="text-xs text-blue-400 font-bold uppercase tracking-wider">Scanning</span>
              </div>
            )}
            {isActive && (
              <div className="absolute inset-0 border-4 border-blue-500/20 animate-pulse pointer-events-none rounded-xl" />
            )}
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6 h-[600px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <CheckSquare className="w-6 h-6 text-emerald-400" />
              <h2 className="text-xl font-bold text-white">Today&apos;s Logs</h2>
            </div>
            <button onClick={fetchTodayLogs} title="Refresh"
              className="text-slate-500 hover:text-slate-300 transition">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {logs.length === 0 ? (
              <div className="text-center py-12">
                <CheckSquare className="w-10 h-10 mx-auto mb-3 text-slate-700" />
                <p className="text-slate-500 text-sm">No attendance logged yet today.</p>
                <p className="text-slate-600 text-xs mt-1">Start the camera to begin.</p>
              </div>
            ) : (
              logs.map((log) => (
                <div key={log.id}
                  className="bg-slate-800/40 border border-slate-700 p-4 rounded-xl flex items-center justify-between">
                  <div>
                    <h4 className="font-bold text-slate-200">{log.student_name}</h4>
                    <p className="text-xs text-slate-400 mt-0.5">{log.time}</p>
                    {log.recognition_confidence != null && (
                      <p className="text-xs text-slate-600 mt-0.5">
                        Confidence: {(log.recognition_confidence * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                  <span className={`px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider ${statusColor(log.status)}`}>
                    {log.status}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
