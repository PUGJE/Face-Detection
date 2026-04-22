"use client";

/**
 * Attendance Page — Live Face Scanning with Blink Verification
 *
 * Opens the webcam, monitors eye landmarks for a blink, and only sends
 * the face crop to the backend once a genuine blink is confirmed.
 *
 * Features:
 *  - Active class banner (polls /api/timetable/active every 60 s)
 *  - Blink gate: attendance is blocked unless a blink is detected
 *  - Continuous scanning at configurable intervals
 *  - Today's attendance log panel (auto-refreshes every 10 s while active)
 *  - Duplicate attendance detection surfaced to the user
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { Camera, ScanFace, CheckSquare, Loader2, RefreshCw, Eye, EyeOff, Clock } from "lucide-react";
import { useMediaPipeDetector } from "@/hooks/useMediaPipeDetector";
import { useBlinkDetector } from "@/hooks/useBlinkDetector";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type AttendanceStatus = "present" | "late" | "absent";

interface LogEntry {
  id: number;
  student_id: string | number;
  student_name: string;
  timestamp: string;
  time: string;
  status: AttendanceStatus;
  recognition_confidence?: number;
  subject_name?: string;
}

interface ActiveClass {
  subject_name: string;
  day_of_week: string;
  start_time: string;
  end_time: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SCAN_INTERVAL_MS       = 500;    // blink-gated server scan
const DETECT_INTERVAL_MS     = 300;    // continuous client-side face stability check
const STREAK_REQUIRED        = 3;      // consecutive frames with face before scan fires
const LOG_POLL_INTERVAL_MS   = 10000;
const CLASS_POLL_INTERVAL_MS = 60000;

function statusBadgeClass(status: AttendanceStatus): string {
  const map: Record<AttendanceStatus, string> = {
    present: "bg-emerald-100 text-[#10B981]",
    late:    "bg-amber-100 text-amber-700",
    absent:  "bg-rose-100 text-rose-700",
  };
  return map[status] ?? "bg-slate-500/10 text-slate-500";
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function AttendancePage() {
  const videoRef        = useRef<HTMLVideoElement>(null);
  const streamRef       = useRef<MediaStream | null>(null);
  const scanTimerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const detectTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollTimerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const classTimerRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const isActiveRef     = useRef(false);
  const streakRef       = useRef(0); // consecutive frames where a face was detected

  const [isActive,       setIsActive]       = useState(false);
  const [logs,           setLogs]           = useState<LogEntry[]>([]);
  const [scanning,       setScanning]       = useState(false);
  const [webcamError,    setWebcamError]    = useState<string | null>(null);
  const [lastScanResult, setLastScanResult] = useState<string | null>(null);
  const [activeClass,    setActiveClass]    = useState<ActiveClass | null>(null);
  const [classLoading,   setClassLoading]   = useState(true);
  const [blinkFlash,     setBlinkFlash]     = useState(false);
  const [faceStable,     setFaceStable]     = useState(false); // streak met

  // Shared MediaPipe face detector (BlazeFace) — loaded once on mount
  const { detectorRef, detectorStatus } = useMediaPipeDetector();

  // Blink detector (Face Landmarker) — loaded once on mount
  const { blinkDetected, resetBlink, startBlink, stopBlink, landmarkerStatus, currentEar } = useBlinkDetector();

  // Mirror blinkDetected into a ref so captureAndScan (inside setInterval)
  // always reads the live value without stale-closure issues.
  const blinkRef = useRef(false);
  useEffect(() => { blinkRef.current = blinkDetected; }, [blinkDetected]);

  useEffect(() => {
    fetchTodayLogs();
    fetchActiveClass();
    return () => {
      stopWebcam();
      if (pollTimerRef.current)  clearInterval(pollTimerRef.current);
      if (classTimerRef.current) clearInterval(classTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Flash the blink indicator whenever blinkDetected flips true
  useEffect(() => {
    if (blinkDetected) {
      setBlinkFlash(true);
      const t = setTimeout(() => setBlinkFlash(false), 700);
      return () => clearTimeout(t);
    }
  }, [blinkDetected]);

  // ------------------------------------------------------------------
  // Data fetching
  // ------------------------------------------------------------------

  async function fetchTodayLogs() {
    try {
      const res = await fetch("/api/attendance/today");
      const data = await res.json();
      if (data.success) setLogs(data.data ?? []);
    } catch (err) {
      console.error("Failed to fetch today's logs:", err);
    }
  }

  async function fetchActiveClass() {
    setClassLoading(true);
    try {
      const res = await fetch("/api/timetable/active");
      const data = await res.json();
      setActiveClass(data.active ? data.data : null);
    } catch {
      setActiveClass(null);
    } finally {
      setClassLoading(false);
    }
  }


  // ------------------------------------------------------------------
  // Continuous face-stability detector (no blink gate, no server call)
  // Fills streakRef; updates faceStable UI indicator
  // ------------------------------------------------------------------

  const detectFaceContinuously = useCallback(async () => {
    if (!isActiveRef.current || !videoRef.current || !detectorRef.current) return;
    if (videoRef.current.readyState < 2) return;

    const canvas = document.createElement("canvas");
    canvas.width  = videoRef.current.videoWidth  || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0);

    try {
      const { detections } = detectorRef.current.detect(canvas);
      if (detections?.length) {
        streakRef.current = Math.min(streakRef.current + 1, STREAK_REQUIRED);
      } else {
        streakRef.current = Math.max(streakRef.current - 1, 0);
      }
      setFaceStable(streakRef.current >= STREAK_REQUIRED);
    } catch { /* ignore */ }
  }, [detectorRef]);

  // ------------------------------------------------------------------
  // Blink-gated server scan (fires only when streak is met + blink)
  // ------------------------------------------------------------------

  const captureAndScan = useCallback(async () => {
    if (!isActiveRef.current || !videoRef.current || !detectorRef.current) return;
    if (videoRef.current.readyState < 2) return;

    // 🔒 Blink gate
    if (!blinkRef.current) return;
    // 🔒 Streak gate — face must have been stable for STREAK_REQUIRED frames
    if (streakRef.current < STREAK_REQUIRED) return;

    blinkRef.current = false;
    resetBlink();
    streakRef.current = 0;
    setFaceStable(false);

    setScanning(true);

    const canvas = document.createElement("canvas");
    canvas.width  = videoRef.current.videoWidth  || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) { setScanning(false); return; }
    ctx.drawImage(videoRef.current, 0, 0);

    try {
      // 1. Client-side face detection
      const { detections } = detectorRef.current.detect(canvas);
      if (!detections?.length) { setScanning(false); return; }

      const face = detections[0].boundingBox;
      if (!face) { setScanning(false); return; }

      // 2. Crop and resize to 112×112 (ArcFace native size — smaller payload)
      const padX = face.width  * 0.25;
      const padY = face.height * 0.25;
      const sx   = Math.max(0, face.originX - padX);
      const sy   = Math.max(0, face.originY - padY);
      const sw   = Math.min(canvas.width  - sx, face.width  + 2 * padX);
      const sh   = Math.min(canvas.height - sy, face.height + 2 * padY);

      const cropCanvas = document.createElement("canvas");
      cropCanvas.width  = 112;
      cropCanvas.height = 112;
      cropCanvas.getContext("2d")?.drawImage(canvas, sx, sy, sw, sh, 0, 0, 112, 112);

      cropCanvas.toBlob(async (blob) => {
        if (!blob) { setScanning(false); return; }

        const form = new FormData();
        form.append("file", blob, "crop.jpg");

        try {
          const res  = await fetch("/api/attendance/mark-crop", { method: "POST", body: form });
          const data = await res.json();

          if (data.success) {
            const label = data.data?.student_name ?? data.data?.student_id ?? "Unknown";
            setLastScanResult(`✅ ${label} marked Present`);
            void fetchTodayLogs();
            setTimeout(() => setLastScanResult(null), 4000);
          } else if (data.duplicate) {
            setLastScanResult("ℹ️ Already marked for this class");
            setTimeout(() => setLastScanResult(null), 2000);
          }
        } catch (err) {
          console.warn("Scan request failed:", err);
        } finally {
          setScanning(false);
        }
      }, "image/jpeg", 0.7);
    } catch (err) {
      console.error("Error during face scan:", err);
      setScanning(false);
    }
  }, [detectorRef, resetBlink]); // blinkDetected read via blinkRef — no dep needed

  // ------------------------------------------------------------------
  // Webcam management
  // ------------------------------------------------------------------

  async function startWebcam() {
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

      // Start rAF-based blink loop
      if (videoRef.current) startBlink(videoRef.current);

      // Continuous face-stability check (no blink needed)
      detectTimerRef.current  = setInterval(detectFaceContinuously, DETECT_INTERVAL_MS);
      // Blink-gated server scan
      scanTimerRef.current    = setInterval(captureAndScan,          SCAN_INTERVAL_MS);
      pollTimerRef.current    = setInterval(fetchTodayLogs,          LOG_POLL_INTERVAL_MS);
      classTimerRef.current   = setInterval(fetchActiveClass,        CLASS_POLL_INTERVAL_MS);
    } catch (err: any) {
      setWebcamError(err?.message ?? "Camera permission denied or unavailable.");
    }
  }

  function stopWebcam() {
    isActiveRef.current = false;
    stopBlink();
    streakRef.current = 0;
    setFaceStable(false);

    [scanTimerRef, detectTimerRef, pollTimerRef, classTimerRef].forEach((r) => {
      if (r.current) { clearInterval(r.current); r.current = null; }
    });

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;

    setIsActive(false);
    setScanning(false);
  }

  // ------------------------------------------------------------------
  // Derived booleans
  // ------------------------------------------------------------------

  const enginesReady  = detectorStatus === "ready" && landmarkerStatus === "ready";
  const enginesLoading = detectorStatus === "loading" || landmarkerStatus === "loading";

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold text-slate-800">Live Attendance</h1>
        <p className="text-slate-500">
          Blink-verified face recognition — scans every {SCAN_INTERVAL_MS / 1000} seconds
        </p>
      </div>

      {/* ── Active class banner ─────────────────────────────────────── */}
      {classLoading ? (
        <div className="glass-panel rounded-2xl px-6 py-4 flex items-center gap-3 text-slate-500 text-sm">
          <Loader2 className="w-4 h-4 animate-spin" />
          Checking class schedule…
        </div>
      ) : activeClass ? (
        <div className="glass-panel rounded-2xl px-6 py-4 flex items-center gap-4 border border-emerald-200 bg-emerald-500/5">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <div className="flex-1">
            <span className="font-semibold text-emerald-300">{activeClass.subject_name}</span>
            <span className="text-slate-500 text-sm ml-3">
              {activeClass.day_of_week} · {activeClass.start_time} – {activeClass.end_time}
            </span>
          </div>
          <span className="text-xs text-emerald-500 font-bold uppercase tracking-wider">
            Class In Session
          </span>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl px-6 py-4 flex items-center gap-4 border border-amber-500/20 bg-amber-500/5">
          <Clock className="w-4 h-4 text-amber-600" />
          <span className="text-amber-300 text-sm font-medium">No class is currently active.</span>
          <span className="text-amber-600 text-xs ml-auto">Attendance is still allowed</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* ── Camera feed ───────────────────────────────────────────── */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-3">
              <ScanFace className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-bold text-slate-800">Scanner Feed</h2>
            </div>

            {/* Engine status pills */}
            <div className="flex items-center gap-2">
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
                detectorStatus === "ready"
                  ? "bg-emerald-100 text-[#10B981] border-emerald-500/20"
                  : "bg-slate-500/10 text-slate-500 border-slate-500/20"
              }`}>
                Face
              </span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
                landmarkerStatus === "ready"
                  ? "bg-purple-50 text-purple-600 border-purple-200"
                  : "bg-slate-500/10 text-slate-500 border-slate-500/20"
              }`}>
                Blink
              </span>
              {isActive && (
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium border transition-colors ${
                  faceStable
                    ? "bg-green-500/10 text-green-400 border-green-500/20"
                    : "bg-slate-500/10 text-slate-500 border-slate-500/20"
                }`}>
                  {faceStable ? "🔒 Stable" : "Detecting…"}
                </span>
              )}
            </div>

            {isActive ? (
              <button
                id="btn-stop-camera"
                onClick={stopWebcam}
                className="bg-red-50 text-red-600 px-4 py-2 rounded-2xl font-medium hover:bg-red-500/30 transition"
              >
                Stop Camera
              </button>
            ) : (
              <button
                id="btn-start-camera"
                onClick={startWebcam}
                disabled={!enginesReady}
                className="bg-emerald-50 text-[#10B981] px-4 py-2 rounded-2xl font-medium hover:bg-emerald-500/30 transition flex items-center gap-2 disabled:opacity-50"
              >
                <Camera className="w-5 h-5" />
                {enginesLoading ? "Loading Engines…" : "Start Camera"}
              </button>
            )}
          </div>

          {/* Error banner */}
          {webcamError && (
            <div className="mb-4 p-3 rounded-2xl bg-red-50 border border-red-200 text-red-600 text-sm">
              ⚠️ {webcamError}
            </div>
          )}

          {/* Last scan result toast */}
          {lastScanResult && (
            <div className="mb-4 p-3 rounded-2xl bg-blue-50 border border-blue-200 text-blue-300 text-sm font-medium">
              {lastScanResult}
            </div>
          )}

          {/* Blink prompt */}
          {isActive && !blinkDetected && (
            <div className="mb-4 p-3 rounded-2xl bg-purple-50 border border-purple-200 text-purple-300 text-sm flex items-center gap-2">
              <Eye className="w-4 h-4" />
              <span>
                {faceStable
                  ? "Face locked — blink to mark attendance"
                  : "Position your face in the camera…"}
              </span>
              {currentEar !== null && (
                <span className="ml-auto text-purple-500 text-xs font-mono">EAR {currentEar}</span>
              )}
            </div>
          )}

          {/* Blink confirmed flash */}
          {isActive && blinkDetected && (
            <div className="mb-4 p-3 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-300 text-sm flex items-center gap-2 animate-pulse">
              <EyeOff className="w-4 h-4" />
              <span>Blink detected! Scanning face…</span>
            </div>
          )}

          {/* Video element */}
          <div className="relative w-full aspect-video bg-slate-100 rounded-2xl overflow-hidden border border-slate-200">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${isActive ? "block" : "hidden"}`}
            />

            {!isActive && (
              <div className="absolute inset-0 flex items-center justify-center flex-col text-slate-500">
                {enginesLoading ? (
                  <>
                    <Loader2 className="w-16 h-16 mb-4 opacity-50 animate-spin" />
                    <p className="font-medium">Warming up ML engines…</p>
                    <p className="text-xs mt-1 text-slate-600">Face + Blink models loading</p>
                  </>
                ) : (
                  <>
                    <Camera className="w-16 h-16 mb-4 opacity-50" />
                    <p className="font-medium">Camera Offline</p>
                    <p className="text-xs mt-1 text-slate-600">
                      Click &quot;Start Camera&quot; to begin scanning
                    </p>
                  </>
                )}
              </div>
            )}

            {/* Scanning indicator */}
            {isActive && scanning && (
              <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-md px-3 py-1.5 rounded-full flex items-center gap-2 border border-slate-200 shadow-sm">
                <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
                <span className="text-xs text-blue-600 font-bold uppercase tracking-wider">
                  Scanning
                </span>
              </div>
            )}

            {/* Blink flash overlay */}
            {blinkFlash && (
              <div className="absolute inset-0 border-4 border-purple-500/60 pointer-events-none rounded-2xl animate-pulse" />
            )}

            {/* Active border pulse */}
            {isActive && !blinkFlash && (
              <div className="absolute inset-0 border-[3px] border-[#10B981] shadow-[0_0_30px_rgba(16,185,129,0.4)] animate-pulse pointer-events-none rounded-2xl" />
            )}
          </div>
        </div>

        {/* ── Today's attendance log ────────────────────────────────── */}
        <div className="glass-panel rounded-2xl p-6 h-[600px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <CheckSquare className="w-6 h-6 text-[#10B981]" />
              <h2 className="text-xl font-bold text-slate-800">Today&apos;s Logs</h2>
            </div>
            <button
              onClick={fetchTodayLogs}
              title="Refresh logs"
              className="text-slate-500 hover:text-slate-800 transition"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {logs.length === 0 ? (
              <div className="text-center py-12">
                <CheckSquare className="w-10 h-10 mx-auto mb-3 text-slate-600" />
                <p className="text-slate-400 text-sm">No attendance logged yet today.</p>
                <p className="text-slate-400 text-xs mt-1">Start the camera and blink.</p>
              </div>
            ) : (
              logs.map((log) => (
                <div
                  key={log.id}
                  className="bg-white border border-slate-200 p-4 rounded-2xl flex items-center justify-between shadow-sm"
                >
                  <div>
                    <h4 className="font-bold text-slate-900">{log.student_name}</h4>
                    <p className="text-xs text-slate-500 mt-0.5 font-mono">{log.time}</p>
                    {log.subject_name && (
                      <p className="text-xs text-blue-600 mt-0.5">{log.subject_name}</p>
                    )}
                    {log.recognition_confidence != null && (
                      <p className="text-xs text-slate-500 mt-0.5 font-mono">
                        Conf: {(log.recognition_confidence * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                  <span
                    className={`px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider ${statusBadgeClass(
                      log.status
                    )}`}
                  >
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
