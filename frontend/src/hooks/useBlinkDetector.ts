/**
 * useBlinkDetector
 *
 * Custom hook that loads the MediaPipe Face Landmarker model (VIDEO mode)
 * and runs an EAR-based blink-detection state machine.
 *
 * EAR formula (Soukupová & Čech, 2016):
 *   EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
 *
 * State machine:
 *   OPEN  → (EAR < EAR_THRESHOLD) → CLOSED
 *   CLOSED → (EAR ≥ EAR_THRESHOLD within BLINK_MAX_MS) → OPEN  ⟹  blink!
 *
 * Design notes:
 *  - CDN version MUST match the installed npm package (0.10.34) to avoid
 *    WASM/JS API mismatches that cause detectForVideo to throw.
 *  - The rAF loop runs inside the hook; callers just call start/stop.
 *  - A 800 ms warmup after model load lets XNNPACK fully initialise before
 *    the first detectForVideo call, preventing the spurious console error.
 *  - All mutation goes through refs; setInterval is never used.
 */

"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type LandmarkerStatus = "loading" | "ready" | "error";

// MediaPipe Face Landmarker landmark indices for both eyes
// https://developers.google.com/mediapipe/solutions/vision/face_landmarker
const LEFT_EYE  = { p1: 33,  p2: 160, p3: 158, p4: 133, p5: 153, p6: 144 };
const RIGHT_EYE = { p1: 362, p2: 385, p3: 387, p4: 263, p5: 373, p6: 380 };

const EAR_THRESHOLD = 0.21;  // below → eye closed
const BLINK_MAX_MS  = 500;   // max blink duration
const WARMUP_MS     = 800;   // let XNNPACK finish init before first call
const BLINK_FPS_MS  = 66;    // ~15 fps for Face Landmarker (keeps CPU free for scanner)

// IMPORTANT: CDN version must match the installed @mediapipe/tasks-vision version
const MEDIAPIPE_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";

const LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function euclidean(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function calcEAR(
  lm: Array<{ x: number; y: number }>,
  eye: typeof LEFT_EYE
): number {
  const v1 = euclidean(lm[eye.p2], lm[eye.p6]);
  const v2 = euclidean(lm[eye.p3], lm[eye.p5]);
  const h  = euclidean(lm[eye.p1], lm[eye.p4]);
  return (v1 + v2) / (2.0 * h);
}

// ---------------------------------------------------------------------------
// Silence MediaPipe's WASM stdout/stderr in dev mode.
// The WASM runtime uses _fd_write → console.log/warn/error to emit INFO and
// WARNING lines. Next.js/Turbopack then decorates every console call with the
// current JS call stack, making routine logs look like unhandled errors.
// We mute those channels for the duration of each MediaPipe call and restore
// them immediately after, so real JS errors are never suppressed.
// ---------------------------------------------------------------------------

type ConsoleFn = typeof console.log;
let _log: ConsoleFn, _warn: ConsoleFn, _error: ConsoleFn;

function muteWasmLogs() {
  _log   = console.log;
  _warn  = console.warn;
  _error = console.error;
  const noop = () => {};
  console.log   = noop;
  console.warn  = noop;
  console.error = noop;
}

function unmuteWasmLogs() {
  console.log   = _log;
  console.warn  = _warn;
  console.error = _error;
}


export interface UseBlinkDetectorResult {
  blinkDetected: boolean;
  resetBlink: () => void;
  /** Start the rAF blink loop against a video element. */
  startBlink: (video: HTMLVideoElement) => void;
  /** Stop the rAF loop. */
  stopBlink: () => void;
  landmarkerStatus: LandmarkerStatus;
  currentEar: number | null;
}

export function useBlinkDetector(): UseBlinkDetectorResult {
  // Model refs — never trigger re-renders
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const landmarkerRef    = useRef<any>(null);
  const landmarkerReady  = useRef(false);
  const lastTsRef        = useRef(-1);     // last submitted timestamp (ms)
  const rafRef           = useRef<number | null>(null);
  const videoTargetRef   = useRef<HTMLVideoElement | null>(null);

  // Blink state machine refs
  const eyeClosedRef  = useRef(false);
  const closeTimeRef  = useRef<number | null>(null);

  // React-visible state
  const [landmarkerStatus, setLandmarkerStatus] = useState<LandmarkerStatus>("loading");
  const [blinkDetected,    setBlinkDetected]    = useState(false);
  const [currentEar,       setCurrentEar]       = useState<number | null>(null);

  // ------------------------------------------------------------------
  // Load Face Landmarker once on mount
  // ------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const { FaceLandmarker, FilesetResolver } = await import(
          "@mediapipe/tasks-vision"
        );
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN);

        muteWasmLogs();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let fl: any;
        try {
          fl = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: LANDMARKER_MODEL_URL,
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: false,
          });
        } finally {
          unmuteWasmLogs();
        }

        if (cancelled) { fl.close?.(); return; }

        landmarkerRef.current = fl;

        // Brief warmup gives XNNPACK time to finish thread-pool init
        setTimeout(() => {
          if (!cancelled) {
            landmarkerReady.current = true;
            setLandmarkerStatus("ready");
          }
        }, WARMUP_MS);
      } catch (err) {
        console.error("FaceLandmarker failed to load:", err);
        if (!cancelled) setLandmarkerStatus("error");
      }
    }

    load();

    return () => {
      cancelled = true;
      landmarkerReady.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close?.();
    };
  }, []);

  // ------------------------------------------------------------------
  // rAF loop — throttled to BLINK_FPS_MS (~15 fps) to keep CPU free for scanner
  // ------------------------------------------------------------------
  const lastRafTs  = useRef(0);
  const rafLoop = useCallback((ts: number) => {
    // Throttle: only process a frame if enough time has elapsed
    if (ts - lastRafTs.current >= BLINK_FPS_MS) {
      lastRafTs.current = ts;

      const video = videoTargetRef.current;
      if (video && landmarkerReady.current && landmarkerRef.current &&
          video.readyState >= 2 && !video.paused && !video.ended) {

        const now = performance.now();
        if (now > lastTsRef.current) {
          lastTsRef.current = now;
          try {
            muteWasmLogs();
            const result = landmarkerRef.current.detectForVideo(video, now);
            unmuteWasmLogs();

            if (result?.faceLandmarks?.length) {
              const lm  = result.faceLandmarks[0] as Array<{ x: number; y: number }>;
              const ear = (calcEAR(lm, LEFT_EYE) + calcEAR(lm, RIGHT_EYE)) / 2;
              setCurrentEar(Math.round(ear * 1000) / 1000);

              if (ear < EAR_THRESHOLD) {
                if (!eyeClosedRef.current) {
                  eyeClosedRef.current = true;
                  closeTimeRef.current = now;
                }
              } else {
                if (eyeClosedRef.current) {
                  const elapsed = now - (closeTimeRef.current ?? now);
                  if (elapsed <= BLINK_MAX_MS && elapsed > 40) setBlinkDetected(true);
                  eyeClosedRef.current = false;
                  closeTimeRef.current = null;
                }
              }
            } else {
              setCurrentEar(null);
            }
          } catch {
            unmuteWasmLogs();
          }
        }
      }
    }

    rafRef.current = requestAnimationFrame(rafLoop);
  }, []); // no deps — all accessed via refs

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------
  const startBlink = useCallback((video: HTMLVideoElement) => {
    videoTargetRef.current = video;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(rafLoop);
  }, [rafLoop]);

  const stopBlink = useCallback(() => {
    videoTargetRef.current = null;
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    setCurrentEar(null);
    eyeClosedRef.current = false;
    closeTimeRef.current = null;
  }, []);

  const resetBlink = useCallback(() => setBlinkDetected(false), []);

  return { blinkDetected, resetBlink, startBlink, stopBlink, landmarkerStatus, currentEar };
}
