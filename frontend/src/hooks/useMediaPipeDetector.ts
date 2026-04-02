/**
 * useMediaPipeDetector
 *
 * Custom hook that lazily loads the MediaPipe BlazeFace short-range model
 * and returns a ref to the detector along with its loading status.
 *
 * Both the Students and Attendance pages share identical MediaPipe
 * initialisation logic — this hook eliminates that duplication.
 *
 * Usage:
 *   const { detectorRef, detectorStatus } = useMediaPipeDetector();
 *
 *   // Use detectorRef.current.detect(canvas) when detectorStatus === "ready"
 */

"use client";

import { useEffect, useRef, useState } from "react";

type DetectorStatus = "loading" | "ready" | "error";

const MEDIAPIPE_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

interface UseMediaPipeDetectorResult {
  /** Ref to the loaded FaceDetector instance. Check detectorStatus before using. */
  detectorRef: React.MutableRefObject<any>;
  /** Current state of the detector. Use detector only when "ready". */
  detectorStatus: React.MutableRefObject<DetectorStatus>;
  /** React state setter — triggers re-renders when status changes. */
  setDetectorStatus: React.Dispatch<React.SetStateAction<DetectorStatus>>;
}

export function useMediaPipeDetector(): {
  detectorRef: React.MutableRefObject<any>;
  detectorStatus: DetectorStatus;
  setDetectorStatus: React.Dispatch<React.SetStateAction<DetectorStatus>>;
} {

  const detectorRef = useRef<any>(null);
  const [detectorStatus, setDetectorStatus] =
    useState<DetectorStatus>("loading");

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const { FaceDetector, FilesetResolver } = await import(
          "@mediapipe/tasks-vision"
        );
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN);
        const fd = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          // IMAGE mode: process one frame at a time (correct for snapshot-based
          // scanning — do NOT use VIDEO mode here as it adds unwanted smoothing).
          runningMode: "IMAGE",
        });

        if (!cancelled) {
          detectorRef.current = fd;
          setDetectorStatus("ready");
        }
      } catch (err) {
        console.error("MediaPipe FaceDetector failed to load:", err);
        if (!cancelled) setDetectorStatus("error");
      }
    }

    init();

    return () => {
      cancelled = true;
      // Clean up the native WASM resource on unmount
      if (detectorRef.current && typeof detectorRef.current.close === "function") {
        detectorRef.current.close();
      }
    };
  }, []); // Run once on mount

  return { detectorRef, detectorStatus, setDetectorStatus };
}
