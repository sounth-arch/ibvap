import { useRef, useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import Tesseract from "tesseract.js";
import * as faceapi from "@vladmandic/face-api";
import { saveEventToFirestore } from "./firebase";

const CLASS_NAMES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "remote","keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
  "toothbrush",
];

const VEHICLE_CLASS_IDS = [2, 3, 5, 7];

const INPUT_SIZE = 640;
const NUM_CLASSES = 80;
const CONFIDENCE_THRESHOLD = 0.6;
const NMS_IOU_THRESHOLD = 0.45;
const TRACK_IOU_THRESHOLD = 0.3;
const DETECTION_INTERVAL = 150;
const MAX_MISSED_FRAMES = 10;

const LOITER_THRESHOLD_SECONDS = 10;
const LOITER_MOVEMENT_RADIUS = 80;

const OCR_COOLDOWN_MS = 4000;
const CAMERA_TIMEOUT_MS = 3000;

// ---- Face recognition settings ----
const FACE_MODEL_URL = "/face-models";
const FACE_CROP_HEIGHT_RATIO = 0.5;   // use the top 50% of a person's box (where the face is)
const FACE_MIN_CROP_PX = 40;          // skip people too small/far away for a usable face
const FACE_COOLDOWN_MS = 1500;        // wait between face attempts on the same person
const FACE_MAX_ATTEMPTS = 15;         // give up on a track after this many "no face found" attempts
const FACE_DETECTION_SCORE = 0.4;     // minimum face-detector confidence
const FACE_MATCH_THRESHOLD = 0.6;     // lower = stricter matching
const FACE_UNKNOWN_AFTER = 3;         // failed matches before someone is called "Unknown"
const SAVE_FACE_SNAPSHOTS = false;    // true = store face crops in Firestore (privacy risk)

function computeIoU(boxA, boxB) {
  const ax1 = boxA.x - boxA.w / 2;
  const ay1 = boxA.y - boxA.h / 2;
  const ax2 = boxA.x + boxA.w / 2;
  const ay2 = boxA.y + boxA.h / 2;
  const bx1 = boxB.x - boxB.w / 2;
  const by1 = boxB.y - boxB.h / 2;
  const bx2 = boxB.x + boxB.w / 2;
  const by2 = boxB.y + boxB.h / 2;
  const x1 = Math.max(ax1, bx1);
  const y1 = Math.max(ay1, by1);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);
  const width = Math.max(0, x2 - x1);
  const height = Math.max(0, y2 - y1);
  const intersection = width * height;
  const areaA = boxA.w * boxA.h;
  const areaB = boxB.w * boxB.h;
  const union = areaA + areaB - intersection;
  if (union <= 0) return 0;
  return intersection / union;
}

function applyNMS(boxes, threshold = NMS_IOU_THRESHOLD) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept = [];
  while (sorted.length > 0) {
    const best = sorted.shift();
    kept.push(best);
    for (let i = sorted.length - 1; i >= 0; i--) {
      const candidate = sorted[i];
      if (computeIoU(best, candidate) > threshold) {
        sorted.splice(i, 1);
      }
    }
  }
  return kept;
}

function matchTracks(existingTracks, newBoxes, nextIdRef) {
  const updatedTracks = [];
  const usedBoxes = new Set();

  existingTracks.forEach((track) => {
    let bestMatch = -1;
    let bestIoU = TRACK_IOU_THRESHOLD;

    newBoxes.forEach((box, index) => {
      if (usedBoxes.has(index)) return;
      if (track.classId !== box.classId) return;
      const iou = computeIoU(track, box);
      if (iou > bestIoU) {
        bestIoU = iou;
        bestMatch = index;
      }
    });

    if (bestMatch !== -1) {
      const matchedBox = newBoxes[bestMatch];
      updatedTracks.push({
        ...matchedBox,
        id: track.id,
        missedFrames: 0,
        firstSeenTime: track.firstSeenTime,
        firstSeenPosition: track.firstSeenPosition,
        lastOcrTime: track.lastOcrTime,
        plateText: track.plateText,
        identity: track.identity,
        lastFaceTime: track.lastFaceTime,
        faceAttempts: track.faceAttempts,
        faceMisses: track.faceMisses,
      });
      usedBoxes.add(bestMatch);
    } else if (track.missedFrames < MAX_MISSED_FRAMES) {
      updatedTracks.push({ ...track, missedFrames: track.missedFrames + 1 });
    }
  });

  newBoxes.forEach((box, index) => {
    if (!usedBoxes.has(index)) {
      const newTrack = {
        ...box,
        id: nextIdRef.current,
        missedFrames: 0,
        firstSeenTime: Date.now(),
        firstSeenPosition: { x: box.x, y: box.y },
        lastOcrTime: 0,
        plateText: null,
        identity: null,
        lastFaceTime: 0,
        faceAttempts: 0,
        faceMisses: 0,
      };
      updatedTracks.push(newTrack);
      nextIdRef.current += 1;
    }
  });

  return updatedTracks;
}

function isInsideZone(x, y, zone) {
  if (!zone) return false;
  return x >= zone.x1 && x <= zone.x2 && y >= zone.y1 && y <= zone.y2;
}

function playAlertSound(freq = 880) {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContextClass();
    const oscillator = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = freq;
    oscillator.connect(gain);
    gain.connect(audioCtx.destination);
    gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    oscillator.start();
    oscillator.stop(audioCtx.currentTime + 0.3);
  } catch (err) {
    console.error("Could not play alert sound: - App.js:168", err);
  }
}

function cleanPlateText(raw) {
  return raw.toUpperCase().replace(/[^A-Z0-9]/g, "").trim();
}

// Builds a small snapshot: the actual video frame with the boxes drawn on top
function captureSnapshot(video, overlayCanvas) {
  try {
    if (!video || video.videoWidth === 0) return null;
    const w = 320;
    const h = Math.round((w * video.videoHeight) / video.videoWidth);
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const cx = c.getContext("2d");
    cx.drawImage(video, 0, 0, w, h);
    if (overlayCanvas) cx.drawImage(overlayCanvas, 0, 0, w, h);
    return c.toDataURL("image/jpeg", 0.6);
  } catch (err) {
    console.error("Snapshot error: - App.js:190", err);
    return null;
  }
}

function createEvent(type, track, extraDetails = {}, snapshot = null) {
  const now = Date.now();
  const event = {
    id: `evt_${now}_${Math.random().toString(36).slice(2, 8)}`,
    type,
    cameraId: "CAM_01",
    trackId: track.id,
    classId: track.classId,
    timestamp: now,
    timeDisplay: new Date(now).toLocaleTimeString(),
    confidence: track.score || 0,
    snapshot,
    details: extraDetails,
  };
  saveEventToFirestore(event);
  return event;
}

function faceDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function matchFace(descriptor, enrolledList) {
  let best = { name: null, distance: Infinity };
  enrolledList.forEach((person) => {
    const d = faceDistance(descriptor, person.descriptor);
    if (d < best.distance) best = { name: person.name, distance: d };
  });
  if (best.distance < FACE_MATCH_THRESHOLD) return best;
  return { name: null, distance: best.distance };
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const processingCanvasRef = useRef(null);
  const plateCanvasRef = useRef(null);
  const faceCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const tracksRef = useRef([]);
  const [tracks, setTracks] = useState([]);
  const nextIdRef = useRef(1);
  const detectingRef = useRef(false);
  const ocrBusyRef = useRef(false);
  const faceBusyRef = useRef(false);
  const facesReadyRef = useRef(false);
  const animationRef = useRef(null);
  const timerRef = useRef(null);
  const lastVideoTimeRef = useRef(0);

  const zoneRef = useRef(null);
  const [zone, setZone] = useState(() => {
    try {
      const saved = localStorage.getItem("ibvap_zone");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const drawStartRef = useRef(null);
  const [drawingRect, setDrawingRect] = useState(null);
  const [events, setEvents] = useState([]);
  const [alertFlash, setAlertFlash] = useState(false);

  const [cameraStatus, setCameraStatus] = useState("CONNECTING");
  const lastFrameUpdateRef = useRef(Date.now());
  const [activeTab, setActiveTab] = useState("live");

  // Which video is selected (remembered in this browser address)
  const [videoSrc] = useState(
    () => localStorage.getItem("ibvap_video") || "/sample.mp4"
  );

  // ---- Face recognition state ----
  const [facesReady, setFacesReady] = useState(false);
  const [faceError, setFaceError] = useState("");
  const enrolledRef = useRef([]);
  const [enrolled, setEnrolled] = useState(() => {
    try {
      const saved = localStorage.getItem("ibvap_faces");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [enrollName, setEnrollName] = useState("");
  const [enrollMsg, setEnrollMsg] = useState("");
  const faceStatsRef = useRef({ attempts: 0, found: 0 });
  const [faceStats, setFaceStats] = useState({ attempts: 0, found: 0 });
  const [lastFace, setLastFace] = useState("No face check yet");

  // Load YOLO
  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        setLoading(true);
        const loadedSession = await ort.InferenceSession.create("/model/yolov8n.onnx");
        if (cancelled) return;
        setSession(loadedSession);
      } catch (err) {
        console.error("MODEL ERROR: - App.js:305", err);
        setError("Could not load YOLO model.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadModel();
    return () => { cancelled = true; };
  }, []);

  // Load face models
  useEffect(() => {
    let cancelled = false;
    async function loadFaceModels() {
      try {
        try {
          if (faceapi.tf && faceapi.tf.ready) await faceapi.tf.ready();
        } catch (e) {
          console.warn("TF ready warning: - App.js:323", e);
        }
        await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(FACE_MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(FACE_MODEL_URL);
        if (cancelled) return;
        facesReadyRef.current = true;
        setFacesReady(true);
      } catch (err) {
        console.error("FACE MODEL ERROR: - App.js:332", err);
        setFaceError("Could not load face models. Check public/face-models.");
      }
    }
    loadFaceModels();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    zoneRef.current = zone;
  }, [zone]);

  // Keep enrolled faces in a ref (for the detection loop) and in localStorage
  useEffect(() => {
    enrolledRef.current = enrolled;
    try {
      localStorage.setItem("ibvap_faces", JSON.stringify(enrolled));
    } catch (err) {
      console.error("Could not save faces: - App.js:350", err);
    }
  }, [enrolled]);

  // Camera status watchdog
  useEffect(() => {
    const watchdog = setInterval(() => {
      const sinceLastFrame = Date.now() - lastFrameUpdateRef.current;
      setCameraStatus(sinceLastFrame > CAMERA_TIMEOUT_MS ? "OFFLINE" : "ONLINE");
    }, 1000);
    return () => clearInterval(watchdog);
  }, []);

  // Main detection loop
  useEffect(() => {
    if (!session) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    const processingCanvas = document.createElement("canvas");
    processingCanvas.width = INPUT_SIZE;
    processingCanvas.height = INPUT_SIZE;
    processingCanvasRef.current = processingCanvas;
    const processingCtx = processingCanvas.getContext("2d", { willReadFrequently: true });

    const plateCanvas = document.createElement("canvas");
    plateCanvasRef.current = plateCanvas;

    const faceCanvas = document.createElement("canvas");
    faceCanvasRef.current = faceCanvas;

    function setupCanvas() {
      if (video.videoWidth === 0 || video.videoHeight === 0) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    function drawBoxes() {
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        animationRef.current = requestAnimationFrame(drawBoxes);
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const scaleX = canvas.width / INPUT_SIZE;
      const scaleY = canvas.height / INPUT_SIZE;

      tracksRef.current.forEach((box) => {
        const x = (box.x - box.w / 2) * scaleX;
        const y = (box.y - box.h / 2) * scaleY;
        const width = box.w * scaleX;
        const height = box.h * scaleY;

        let color = "#00ff00";
        if (box.isInside) color = "#ff0000";
        else if (box.isLoitering) color = "#ffaa00";
        else if (VEHICLE_CLASS_IDS.includes(box.classId)) color = "#00aaff";

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
      });

      animationRef.current = requestAnimationFrame(drawBoxes);
    }

    async function runOcrOnVehicle(track) {
      if (ocrBusyRef.current) return;
      ocrBusyRef.current = true;

      try {
        const scaleX = video.videoWidth / INPUT_SIZE;
        const scaleY = video.videoHeight / INPUT_SIZE;

        const boxX = (track.x - track.w / 2) * scaleX;
        const boxY = (track.y - track.h / 2) * scaleY;
        const boxW = track.w * scaleX;
        const boxH = track.h * scaleY;

        if (boxW < 20 || boxH < 20) return;

        plateCanvasRef.current.width = boxW;
        plateCanvasRef.current.height = boxH;
        const pctx = plateCanvasRef.current.getContext("2d");
        pctx.drawImage(video, boxX, boxY, boxW, boxH, 0, 0, boxW, boxH);

        const dataUrl = plateCanvasRef.current.toDataURL("image/png");
        const result = await Tesseract.recognize(dataUrl, "eng");
        const cleaned = cleanPlateText(result.data.text);

        const live = tracksRef.current.find((t) => t.id === track.id);
        if (cleaned.length >= 4 && live) {
          live.plateText = cleaned;
          const event = createEvent(
            "ANPR",
            live,
            { plateText: cleaned, ocrConfidence: result.data.confidence },
            captureSnapshot(video, canvas)
          );
          setEvents((prev) => [event, ...prev].slice(0, 100));
        }
      } catch (err) {
        console.error("OCR ERROR: - App.js:453", err);
      } finally {
        ocrBusyRef.current = false;
      }
    }

    async function runFaceCheck(track) {
      if (faceBusyRef.current || !facesReadyRef.current) return;
      faceBusyRef.current = true;

      try {
        const scaleX = video.videoWidth / INPUT_SIZE;
        const scaleY = video.videoHeight / INPUT_SIZE;

        const boxX = Math.max(0, (track.x - track.w / 2) * scaleX);
        const boxY = Math.max(0, (track.y - track.h / 2) * scaleY);
        const boxW = Math.min(track.w * scaleX, video.videoWidth - boxX);
        const boxH = Math.min(
          track.h * scaleY * FACE_CROP_HEIGHT_RATIO,
          video.videoHeight - boxY
        );

        // Too small/far away to contain a usable face: skip
        if (boxW < FACE_MIN_CROP_PX || boxH < FACE_MIN_CROP_PX) {
          setLastFace((prev) =>
            prev === "No face check yet" || prev.includes("too small")
              ? `Track #${track.id}: person too small for a face check (${Math.round(boxW)}x${Math.round(boxH)} px, needs ${FACE_MIN_CROP_PX})`
              : prev
          );
          return;
        }

        const fc = faceCanvasRef.current;
        fc.width = boxW;
        fc.height = boxH;
        fc.getContext("2d").drawImage(video, boxX, boxY, boxW, boxH, 0, 0, boxW, boxH);

        const result = await faceapi
          .detectSingleFace(
            fc,
            new faceapi.TinyFaceDetectorOptions({
              inputSize: 320,
              scoreThreshold: FACE_DETECTION_SCORE,
            })
          )
          .withFaceLandmarks()
          .withFaceDescriptor();

        faceStatsRef.current.attempts += 1;

        const live = tracksRef.current.find((t) => t.id === track.id);
        if (!live) {
          setFaceStats({ ...faceStatsRef.current });
          return;
        }

        if (!result) {
          live.faceAttempts = (live.faceAttempts || 0) + 1;
          setFaceStats({ ...faceStatsRef.current });
          setLastFace(
            `Track #${track.id}: no face found in the head area (${Math.round(boxW)}x${Math.round(boxH)} px)`
          );
          return;
        }

        faceStatsRef.current.found += 1;
        setFaceStats({ ...faceStatsRef.current });

        const logFace = (type, details) => {
          const snapshot = SAVE_FACE_SNAPSHOTS ? fc.toDataURL("image/jpeg", 0.7) : null;
          const event = createEvent(
            type,
            { ...live, score: result.detection.score },
            details,
            snapshot
          );
          setEvents((prev) => [event, ...prev].slice(0, 100));
        };

        const list = enrolledRef.current;

        // Nobody enrolled yet: just log that a face was seen
        if (list.length === 0) {
          live.identity = "Face";
          setLastFace(`Track #${track.id}: face found, but nobody is enrolled yet`);
          logFace("FACE_DETECTED", {});
          return;
        }

        const m = matchFace(result.descriptor, list);

        // Recognised: lock in the name
        if (m.name) {
          live.identity = m.name;
          setLastFace(`Track #${track.id}: MATCH ${m.name} (distance ${m.distance.toFixed(2)})`);
          logFace("KNOWN_FACE", {
            name: m.name,
            distance: Number(m.distance.toFixed(2)),
          });
          return;
        }

        // Not recognised: try a few more times before calling the person Unknown
        live.faceMisses = (live.faceMisses || 0) + 1;
        setLastFace(
          `Track #${track.id}: no match (distance ${m.distance.toFixed(2)}, must be below ${FACE_MATCH_THRESHOLD}), try ${live.faceMisses} of ${FACE_UNKNOWN_AFTER}`
        );
        if (live.faceMisses >= FACE_UNKNOWN_AFTER) {
          live.identity = "Unknown";
          logFace("UNKNOWN_FACE", { distance: Number(m.distance.toFixed(2)) });
          playAlertSound(660);
        }
      } catch (err) {
        console.error("FACE ERROR: - App.js:566", err);
        setLastFace(`Error while checking face: ${err.message || err}`);
      } finally {
        faceBusyRef.current = false;
      }
    }

    async function detect() {
      if (detectingRef.current) return;
      if (video.readyState < 2) return;
      if (video.videoWidth === 0) return;

      detectingRef.current = true;
      lastFrameUpdateRef.current = Date.now();

      // Reset the tracker cleanly when the video loops
      const currentVideoTime = video.currentTime;
      const looped = currentVideoTime < lastVideoTimeRef.current - 0.5;
      lastVideoTimeRef.current = currentVideoTime;

      if (looped) {
        tracksRef.current = [];
        nextIdRef.current = 1;
        setTracks([]);
        detectingRef.current = false;
        return;
      }

      try {
        processingCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
        const imageData = processingCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
        const data = imageData.data;
        const channelSize = INPUT_SIZE * INPUT_SIZE;
        const input = new Float32Array(channelSize * 3);

        for (let i = 0; i < channelSize; i++) {
          const pixel = i * 4;
          input[i] = data[pixel] / 255;
          input[channelSize + i] = data[pixel + 1] / 255;
          input[channelSize * 2 + i] = data[pixel + 2] / 255;
        }

        const tensor = new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
        const results = await session.run({ images: tensor });
        const output = results.output0.data;
        const boxes = [];

        for (let i = 0; i < 8400; i++) {
          let maxScore = 0;
          let classId = -1;
          for (let c = 0; c < NUM_CLASSES; c++) {
            const score = output[(4 + c) * 8400 + i];
            if (score > maxScore) {
              maxScore = score;
              classId = c;
            }
          }
          if (maxScore < CONFIDENCE_THRESHOLD) continue;

          const x = output[i];
          const y = output[8400 + i];
          const w = output[16800 + i];
          const h = output[25200 + i];
          boxes.push({ x, y, w, h, classId, score: maxScore });
        }

        const finalBoxes = applyNMS(boxes);

        const oldTracks = tracksRef.current;
        const oldStatusMap = new Map(
          oldTracks.map((t) => [t.id, { isInside: t.isInside, isLoitering: t.isLoitering }])
        );

        const updatedTracks = matchTracks(oldTracks, finalBoxes, nextIdRef);
        const currentZone = zoneRef.current;
        const newEventsBatch = [];
        const now = Date.now();

        updatedTracks.forEach((track) => {
          const footX = track.x;
          const footY = track.y + track.h / 2;
          const inside = isInsideZone(footX, footY, currentZone);
          track.isInside = inside;

          const oldStatus = oldStatusMap.get(track.id) || {};
          if (inside && !oldStatus.isInside) {
            const event = createEvent(
              "INTRUSION",
              track,
              { zone: currentZone },
              captureSnapshot(video, canvas)
            );
            newEventsBatch.push(event);
          }

          if (track.classId === 0) {
            const dx = track.x - track.firstSeenPosition.x;
            const dy = track.y - track.firstSeenPosition.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const elapsedSeconds = (now - track.firstSeenTime) / 1000;

            const stillInArea = distance <= LOITER_MOVEMENT_RADIUS;
            const longEnough = elapsedSeconds >= LOITER_THRESHOLD_SECONDS;

            track.isLoitering = stillInArea && longEnough;

            if (track.isLoitering && !oldStatus.isLoitering) {
              const event = createEvent(
                "LOITERING",
                track,
                { durationSeconds: Math.round(elapsedSeconds) },
                captureSnapshot(video, canvas)
              );
              newEventsBatch.push(event);
            }

            // Face check for people we haven't identified yet
            if (
              !track.identity &&
              (track.faceAttempts || 0) < FACE_MAX_ATTEMPTS &&
              now - (track.lastFaceTime || 0) > FACE_COOLDOWN_MS &&
              !faceBusyRef.current &&
              facesReadyRef.current
            ) {
              track.lastFaceTime = now;
              runFaceCheck(track);
            }
          } else {
            track.isLoitering = false;
          }

          if (
            VEHICLE_CLASS_IDS.includes(track.classId) &&
            !track.plateText &&
            now - (track.lastOcrTime || 0) > OCR_COOLDOWN_MS &&
            !ocrBusyRef.current
          ) {
            track.lastOcrTime = now;
            runOcrOnVehicle(track);
          }
        });

        tracksRef.current = updatedTracks;
        setTracks([...updatedTracks]);

        if (newEventsBatch.length > 0) {
          setEvents((prev) => [...newEventsBatch, ...prev].slice(0, 100));
          const hasIntrusion = newEventsBatch.some((e) => e.type === "INTRUSION");
          playAlertSound(hasIntrusion ? 880 : 440);
          if (hasIntrusion) {
            setAlertFlash(true);
            setTimeout(() => setAlertFlash(false), 2000);
          }
        }
      } catch (err) {
        console.error("DETECTION ERROR: - App.js:721", err);
      } finally {
        detectingRef.current = false;
      }
    }

    function startSystem() {
      setupCanvas();
      animationRef.current = requestAnimationFrame(drawBoxes);
      timerRef.current = setInterval(detect, DETECTION_INTERVAL);
    }

    function handleMetadata() {
      setupCanvas();
      startSystem();
    }

    video.addEventListener("loadedmetadata", handleMetadata);
    if (video.readyState >= 1) handleMetadata();

    return () => {
      video.removeEventListener("loadedmetadata", handleMetadata);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      tracksRef.current = [];
      setTracks([]);
      detectingRef.current = false;
    };
  }, [session]);

  // ---- Zone drawing ----
  function getRelativePoint(e) {
    const rect = containerRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    return {
      x: (px / rect.width) * INPUT_SIZE,
      y: (py / rect.height) * INPUT_SIZE,
    };
  }

  function handleMouseDown(e) {
    const point = getRelativePoint(e);
    drawStartRef.current = point;
    setDrawingRect({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  }

  function handleMouseMove(e) {
    if (!drawStartRef.current) return;
    const point = getRelativePoint(e);
    setDrawingRect({
      x1: Math.min(drawStartRef.current.x, point.x),
      y1: Math.min(drawStartRef.current.y, point.y),
      x2: Math.max(drawStartRef.current.x, point.x),
      y2: Math.max(drawStartRef.current.y, point.y),
    });
  }

  function handleMouseUp(e) {
    if (!drawStartRef.current) return;
    const point = getRelativePoint(e);
    const newZone = {
      x1: Math.min(drawStartRef.current.x, point.x),
      y1: Math.min(drawStartRef.current.y, point.y),
      x2: Math.max(drawStartRef.current.x, point.x),
      y2: Math.max(drawStartRef.current.y, point.y),
    };
    setZone(newZone);
    zoneRef.current = newZone;
    try {
      localStorage.setItem("ibvap_zone", JSON.stringify(newZone));
    } catch (err) {
      console.error("Could not save zone: - App.js:793", err);
    }
    drawStartRef.current = null;
    setDrawingRect(null);
  }

  function clearZone() {
    setZone(null);
    zoneRef.current = null;
    localStorage.removeItem("ibvap_zone");
  }

  // ---- Face enrollment ----
  async function handleEnroll() {
    const file = fileInputRef.current && fileInputRef.current.files[0];
    const name = enrollName.trim();

    if (!facesReady) {
      setEnrollMsg("Face models are still loading. Try again in a moment.");
      return;
    }
    if (!name) {
      setEnrollMsg("Type a name first.");
      return;
    }
    if (!file) {
      setEnrollMsg("Choose a photo first.");
      return;
    }

    setEnrollMsg("Processing photo...");
    try {
      const img = await faceapi.bufferToImage(file);
      const result = await faceapi
        .detectSingleFace(
          img,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 })
        )
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!result) {
        setEnrollMsg("No face found in that photo. Try a clearer, front-facing one.");
        return;
      }

      const entry = { name, descriptor: Array.from(result.descriptor) };
      setEnrolled((prev) => [...prev, entry]);
      setEnrollMsg(`Enrolled a photo for ${name}.`);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      console.error("ENROLL ERROR: - App.js:844", err);
      setEnrollMsg("Something went wrong reading that photo.");
    }
  }

  function removePerson(name) {
    setEnrolled((prev) => prev.filter((p) => p.name !== name));
  }

  const enrolledSummary = Object.entries(
    enrolled.reduce((acc, p) => {
      acc[p.name] = (acc[p.name] || 0) + 1;
      return acc;
    }, {})
  );

  // ---- Statistics ----
  const stats = {
    total: events.length,
    intrusions: events.filter((e) => e.type === "INTRUSION").length,
    loitering: events.filter((e) => e.type === "LOITERING").length,
    anpr: events.filter((e) => e.type === "ANPR").length,
    knownFaces: events.filter((e) => e.type === "KNOWN_FACE").length,
    unknownFaces: events.filter((e) => e.type === "UNKNOWN_FACE").length,
    activeTracks: tracks.length,
    uniquePlates: [
      ...new Set(events.filter((e) => e.type === "ANPR").map((e) => e.details.plateText)),
    ].length,
    enrolledPeople: enrolledSummary.length,
  };

  const tabButtonStyle = (tab) => ({
    padding: "10px 20px",
    fontSize: "14px",
    fontWeight: "bold",
    border: "none",
    borderBottom: activeTab === tab ? "3px solid #007bff" : "3px solid transparent",
    background: "transparent",
    color: activeTab === tab ? "#007bff" : "#555",
    cursor: "pointer",
  });

  const panelStyle = {
    marginTop: "20px",
    width: "100%",
    maxWidth: "900px",
    padding: "15px",
    border: "1px solid #ccc",
    borderRadius: "8px",
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>IBVAP Dashboard</h2>

      {alertFlash && (
        <div style={{
          background: "#ff0000", color: "white", fontSize: "22px",
          fontWeight: "bold", padding: "12px", marginBottom: "10px",
          textAlign: "center",
        }}>
          🚨 INTRUSION DETECTED
        </div>
      )}

      <div style={{
        display: "flex", alignItems: "center", gap: "10px",
        padding: "10px", marginBottom: "10px", borderRadius: "6px",
        background: cameraStatus === "ONLINE" ? "#e5ffe5" : cameraStatus === "OFFLINE" ? "#ffe5e5" : "#fff3cd",
        border: `1px solid ${cameraStatus === "ONLINE" ? "#2ecc71" : cameraStatus === "OFFLINE" ? "#e74c3c" : "#f1c40f"}`,
      }}>
        <div style={{
          width: "12px", height: "12px", borderRadius: "50%",
          background: cameraStatus === "ONLINE" ? "#2ecc71" : cameraStatus === "OFFLINE" ? "#e74c3c" : "#f1c40f",
        }} />
        <strong>CAM_01</strong> — Status: {cameraStatus}
        {loading && " (loading model...)"}
        {error && ` — ${error}`}
      </div>

      <div style={{ display: "flex", borderBottom: "1px solid #ccc", marginBottom: "15px" }}>
        <button style={tabButtonStyle("live")} onClick={() => setActiveTab("live")}>Live View</button>
        <button style={tabButtonStyle("history")} onClick={() => setActiveTab("history")}>Event History</button>
        <button style={tabButtonStyle("stats")} onClick={() => setActiveTab("stats")}>Statistics</button>
      </div>

      {/* ================= LIVE TAB ================= */}
      {activeTab === "live" && (
        <div>
          <div style={{
            background: "red", color: "white", fontSize: "16px",
            fontWeight: "bold", padding: "6px", marginBottom: "10px", display: "inline-block"
          }}>
            DEBUG: {tracks.length} tracks currently active
          </div>

          {session && !loading && <p style={{ color: "green" }}>✓ YOLO model loaded</p>}

          <p style={{ fontSize: "14px", color: "#555" }}>
            🖱️ Click and drag directly on the video below to draw a restricted zone.
          </p>
          <button onClick={clearZone} style={{ marginBottom: "10px", padding: "6px 12px" }}>
            Clear Zone
          </button>
          <br />
          <select
            value={videoSrc}
            onChange={(e) => {
              localStorage.setItem("ibvap_video", e.target.value);
              window.location.reload();
            }}
            style={{ marginBottom: "10px", padding: "6px" }}
          >
            <option value="/sample.mp4">Street clip (sample.mp4)</option>
            <option value="/face-test.mp4">Face test (face-test.mp4)</option>
          </select>

          <div
            ref={containerRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{ position: "relative", width: "480px", maxWidth: "100%", cursor: "crosshair" }}
          >
            <video
              ref={videoRef}
              src={videoSrc}
              controls
              autoPlay
              muted
              loop
              playsInline
              draggable={false}
              crossOrigin="anonymous"
              style={{ display: "block", width: "100%", height: "auto" }}
            />
            <canvas
              ref={canvasRef}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
            />

            {zone && (
              <div style={{
                position: "absolute",
                left: `${(zone.x1 / INPUT_SIZE) * 100}%`,
                top: `${(zone.y1 / INPUT_SIZE) * 100}%`,
                width: `${((zone.x2 - zone.x1) / INPUT_SIZE) * 100}%`,
                height: `${((zone.y2 - zone.y1) / INPUT_SIZE) * 100}%`,
                border: "2px dashed red",
                background: "rgba(255,0,0,0.15)",
                pointerEvents: "none",
              }} />
            )}

            {drawingRect && (
              <div style={{
                position: "absolute",
                left: `${(drawingRect.x1 / INPUT_SIZE) * 100}%`,
                top: `${(drawingRect.y1 / INPUT_SIZE) * 100}%`,
                width: `${((drawingRect.x2 - drawingRect.x1) / INPUT_SIZE) * 100}%`,
                height: `${((drawingRect.y2 - drawingRect.y1) / INPUT_SIZE) * 100}%`,
                border: "2px dashed orange",
                background: "rgba(255,165,0,0.15)",
                pointerEvents: "none",
              }} />
            )}

            {tracks.map((track) => {
              const left = ((track.x - track.w / 2) / INPUT_SIZE) * 100;
              const top = ((track.y - track.h / 2) / INPUT_SIZE) * 100;
              let bg = "#000000";
              let border = "#00ff00";
              if (track.isInside) { bg = "#ff0000"; border = "#ff0000"; }
              else if (track.isLoitering) { bg = "#cc7a00"; border = "#ffaa00"; }
              else if (VEHICLE_CLASS_IDS.includes(track.classId)) { bg = "#0077aa"; border = "#00aaff"; }

              return (
                <div
                  key={`${track.id}-${track.classId}`}
                  style={{
                    position: "absolute",
                    left: `${left}%`,
                    top: `${top}%`,
                    transform: "translateY(-100%)",
                    background: bg,
                    color: "#ffffff",
                    padding: "2px 6px",
                    fontSize: "13px",
                    fontWeight: "bold",
                    border: `1px solid ${border}`,
                    borderRadius: "4px",
                    whiteSpace: "nowrap",
                    pointerEvents: "none",
                    zIndex: 10,
                  }}
                >
                  #{track.id}
                  {track.isLoitering ? " ⏱" : ""}
                  {track.plateText ? ` ${track.plateText}` : ""}
                  {track.identity ? ` 👤${track.identity}` : ""}
                </div>
              );
            })}
          </div>

          {/* ---- Face recognition panel ---- */}
          <div style={panelStyle}>
            <h3>Face Recognition</h3>
            <p style={{ fontSize: "13px", color: faceError ? "red" : facesReady ? "green" : "orange" }}>
              {faceError ? faceError : facesReady ? "✓ Face models loaded" : "Loading face models..."}
            </p>
            <p style={{ fontSize: "12px", color: "#777" }}>
              Only enroll people who have agreed to it. Face data stays in this browser.
            </p>

            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center", marginBottom: "8px" }}>
              <input
                type="text"
                placeholder="Person's name"
                value={enrollName}
                onChange={(e) => setEnrollName(e.target.value)}
                style={{ padding: "6px" }}
              />
              <input type="file" accept="image/*" ref={fileInputRef} />
              <button onClick={handleEnroll} style={{ padding: "6px 12px" }}>Enroll photo</button>
            </div>
            {enrollMsg && <p style={{ fontSize: "13px" }}>{enrollMsg}</p>}

            <div style={{ marginTop: "8px" }}>
              <strong>Enrolled people ({enrolledSummary.length})</strong>
              {enrolledSummary.length === 0 ? (
                <p style={{ fontSize: "13px" }}>
                  No one enrolled yet. Faces will be logged as "FACE_DETECTED" until you enroll someone.
                </p>
              ) : (
                enrolledSummary.map(([name, count]) => (
                  <div key={name} style={{ padding: "4px 0", fontSize: "14px" }}>
                    👤 {name} — {count} photo{count > 1 ? "s" : ""}{" "}
                    <button onClick={() => removePerson(name)} style={{ marginLeft: "8px" }}>Remove</button>
                  </div>
                ))
              )}
            </div>

            <p style={{ fontSize: "12px", color: "#555", marginTop: "10px" }}>
              Face checks run: {faceStats.attempts} · Faces found: {faceStats.found}
            </p>
            <p style={{ fontSize: "13px", fontWeight: "bold" }}>
              Last result: {lastFace}
            </p>
          </div>

          <div style={panelStyle}>
            <h3>Active Tracking IDs ({tracks.length})</h3>
            {tracks.length === 0 ? (
              <p>No objects detected.</p>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
                <thead>
                  <tr style={{ background: "#333", color: "#fff", textAlign: "left" }}>
                    <th style={{ padding: "8px" }}>Track ID</th>
                    <th style={{ padding: "8px" }}>Object</th>
                    <th style={{ padding: "8px" }}>Confidence</th>
                    <th style={{ padding: "8px" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {tracks.map((track) => {
                    let status = "Normal";
                    let rowBg = "#fff";
                    if (track.isInside) { status = "🚨 Inside Zone"; rowBg = "#ffcccc"; }
                    else if (track.isLoitering) { status = "⏱ Loitering"; rowBg = "#fff3cd"; }
                    if (track.plateText) status += ` | 🚗 ${track.plateText}`;
                    if (track.identity) status += ` | 👤 ${track.identity}`;

                    return (
                      <tr key={track.id} style={{ background: rowBg, borderBottom: "1px solid #ddd" }}>
                        <td style={{ padding: "8px" }}>#{track.id}</td>
                        <td style={{ padding: "8px" }}>{CLASS_NAMES[track.classId]}</td>
                        <td style={{ padding: "8px" }}>{(track.score * 100).toFixed(0)}%</td>
                        <td style={{ padding: "8px" }}>{status}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* ================= HISTORY TAB ================= */}
      {activeTab === "history" && (
        <div style={{ width: "100%", maxWidth: "1100px", overflowX: "auto" }}>
          <h3>Event History ({events.length})</h3>
          {events.length === 0 ? (
            <p>No events yet.</p>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <thead>
                <tr style={{ background: "#333", color: "#fff", textAlign: "left" }}>
                  <th style={{ padding: "6px" }}>ID</th>
                  <th style={{ padding: "6px" }}>Type</th>
                  <th style={{ padding: "6px" }}>Camera ID</th>
                  <th style={{ padding: "6px" }}>Track ID</th>
                  <th style={{ padding: "6px" }}>Class ID</th>
                  <th style={{ padding: "6px" }}>Timestamp</th>
                  <th style={{ padding: "6px" }}>Time Display</th>
                  <th style={{ padding: "6px" }}>Confidence</th>
                  <th style={{ padding: "6px" }}>Snapshot</th>
                  <th style={{ padding: "6px" }}>Details</th>
                </tr>
              </thead>
              <tbody>
                {events.map((event, idx) => {
                  let rowColor = "#f9f9f9";
                  if (event.type === "INTRUSION") rowColor = "#ffe5e5";
                  else if (event.type === "LOITERING") rowColor = "#fff3cd";
                  else if (event.type === "ANPR") rowColor = "#e5f3ff";
                  else if (event.type === "KNOWN_FACE") rowColor = "#e5ffe5";
                  else if (event.type === "UNKNOWN_FACE") rowColor = "#f3e5ff";
                  else if (event.type === "FACE_DETECTED") rowColor = "#eef6ee";

                  return (
                    <tr key={event.id} style={{ background: idx % 2 === 0 ? rowColor : "#fff", borderBottom: "1px solid #ddd" }}>
                      <td style={{ padding: "6px", fontFamily: "monospace", fontSize: "11px" }}>{event.id}</td>
                      <td style={{ padding: "6px", fontWeight: "bold" }}>{event.type}</td>
                      <td style={{ padding: "6px" }}>{event.cameraId}</td>
                      <td style={{ padding: "6px" }}>{event.trackId}</td>
                      <td style={{ padding: "6px" }}>{event.classId} ({CLASS_NAMES[event.classId]})</td>
                      <td style={{ padding: "6px", fontFamily: "monospace", fontSize: "11px" }}>{event.timestamp}</td>
                      <td style={{ padding: "6px" }}>{event.timeDisplay}</td>
                      <td style={{ padding: "6px" }}>{(event.confidence * 100).toFixed(0)}%</td>
                      <td style={{ padding: "6px" }}>
                        {event.snapshot ? (
                          <img src={event.snapshot} alt="snapshot" style={{ width: "60px", height: "auto", border: "1px solid #999" }} />
                        ) : "—"}
                      </td>
                      <td style={{ padding: "6px", fontFamily: "monospace", fontSize: "11px" }}>
                        {JSON.stringify(event.details)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* ================= STATS TAB ================= */}
      {activeTab === "stats" && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "15px", maxWidth: "1100px" }}>
          {[
            { label: "Total Events", value: stats.total, color: "#333" },
            { label: "Intrusions", value: stats.intrusions, color: "#e74c3c" },
            { label: "Loitering Events", value: stats.loitering, color: "#f1c40f" },
            { label: "Plates Read (ANPR)", value: stats.anpr, color: "#3498db" },
            { label: "Unique Plates", value: stats.uniquePlates, color: "#3498db" },
            { label: "Known Faces", value: stats.knownFaces, color: "#27ae60" },
            { label: "Unknown Faces", value: stats.unknownFaces, color: "#8e44ad" },
            { label: "Enrolled People", value: stats.enrolledPeople, color: "#27ae60" },
            { label: "Currently Active Tracks", value: stats.activeTracks, color: "#2ecc71" },
          ].map((card) => (
            <div key={card.label} style={{
              flex: "1 1 200px", padding: "20px", borderRadius: "8px",
              border: `2px solid ${card.color}`, textAlign: "center"
            }}>
              <div style={{ fontSize: "32px", fontWeight: "bold", color: card.color }}>{card.value}</div>
              <div style={{ fontSize: "14px", color: "#555", marginTop: "5px" }}>{card.label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;