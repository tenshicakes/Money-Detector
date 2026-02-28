import './App.css'
import { useEffect, useRef, useState, useCallback } from 'react'

/* ──────────────────────────────────────────────
   Roboflow inferencejs — client-side only
   The UMD bundle is loaded via <script> in index.html
   and exposes window.InferenceEngine & window.CVImage.
   ────────────────────────────────────────────── */

const RF_API_KEY   = 'rf_J8z0Ag3yFvdfvLiDpmtOPMVcNQs2'
const RF_MODEL     = 'bill-detection-emid4'
const RF_VERSION   = 2
const VALID_CLASSES = ['20', '50', '100', '200', '500']
const CONFIDENCE_THRESHOLD = 0.25
const TRIPLE_CHECK_COUNT = 3

/* ── colour per denomination for bounding boxes ── */
const CLASS_COLORS = {
  '20':  '#00e5ff',
  '50':  '#76ff03',
  '100': '#ffea00',
  '200': '#ff9100',
  '500': '#ff1744',
}
const DEFAULT_COLOR = '#00ff00'

/* ── helper: normalise any prediction structure ── */
function normalisePrediction(p) {
  return {
    class:      p.class ?? p.label ?? p.name ?? 'unknown',
    confidence: p.confidence ?? p.score ?? 0,
    x:          p.bbox?.x      ?? p.x      ?? 0,
    y:          p.bbox?.y      ?? p.y      ?? 0,
    width:      p.bbox?.width  ?? p.bbox?.w ?? p.width  ?? 0,
    height:     p.bbox?.height ?? p.bbox?.h ?? p.height ?? 0,
  }
}

export default function App() {
  /* ── refs ── */
  const videoRef       = useRef(null)
  const overlayRef     = useRef(null)
  const imageCanvasRef = useRef(null)
  const imageOverlayRef= useRef(null)
  const engineRef      = useRef(null)
  const workerRef      = useRef(null)
  const detectBuffer   = useRef([])
  const loopActiveRef  = useRef(false)
  const fileInputRef   = useRef(null)

  /* ── state ── */
  const [showNotification, setShowNotification] = useState(false)
  const [notificationMsg, setNotificationMsg]   = useState('')
  const [isDarkMode, setIsDarkMode]             = useState(true)
  const [detection, setDetection]               = useState(null)
  const [mode, setMode]                         = useState('realtime') // 'realtime' | 'image'
  const [modelReady, setModelReady]             = useState(false)
  const [uploadedImage, setUploadedImage]       = useState(null)
  const [imagePreds, setImagePreds]             = useState([])
  const [isDetecting, setIsDetecting]           = useState(false)

  /* ── notification helper ── */
  const notify = useCallback((msg) => {
    setNotificationMsg(msg)
    setShowNotification(true)
    setTimeout(() => setShowNotification(false), 3000)
  }, [])

  /* ── auto-speak helper (avoids repeating same detection) ── */
  const lastSpokenRef = useRef(null)
  const speak = useCallback((denomination) => {
    if (!denomination || !window.speechSynthesis) return
    // Don't repeat the same denomination back-to-back
    if (lastSpokenRef.current === denomination) return
    lastSpokenRef.current = denomination
    // Cancel any ongoing speech first
    window.speechSynthesis.cancel()
    const utter = new SpeechSynthesisUtterance(`Detected ${denomination} pesos`)
    utter.rate = 1.1
    window.speechSynthesis.speak(utter)
    // Reset after 5s so same bill can be announced again if re-detected
    setTimeout(() => { lastSpokenRef.current = null }, 5000)
  }, [])

  /* ────────────────────────────────────────────
     1. Camera init
     ──────────────────────────────────────────── */
  useEffect(() => {
    let stream = null
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => {
            if (overlayRef.current) {
              overlayRef.current.width  = videoRef.current.videoWidth
              overlayRef.current.height = videoRef.current.videoHeight
            }
          }
        }
      } catch (err) {
        console.error('Camera error:', err)
        alert('Could not access camera. Please check permissions.')
      }
    }
    startCamera()
    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop())
    }
  }, [])

  /* ────────────────────────────────────────────
     2. inferencejs engine init (client-side)
     Uses the UMD global loaded in index.html:
       window.InferenceEngine, window.CVImage
     ──────────────────────────────────────────── */
  useEffect(() => {
    let cancelled = false
    const init = async () => {
      try {
        // Resolve InferenceEngine constructor — UMD global OR npm import
        let EngineCtor = null
        if (typeof window !== 'undefined' && window.InferenceEngine) {
          EngineCtor = window.InferenceEngine
        } else {
          try {
            const mod = await import('inferencejs')
            EngineCtor = mod.InferenceEngine || mod.default?.InferenceEngine || mod.default || mod
          } catch { /* not available */ }
        }

        if (!EngineCtor) {
          console.warn('InferenceEngine not available — will use Roboflow API fallback')
          if (!cancelled) setModelReady(true) // allow fallback mode
          return
        }

        const engine = new EngineCtor()
        engineRef.current = engine

        const workerId = await engine.startWorker(RF_MODEL, RF_VERSION, RF_API_KEY)
        workerRef.current = workerId

        if (!cancelled) {
          setModelReady(true)
          console.info(`[inferencejs] worker ${workerId} started for ${RF_MODEL}/${RF_VERSION}`)
        }
      } catch (err) {
        console.error('Engine init error:', err)
        if (!cancelled) setModelReady(true) // proceed with fallback
      }
    }
    init()
    return () => { cancelled = true }
  }, [])

  /* ────────────────────────────────────────────
     3. Run a single inference on a canvas or
        HTMLVideoElement / HTMLImageElement
     CVImage only supports: ImageBitmap, HTMLImageElement, tf.Tensor
     So for <video> or <canvas> we must first create an ImageBitmap.
     ──────────────────────────────────────────── */
  const runInference = useCallback(async (source) => {
    const engine   = engineRef.current
    const workerId = workerRef.current

    if (!engine || workerId == null || typeof engine.infer !== 'function') {
      console.warn('[runInference] inferencejs engine not ready yet — skipping frame')
      return []
    }

    try {
      // CVImage accepts ImageBitmap | HTMLImageElement | tf.Tensor
      // For <video> or <canvas> elements, convert to ImageBitmap first
      let inferInput

      if (source instanceof HTMLVideoElement) {
        // Grab current video frame as ImageBitmap
        if (source.readyState < 2) return [] // not enough data
        const bitmap = await createImageBitmap(source)
        inferInput = bitmap
      } else if (source instanceof HTMLCanvasElement) {
        const bitmap = await createImageBitmap(source)
        inferInput = bitmap
      } else if (source instanceof HTMLImageElement) {
        // HTMLImageElement is directly supported by CVImage
        inferInput = source
      } else {
        // Assume ImageBitmap or other supported type
        inferInput = source
      }

      const result = await engine.infer(workerId, inferInput)

      // result is an array of predictions for object-detection models
      const preds = Array.isArray(result) ? result : (result?.predictions || result?.pred || [])
      return preds.map(normalisePrediction)
    } catch (err) {
      console.error('[inferencejs] infer error:', err)
      return []
    }
  }, [])

  /* ────────────────────────────────────────────
     4. Drawing bounding boxes on an overlay canvas
     The Roboflow API returns centre‐origin coords:
       (x, y) = centre of box, (width, height) = size
     ──────────────────────────────────────────── */
  const drawBoxes = useCallback((overlay, preds, sourceW, sourceH) => {
    if (!overlay) return
    const ctx = overlay.getContext('2d')
    overlay.width = sourceW
    overlay.height = sourceH
    ctx.clearRect(0, 0, overlay.width, overlay.height)

    preds.forEach(p => {
      // convert centre coords to top-left
      const bx = p.x - p.width / 2
      const by = p.y - p.height / 2

      const color = CLASS_COLORS[p.class] || p.color || DEFAULT_COLOR

      // box
      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.strokeRect(bx, by, p.width, p.height)

      // label background
      const label = `${p.class}  ${(p.confidence * 100).toFixed(0)}%`
      ctx.font = 'bold 16px sans-serif'
      const metrics = ctx.measureText(label)
      const textH = 20
      ctx.fillStyle = color
      ctx.globalAlpha = 0.75
      ctx.fillRect(bx, by - textH - 4, metrics.width + 10, textH + 4)
      ctx.globalAlpha = 1

      // label text
      ctx.fillStyle = '#000'
      ctx.textBaseline = 'top'
      ctx.fillText(label, bx + 5, by - textH - 1)
    })
  }, [])

  /* ────────────────────────────────────────────
     5. Real-time detection loop (requestAnimationFrame-based)
     Runs continuously while mode === 'realtime' and model is ready.
     Uses a triple-check buffer: detection is accepted only
     when 3 consecutive frames agree on the same top class.
     ──────────────────────────────────────────── */
  useEffect(() => {
    if (mode !== 'realtime' || !modelReady) return
    let rafId = null
    let running = true
    loopActiveRef.current = true

    const loop = async () => {
      if (!running) return
      const video = videoRef.current
      if (video && video.readyState >= 2) {
        try {
          const preds = await runInference(video)

          // filter to valid denominations & confidence threshold
          const validPreds = preds.filter(
            p => VALID_CLASSES.includes(p.class) && p.confidence >= CONFIDENCE_THRESHOLD
          )

          // overlay
          drawBoxes(
            overlayRef.current,
            validPreds,
            video.videoWidth || 640,
            video.videoHeight || 480,
          )

          // triple-check buffer
          if (validPreds.length > 0) {
            const top = validPreds.reduce((a, b) => (a.confidence > b.confidence ? a : b))
            detectBuffer.current.push(top.class)
            if (detectBuffer.current.length > TRIPLE_CHECK_COUNT) detectBuffer.current.shift()

            if (
              detectBuffer.current.length === TRIPLE_CHECK_COUNT &&
              detectBuffer.current.every(l => l === top.class)
            ) {
              setDetection(top.class)
              speak(top.class)
            }
          } else {
            detectBuffer.current.push(null)
            if (detectBuffer.current.length > TRIPLE_CHECK_COUNT) detectBuffer.current.shift()
          }
        } catch (e) {
          console.error('detection loop error', e)
        }
      }

      // throttle to ~2-4 fps to be kind to the inference worker
      if (running) rafId = setTimeout(() => requestAnimationFrame(loop), 300)
    }

    requestAnimationFrame(loop)
    return () => {
      running = false
      loopActiveRef.current = false
      if (rafId) clearTimeout(rafId)
    }
  }, [mode, modelReady, runInference, drawBoxes, speak])

  /* ────────────────────────────────────────────
     6. Image upload handler
     ──────────────────────────────────────────── */
  const handleImageUpload = useCallback((e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setUploadedImage(url)
    setImagePreds([])
    setDetection(null)
    detectBuffer.current = []
  }, [])

  /* ────────────────────────────────────────────
     7. Detect on uploaded image (triple‑checked)
     Runs inference 3 times; if all 3 agree on top class → confirmed.
     ──────────────────────────────────────────── */
  const detectOnImage = useCallback(async () => {
    if (!uploadedImage) return
    setIsDetecting(true)
    setDetection(null)
    setImagePreds([])

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = uploadedImage

    await new Promise((res) => { img.onload = res })

    const canvas = document.createElement('canvas')
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    canvas.getContext('2d').drawImage(img, 0, 0)

    // Run inference 3 times for triple-check
    const results = []
    for (let i = 0; i < TRIPLE_CHECK_COUNT; i++) {
      const preds = await runInference(canvas)
      const valid = preds.filter(
        p => VALID_CLASSES.includes(p.class) && p.confidence >= CONFIDENCE_THRESHOLD
      )
      if (valid.length > 0) {
        const top = valid.reduce((a, b) => (a.confidence > b.confidence ? a : b))
        results.push({ label: top.class, preds: valid })
      } else {
        results.push({ label: null, preds: [] })
      }
    }

    // Use the preds from the last run for drawing
    const lastValid = results[results.length - 1].preds
    setImagePreds(lastValid)

    // Draw boxes on image overlay
    drawBoxes(imageOverlayRef.current, lastValid, img.naturalWidth, img.naturalHeight)

    // Triple-check: all 3 must agree
    const labels = results.map(r => r.label)
    if (labels.every(l => l !== null && l === labels[0])) {
      setDetection(labels[0])
      notify(`Confirmed: ₱${labels[0]} bill detected (3/3 checks passed)`)
      speak(labels[0])
    } else {
      const counts = {}
      labels.forEach(l => { if (l) counts[l] = (counts[l] || 0) + 1 })
      const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
      if (best) {
        setDetection(`${best[0]} (${best[1]}/${TRIPLE_CHECK_COUNT} checks)`)
        notify(`Likely: ₱${best[0]} bill (${best[1]}/${TRIPLE_CHECK_COUNT} checks matched)`)
        speak(best[0])
      } else {
        notify('No bill detected in image.')
      }
    }

    setIsDetecting(false)
  }, [uploadedImage, runInference, drawBoxes, notify, speak])

  /* ────────────────────────────────────────────
     8. Mode switch handler
     ──────────────────────────────────────────── */
  const switchMode = useCallback((newMode) => {
    setMode(newMode)
    setDetection(null)
    detectBuffer.current = []
    setImagePreds([])
    // Clear overlays
    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext('2d')
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
    }
  }, [])

  /* ────────────────────────────────────────────
     9. Capture = take a snapshot from camera,
        do 3× inference on that snapshot, confirm.
     ──────────────────────────────────────────── */
  const captureImage = useCallback(async () => {
    const video = videoRef.current
    if (!video) return
    notify('Capturing & analysing…')

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)

    // Run triple-check on the snapshot
    const results = []
    for (let i = 0; i < TRIPLE_CHECK_COUNT; i++) {
      const preds = await runInference(canvas)
      const valid = preds.filter(
        p => VALID_CLASSES.includes(p.class) && p.confidence >= CONFIDENCE_THRESHOLD
      )
      if (valid.length > 0) {
        const top = valid.reduce((a, b) => (a.confidence > b.confidence ? a : b))
        results.push(top.class)
      } else {
        results.push(null)
      }
    }

    const labels = results.filter(Boolean)
    if (labels.length === TRIPLE_CHECK_COUNT && labels.every(l => l === labels[0])) {
      setDetection(labels[0])
      notify(`Confirmed: ₱${labels[0]} bill (3/3 checks)`)
      speak(labels[0])
    } else if (labels.length > 0) {
      const counts = {}
      labels.forEach(l => { counts[l] = (counts[l] || 0) + 1 })
      const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
      notify(`Likely: ₱${best[0]} bill (${best[1]}/${TRIPLE_CHECK_COUNT} checks)`)
      setDetection(best[0])
      speak(best[0])
    } else {
      notify('No bill detected. Try again.')
    }
  }, [runInference, notify, speak])

  /* ──────────── RENDER ──────────── */
  return (
    <div className={`app-container ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>

      {/* Theme toggle */}
      <div className="theme-toggle">
        <input
          type="checkbox"
          id="theme-switch"
          checked={isDarkMode}
          onChange={(e) => setIsDarkMode(e.target.checked)}
          className="theme-checkbox"
        />
        <label htmlFor="theme-switch" className="theme-slider"></label>
      </div>

      {/* Notification */}
      {showNotification && (
        <div className="notification">
          {notificationMsg}
        </div>
      )}

      {/* Header */}
      <header className="header">
        <h1>Currency Recognition System for the Visually Impaired</h1>
      </header>

      {/* Mode toggle */}
      <div className="mode-toggle-container">
        <button
          className={`mode-btn ${mode === 'realtime' ? 'mode-active' : ''}`}
          onClick={() => switchMode('realtime')}
        >
          🎥 Real-time
        </button>
        <button
          className={`mode-btn ${mode === 'image' ? 'mode-active' : ''}`}
          onClick={() => switchMode('image')}
        >
          🖼️ Image Upload
        </button>
      </div>

      {/* Model loading indicator */}
      {!modelReady && (
        <p className="guide-text" style={{ textAlign: 'center' }}>
          ⏳ Loading detection model…
        </p>
      )}

      {/* ─── REALTIME MODE ─── */}
      {mode === 'realtime' && (
        <div className="camera-section">
          <p className="guide-text">
            Point the camera at a bill — detection runs automatically.
          </p>

          <input type="range" className="volume-slider" />

          <div className="camera-preview" style={{ position: 'relative' }}>
            <video ref={videoRef} autoPlay playsInline></video>
            <canvas
              ref={overlayRef}
              className="overlay"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                pointerEvents: 'none',
              }}
            ></canvas>
          </div>
        </div>
      )}

      {/* ─── IMAGE MODE ─── */}
      {mode === 'image' && (
        <div className="camera-section">
          <p className="guide-text">
            Upload an image of a bill to detect its denomination.
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="file-input"
          />

          {uploadedImage && (
            <div className="camera-preview" style={{ position: 'relative' }}>
              <img
                src={uploadedImage}
                alt="Uploaded bill"
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />
              <canvas
                ref={imageOverlayRef}
                className="overlay"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  pointerEvents: 'none',
                }}
              ></canvas>
            </div>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="button-container">
        {mode === 'realtime' && (
          <button className="capture-btn" onClick={captureImage}>
            Capture &amp; Verify
          </button>
        )}
        {mode === 'image' && (
          <button
            className="capture-btn"
            onClick={detectOnImage}
            disabled={!uploadedImage || isDetecting}
          >
            {isDetecting ? 'Detecting…' : 'Detect Bill'}
          </button>
        )}
        <button className="audio-btn" onClick={() => {
          if (detection && window.speechSynthesis) {
            const utter = new SpeechSynthesisUtterance(`Detected ${detection} pesos`)
            window.speechSynthesis.speak(utter)
          }
        }}>
          Replay Audio
        </button>
      </div>

      {/* Detection result */}
      <div className="captions-container">
        <button className="clear-btn" onClick={() => { setDetection(null); detectBuffer.current = []; lastSpokenRef.current = null }}>
          Clear
        </button>
        <p className="captions-text">
          {detection ? `Detected: ₱${detection}` : 'Captions will appear here...'}
        </p>
      </div>

      {/* Hidden video for image-mode (camera stays alive for switching back) */}
      {mode === 'image' && (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ display: 'none' }}
        ></video>
      )}
    </div>
  )
}
