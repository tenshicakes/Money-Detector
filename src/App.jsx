import './App.css'
import { useEffect, useRef, useState } from 'react'
import * as InferenceJS from 'inferencejs'


export default function App() {
  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const [showNotification, setShowNotification] = useState(false)
  const [isDarkMode, setIsDarkMode] = useState(true)
  // inference engine refs
  const engineRef = useRef(null)
  const workerRef = useRef(null)
  const [detection, setDetection] = useState(null)
  const detectBuffer = useRef([])

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          // when metadata arrives, size overlay canvas
          videoRef.current.onloadedmetadata = () => {
            if (overlayRef.current) {
              overlayRef.current.width = videoRef.current.videoWidth
              overlayRef.current.height = videoRef.current.videoHeight
            }
          }
        }
      } catch (error) {
        console.error('Camera error:', error)
        alert('Could not access camera. Please check permissions.')
      }
    }

    const initEngine = async () => {
      try {
        const API_KEY = 'rf_J8z0Ag3yFvdfvLiDpmtOPMVcNQs2';
        const MODEL_NAME = 'bill-detection-emid4';
        const MODEL_VERSION = 2;

        // use the npm-provided inferencejs module
        const EngineCtor = InferenceJS?.InferenceEngine || InferenceJS?.default?.InferenceEngine || InferenceJS?.default || InferenceJS;
        if (!EngineCtor) throw new Error('Could not resolve InferenceEngine from inferencejs package');
        const engine = new EngineCtor();
        engineRef.current = engine;
        // start worker for the Roboflow model using the API key
        const workerId = await engine.startWorker(MODEL_NAME, MODEL_VERSION, API_KEY);
        workerRef.current = workerId;
      } catch (err) {
        console.error('engine initialization error', err);
      }
    }

    startCamera()
    initEngine()

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const showNotificationMessage = () => {
    setShowNotification(true)
    setTimeout(() => {
      setShowNotification(false)
    }, 3000)
  }

  const captureImage = () => {
    // simply flash notification when user presses capture
    showNotificationMessage()
  }

  // detection loop using inferencejs engine + triple‑check filter
  useEffect(() => {
    let intervalId;
    const runDetection = async () => {
      const engine = engineRef.current;
      const workerId = workerRef.current;
      if (!engine || !workerId || !videoRef.current) return;

      // draw current frame to temporary canvas
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);

      try {
        const res = await engine.infer(workerId, canvas);
        const preds = res?.predictions || res?.pred || [];

        // draw any boxes on overlay
        const overlay = overlayRef.current;
        if (overlay) {
          const octx = overlay.getContext('2d');
          octx.clearRect(0, 0, overlay.width, overlay.height);
          preds.forEach(p => {
            const x = p.x || p.bbox?.x || 0;
            const y = p.y || p.bbox?.y || 0;
            const w = p.width || p.bbox?.width || p.bbox?.w || 0;
            const h = p.height || p.bbox?.height || p.bbox?.h || 0;
            octx.strokeStyle = 'lime';
            octx.lineWidth = 2;
            octx.strokeRect(x, y, w, h);
            octx.fillStyle = 'lime';
            const label = p.class || p.label || p.name;
            if (label) octx.fillText(label, x + 2, y + 12);
          });
        }

        if (preds.length > 0) {
          const top = preds.reduce((a, b) => (a.confidence > b.confidence ? a : b));
          const label = top.class || top.label || top.name;

          detectBuffer.current.push(label);
          if (detectBuffer.current.length > 3) detectBuffer.current.shift();

          if (
            detectBuffer.current.length === 3 &&
            detectBuffer.current.every(l => l === label)
          ) {
            setDetection(label);
          }
        } else {
          // no prediction, push null to history
          detectBuffer.current.push(null);
          if (detectBuffer.current.length > 3) detectBuffer.current.shift();
        }
      } catch (e) {
        console.error('detection error', e);
      }
    };
    intervalId = setInterval(runDetection, 500);
    return () => clearInterval(intervalId);
  }, []);

  return (

    <div className={`app-container ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
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

      {showNotification && (

        <div className="notification">
        Image captured!
        </div>
      )}

      <header className="header">
        <h1>Currency Recognition System for the Visually Impaired</h1>
      </header>

      <div className="camera-section">

        <p className="guide-text">Place the bill in front of the camera and press Capture.</p>

        <input 
        type='range' 
        className="volume-slider">
          
        </input>

        <div className="camera-preview" style={{ position: 'relative' }}>
          <video ref={videoRef} autoPlay playsInline></video>
          <canvas
            ref={overlayRef}
            className="overlay"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              pointerEvents: 'none'
            }}
          ></canvas>
        </div>

      </div>

      <div className="button-container">
        <button className="capture-btn" onClick={captureImage}>Capture Image</button>
        <button className="audio-btn" onClick={() => {}}>Replay Audio</button>
      </div>

      <div className="captions-container">
        <button className="clear-btn" onClick={() => setDetection(null)}>Clear</button>
        <p className="captions-text">
          {detection ? `Detected: ${detection}` : 'Captions will appear here...'}
        </p>
      </div>
      
    </div>
  )
}
