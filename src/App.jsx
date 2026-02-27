import './App.css'
import { useEffect, useRef, useState } from 'react'


export default function App() {
  const videoRef = useRef(null)
  const [showNotification, setShowNotification] = useState(false)
  const [isDarkMode, setIsDarkMode] = useState(true)

  useEffect(() => {
    const startCamera = async () => {
      try {

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false
        })
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (error) {
        console.error('Camera error:', error)
        alert('Could not access camera. Please check permissions.')
      }
    }

    startCamera()

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
    if (videoRef.current) {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const context = canvas.getContext('2d')
      context.drawImage(videoRef.current, 0, 0)

      showNotificationMessage()
    }
  }

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

        <div className="camera-preview">
          <video ref={videoRef} autoPlay playsInline></video>
        </div>

      </div>

      <div className="button-container">
        <button className="capture-btn" onClick={captureImage}>Capture Image</button>
        <button className="audio-btn" onClick={() => {}}>Replay Audio</button>
      </div>

      <div className="captions-container">
      <button className="clear-btn" onClick={() => {}}>Clear</button>
        <p className="captions-text">Captions will appear here...</p> 
      </div>
      
    </div>
  )
}
