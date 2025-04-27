import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import OverlayCanvas from './OverlayCanvas';
import { Cluster, AugmentedDetectedFace } from './App';

// Define the props for WebcamTile
interface WebcamTileProps {
  onFrame: (imageDataUrl: string) => void; // Callback to send frame data
  analysisResults?: any; // Data from backend analysis (YOLO, ArcFace)
  clusters?: Cluster[];
  allDetectedFaces?: Record<string, AugmentedDetectedFace>;
}

// Define the interface for the functions exposed via the ref
export interface WebcamTileRef {
  captureCurrentImage: () => string | null;
}

// Type for the rendered dimensions state
interface RenderedDimensions {
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
  originalWidth: number;
  originalHeight: number;
}

// Use forwardRef to allow parent components to get a ref to this component
const WebcamTile = forwardRef<WebcamTileRef, WebcamTileProps>((
  { onFrame, analysisResults, clusters, allDetectedFaces },
  ref
) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaContainerRef = useRef<HTMLDivElement>(null); // Ref for the container div
  const [error, setError] = useState<string | null>(null);
  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [renderedDimensions, setRenderedDimensions] = useState<RenderedDimensions | null>(null);

  useImperativeHandle(ref, () => ({
    captureCurrentImage: () => {
      if (!videoRef.current || !streamActive) return null;
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        try { return canvas.toDataURL('image/jpeg'); }
        catch (e) { console.error("Error getting data URL:", e); return null; }
      }
      return null;
    }
  }));

  // Effect to calculate rendered video dimensions
  useEffect(() => {
    const video = videoRef.current;
    const container = mediaContainerRef.current;
    if (!video || !container || !streamActive) return;

    let observer: ResizeObserver | null = null;
    let intervalId: number | null = null;

    const calculateDimensions = () => {
      if (!video || !container || video.readyState < video.HAVE_METADATA) return;

      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      const originalWidth = video.videoWidth;
      const originalHeight = video.videoHeight;

      if (originalWidth === 0 || originalHeight === 0 || containerWidth === 0 || containerHeight === 0) {
        setRenderedDimensions(null);
        return;
      }

      const containerRatio = containerWidth / containerHeight;
      const videoRatio = originalWidth / originalHeight;

      let renderedWidth = 0;
      let renderedHeight = 0;
      let offsetX = 0;
      let offsetY = 0;

      if (videoRatio > containerRatio) {
        renderedWidth = containerWidth;
        renderedHeight = containerWidth / videoRatio;
        offsetX = 0;
        offsetY = (containerHeight - renderedHeight) / 2;
      } else {
        renderedHeight = containerHeight;
        renderedWidth = containerHeight * videoRatio;
        offsetY = 0;
        offsetX = (containerWidth - renderedWidth) / 2;
      }

      setRenderedDimensions(prev => {
        // Avoid unnecessary state updates if dimensions haven't changed significantly
        if (prev && Math.abs(prev.width - renderedWidth) < 1 && Math.abs(prev.height - renderedHeight) < 1) {
          return prev;
        }
        return {
          width: renderedWidth,
          height: renderedHeight,
          offsetX: offsetX,
          offsetY: offsetY,
          originalWidth: originalWidth,
          originalHeight: originalHeight,
        };
      });
    };

    // Video metadata might take time, check periodically and on resize
    video.addEventListener('loadedmetadata', calculateDimensions);
    video.addEventListener('play', calculateDimensions); // Also check when play starts
    observer = new ResizeObserver(calculateDimensions);
    observer.observe(container);
    intervalId = window.setInterval(calculateDimensions, 500); // Check periodically

    calculateDimensions(); // Initial check

    return () => {
      video.removeEventListener('loadedmetadata', calculateDimensions);
      video.removeEventListener('play', calculateDimensions);
      if (observer) observer.disconnect();
      if (intervalId) clearInterval(intervalId);
      setRenderedDimensions(null);
    };
  }, [streamActive]); // Re-run when stream becomes active

  useEffect(() => {
    let stream: MediaStream | null = null;
    let animationFrameId: number | null = null;
    let lastFrameTime = 0;
    const frameInterval = 100; // ms, ~10 FPS. Adjust as needed.

    const startWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setStreamActive(true);
            setError(null);
            requestAnimationFrame(captureFrame);
          };
        }
      } catch (err) {
        console.error("Error accessing webcam: ", err);
        setError(`Error accessing webcam: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setStreamActive(false);
      }
    };

    const captureFrame = (timestamp: number) => {
      if (!videoRef.current || !streamActive) return;

      if (timestamp - lastFrameTime >= frameInterval) {
        lastFrameTime = timestamp;
        const video = videoRef.current;
        // Use a temporary canvas for analysis frame to avoid conflicts
        const analysisCanvas = document.createElement('canvas');
        analysisCanvas.width = video.videoWidth;
        analysisCanvas.height = video.videoHeight;
        const ctx = analysisCanvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);
          const imageDataUrl = analysisCanvas.toDataURL('image/jpeg');
          onFrame(imageDataUrl);
        }
      }

      // Continue the loop
      animationFrameId = requestAnimationFrame(captureFrame);
    };

    startWebcam();

    // Cleanup
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if(videoRef.current) {
        videoRef.current.srcObject = null; // Release the stream
      }
       setStreamActive(false);
    };
  }, [onFrame]); // Re-run effect if onFrame changes

  return (
    <div className="tile webcam-tile">
      {/* Container for video and overlay */}
      <div ref={mediaContainerRef} className="media-container">
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <video ref={videoRef} muted playsInline style={{ display: streamActive ? 'block' : 'none' }} />

        {!streamActive && !error && <p>Starting webcam...</p>}

        {/* Render canvas only when dimensions are calculated */}
        {streamActive && renderedDimensions && videoRef.current && (
          <OverlayCanvas
            results={analysisResults}
            renderedContentWidth={renderedDimensions.width}
            renderedContentHeight={renderedDimensions.height}
            renderedContentOffsetX={renderedDimensions.offsetX}
            renderedContentOffsetY={renderedDimensions.offsetY}
            originalWidth={renderedDimensions.originalWidth}
            originalHeight={renderedDimensions.originalHeight}
            clusters={clusters}
            allDetectedFaces={allDetectedFaces}
          />
        )}
      </div>
      {/* No Gemini controls needed here */}
    </div>
  );
});

export default WebcamTile;