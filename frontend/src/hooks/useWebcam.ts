import { useRef, useEffect, useState } from 'react';

const useWebcam = (width = 320, height = 240) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    const getMedia = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width, height } });
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
        setStream(mediaStream);
      } catch (err) {
        console.error("Error accessing webcam: ", err);
      }
    };

    getMedia();

    return () => {
      // Cleanup: stop the stream when the component unmounts
      stream?.getTracks().forEach(track => track.stop());
    };
    // Re-run effect if width/height props change (though unlikely for this simple hook)
  }, [width, height]);

  const captureImage = (): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas && video.readyState === 4) { // readyState 4 means HAVE_ENOUGH_DATA
      const context = canvas.getContext('2d');
      if (context) {
        // Set canvas dimensions to match video stream
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Return image as data URL
        return canvas.toDataURL('image/jpeg');
      }
    }
    return null;
  };

  return { videoRef, canvasRef, captureImage, stream };
};

export default useWebcam; 