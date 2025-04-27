import React, { useState } from 'react';
import { ImageTileData } from '../types';
import * as api from '../services/api';

interface ImageTileProps {
  image: ImageTileData;
}

const ImageTile: React.FC<ImageTileProps> = ({ image }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [geminiDescription, setGeminiDescription] = useState<string | null>(null);
  const [geminiQuestion, setGeminiQuestion] = useState("");
  const [geminiResponse, setGeminiResponse] = useState<string | null>(null);

  const handleApiCall = async (apiFunction: () => Promise<any>, resultSetter?: (result: any) => void) => {
      setIsLoading(true);
      setError(null);
      try {
          const result = await apiFunction();
          console.log(`API Result for ${image.id}:`, result);
          if (resultSetter) {
              resultSetter(result);
          }
          // TODO: Update image state in TileGrid if needed (e.g., for overlays)
      } catch (err) {
          console.error(`API Error for ${image.id}:`, err);
          setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
          setIsLoading(false);
      }
  };

  const handleDescribe = () => {
      handleApiCall(() => api.postGeminiDescription(image.src), (res) => setGeminiDescription(res.description));
  };

  const handleAsk = () => {
      if (!geminiQuestion.trim()) return;
      handleApiCall(() => api.postGeminiQuestion(image.src, geminiQuestion), (res) => setGeminiResponse(res.response));
  };

  const handleYoloEntities = () => {
      handleApiCall(() => api.postYoloEntity(image.src));
      // TODO: Display YOLO entity results (e.g., draw boxes)
  }
  const handleYoloPosture = () => {
      handleApiCall(() => api.postYoloPosture(image.src));
      // TODO: Display YOLO posture results (e.g., draw keypoints)
  }
  const handleArcFace = () => {
      handleApiCall(() => api.postArcfaceDetect(image.src));
      // TODO: Display ArcFace results & implement clustering
  }

  return (
    <div className="image-tile" style={tileStyle}>
      <img src={image.src} alt="Captured or Uploaded" style={imageStyle} />
      <div style={controlsStyle}>
        {/* --- Analysis Buttons --- */}
        <button onClick={handleYoloEntities} disabled={isLoading}>YOLO Entity</button>
        <button onClick={handleYoloPosture} disabled={isLoading}>YOLO Pose</button>
        <button onClick={handleArcFace} disabled={isLoading}>ArcFace</button>
        <button onClick={handleDescribe} disabled={isLoading}>Describe (Gemini)</button>

        {/* --- Gemini Ask --- */}
        <div>
            <input
                type="text"
                value={geminiQuestion}
                onChange={(e) => setGeminiQuestion(e.target.value)}
                placeholder="Ask Gemini about image..."
                disabled={isLoading}
                style={{ width: 'calc(100% - 60px)', marginRight: '5px' }}
            />
            <button onClick={handleAsk} disabled={isLoading || !geminiQuestion.trim()}>Ask</button>
        </div>

        {/* --- Results Display --- */}
        {isLoading && <p style={{ color: 'blue'}}>Loading...</p>}
        {error && <p style={{ color: 'red'}}>Error: {error}</p>}
        {geminiDescription && <p><b>Desc:</b> {geminiDescription}</p>}
        {geminiResponse && <p><b>Resp:</b> {geminiResponse}</p>}
        {/* TODO: Add overlay rendering for visual results */}
      </div>
    </div>
  );
};

// Basic styles
const tileStyle: React.CSSProperties = {
    border: '1px solid black',
    width: '320px',
    // height: 'auto', // Adjust height based on content
    position: 'relative',
    display: 'flex',
    flexDirection: 'column'
};
const imageStyle: React.CSSProperties = {
    width: '100%',
    height: '240px', // Fixed height for the image part
    objectFit: 'cover'
};
const controlsStyle: React.CSSProperties = {
    padding: '5px',
    fontSize: '0.8em'
};


export default ImageTile; 