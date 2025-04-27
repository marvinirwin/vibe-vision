import React, { useState, useRef, useEffect } from 'react';
import OverlayCanvas from './OverlayCanvas';
import { Cluster, AugmentedDetectedFace } from './App'; // Import necessary types
import * as api from '../services/api'; // Import API functions

interface StaticImageTileProps {
  imageId: string; // Need the ID to update the image URL
  imageUrl: string;
  analysisResults?: any;
  altText?: string;
  clusters?: Cluster[];
  allDetectedFaces?: Record<string, AugmentedDetectedFace>;
  onImageUpdate: (imageId: string, newImageUrl: string) => void; // Callback to App
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

const StaticImageTile: React.FC<StaticImageTileProps> = ({ imageId, imageUrl, analysisResults, altText = "Uploaded image", clusters, allDetectedFaces, onImageUpdate }) => {
  const imgRef = React.useRef<HTMLImageElement>(null);
  const mediaContainerRef = React.useRef<HTMLDivElement>(null); // Ref for the container div
  // State to hold the calculated dimensions of the contained image
  const [renderedDimensions, setRenderedDimensions] = React.useState<RenderedDimensions | null>(null);

  // --- Per-Tile Gemini State ---
  const [geminiPrompt, setGeminiPrompt] = useState<string>("");
  const [geminiResponseText, setGeminiResponseText] = useState<string | null>(null);
  const [geminiResponseImage, setGeminiResponseImage] = useState<string | null>(null);
  const [isGeminiLoading, setIsGeminiLoading] = useState<boolean>(false);
  const [geminiError, setGeminiError] = useState<string | null>(null);
  // Flag to track if initial description was requested
  const initialDescriptionRequested = useRef<boolean>(false);

  // Effect to calculate rendered image dimensions
  useEffect(() => {
    const img = imgRef.current;
    const container = mediaContainerRef.current;
    if (!img || !container) return;

    let observer: ResizeObserver | null = null;

    const calculateDimensions = () => {
      if (!img.complete || !container) return;

      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      const originalWidth = img.naturalWidth;
      const originalHeight = img.naturalHeight;

      if (originalWidth === 0 || originalHeight === 0 || containerWidth === 0 || containerHeight === 0) {
        setRenderedDimensions(null); // Not ready
        return;
      }

      const containerRatio = containerWidth / containerHeight;
      const imageRatio = originalWidth / originalHeight;

      let renderedWidth = 0;
      let renderedHeight = 0;
      let offsetX = 0;
      let offsetY = 0;

      if (imageRatio > containerRatio) {
        // Image wider than container (letterboxed)
        renderedWidth = containerWidth;
        renderedHeight = containerWidth / imageRatio;
        offsetX = 0;
        offsetY = (containerHeight - renderedHeight) / 2;
      } else {
        // Image taller than or same ratio as container (pillarboxed)
        renderedHeight = containerHeight;
        renderedWidth = containerHeight * imageRatio;
        offsetY = 0;
        offsetX = (containerWidth - renderedWidth) / 2;
      }

      setRenderedDimensions({
        width: renderedWidth,
        height: renderedHeight,
        offsetX: offsetX,
        offsetY: offsetY,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
      });
    };

    // Run calculation on load and resize
    if (img.complete) {
      calculateDimensions();
    } else {
      img.addEventListener('load', calculateDimensions);
    }

    observer = new ResizeObserver(calculateDimensions);
    observer.observe(container);

    return () => {
      img.removeEventListener('load', calculateDimensions);
      if (observer) {
        observer.disconnect();
      }
      setRenderedDimensions(null); // Reset on cleanup/URL change
    };
  }, [imageUrl]); // Re-run when imageUrl changes

  // --- Effect for Initial Gemini Description ---
  useEffect(() => {
    // Only run once per component instance after the image URL is set
    // and if we haven't requested it yet and there isn't already a response.
    if (imageUrl && !initialDescriptionRequested.current && !geminiResponseText && !geminiResponseImage) {
      console.log(`[${imageId}] Requesting initial description...`);
      initialDescriptionRequested.current = true; // Mark as requested
      setIsGeminiLoading(true);
      setGeminiError(null);

      api.postGeminiDescribe(imageUrl)
        .then(result => {
          setGeminiResponseText(result.description);
        })
        .catch(err => {
          console.error(`[${imageId}] Initial Gemini Describe Error:`, err);
          setGeminiError(err instanceof Error ? err.message : "Failed initial description");
          // Clear text response if error occurred
          setGeminiResponseText(null);
        })
        .finally(() => {
          setIsGeminiLoading(false);
        });
    }
  }, [imageUrl, imageId, geminiResponseText, geminiResponseImage]); // Dependencies

  // --- Gemini Handlers ---
  const handleGeminiTextOnly = async () => {
    if (!geminiPrompt) return;
    console.log(`[${imageId}] Asking Gemini (Text): ${geminiPrompt}`);
    setIsGeminiLoading(true);
    setGeminiError(null);
    setGeminiResponseText(null);
    setGeminiResponseImage(null);
    try {
      // We use the 'ask' endpoint which expects text only response
      const result = await api.postGeminiAsk(imageUrl, geminiPrompt);
      setGeminiResponseText(result.response);
    } catch (err) {
      console.error(`[${imageId}] Gemini Text Ask Error:`, err);
      setGeminiError(err instanceof Error ? err.message : "Failed text request");
    } finally {
      setIsGeminiLoading(false);
    }
  };

  const handleGeminiWithImage = async () => {
    if (!geminiPrompt) return;
    console.log(`[${imageId}] Asking Gemini (Text+Image): ${geminiPrompt}`);
    setIsGeminiLoading(true);
    setGeminiError(null);
    setGeminiResponseText(null);
    setGeminiResponseImage(null);
    try {
      // Use the new 'generate' endpoint
      const result = await api.postGeminiGenerate(imageUrl, geminiPrompt);
      if (result.response_image_b64) {
        console.log(`[${imageId}] Gemini returned image.`);
        setGeminiResponseImage(result.response_image_b64);
        // Update the image in App state
        onImageUpdate(imageId, result.response_image_b64);
      } else if (result.response_text) {
        console.log(`[${imageId}] Gemini returned text.`);
        setGeminiResponseText(result.response_text);
      } else {
           setGeminiError("Gemini returned no response.");
      }
    } catch (err) {
      console.error(`[${imageId}] Gemini Generate Error:`, err);
      setGeminiError(err instanceof Error ? err.message : "Failed image request");
    } finally {
      setIsGeminiLoading(false);
    }
  };

  // Determine which image source to display
  const displayImageUrl = geminiResponseImage || imageUrl;

  return (
    <div className="tile static-image-tile">
      {/* Container for image and overlay */}
      <div ref={mediaContainerRef} className="media-container">
        <img
          ref={imgRef}
          src={displayImageUrl}
          alt={altText}
          style={{ objectFit: 'contain' }}
        />
        {/* Render canvas only when dimensions are calculated */} 
        {analysisResults && renderedDimensions && !geminiResponseImage && (
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

      {/* --- Per-Tile Gemini Interaction --- */}
      <div className="tile-gemini-controls">
        <textarea
          placeholder="Ask Gemini..."
          value={geminiPrompt}
          onChange={(e) => setGeminiPrompt(e.target.value)}
          rows={2}
          disabled={isGeminiLoading}
        />
        <div className="tile-gemini-buttons">
          <button onClick={handleGeminiTextOnly} disabled={isGeminiLoading || !geminiPrompt} title="Send prompt, expect text response">
            {isGeminiLoading ? "..." : "Ask (Text)"}
          </button>
          <button onClick={handleGeminiWithImage} disabled={isGeminiLoading || !geminiPrompt} title="Send prompt + image, expect text or image response">
            {isGeminiLoading ? "..." : "Ask (Img)"}
          </button>
        </div>
        {isGeminiLoading && !geminiResponseText && !geminiResponseImage && <span className="tile-gemini-status">Getting description...</span>}
        {isGeminiLoading && (geminiResponseText || geminiResponseImage) && <span className="tile-gemini-status">Processing...</span>}
        {geminiError && <span className="tile-gemini-status error">Error: {geminiError}</span>}
        {geminiResponseText && <p className="tile-gemini-response">{geminiResponseText}</p>}
      </div>

      {/* Removed central interaction buttons */}
      {/* <div className="tile-actions"> ... </div> */}
    </div>
  );
};

export default StaticImageTile; 