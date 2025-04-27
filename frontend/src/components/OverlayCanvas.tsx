import React, { useRef, useEffect } from 'react';
import { Cluster, AugmentedDetectedFace } from './App'; // Import types from App

interface OverlayCanvasProps {
  results: any; // Combined results (YOLO, ArcFace etc., faces include faceId)
  clusters?: Cluster[]; // Pass down cluster data from App
  allDetectedFaces?: Record<string, AugmentedDetectedFace>; // Pass down all faces data
  // Dimensions/position of the actual rendered image/video content
  renderedContentWidth: number;
  renderedContentHeight: number;
  renderedContentOffsetX: number;
  renderedContentOffsetY: number;
  // Original media dimensions (for scaling calculations)
  originalWidth: number;
  originalHeight: number;
}

const OverlayCanvas: React.FC<OverlayCanvasProps> = ({ results, clusters, allDetectedFaces, renderedContentWidth, renderedContentHeight, renderedContentOffsetX, renderedContentOffsetY, originalWidth, originalHeight }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !canvas || !results || renderedContentWidth <= 0 || renderedContentHeight <= 0 || originalWidth <= 0 || originalHeight <= 0) {
      // Clear canvas if dimensions become invalid
      if (canvas) {
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    // --- Set Canvas Size and Position --- 
    canvas.width = renderedContentWidth;
    canvas.height = renderedContentHeight;
    // Position the canvas precisely over the rendered content
    canvas.style.position = 'absolute';
    canvas.style.left = `${renderedContentOffsetX}px`;
    canvas.style.top = `${renderedContentOffsetY}px`;
    canvas.style.width = `${renderedContentWidth}px`;
    canvas.style.height = `${renderedContentHeight}px`;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // --- Calculate Scaling Factor --- 
    // Scale based on the ratio between rendered content and original content size
    const scale = renderedContentWidth / originalWidth;
    // Note: Since we've sized the canvas to the rendered content,
    // offsetX and offsetY for drawing *within* the canvas are now 0.

    // --- Draw YOLO Entity Detections ---
    if (results.detectedEntities) {
      ctx.strokeStyle = '#ff0000'; // Red for entities
      ctx.lineWidth = 2;
      ctx.font = '12px Arial';
      ctx.fillStyle = '#ff0000';

      results.detectedEntities.forEach((entity: any) => {
        const [x1, y1, x2, y2] = entity.box;
        // Apply scale (offsets within canvas are 0)
        const drawX = x1 * scale;
        const drawY = y1 * scale;
        const drawWidth = (x2 - x1) * scale;
        const drawHeight = (y2 - y1) * scale;

        // Draw bounding box
        ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);

        // Draw label and confidence
        const label = `${entity.label} (${entity.confidence.toFixed(2)})`;
        // Position label relative to the drawn box
        ctx.fillText(label, drawX, drawY > 10 ? drawY - 5 : drawY + 10);
      });
    }

    // --- Draw YOLO Posture Keypoints ---
    if (results.detectedPosture?.keypoints) {
        ctx.fillStyle = '#00ff00'; // Green for keypoints
        ctx.font = '10px Arial';   // Font for labels
        const keypointRadius = 3;

        results.detectedPosture.keypoints.forEach((kp: any) => {
            if (kp.score > 0.3) { // Confidence threshold
                // Apply scale (offsets within canvas are 0)
                const drawX = kp.x * scale;
                const drawY = kp.y * scale;

                // Draw the keypoint circle
                ctx.beginPath();
                ctx.arc(drawX, drawY, keypointRadius, 0, 2 * Math.PI);
                ctx.fill();

                // Draw the keypoint name label
                if (kp.name) {
                    ctx.fillStyle = '#00cc00'; // Slightly darker green for text
                    // Position the text slightly offset from the drawn point
                    ctx.fillText(kp.name, drawX + keypointRadius + 2, drawY + keypointRadius + 2);
                }
            }
        });
        // Reset fill style just in case
        ctx.fillStyle = '#0000ff'; // Reset to blue for faces section
        // TODO: Add drawing connections if provided by backend
    }

    // --- Draw ArcFace Face Detections (with Cluster Info) ---
    if (results.detectedFaces) {
        ctx.strokeStyle = '#0000ff'; // Blue for faces
        ctx.fillStyle = '#0000ff'; // Blue for landmarks
        ctx.lineWidth = 2;
        const landmarkRadius = 2;

        results.detectedFaces.forEach((face: any) => { // face here includes faceId
            const [x1, y1, x2, y2] = face.box;
            // Apply scale (offsets within canvas are 0)
            const drawX1 = x1 * scale;
            const drawY1 = y1 * scale;
            const drawWidth = (x2 - x1) * scale;
            const drawHeight = (y2 - y1) * scale;

            // Draw bounding box
            ctx.strokeRect(drawX1, drawY1, drawWidth, drawHeight);

            // Find cluster info for this face
            let clusterName: string | null = null;
            if (face.faceId && clusters && allDetectedFaces) {
                const fullFaceData = allDetectedFaces[face.faceId];
                if (fullFaceData?.clusterId) {
                    const cluster = clusters.find(c => c.id === fullFaceData.clusterId);
                    if (cluster) {
                        clusterName = cluster.name;
                    }
                }
            }

            // Draw cluster name above the box if found
            if (clusterName) {
                ctx.fillStyle = '#0000ff'; // Blue text
                ctx.font = 'bold 14px Arial';
                ctx.fillText(clusterName, drawX1, drawY1 > 15 ? drawY1 - 5 : drawY1 + drawHeight + 15);
            }

            // Draw landmarks
            if (face.landmarks) {
                ctx.fillStyle = '#0000ff'; // Reset fill style for landmarks
                face.landmarks.forEach((lm: [number, number]) => {
                    // Apply scale (offsets within canvas are 0)
                    const drawX = lm[0] * scale;
                    const drawY = lm[1] * scale;
                    ctx.beginPath();
                    ctx.arc(drawX, drawY, landmarkRadius, 0, 2 * Math.PI);
                    ctx.fill();
                });
            }
        });
    }

  }, [results, clusters, allDetectedFaces, renderedContentWidth, renderedContentHeight, renderedContentOffsetX, renderedContentOffsetY, originalWidth, originalHeight]); // Redraw when results or dimensions change

  return (
    <canvas
      ref={canvasRef}
      className="overlay-canvas"
    />
  );
};

export default OverlayCanvas; 