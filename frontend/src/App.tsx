import React, { useState, useCallback, useRef, useEffect } from 'react';
import WebcamTile, { WebcamTileRef } from './components/WebcamTile';
import StaticImageTile from './components/StaticImageTile';
import * as api from './services/api'; // Import API functions
import { cosineSimilarity } from './utils/similarity'; // Import similarity util

// import './App.css'; // Removed as the file doesn't exist yet

// --- Constants ---
const FACE_CLUSTER_THRESHOLD = 0.45; // Increased threshold further

// --- Type Definitions ---

// Extend API's DetectedFace to include local app data
interface AugmentedDetectedFace extends api.DetectedFace {
    faceId: string; // Unique ID for this detected face instance
    imageId: string; // ID of the image this face came from (StaticImageData id or 'webcam')
    imageUrl: string; // URL of the image this face came from (for potential display)
    clusterId?: string; // ID of the cluster this face belongs to
}

// Structure for storing face cluster information
interface Cluster {
    id: string; // Unique cluster ID (e.g., cluster-timestamp)
    name: string; // User-assigned name (or default)
    faceIds: string[]; // List of faceIds belonging to this cluster
    representativeEmbedding: number[]; // Embedding of the first face added
}

// Modified StaticImageData to hold face IDs instead of full objects
interface StaticImageData {
    id: string;
    url: string;
    analysisResults?: AnalysisResultsBundle; // Contains potentially augmented face data initially
    faceIds?: string[]; // Store only the IDs of faces found in this image
    isLoading: boolean;
    error?: string;
}

// Analysis bundle - Face data here might include faceId after processing
interface AnalysisResultsBundle {
    detectedEntities?: api.YoloEntity[];
    detectedPosture?: api.YoloPostureDetectionResponse['detectedPosture'];
    detectedFaces?: (api.DetectedFace & { faceId?: string })[]; // Allow faceId here
}

function App() {
    // Remove Gemini State
    // const [geminiTargetImageUrl, setGeminiTargetImageUrl] = useState<string | null>(null);
    // const [geminiPrompt, setGeminiPrompt] = useState<string>("");
    // const [geminiResponse, setGeminiResponse] = useState<string | null>(null);
    // const [isGeminiLoading, setIsGeminiLoading] = useState<boolean>(false);
    // const [geminiError, setGeminiError] = useState<string | null>(null);
    // const [geminiMode, setGeminiMode] = useState<'describe' | 'ask' | null>(null);

    // Other state remains
    const [webcamAnalysisResults, setWebcamAnalysisResults] = useState<AnalysisResultsBundle | null>(null);
    const [staticImages, setStaticImages] = useState<StaticImageData[]>([]);
    const [isProcessingWebcam, setIsProcessingWebcam] = useState<boolean>(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const webcamTileRef = useRef<WebcamTileRef>(null);
    const [clusters, setClusters] = useState<Cluster[]>([]);
    const [allDetectedFaces, setAllDetectedFaces] = useState<Record<string, AugmentedDetectedFace>>({});
    const needsClusteringUpdate = useRef<boolean>(false);

    // --- Webcam Frame Handling ---
    const handleWebcamFrame = useCallback(async (imageDataUrl: string) => {
        if (isProcessingWebcam) return;
        setIsProcessingWebcam(true);
        let resultsBundle: AnalysisResultsBundle = {}; // Initialize empty
        try {
            const [entitiesResult, postureResult, facesResult] = await Promise.all([
                api.postYoloDetectEntities(imageDataUrl).catch(e => { console.error("W: YOLO Entity Error:", e); return null; }),
                api.postYoloDetectPosture(imageDataUrl).catch(e => { console.error("W: YOLO Posture Error:", e); return null; }),
                api.postArcfaceDetect(imageDataUrl).catch(e => { console.error("W: ArcFace Error:", e); return null; })
            ]);

            resultsBundle = {
                detectedEntities: entitiesResult?.detectedEntities,
                detectedPosture: postureResult?.detectedPosture,
                // Process faces immediately if found
                detectedFaces: facesResult?.detectedFaces.map(face => ({
                    ...face,
                    faceId: `face-webcam-${Date.now()}-${Math.random().toString(16).slice(2)}` // Assign temporary ID
                }))
            };

            setWebcamAnalysisResults(resultsBundle);

            // --- Trigger clustering for webcam faces ---
            if (resultsBundle.detectedFaces && resultsBundle.detectedFaces.length > 0) {
                const newWebcamFaces: Record<string, AugmentedDetectedFace> = {};
                resultsBundle.detectedFaces.forEach(face => {
                    if (face.faceId && face.embedding) { // Ensure faceId and embedding exist
                        newWebcamFaces[face.faceId] = {
                            ...face,
                            embedding: face.embedding, // Ensure embedding is present
                            imageId: 'webcam', // Special ID for webcam
                            imageUrl: imageDataUrl, // Use current frame data URL
                        };
                    }
                });
                // Update all faces state - Note: this overwrites previous webcam faces
                // If persistence is needed, merge instead of overwriting.
                setAllDetectedFaces(prev => ({ ...prev, ...newWebcamFaces }));
                needsClusteringUpdate.current = true;
            }

        } catch (error) {
            console.error("Error during webcam analysis:", error);
            setWebcamAnalysisResults(null);
        } finally {
            setIsProcessingWebcam(false);
        }
    }, [isProcessingWebcam]);

    // --- Static Image Handling ---
    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (loadEvent) => {
            const imageDataUrl = loadEvent.target?.result as string;
            if (imageDataUrl) {
                addStaticImage(imageDataUrl); // Use helper function
            }
        };
        reader.onerror = (error) => console.error("Error reading file:", error);
        reader.readAsDataURL(file);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const analyzeStaticImage = async (imageId: string, imageDataUrl: string) => {
        console.log(`Analyzing static image: ${imageId}`);
        let resultsBundle: AnalysisResultsBundle = {};
        let currentImageFaceIds: string[] = [];
        let newStaticFaces: Record<string, AugmentedDetectedFace> = {};

        try {
            const [entitiesResult, postureResult, facesResult] = await Promise.all([
                api.postYoloDetectEntities(imageDataUrl).catch(e => { console.error(`S: YOLO Entity Error (${imageId}):`, e); return null; }),
                api.postYoloDetectPosture(imageDataUrl).catch(e => { console.error(`S: YOLO Posture Error (${imageId}):`, e); return null; }),
                api.postArcfaceDetect(imageDataUrl).catch(e => { console.error(`S: ArcFace Error (${imageId}):`, e); return null; })
            ]);

            resultsBundle = {
                detectedEntities: entitiesResult?.detectedEntities,
                detectedPosture: postureResult?.detectedPosture,
                detectedFaces: facesResult?.detectedFaces.map(face => ({
                    ...face,
                    faceId: `face-${imageId}-${Date.now()}-${Math.random().toString(16).slice(2)}`
                }))
            };

            // Prepare faces for clustering and store IDs
            if (resultsBundle.detectedFaces) {
                resultsBundle.detectedFaces.forEach(face => {
                    if (face.faceId && face.embedding) {
                        currentImageFaceIds.push(face.faceId);
                        newStaticFaces[face.faceId] = {
                            ...face,
                            embedding: face.embedding,
                            imageId: imageId,
                            imageUrl: imageDataUrl
                        };
                    }
                });
            }

            // Update the specific image's data in state
            setStaticImages(prev => prev.map(img =>
                img.id === imageId
                    ? { ...img, analysisResults: resultsBundle, faceIds: currentImageFaceIds, isLoading: false, error: undefined }
                    : img
            ));

            // Add new faces to the global pool and mark for clustering
            if (Object.keys(newStaticFaces).length > 0) {
                setAllDetectedFaces(prev => ({ ...prev, ...newStaticFaces }));
                needsClusteringUpdate.current = true;
            }

        } catch (error) {
            console.error(`Error analyzing static image ${imageId}:`, error);
            setStaticImages(prev => prev.map(img =>
                img.id === imageId
                    ? { ...img, isLoading: false, error: error instanceof Error ? error.message : "Analysis failed" }
                    : img
            ));
        }
    };

    // --- Clustering Logic ---
    const updateClusters = useCallback(() => {
        console.log("Updating clusters... Threshold:", FACE_CLUSTER_THRESHOLD);
        let updatedClusters = [...clusters];
        let updatedFaces = { ...allDetectedFaces };
        let changed = false;

        const facesToCluster = Object.values(updatedFaces).filter(face => !face.clusterId && face.embedding);
        console.log(`Found ${facesToCluster.length} faces to cluster.`);

        facesToCluster.forEach(face => {
            if (!face.embedding) return;

            let bestMatchClusterId: string | null = null;
            let highestSimilarity = -1; // Start below valid range

            updatedClusters.forEach(cluster => {
                if (!cluster.representativeEmbedding || cluster.representativeEmbedding.length === 0) {
                     console.warn(`Cluster ${cluster.id} (${cluster.name}) has invalid representativeEmbedding.`);
                     return;
                }

                // --- MORE DEBUG LOGGING: Check vectors before comparison ---
                const faceEmbedding = face.embedding!;
                const clusterEmbedding = cluster.representativeEmbedding;
                console.log(`   -> Comparing Face ${face.faceId.slice(-6)} Embedding[0..3]: [${faceEmbedding.slice(0,4).map(v => v.toFixed(4)).join(', ')}]`);
                console.log(`   ->   vs Cluster ${cluster.id.slice(-6)} Embedding[0..3]: [${clusterEmbedding.slice(0,4).map(v => v.toFixed(4)).join(', ')}]`);
                // --- END MORE DEBUG LOGGING ---

                const similarity = cosineSimilarity(faceEmbedding, clusterEmbedding);
                console.log(` - Comparing Face ${face.faceId.slice(-6)} (img: ${face.imageId}) to Cluster ${cluster.id.slice(-6)} (${cluster.name}): Similarity = ${similarity.toFixed(4)}`);

                if (isNaN(similarity)) {
                    console.warn(`   -> NaN similarity score detected! Skipping comparison.`);
                    return;
                }

                if (similarity > FACE_CLUSTER_THRESHOLD && similarity > highestSimilarity) {
                    highestSimilarity = similarity;
                    bestMatchClusterId = cluster.id;
                    console.log(`   -> Potential best match found: Cluster ${cluster.id.slice(-6)} (${cluster.name}) with similarity ${similarity.toFixed(4)}`);
                }
            });

            if (bestMatchClusterId) {
                const clusterIndex = updatedClusters.findIndex(c => c.id === bestMatchClusterId);
                if (clusterIndex !== -1) {
                    updatedClusters[clusterIndex].faceIds.push(face.faceId);
                    updatedFaces[face.faceId].clusterId = bestMatchClusterId;
                    console.log(` -> Assigned Face ${face.faceId.slice(-6)} to existing Cluster ${bestMatchClusterId.slice(-6)}`);
                    changed = true;
                }
            } else {
                const newClusterId = `cluster-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                const newCluster: Cluster = {
                    id: newClusterId,
                    name: `Cluster ${updatedClusters.length + 1}`,
                    faceIds: [face.faceId],
                    representativeEmbedding: [...face.embedding!], 
                };
                updatedClusters.push(newCluster);
                updatedFaces[face.faceId].clusterId = newClusterId;
                console.log(` -> Created new Cluster ${newClusterId.slice(-6)} for Face ${face.faceId.slice(-6)}`);
                changed = true;
            }
        });

        if (changed) {
            console.log("Cluster update complete. New cluster count:", updatedClusters.length);
            setClusters(updatedClusters);
            setAllDetectedFaces(updatedFaces);
        }
        needsClusteringUpdate.current = false;
    }, [allDetectedFaces, clusters]);

    // Effect to run clustering when needed
  useEffect(() => {
        if (needsClusteringUpdate.current) {
            updateClusters();
        }
    }, [allDetectedFaces, updateClusters]); // Run when allDetectedFaces changes

    // --- Helper function to add a static image (from upload or capture) ---
    const addStaticImage = (imageDataUrl: string) => {
        if (!imageDataUrl) return;
        const newImageId = `static-${Date.now()}-${Math.random()}`;
        const newImageData: StaticImageData = {
            id: newImageId,
            url: imageDataUrl,
            isLoading: true // Start loading
        };
        setStaticImages(prev => [...prev, newImageData]);
        analyzeStaticImage(newImageId, imageDataUrl);
    };

    // --- Capture Photo Handler ---
    const handleCapturePhoto = () => {
        const capturedDataUrl = webcamTileRef.current?.captureCurrentImage();
        if (capturedDataUrl) {
            console.log("Photo captured from webcam.");
            addStaticImage(capturedDataUrl); // Add like an uploaded image
        } else {
            console.error("Failed to capture photo from webcam.");
            // Optionally show user feedback
        }
    };

    // --- NEW: Function to update a static image URL ---
    const updateStaticImageUrl = (imageId: string, newImageUrl: string) => {
        setStaticImages(prev =>
            prev.map(img =>
                img.id === imageId ? { ...img, url: newImageUrl, analysisResults: undefined, faceIds: [], isLoading: true, error: undefined } : img
            )
        );
        // Trigger re-analysis for the new image content
        analyzeStaticImage(imageId, newImageUrl);
    };

    // --- Cluster Naming Handler ---
    const handleClusterNameChange = (clusterId: string, newName: string) => {
        setClusters(prevClusters =>
            prevClusters.map(cluster =>
                cluster.id === clusterId ? { ...cluster, name: newName } : cluster
            )
        );
        // Optional: Update name in allDetectedFaces as well if needed elsewhere
        // setAllDetectedFaces(prevFaces => ...);
    };

    // Trigger file input click
    const triggerUpload = () => {
        fileInputRef.current?.click();
    };

  return (
    <div className="App">
      <h1>Vibe Vision Workshop</h1>

            {/* --- Controls --- */}
            <div className="controls">
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    ref={fileInputRef}
                    style={{ display: 'none' }} // Hide default input
                />
                <button onClick={triggerUpload}>Upload Image</button>
                <button onClick={handleCapturePhoto}>Capture Photo</button>
            </div>

            {/* --- Tile Grid --- */}
            <div className="tile-grid">
                <WebcamTile
                    ref={webcamTileRef}
                    onFrame={handleWebcamFrame}
                    analysisResults={webcamAnalysisResults}
                    clusters={clusters}
                    allDetectedFaces={allDetectedFaces}
                />
                {staticImages.map((imgData) => (
                    <StaticImageTile
                        key={imgData.id}
                        imageId={imgData.id}
                        imageUrl={imgData.url}
                        analysisResults={imgData.analysisResults}
                        altText={`Uploaded image ${imgData.id}`}
                        clusters={clusters}
                        allDetectedFaces={allDetectedFaces}
                        onImageUpdate={updateStaticImageUrl}
                    />
                ))}
            </div>

            {/* Face Clustering Section */}
            {clusters.length > 0 && (
                <div className="clusters-section">
                    <h2>Face Clusters</h2>
                    {clusters.map((cluster) => (
                        <div key={cluster.id} className="cluster-item">
                            <div className="cluster-header">
                                <input
                                    type="text"
                                    value={cluster.name}
                                    onChange={(e) => handleClusterNameChange(cluster.id, e.target.value)}
                                    placeholder="Enter name..."
                                    title={`Cluster ID: ${cluster.id}`}
                                />
                                <span>({cluster.faceIds.length} faces)</span>
                            </div>
                            {/* List the faces within the cluster */} 
                            <ul className="cluster-faces-list">
                                {cluster.faceIds.map((faceId) => {
                                    const faceData = allDetectedFaces[faceId];
                                    // Display face ID suffix and image ID (webcam or static ID suffix)
                                    const faceIdSuffix = faceId.split('-').pop()?.substring(0, 6) || 'N/A';
                                    const imageIdSuffix = faceData?.imageId === 'webcam' ? 'webcam' : (faceData?.imageId.split('-').pop()?.substring(0, 6) || 'unknown');
                                    return (
                                        <li key={faceId} className="cluster-face-item" title={`Face ID: ${faceId}, Image ID: ${faceData?.imageId}`}>
                                            Face: {faceIdSuffix} (from Img: {imageIdSuffix})
                                        </li>
                                    );
                                })}
                            </ul>
                        </div>
                    ))}
                </div>
            )}

            {/* TODO: Add section for face clustering/naming */}
    </div>
  );
}

export default App; 