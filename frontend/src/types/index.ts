// --- Common --- 
export interface BoundingBox {
    box: [number, number, number, number]; // [x1, y1, x2, y2]
    label: string;
    confidence?: number;
}

export interface Keypoint {
    x: number;
    y: number;
    score?: number;
    name?: string;
}

// --- API Response Payloads ---

// YOLO
export interface YoloEntityDetectionResponse {
    detectedEntities: BoundingBox[];
}

export interface YoloPostureDetectionResponse {
    detectedPosture: {
        keypoints: Keypoint[];
        connections?: [number, number][]; // Optional: indices of connected keypoints
    };
}

// ArcFace
export interface ArcFaceDetectionResponse {
    detectedFaces: {
        box: [number, number, number, number]; // [x1, y1, x2, y2]
        landmarks: [number, number][]; // Array of [x, y] for landmarks
        embedding?: number[]; // High-dimensional embedding vector (optional for frontend)
    }[];
}

// Gemini
export interface GeminiDescriptionResponse {
    description: string;
}

export interface GeminiQuestionResponse {
    response: string;
}

// --- Frontend State ---
export interface ImageTileData {
    id: string;
    src: string;
    analysisResults?: {
        yoloEntities?: YoloEntityDetectionResponse;
        yoloPosture?: YoloPostureDetectionResponse;
        arcface?: ArcFaceDetectionResponse;
        geminiDescription?: GeminiDescriptionResponse;
        geminiQuestion?: GeminiQuestionResponse;
    };
    isLoading?: {
        yoloEntities?: boolean;
        yoloPosture?: boolean;
        arcface?: boolean;
        geminiDescription?: boolean;
        geminiQuestion?: boolean;
    };
} 