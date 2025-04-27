import {
    YoloEntityDetectionResponse,
    YoloPostureDetectionResponse,
    ArcFaceDetectionResponse,
    GeminiDescriptionResponse,
    GeminiQuestionResponse
} from '../types';

const API_BASE_URL = 'http://localhost:3001/api'; // Backend API base URL

// Defines the structure for image analysis API calls
interface ImageDataPayload {
    imageData: string; // Base64 encoded image data
}

// Generic function to handle API requests
async function fetchApi<T>(endpoint: string, method: string = 'GET', body?: any): Promise<T> {
    const options: RequestInit = {
        method,
        headers: {},
    };

    if (body) {
        if (method === 'POST' || method === 'PUT') {
            options.headers = { ...options.headers, 'Content-Type': 'application/json' };
            options.body = JSON.stringify(body);
        }
        // For GET or other methods, data might be sent as query params (not handled here)
    }

    try {
        const response = await fetch(endpoint, options);

        if (!response.ok) {
            // Attempt to read error message from backend response
            let errorMessage = `HTTP error! status: ${response.status}`;
            try {
                const errorBody = await response.json();
                errorMessage = errorBody.message || errorMessage;
            } catch (e) {
                // Ignore if response is not JSON or empty
            }
            throw new Error(errorMessage);
        }
        return await response.json() as T;
    } catch (error) {
        console.error(`API call to ${endpoint} failed:`, error);
        // Re-throw the error so the calling component can handle it
        throw error;
    }
}

// Health check (adjust endpoint if needed, it's outside /api)
export const healthCheck = async (): Promise<string> => {
    try {
        const response = await fetch('/'); // Assuming the root path '/' of the backend returns the health message
        if (!response.ok) {
            throw new Error(`Backend not responding. Status: ${response.status}`);
        }
        const text = await response.text();
        console.log("Backend health check:", text);
        return text || "Backend connected.";
    } catch (error) {
        console.error("Health check failed:", error);
        return "Backend connection failed.";
    }
};

// --- Implemented API Functions ---

// Define expected response structure for YOLO entity detection
export interface YoloEntity {
    box: [number, number, number, number]; // [x1, y1, x2, y2]
    label: string;
    confidence: number;
}
export interface YoloEntityDetectionResponse {
    detectedEntities: YoloEntity[];
}

export function postYoloDetectEntities(imageData: string): Promise<YoloEntityDetectionResponse> {
    const payload: ImageDataPayload = { imageData };
    return fetchApi<YoloEntityDetectionResponse>('/api/yolo/detect-entities', 'POST', payload);
}

// Define expected response structure for YOLO posture detection
export interface YoloKeypoint {
    x: number;
    y: number;
    score: number;
    name?: string; // Optional name (e.g., 'nose', 'left_shoulder')
}
export interface YoloPostureDetectionResponse {
    detectedPosture: {
        keypoints: YoloKeypoint[];
        connections: any[]; // Define structure if backend provides connections
    };
}

export function postYoloDetectPosture(imageData: string): Promise<YoloPostureDetectionResponse> {
    const payload: ImageDataPayload = { imageData };
    return fetchApi<YoloPostureDetectionResponse>('/api/yolo/detect-posture', 'POST', payload);
}

// --- ArcFace Endpoint --- //

// Define expected response structure for ArcFace detection
export interface ArcFaceLandmark extends Array<number> {} // e.g., [x, y]

export interface DetectedFace {
    box: [number, number, number, number]; // [x1, y1, x2, y2]
    landmarks: ArcFaceLandmark[];
    embedding?: number[]; // Optional embedding
}
export interface ArcFaceDetectionResponse {
    detectedFaces: DetectedFace[];
}

export function postArcfaceDetect(imageData: string): Promise<ArcFaceDetectionResponse> {
    const payload: ImageDataPayload = { imageData };
    return fetchApi<ArcFaceDetectionResponse>('/api/arcface/detect-faces', 'POST', payload);
}

// --- Gemini Endpoints --- //

// Define expected response structure for Gemini description
export interface GeminiDescriptionResponse {
    description: string;
}

export function postGeminiDescribe(imageData: string): Promise<GeminiDescriptionResponse> {
    const payload: ImageDataPayload = { imageData };
    return fetchApi<GeminiDescriptionResponse>('/api/gemini/describe-image', 'POST', payload);
}

// Define expected response structure for Gemini question
export interface GeminiAskResponse {
    response: string;
}

export interface GeminiAskPayload extends ImageDataPayload {
    prompt: string;
}

export function postGeminiAsk(imageData: string, prompt: string): Promise<GeminiAskResponse> {
    const payload: GeminiAskPayload = { imageData, prompt };
    return fetchApi<GeminiAskResponse>('/api/gemini/ask-about-image', 'POST', payload);
}

// --- Gemini Generation/Modification Endpoint --- //

export interface GeminiGeneratePayload extends ImageDataPayload {
    prompt: string;
}

// Response can contain either text OR an image data URL
export interface GeminiGenerateResponse {
    response_text: string | null;
    response_image_b64: string | null;
}

export function postGeminiGenerate(imageData: string, prompt: string): Promise<GeminiGenerateResponse> {
    const payload: GeminiGeneratePayload = { imageData, prompt };
    return fetchApi<GeminiGenerateResponse>('/api/gemini/generate-with-image', 'POST', payload);
}

// TODO: Add Gemini image modification endpoint when backend supports it

// TODO: Implement functions to call backend endpoints:
// - postYoloEntity(imageData: string): Promise<any>
// - postYoloPosture(imageData: string): Promise<any>
// - postArcfaceDetect(imageData: string): Promise<any>
// - postGeminiText(imageData: string): Promise<string>
// - postGeminiImage(imageData: string, prompt: string): Promise<string> // Assuming image response is base64 string 