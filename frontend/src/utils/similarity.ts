/**
 * Calculates the dot product of two vectors.
 * @param vecA First vector (array of numbers).
 * @param vecB Second vector (array of numbers).
 * @returns The dot product, or 0 if vectors are invalid or lengths differ.
 */
function dotProduct(vecA: number[], vecB: number[]): number {
    if (!vecA || !vecB || vecA.length !== vecB.length) {
        return 0;
    }
    let product = 0;
    for (let i = 0; i < vecA.length; i++) {
        const term = vecA[i] * vecB[i];
        product += term;
    }
    return product;
}

/**
 * Calculates the magnitude (Euclidean norm) of a vector.
 * @param vec Vector (array of numbers).
 * @returns The magnitude, or 0 if vector is invalid.
 */
function magnitude(vec: number[]): number {
    if (!vec) {
        return 0;
    }
    let sumOfSquares = 0;
    for (let i = 0; i < vec.length; i++) {
        const term = vec[i] * vec[i];
        sumOfSquares += term;
    }
    const mag = Math.sqrt(sumOfSquares);
    return mag;
}

/**
 * Calculates the cosine similarity between two vectors.
 * Ranges from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.
 * Higher values mean more similarity for typical embeddings like ArcFace.
 * @param vecA First vector (array of numbers).
 * @param vecB Second vector (array of numbers).
 * @returns The cosine similarity (between -1 and 1), or 0 if vectors are invalid.
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
    const magA = magnitude(vecA);
    const magB = magnitude(vecB);

    if (magA === 0 || magB === 0) {
        if (magA === 0 && magB === 0) return 1;
        console.warn("Cannot compute cosine similarity with zero vector(s).");
        return 0;
    }

    const dot = dotProduct(vecA, vecB);

    const similarity = dot / (magA * magB);

    return similarity;
} 