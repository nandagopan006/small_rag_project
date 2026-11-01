/**
 * Centralized API service for communicating with the Django REST Framework backend.
 */

// Read base URL from environment or fallback to Vite dev proxy path
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "/api"
).replace(/\/+$/, "");

/**
 * Ask a question about the college policy handbook.
 *
 * @param {string} question - The student's question text.
 * @param {number} [topK=5] - Number of top chunks to retrieve.
 * @returns {Promise<{answer: string, sources: Array, retrievedChunkCount: number, includedChunkCount: number, excludedChunkCount: number}>}
 */
export async function askPolicyQuestion(question, topK = 5) {
  const cleanQuestion = question ? question.trim() : "";
  if (!cleanQuestion) {
    throw new Error("Question cannot be empty.");
  }

  const endpoint = `${API_BASE_URL}/chat/ask/`;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        question: cleanQuestion,
        top_k: topK,
      }),
    });

    // Attempt to parse JSON response safely
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      // Handle known HTTP error codes
      if (response.status === 400) {
        const message =
          (data && (data.detail || data.error || (data.question && data.question[0]))) ||
          "Please check your question and try again.";
        throw new Error(message);
      }

      if (response.status === 503) {
        throw new Error(
          (data && (data.detail || data.error)) ||
            "The AI assistant is momentarily busy. Please try again shortly."
        );
      }

      if (response.status === 500) {
        throw new Error(
          "An unexpected server error occurred while retrieving policy answers. Please try again later."
        );
      }

      throw new Error(
        (data && (data.detail || data.error)) ||
          `Server returned status ${response.status}. Please try again.`
      );
    }

    // Return structured, normalized payload matching backend response
    return {
      answer: data?.answer || "No answer was returned by the system.",
      sources: Array.isArray(data?.sources) ? data.sources : [],
      retrievedChunkCount: data?.retrieved_chunk_count ?? 0,
      includedChunkCount: data?.included_chunk_count ?? 0,
      excludedChunkCount: data?.excluded_chunk_count ?? 0,
    };
  } catch (error) {
    // Network failures (e.g. backend server not running or CORS blocked)
    if (error.name === "TypeError" && error.message.includes("fetch")) {
      throw new Error(
        "Unable to connect to the backend server. Please ensure the Django backend is running on http://127.0.0.1:8000."
      );
    }
    throw error;
  }
}

/**
 * Fetch available policy documents metadata from the backend.
 * Non-blocking helper used to display real handbook stats.
 *
 * @returns {Promise<Array>} List of policy document objects.
 */
export async function fetchDocuments() {
  const endpoint = `${API_BASE_URL}/documents/`;

  try {
    const response = await fetch(endpoint, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      return [];
    }

    const data = await response.json();
    return Array.isArray(data) ? data : [];
  } catch {
    // Graceful fallback if endpoint is not reachable
    return [];
  }
}
