const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:21115"

export interface GenerateRequest {
  prompt: string
  negative_prompt?: string
  height?: number
  width?: number
  seed?: number
  num_inference_steps?: number
  guidance_scale?: number
}

export interface GenerateResponse {
  image_base64: string
  seed: number
  generation_time_ms: number
  width: number
  height: number
  image_id?: string
}

export async function generateImage(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const response = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: request.prompt,
      negative_prompt: request.negative_prompt || "",
      height: request.height || 1024,
      width: request.width || 1024,
      seed: request.seed === undefined ? -1 : request.seed,
      num_inference_steps: request.num_inference_steps || 9,
      guidance_scale: request.guidance_scale || 0.0,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }))
    throw new Error(error.detail || `HTTP error! status: ${response.status}`)
  }

  return response.json()
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}

