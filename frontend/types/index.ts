export interface GeneratedImage {
  id: string
  imageBase64: string
  prompt: string
  negativePrompt: string
  seed: number
  width: number
  height: number
  steps: number
  generationTimeMs: number
  timestamp: number
}

export type ResolutionPreset = {
  label: string
  width: number
  height: number
  hint: string
}

export const RESOLUTION_PRESETS: ResolutionPreset[] = [
  { label: "Square", width: 1024, height: 1024, hint: "Best for Instagram & avatars" },
  { label: "Portrait", width: 768, height: 1152, hint: "Ideal for phone wallpapers & portraits" },
  { label: "Landscape", width: 1152, height: 768, hint: "Great for desktop wallpapers & scenery" },
  { label: "Wide", width: 1344, height: 768, hint: "Cinematic 16:9 look" },
  { label: "Tall", width: 768, height: 1344, hint: "Full vertical for stories" },
]

