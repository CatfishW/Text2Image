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
}

export const RESOLUTION_PRESETS: ResolutionPreset[] = [
  { label: "Square (1024×1024)", width: 1024, height: 1024 },
  { label: "Portrait (768×1152)", width: 768, height: 1152 },
  { label: "Landscape (1152×768)", width: 1152, height: 768 },
  { label: "Wide (1344×768)", width: 1344, height: 768 },
  { label: "Tall (768×1344)", width: 768, height: 1344 },
]

