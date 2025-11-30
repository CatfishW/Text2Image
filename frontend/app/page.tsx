"use client"

import { useState, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { generateImage, checkHealth } from "@/lib/api"
import { RESOLUTION_PRESETS, type GeneratedImage } from "@/types"
import { toast } from "sonner"
import {
  Loader2,
  Sparkles,
  Copy,
  Download,
  ChevronDown,
  ChevronUp,
  Shuffle,
  Image as ImageIcon,
  AlertCircle,
  Zap,
  Wand2,
  TrendingUp,
  Clock,
  Layers,
  Palette,
  Info,
  Lightbulb,
  Star,
  Video,
} from "lucide-react"

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

const cardVariants = {
  hidden: { opacity: 0, scale: 0.95, y: 20 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

const imageVariants = {
  hidden: { opacity: 0, scale: 0.8, rotate: -5 },
  visible: {
    opacity: 1,
    scale: 1,
    rotate: 0,
    transition: {
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

const galleryItemVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: (i: number) => ({
    opacity: 1,
    scale: 1,
    transition: {
      delay: i * 0.05,
      duration: 0.3,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  }),
  hover: {
    scale: 1.05,
    rotate: 2,
    transition: {
      duration: 0.2,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

type Tab = "text-to-image" | "image-to-video" | "image-to-image"

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("text-to-image")
  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("")
  const [showNegativePrompt, setShowNegativePrompt] = useState(false)
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [seed, setSeed] = useState<number>(-1)
  const [steps, setSteps] = useState(9)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<GeneratedImage | null>(null)
  const [gallery, setGallery] = useState<GeneratedImage[]>([])
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null)
  const [apiHealth, setApiHealth] = useState<boolean | null>(null)
  const [progressValue, setProgressValue] = useState(0)
  const [showTips, setShowTips] = useState(true)

  // Check API health on mount
  useEffect(() => {
    checkHealth().then(setApiHealth)
    const interval = setInterval(() => {
      checkHealth().then(setApiHealth)
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  // Simulate progress during generation
  useEffect(() => {
    if (isGenerating) {
      setProgressValue(0)
      const interval = setInterval(() => {
        setProgressValue((prev) => Math.min(prev + Math.random() * 15, 90))
      }, 500)
      return () => clearInterval(interval)
    } else {
      setProgressValue(100)
      setTimeout(() => setProgressValue(0), 500)
    }
  }, [isGenerating])

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Please enter a prompt")
      return
    }

    if (apiHealth === false) {
      toast.error("API is not available. Please check the backend connection.")
      return
    }

    setIsGenerating(true)
    setProgressValue(0)
    try {
      const startTime = Date.now()
      const response = await generateImage({
        prompt,
        negative_prompt: negativePrompt,
        width,
        height,
        seed: seed >= 0 ? seed : undefined,
        num_inference_steps: steps,
      })

      setProgressValue(100)

      const generatedImage: GeneratedImage = {
        id: response.image_id || `img-${Date.now()}`,
        imageBase64: response.image_base64,
        prompt,
        negativePrompt,
        seed: response.seed,
        width: response.width,
        height: response.height,
        steps,
        generationTimeMs: response.generation_time_ms,
        timestamp: startTime,
      }

      setCurrentImage(generatedImage)
      setGallery((prev) => [generatedImage, ...prev].slice(0, 12))
      toast.success(`✨ Image generated in ${response.generation_time_ms}ms`)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to generate image"
      toast.error(message)
      console.error("Generation error:", error)
    } finally {
      setIsGenerating(false)
    }
  }, [prompt, negativePrompt, width, height, seed, steps, apiHealth])

  const handleRandomSeed = () => {
    const newSeed = Math.floor(Math.random() * 2 ** 32)
    setSeed(newSeed)
    toast.success(`🎲 Random seed: ${newSeed}`)
  }

  const handleCopyPrompt = () => {
    navigator.clipboard.writeText(prompt)
    toast.success("📋 Prompt copied to clipboard")
  }

  const handleCopySeed = () => {
    if (currentImage) {
      navigator.clipboard.writeText(currentImage.seed.toString())
      toast.success("📋 Seed copied to clipboard")
    }
  }

  const handleDownload = (image: GeneratedImage) => {
    const link = document.createElement("a")
    link.href = image.imageBase64
    link.download = `image-${image.seed}-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    toast.success("💾 Image downloaded")
  }

  const handlePresetSelect = (preset: typeof RESOLUTION_PRESETS[0]) => {
    setWidth(preset.width)
    setHeight(preset.height)
    toast.success(`📐 Resolution set to ${preset.label}`)
  }

  const examplePrompts = [
    "A futuristic cityscape at sunset with flying cars",
    "A serene Japanese garden with cherry blossoms",
    "A cyberpunk street with neon lights and rain",
    "A magical forest with glowing mushrooms",
    "An astronaut floating in space",
  ]

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} apiHealth={apiHealth} />

      <motion.div
        className="flex-1 relative overflow-hidden"
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        {/* Rowan University themed background gradient */}
        <div className="fixed inset-0 -z-10">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/3 via-background to-secondary/3" />
          <motion.div
            className="absolute inset-0 opacity-20"
            animate={{
              background: [
                "radial-gradient(circle at 20% 50%, rgba(122, 69, 41, 0.08) 0%, transparent 50%)",
                "radial-gradient(circle at 80% 80%, rgba(204, 153, 0, 0.08) 0%, transparent 50%)",
                "radial-gradient(circle at 40% 20%, rgba(122, 69, 41, 0.06) 0%, transparent 50%)",
                "radial-gradient(circle at 60% 40%, rgba(204, 153, 0, 0.06) 0%, transparent 50%)",
                "radial-gradient(circle at 20% 50%, rgba(122, 69, 41, 0.08) 0%, transparent 50%)",
              ],
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        </div>

        <div className="container mx-auto px-4 py-6 md:py-8 max-w-7xl relative z-10">
          <AnimatePresence mode="wait">
            {activeTab === "text-to-image" && (
              <motion.div
                key="text-to-image"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="space-y-6"
              >
                {/* Stats Cards - Rowan University Style */}
                <motion.div
                  className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4"
                  variants={itemVariants}
                >
                  <motion.div
                    variants={cardVariants}
                    whileHover={{ scale: 1.03, y: -3 }}
                    className="bg-card border-2 border-primary/20 rounded-lg p-4 shadow-md hover:shadow-lg transition-shadow"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-primary/15 rounded-lg border border-primary/20">
                        <TrendingUp className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs md:text-sm text-muted-foreground font-medium">Total Generated</p>
                        <p className="text-xl md:text-2xl font-bold text-primary">{gallery.length}</p>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    variants={cardVariants}
                    whileHover={{ scale: 1.03, y: -3 }}
                    className="bg-card border-2 border-secondary/20 rounded-lg p-4 shadow-md hover:shadow-lg transition-shadow"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-secondary/15 rounded-lg border border-secondary/20">
                        <Clock className="h-5 w-5 text-secondary-foreground" />
                      </div>
                      <div>
                        <p className="text-xs md:text-sm text-muted-foreground font-medium">Avg Time</p>
                        <p className="text-xl md:text-2xl font-bold text-secondary-foreground">
                          {gallery.length > 0
                            ? Math.round(
                                gallery.reduce((acc, img) => acc + img.generationTimeMs, 0) /
                                  gallery.length
                              )
                            : 0}
                          ms
                        </p>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    variants={cardVariants}
                    whileHover={{ scale: 1.03, y: -3 }}
                    className="bg-card border-2 border-primary/20 rounded-lg p-4 shadow-md hover:shadow-lg transition-shadow"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-primary/15 rounded-lg border border-primary/20">
                        <Layers className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs md:text-sm text-muted-foreground font-medium">Resolution</p>
                        <p className="text-xl md:text-2xl font-bold text-primary">
                          {width}×{height}
                        </p>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    variants={cardVariants}
                    whileHover={{ scale: 1.03, y: -3 }}
                    className="bg-card border-2 border-secondary/20 rounded-lg p-4 shadow-md hover:shadow-lg transition-shadow"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-secondary/15 rounded-lg border border-secondary/20">
                        <Palette className="h-5 w-5 text-secondary-foreground" />
                      </div>
                      <div>
                        <p className="text-xs md:text-sm text-muted-foreground font-medium">Steps</p>
                        <p className="text-xl md:text-2xl font-bold text-secondary-foreground">{steps}</p>
                      </div>
                    </div>
                  </motion.div>
                </motion.div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left Column: Controls */}
                  <motion.div
                    className="lg:col-span-1 space-y-6"
                    variants={itemVariants}
                  >
                    {/* Tips Card - Rowan Style */}
                    {showTips && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="bg-gradient-to-br from-primary/8 to-secondary/8 backdrop-blur-sm border-2 border-primary/30 rounded-lg p-4 shadow-md"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <div className="p-1.5 bg-primary/20 rounded-lg">
                              <Lightbulb className="h-4 w-4 text-primary" />
                            </div>
                            <h3 className="font-semibold text-primary">Pro Tips</h3>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setShowTips(false)}
                            className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                          >
                            ×
                          </Button>
                        </div>
                        <ul className="space-y-2.5 text-sm text-muted-foreground">
                          <li className="flex items-start gap-2.5">
                            <Star className="h-4 w-4 text-secondary mt-0.5 flex-shrink-0" />
                            <span>Be specific and descriptive in your prompts</span>
                          </li>
                          <li className="flex items-start gap-2.5">
                            <Star className="h-4 w-4 text-secondary mt-0.5 flex-shrink-0" />
                            <span>Use negative prompts to exclude unwanted elements</span>
                          </li>
                          <li className="flex items-start gap-2.5">
                            <Star className="h-4 w-4 text-secondary mt-0.5 flex-shrink-0" />
                            <span>Higher steps = better quality but slower generation</span>
                          </li>
                        </ul>
                      </motion.div>
                    )}

                    {/* Example Prompts */}
                    <motion.div variants={cardVariants}>
                      <Card className="bg-card border-2 border-primary/20 shadow-md">
                        <CardHeader className="pb-3">
                          <CardTitle className="flex items-center gap-2 text-base">
                            <div className="p-1.5 bg-primary/15 rounded-lg">
                              <Info className="h-4 w-4 text-primary" />
                            </div>
                            Example Prompts
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2 pt-0">
                          {examplePrompts.map((example, index) => (
                            <motion.div
                              key={index}
                              whileHover={{ scale: 1.02, x: 3 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              <Button
                                variant="ghost"
                                size="sm"
                                className="w-full justify-start text-left h-auto py-2.5 px-3 text-xs hover:bg-primary/10 hover:text-primary border border-transparent hover:border-primary/20 transition-all"
                                onClick={() => setPrompt(example)}
                              >
                                <span className="truncate">{example}</span>
                              </Button>
                            </motion.div>
                          ))}
                        </CardContent>
                      </Card>
                    </motion.div>

                    <motion.div variants={cardVariants}>
                      <Card className="bg-card border-2 border-primary/20 shadow-md">
                        <CardHeader className="border-b border-border/50">
                          <CardTitle className="flex items-center gap-2 text-lg">
                            <div className="p-2 bg-primary/15 rounded-lg border border-primary/20">
                              <Wand2 className="h-5 w-5 text-primary" />
                            </div>
                            Generation Settings
                          </CardTitle>
                          <CardDescription className="mt-1">Configure your image generation parameters</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                          {/* Prompt */}
                          <motion.div
                            className="space-y-2"
                            whileHover={{ scale: 1.01 }}
                            transition={{ type: "spring", stiffness: 300 }}
                          >
                            <div className="flex items-center justify-between">
                              <Label htmlFor="prompt">Prompt</Label>
                              <motion.span
                                className="text-xs text-muted-foreground"
                                key={prompt.length}
                                initial={{ scale: 1.2, color: "var(--primary)" }}
                                animate={{ scale: 1, color: "var(--muted-foreground)" }}
                                transition={{ duration: 0.3 }}
                              >
                                {prompt.length}/2000
                              </motion.span>
                            </div>
                            <Textarea
                              id="prompt"
                              placeholder="Enter your prompt here... ✨"
                              value={prompt}
                              onChange={(e) => setPrompt(e.target.value.slice(0, 2000))}
                              className="min-h-[120px] resize-none transition-all focus:ring-2 focus:ring-primary/50"
                              maxLength={2000}
                            />
                            <div className="flex gap-2">
                              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={handleCopyPrompt}
                                  disabled={!prompt.trim()}
                                  className="w-full"
                                >
                                  <Copy className="h-4 w-4 mr-2" />
                                  Copy Prompt
                                </Button>
                              </motion.div>
                            </div>
                          </motion.div>

                          {/* Negative Prompt */}
                          <motion.div
                            className="space-y-2"
                            initial={false}
                            animate={{
                              height: showNegativePrompt ? "auto" : "auto",
                            }}
                          >
                            <motion.div
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowNegativePrompt(!showNegativePrompt)}
                                className="w-full justify-between"
                              >
                                <span>Negative Prompt (Optional)</span>
                                <motion.div
                                  animate={{ rotate: showNegativePrompt ? 180 : 0 }}
                                  transition={{ duration: 0.3 }}
                                >
                                  {showNegativePrompt ? (
                                    <ChevronUp className="h-4 w-4" />
                                  ) : (
                                    <ChevronDown className="h-4 w-4" />
                                  )}
                                </motion.div>
                              </Button>
                            </motion.div>
                            <AnimatePresence>
                              {showNegativePrompt && (
                                <motion.div
                                  initial={{ opacity: 0, height: 0 }}
                                  animate={{ opacity: 1, height: "auto" }}
                                  exit={{ opacity: 0, height: 0 }}
                                  transition={{ duration: 0.3 }}
                                >
                                  <Textarea
                                    placeholder="What to avoid in the image..."
                                    value={negativePrompt}
                                    onChange={(e) => setNegativePrompt(e.target.value.slice(0, 2000))}
                                    className="min-h-[80px] resize-none"
                                    maxLength={2000}
                                  />
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </motion.div>

                          {/* Resolution Presets */}
                          <motion.div className="space-y-2" variants={itemVariants}>
                            <Label>Resolution</Label>
                            <div className="grid grid-cols-2 gap-2">
                              {RESOLUTION_PRESETS.map((preset, index) => (
                                <motion.div
                                  key={preset.label}
                                  whileHover={{ scale: 1.05, y: -2 }}
                                  whileTap={{ scale: 0.95 }}
                                  initial={{ opacity: 0, y: 10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  transition={{ delay: index * 0.05 }}
                                >
                                  <Button
                                    variant={width === preset.width && height === preset.height ? "default" : "outline"}
                                    size="sm"
                                    onClick={() => handlePresetSelect(preset)}
                                    className="text-xs w-full transition-all"
                                  >
                                    {preset.label}
                                  </Button>
                                </motion.div>
                              ))}
                            </div>
                            <div className="grid grid-cols-2 gap-2 mt-2">
                              <motion.div
                                className="space-y-1"
                                whileHover={{ scale: 1.02 }}
                              >
                                <Label htmlFor="width" className="text-xs">Width</Label>
                                <Input
                                  id="width"
                                  type="number"
                                  value={width}
                                  onChange={(e) => setWidth(Math.max(256, Math.min(2048, parseInt(e.target.value) || 1024)))}
                                  min={256}
                                  max={2048}
                                  className="transition-all focus:ring-2 focus:ring-primary/50"
                                />
                              </motion.div>
                              <motion.div
                                className="space-y-1"
                                whileHover={{ scale: 1.02 }}
                              >
                                <Label htmlFor="height" className="text-xs">Height</Label>
                                <Input
                                  id="height"
                                  type="number"
                                  value={height}
                                  onChange={(e) => setHeight(Math.max(256, Math.min(2048, parseInt(e.target.value) || 1024)))}
                                  min={256}
                                  max={2048}
                                  className="transition-all focus:ring-2 focus:ring-primary/50"
                                />
                              </motion.div>
                            </div>
                          </motion.div>

                          {/* Seed */}
                          <motion.div className="space-y-2" variants={itemVariants}>
                            <div className="flex items-center justify-between">
                              <Label htmlFor="seed">Seed</Label>
                              <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9, rotate: 180 }}>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={handleRandomSeed}
                                >
                                  <Shuffle className="h-4 w-4 mr-2" />
                                  Random
                                </Button>
                              </motion.div>
                            </div>
                            <Input
                              id="seed"
                              type="number"
                              placeholder="Random"
                              value={seed >= 0 ? seed : ""}
                              onChange={(e) => {
                                const val = e.target.value
                                setSeed(val === "" ? -1 : parseInt(val) || -1)
                              }}
                              min={0}
                              max={2 ** 32 - 1}
                              className="transition-all focus:ring-2 focus:ring-primary/50"
                            />
                          </motion.div>

                          {/* Steps */}
                          <motion.div className="space-y-2" variants={itemVariants}>
                            <div className="flex items-center justify-between">
                              <Label htmlFor="steps">Steps: {steps}</Label>
                              <motion.span
                                key={steps}
                                initial={{ scale: 1.2, color: "var(--primary)" }}
                                animate={{ scale: 1, color: "var(--foreground)" }}
                                className="text-sm font-semibold"
                              >
                                {steps}
                              </motion.span>
                            </div>
                            <Slider
                              id="steps"
                              min={6}
                              max={12}
                              step={1}
                              value={[steps]}
                              onValueChange={(value) => setSteps(value[0])}
                              className="cursor-pointer"
                            />
                          </motion.div>

                          {/* Generate Button - Rowan Style */}
                          <motion.div
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            <Button
                              onClick={handleGenerate}
                              disabled={isGenerating || !prompt.trim() || apiHealth === false}
                              className="w-full relative overflow-hidden group rowan-gradient text-primary-foreground font-semibold hover:opacity-90 transition-opacity shadow-lg hover:shadow-xl"
                              size="lg"
                            >
                              <motion.div
                                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                                animate={{
                                  x: isGenerating ? ["-100%", "100%"] : "-100%",
                                }}
                                transition={{
                                  duration: 2,
                                  repeat: isGenerating ? Infinity : 0,
                                  ease: "linear",
                                }}
                              />
                              {isGenerating ? (
                                <>
                                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                                  Generating...
                                </>
                              ) : (
                                <>
                                  <Sparkles className="h-5 w-5 mr-2" />
                                  Generate Image
                                </>
                              )}
                            </Button>
                            {/* Progress bar */}
                            <AnimatePresence>
                              {isGenerating && (
                                <motion.div
                                  initial={{ opacity: 0, height: 0 }}
                                  animate={{ opacity: 1, height: 4 }}
                                  exit={{ opacity: 0, height: 0 }}
                                  className="mt-2 w-full bg-muted rounded-full overflow-hidden"
                                >
                                  <motion.div
                                    className="h-full bg-gradient-to-r from-primary via-primary/80 to-primary"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${progressValue}%` }}
                                    transition={{ type: "spring", stiffness: 100, damping: 30 }}
                                  />
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </motion.div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  </motion.div>

                  {/* Right Column: Image Display & Gallery */}
                  <motion.div
                    className="lg:col-span-2 space-y-6"
                    variants={itemVariants}
                  >
                    {/* Current Image */}
                    <AnimatePresence mode="wait">
                      {currentImage && (
                        <motion.div
                          key={currentImage.id}
                          variants={cardVariants}
                          initial="hidden"
                          animate="visible"
                          exit="hidden"
                        >
                          <Card className="bg-card border-2 border-primary/20 shadow-md">
                            <CardHeader className="border-b border-border/50">
                              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                                <div>
                                  <CardTitle className="flex items-center gap-2 text-lg">
                                    <div className="p-1.5 bg-secondary/15 rounded-lg border border-secondary/20">
                                      <Zap className="h-5 w-5 text-secondary-foreground" />
                                    </div>
                                    Generated Image
                                  </CardTitle>
                                  <CardDescription className="mt-1.5">
                                    Seed: {currentImage.seed} • {currentImage.width}×{currentImage.height} • {currentImage.steps} steps • {currentImage.generationTimeMs}ms
                                  </CardDescription>
                                </div>
                                <div className="flex gap-2">
                                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      onClick={handleCopySeed}
                                      className="border-primary/30 hover:bg-primary/10 hover:border-primary/50"
                                    >
                                      <Copy className="h-4 w-4 mr-2" />
                                      Copy Seed
                                    </Button>
                                  </motion.div>
                                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      onClick={() => handleDownload(currentImage)}
                                      className="border-secondary/30 hover:bg-secondary/10 hover:border-secondary/50"
                                    >
                                      <Download className="h-4 w-4 mr-2" />
                                      Download
                                    </Button>
                                  </motion.div>
                                </div>
                              </div>
                            </CardHeader>
                            <CardContent>
                              <motion.div
                                className="relative w-full aspect-square bg-muted rounded-lg overflow-hidden group"
                                variants={imageVariants}
                                whileHover={{ scale: 1.02 }}
                                transition={{ type: "spring", stiffness: 300 }}
                              >
                                <motion.img
                                  src={currentImage.imageBase64}
                                  alt={currentImage.prompt}
                                  className="w-full h-full object-contain"
                                  initial={{ opacity: 0, scale: 0.9 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  transition={{ duration: 0.5, delay: 0.2 }}
                                />
                                <motion.div
                                  className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                                  initial={false}
                                />
                              </motion.div>
                            </CardContent>
                          </Card>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Gallery */}
                    <AnimatePresence>
                      {gallery.length > 0 && (
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.4 }}
                        >
                          <Card className="bg-card border-2 border-primary/20 shadow-md">
                            <CardHeader className="border-b border-border/50">
                              <CardTitle className="text-lg">Recent Images</CardTitle>
                              <CardDescription>Click to view full size</CardDescription>
                            </CardHeader>
                            <CardContent>
                              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                                {gallery.map((img, index) => (
                                  <motion.div
                                    key={img.id}
                                    custom={index}
                                    variants={galleryItemVariants}
                                    initial="hidden"
                                    animate="visible"
                                    whileHover="hover"
                                    className="relative aspect-square bg-muted rounded-lg overflow-hidden cursor-pointer group"
                                    onClick={() => setSelectedImage(img)}
                                  >
                                    <motion.img
                                      src={img.imageBase64}
                                      alt={img.prompt}
                                      className="w-full h-full object-cover"
                                      whileHover={{ scale: 1.1 }}
                                      transition={{ duration: 0.3 }}
                                    />
                                    <motion.div
                                      className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center"
                                      initial={false}
                                    >
                                      <motion.div
                                        initial={{ opacity: 0, scale: 0.5 }}
                                        whileHover={{ opacity: 1, scale: 1 }}
                                        transition={{ duration: 0.2 }}
                                      >
                                        <ImageIcon className="h-8 w-8 text-white" />
                                      </motion.div>
                                    </motion.div>
                                  </motion.div>
                                ))}
                              </div>
                            </CardContent>
                          </Card>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Empty State */}
                    <AnimatePresence>
                      {!currentImage && !isGenerating && (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.9 }}
                          transition={{ duration: 0.4 }}
                        >
                          <Card className="bg-card border-2 border-primary/20 shadow-md">
                            <CardContent className="flex flex-col items-center justify-center py-16">
                              <motion.div
                                animate={{
                                  y: [0, -10, 0],
                                  rotate: [0, 5, -5, 0],
                                }}
                                transition={{
                                  duration: 3,
                                  repeat: Infinity,
                                  ease: "easeInOut",
                                }}
                              >
                                <div className="relative">
                                  <ImageIcon className="h-20 w-20 text-muted-foreground mb-4" />
                                  <motion.div
                                    className="absolute -inset-4 bg-primary/10 rounded-full blur-xl"
                                    animate={{
                                      scale: [1, 1.2, 1],
                                      opacity: [0.5, 0.8, 0.5],
                                    }}
                                    transition={{
                                      duration: 3,
                                      repeat: Infinity,
                                      ease: "easeInOut",
                                    }}
                                  />
                                </div>
                              </motion.div>
                              <p className="text-muted-foreground text-center text-lg font-medium">
                                Enter a prompt and click Generate to create an image ✨
                              </p>
                            </CardContent>
                          </Card>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {/* Placeholder tabs */}
            {(activeTab === "image-to-video" || activeTab === "image-to-image") && (
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="flex items-center justify-center min-h-[60vh]"
              >
                <Card className="bg-card border-2 border-primary/20 shadow-md max-w-md">
                  <CardContent className="flex flex-col items-center justify-center py-16 px-8">
                    <motion.div
                      animate={{
                        y: [0, -10, 0],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                    >
                      <div className="relative">
                        {activeTab === "image-to-video" ? (
                          <Video className="h-16 w-16 text-muted-foreground mb-4" />
                        ) : (
                          <ImageIcon className="h-16 w-16 text-muted-foreground mb-4" />
                        )}
                        <motion.div
                          className="absolute -inset-4 bg-secondary/10 rounded-full blur-xl"
                          animate={{
                            scale: [1, 1.2, 1],
                            opacity: [0.5, 0.8, 0.5],
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            ease: "easeInOut",
                          }}
                        />
                      </div>
                    </motion.div>
                    <h2 className="text-2xl font-bold mb-2 rowan-gradient-text">
                      {activeTab === "image-to-video" ? "Image to Video" : "Image to Image Edit"}
                    </h2>
                    <p className="text-muted-foreground text-center font-medium">
                      This feature is coming soon! 🚀
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Image Dialog with animations */}
      <Dialog open={!!selectedImage} onOpenChange={() => setSelectedImage(null)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Image Details</DialogTitle>
            <DialogDescription>
              Seed: {selectedImage?.seed} • {selectedImage?.width}×{selectedImage?.height} • {selectedImage?.steps} steps
            </DialogDescription>
          </DialogHeader>
          <AnimatePresence mode="wait">
            {selectedImage && (
              <motion.div
                key={selectedImage.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="space-y-4"
              >
                <motion.div
                  className="relative w-full bg-muted rounded-lg overflow-hidden"
                  initial={{ scale: 0.95 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <img
                    src={selectedImage.imageBase64}
                    alt={selectedImage.prompt}
                    className="w-full h-auto"
                  />
                </motion.div>
                <div className="space-y-2">
                  <div>
                    <Label className="text-xs text-muted-foreground">Prompt</Label>
                    <p className="text-sm">{selectedImage.prompt}</p>
                  </div>
                  {selectedImage.negativePrompt && (
                    <div>
                      <Label className="text-xs text-muted-foreground">Negative Prompt</Label>
                      <p className="text-sm">{selectedImage.negativePrompt}</p>
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      variant="outline"
                      onClick={() => {
                        navigator.clipboard.writeText(selectedImage.seed.toString())
                        toast.success("Seed copied")
                      }}
                    >
                      <Copy className="h-4 w-4 mr-2" />
                      Copy Seed
                    </Button>
                  </motion.div>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      variant="outline"
                      onClick={() => handleDownload(selectedImage)}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </motion.div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </DialogContent>
      </Dialog>

      {/* Footer */}
      <Footer />
    </div>
  )
}
