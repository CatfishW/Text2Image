"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useEffect, useState } from "react"
import { Printer } from "lucide-react"

interface PrinterAnimationProps {
    prompt: string
    image: string | null
    onComplete: () => void
}

export function PrinterAnimation({ prompt, image, onComplete }: PrinterAnimationProps) {
    const [typedPrompt, setTypedPrompt] = useState("")
    const [isPrinting, setIsPrinting] = useState(false)

    // Typewriter effect
    useEffect(() => {
        if (isPrinting) return

        let i = 0
        const interval = setInterval(() => {
            setTypedPrompt(prompt.slice(0, i))
            i++
            if (i > prompt.length) {
                clearInterval(interval)
            }
        }, 30) // Speed of typing

        return () => clearInterval(interval)
    }, [prompt, isPrinting])

    // Trigger printing when image is available
    useEffect(() => {
        if (image) {
            setIsPrinting(true)
            // Wait for print animation to finish before completing
            const timer = setTimeout(() => {
                onComplete()
            }, 3000) // Duration of print animation
            return () => clearTimeout(timer)
        }
    }, [image, onComplete])

    return (
        <div className="relative flex flex-col items-center justify-center h-full w-full max-w-2xl mx-auto">
            <div className="relative z-10 w-64 h-32 bg-card border-2 border-primary/20 rounded-lg shadow-2xl flex items-center justify-center">
                {/* Printer Body */}
                <div className="absolute inset-0 bg-gradient-to-b from-muted to-card rounded-lg" />
                <div className="absolute bottom-0 left-4 right-4 h-4 bg-black/20 rounded-full blur-md" />

                {/* Paper Slot */}
                <div className="absolute top-0 left-8 right-8 h-2 bg-black/80 rounded-b-sm" />

                {/* Printer Icon/Logo */}
                <Printer className="w-12 h-12 text-primary/50 relative z-10" />

                {/* Status Light */}
                <div className="absolute top-4 right-4 flex gap-2">
                    <motion.div
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="w-2 h-2 rounded-full bg-green-500"
                    />
                </div>
            </div>

            {/* Paper with Prompt (Input) */}
            <AnimatePresence>
                {!isPrinting && (
                    <motion.div
                        initial={{ y: 0, opacity: 0 }}
                        animate={{ y: -80, opacity: 1 }}
                        exit={{ y: 0, opacity: 0 }}
                        className="absolute top-1/2 left-1/2 -translate-x-1/2 w-48 bg-white text-black p-3 text-xs font-mono shadow-sm border border-gray-200 -z-10"
                        style={{ transformOrigin: "bottom center" }}
                    >
                        <p className="opacity-70">PROMPT_REQ_01</p>
                        <div className="h-[1px] w-full bg-gray-200 my-1" />
                        <p className="break-words line-clamp-4">{typedPrompt}<span className="animate-pulse">_</span></p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Printed Image (Output) */}
            <AnimatePresence>
                {isPrinting && image && (
                    <motion.div
                        initial={{ y: -50, scale: 0.9, opacity: 0 }}
                        animate={{ y: 180, scale: 1, opacity: 1 }}
                        transition={{ duration: 2.5, ease: "easeInOut" }}
                        className="absolute top-1/2 left-1/2 -translate-x-1/2 w-64 h-64 bg-white p-2 shadow-xl border border-gray-200 -z-10"
                    >
                        <div className="w-full h-full overflow-hidden bg-gray-100 relative">
                            {/* Scanning effect line */}
                            <motion.div
                                initial={{ top: 0 }}
                                animate={{ top: "100%" }}
                                transition={{ duration: 2.5, ease: "linear" }}
                                className="absolute left-0 right-0 h-1 bg-primary/50 shadow-[0_0_10px_rgba(0,0,0,0.2)] z-10"
                            />
                            <img src={image} alt="Generated" className="w-full h-full object-cover" />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
