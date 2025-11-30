"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useEffect, useState } from "react"
import { Printer, Terminal, Zap } from "lucide-react"

interface PrinterAnimationProps {
    prompt: string
    image: string | null
    onComplete: () => void
}

export function PrinterAnimation({ prompt, image, onComplete }: PrinterAnimationProps) {
    const [typedPrompt, setTypedPrompt] = useState("")
    const [isPrinting, setIsPrinting] = useState(false)
    const [showFinal, setShowFinal] = useState(false)

    // Typewriter effect for the CRT screen
    useEffect(() => {
        if (isPrinting) return

        let i = 0
        setTypedPrompt("")
        const interval = setInterval(() => {
            setTypedPrompt(prompt.slice(0, i))
            i++
            if (i > prompt.length) {
                clearInterval(interval)
            }
        }, 50)

        return () => clearInterval(interval)
    }, [prompt, isPrinting])

    // Trigger printing sequence when image is available
    useEffect(() => {
        if (image) {
            setIsPrinting(true)

            // Sequence:
            // 1. Image slides out (approx 2s)
            // 2. Printer fades, Image flies to center (approx 1s)
            // 3. Complete

            const flyTimer = setTimeout(() => {
                setShowFinal(true)
            }, 2500)

            const completeTimer = setTimeout(() => {
                onComplete()
            }, 4000)

            return () => {
                clearTimeout(flyTimer)
                clearTimeout(completeTimer)
            }
        }
    }, [image, onComplete])

    return (
        <div className="relative flex flex-col items-center justify-center h-full w-full overflow-hidden">

            {/* 3D Fallout Printer Container */}
            <AnimatePresence>
                {!showFinal && (
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.8, opacity: 0, y: 100 }}
                        transition={{ duration: 0.5 }}
                        className="relative z-20 flex flex-col items-center"
                    >
                        {/* Printer Body */}
                        <div className="relative w-96 h-64 bg-[#2a2a2a] rounded-xl shadow-2xl border-4 border-[#3d3d3d] flex flex-col overflow-hidden">
                            {/* Top Panel (Bolts & Industrial look) */}
                            <div className="h-4 bg-[#1a1a1a] border-b border-[#3d3d3d] flex items-center justify-between px-2">
                                <div className="flex gap-1">
                                    <div className="w-1.5 h-1.5 rounded-full bg-[#555]" />
                                    <div className="w-1.5 h-1.5 rounded-full bg-[#555]" />
                                </div>
                                <div className="flex gap-1">
                                    <div className="w-1.5 h-1.5 rounded-full bg-[#555]" />
                                    <div className="w-1.5 h-1.5 rounded-full bg-[#555]" />
                                </div>
                            </div>

                            {/* Main Face */}
                            <div className="flex-1 p-4 flex flex-col gap-4 relative">
                                {/* CRT Screen */}
                                <div className="w-full h-32 bg-black rounded-lg border-2 border-[#4a4a4a] shadow-[inset_0_0_20px_rgba(0,0,0,1)] relative overflow-hidden p-3">
                                    <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-10 pointer-events-none bg-[length:100%_2px,3px_100%]" />
                                    <div className="font-mono text-green-500 text-xs leading-relaxed break-words relative z-0 opacity-90 shadow-[0_0_5px_rgba(74,222,128,0.5)]">
                                        <span className="text-green-700 mr-2">{">"}</span>
                                        {typedPrompt}
                                        <motion.span
                                            animate={{ opacity: [0, 1, 0] }}
                                            transition={{ duration: 0.8, repeat: Infinity }}
                                            className="inline-block w-2 h-4 bg-green-500 ml-1 align-middle"
                                        />
                                    </div>
                                </div>

                                {/* Controls & Output Slot */}
                                <div className="flex items-center justify-between mt-auto">
                                    {/* Buttons/Dials */}
                                    <div className="flex gap-3">
                                        <div className="w-8 h-8 rounded-full bg-[#1a1a1a] border-2 border-[#3d3d3d] flex items-center justify-center shadow-inner">
                                            <div className={`w-2 h-2 rounded-full ${isPrinting ? "bg-amber-500 animate-pulse" : "bg-green-900"}`} />
                                        </div>
                                        <div className="w-8 h-8 rounded-full bg-[#1a1a1a] border-2 border-[#3d3d3d] flex items-center justify-center">
                                            <Zap className="w-4 h-4 text-yellow-600" />
                                        </div>
                                    </div>

                                    {/* Output Slot Label */}
                                    <div className="text-[10px] font-mono text-[#666] uppercase tracking-widest border border-[#444] px-2 py-0.5 rounded">
                                        Output_V2
                                    </div>
                                </div>
                            </div>

                            {/* Paper Slot (Bottom) */}
                            <div className="h-2 bg-black w-3/4 mx-auto rounded-full blur-[1px] relative top-1" />
                        </div>

                        {/* The Printed Image (Sliding Out) */}
                        <AnimatePresence>
                            {isPrinting && image && !showFinal && (
                                <motion.div
                                    initial={{ y: -200, scale: 0.9, opacity: 0, zIndex: 10 }}
                                    animate={{ y: 60, scale: 1, opacity: 1 }}
                                    exit={{ opacity: 0 }} // Fades out when showFinal becomes true
                                    transition={{ duration: 2, ease: "linear" }}
                                    className="absolute top-[100%] w-64 h-64 bg-white p-2 shadow-xl border border-gray-200"
                                    style={{ marginTop: "-20px" }} // Start tucked in
                                >
                                    <div className="w-full h-full overflow-hidden bg-gray-100 relative">
                                        {/* Scanning Line */}
                                        <motion.div
                                            initial={{ top: 0 }}
                                            animate={{ top: "100%" }}
                                            transition={{ duration: 2, ease: "linear" }}
                                            className="absolute left-0 right-0 h-1 bg-green-500/50 shadow-[0_0_15px_rgba(74,222,128,0.8)] z-10"
                                        />
                                        <img src={image} alt="Generated" className="w-full h-full object-cover" />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Final Flying Image */}
            <AnimatePresence>
                {showFinal && image && (
                    <motion.div
                        initial={{ scale: 0.5, y: 100, opacity: 0 }}
                        animate={{ scale: 1.5, y: 0, opacity: 1 }}
                        transition={{ duration: 1.2, type: "spring", bounce: 0.4 }}
                        className="absolute z-50 shadow-2xl rounded-lg overflow-hidden border-4 border-white/10"
                    >
                        <img src={image} alt="Final" className="max-w-[80vw] max-h-[80vh] object-contain shadow-2xl" />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
