"use client"

import { motion } from "framer-motion"
import { Mail, GraduationCap, Heart } from "lucide-react"

export function Footer() {
  return (
    <motion.footer
      className="w-full border-t-2 border-primary/20 bg-background/95 backdrop-blur-sm mt-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="container mx-auto px-4 py-6 max-w-7xl">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Author Information */}
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg border border-primary/20">
              <GraduationCap className="h-5 w-5 text-primary" />
            </div>
            <div className="flex flex-col sm:flex-row sm:items-center gap-2">
              <span className="text-sm font-medium text-foreground">
                Author: <span className="text-primary font-semibold">Yanlai Wu</span>
              </span>
              <span className="hidden sm:inline text-muted-foreground">•</span>
              <a
                href="mailto:wuyanl37@rowan.edu"
                className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-primary transition-colors"
              >
                <Mail className="h-4 w-4" />
                <span>wuyanl37@rowan.edu</span>
              </a>
            </div>
          </div>

          {/* Rowan University Branding */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Made with</span>
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
            >
              <Heart className="h-4 w-4 text-secondary fill-secondary" />
            </motion.div>
            <span>at</span>
            <span className="rowan-gradient-text font-semibold">Rowan University</span>
          </div>
        </div>
      </div>
    </motion.footer>
  )
}
