"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Sparkles, Video, ImageIcon, Wand2, Settings, GraduationCap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"

type Tab = "text-to-image" | "image-to-video" | "image-to-image"

interface NavbarProps {
  activeTab: Tab
  onTabChange: (tab: Tab) => void
  apiHealth: boolean | null
}

export function Navbar({ activeTab, onTabChange, apiHealth }: NavbarProps) {
  const tabs: { id: Tab; label: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: "text-to-image",
      label: "Text to Image",
      icon: <Sparkles className="h-4 w-4" />,
    },
    {
      id: "image-to-video",
      label: "Image to Video",
      icon: <Video className="h-4 w-4" />,
      badge: "Coming Soon",
    },
    {
      id: "image-to-image",
      label: "Image to Image Edit",
      icon: <ImageIcon className="h-4 w-4" />,
      badge: "Coming Soon",
    },
  ]

  return (
    <motion.nav
      className="sticky top-0 z-50 w-full glass border-b border-white/10"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      style={{ position: 'sticky' }}
    >
      <div className="container mx-auto px-4 max-w-7xl">
        <div className="flex flex-col md:flex-row md:h-20 items-center justify-between relative py-2 md:py-0 gap-4 md:gap-0">

          {/* Top Row (Mobile) / Left & Center & Right (Desktop) */}
          <div className="w-full md:w-auto flex items-center justify-between md:contents">

            {/* Left side - API Health & Theme Toggle */}
            <div className="flex-1 flex items-center justify-start gap-3">
              {apiHealth === false && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex items-center gap-2 text-destructive text-sm bg-destructive/10 px-3 py-1.5 rounded-full border border-destructive/20"
                >
                  <motion.div
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  >
                    <span className="h-2 w-2 bg-destructive rounded-full block" />
                  </motion.div>
                  <span className="hidden sm:inline">API Offline</span>
                </motion.div>
              )}
              <ThemeToggle />
            </div>

            {/* Center - Logo */}
            <div className="flex-0 flex flex-col items-center gap-1 mx-4 absolute left-1/2 -translate-x-1/2 md:static md:translate-x-0">
              <motion.div
                className="flex items-center gap-3"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
              >
                <motion.div
                  className="relative"
                  whileHover={{ rotate: [0, -10, 10, 0] }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="relative">
                    <GraduationCap className="h-6 w-6 md:h-8 md:w-8 text-primary" />
                    <motion.div
                      className="absolute -inset-1 bg-secondary/20 rounded-full blur"
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
                <div className="flex flex-col items-center md:items-start">
                  <span className="text-sm md:text-lg font-bold rowan-gradient-text leading-tight whitespace-nowrap">
                    Rowan University
                  </span>
                  <span className="text-[10px] md:text-xs text-muted-foreground font-medium whitespace-nowrap">
                    AI Image Generation Studio
                  </span>
                  <span className="hidden md:inline text-[10px] text-muted-foreground/60 whitespace-nowrap">
                    Creator: Yanlai Wu (wuyanl37@rowan.edu)
                  </span>
                </div>
              </motion.div>
            </div>

            {/* Right Spacer for Mobile Balance */}
            <div className="w-10 md:hidden" />
          </div>

          {/* Right side - Tabs (Desktop) */}
          <div className="hidden md:flex flex-1 items-center justify-end gap-1">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id
              const isDisabled = tab.badge !== undefined

              return (
                <motion.div
                  key={tab.id}
                  className="relative"
                  whileHover={!isDisabled ? { scale: 1.05 } : {}}
                  whileTap={!isDisabled ? { scale: 0.95 } : {}}
                >
                  <Button
                    variant={isActive ? "default" : "ghost"}
                    className={`relative px-4 py-2 h-9 text-xs gap-2 font-medium ${isDisabled ? "opacity-50 cursor-not-allowed" : ""
                      }`}
                    onClick={() => !isDisabled && onTabChange(tab.id)}
                    disabled={isDisabled}
                  >
                    {tab.icon}
                    <span className="hidden lg:inline">{tab.label}</span>
                    <span className="lg:hidden">{tab.label.split(" ")[0]}</span>
                    {tab.badge && (
                      <motion.span
                        className="ml-1 text-[10px] px-1 py-0.5 rounded-full bg-secondary/20 text-secondary-foreground"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.2 }}
                      >
                        {tab.badge}
                      </motion.span>
                    )}
                    {isActive && (
                      <motion.div
                        className="absolute bottom-0 left-0 right-0 h-1 bg-secondary rounded-t-full"
                        layoutId="activeTab"
                        initial={false}
                        transition={{
                          type: "spring",
                          stiffness: 500,
                          damping: 30,
                        }}
                      />
                    )}
                  </Button>
                </motion.div>
              )
            })}
          </div>

          {/* Mobile Tabs Row */}
          <div className="flex md:hidden w-full items-center justify-center gap-2 overflow-x-auto pb-2">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id
              const isDisabled = tab.badge !== undefined
              return (
                <Button
                  key={tab.id}
                  variant={isActive ? "default" : "ghost"}
                  size="sm"
                  className={`text-xs gap-2 ${isDisabled ? "opacity-50" : ""}`}
                  onClick={() => !isDisabled && onTabChange(tab.id)}
                  disabled={isDisabled}
                >
                  {tab.icon}
                  <span>{tab.label.split(" ")[0]}</span>
                </Button>
              )
            })}
          </div>
        </div>
      </div>
    </motion.nav>
  )
}
