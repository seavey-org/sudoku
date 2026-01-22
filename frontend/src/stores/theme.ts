import { defineStore } from 'pinia'

const STORAGE_KEY = 'sudoku-theme'

type ThemeMode = 'system' | 'light' | 'dark'

export const useThemeStore = defineStore('theme', {
  state: () => ({
    currentTheme: (localStorage.getItem(STORAGE_KEY) || 'system') as ThemeMode
  }),

  getters: {
    isDark: (state): boolean => {
      if (state.currentTheme === 'dark') return true
      if (state.currentTheme === 'light') return false
      // System preference
      return window.matchMedia('(prefers-color-scheme: dark)').matches
    }
  },

  actions: {
    setTheme(theme: ThemeMode) {
      this.currentTheme = theme
      localStorage.setItem(STORAGE_KEY, theme)
      this.applyTheme()
    },

    cycleTheme() {
      const themes: ThemeMode[] = ['system', 'light', 'dark']
      const currentIndex = themes.indexOf(this.currentTheme)
      const nextIndex = (currentIndex + 1) % themes.length
      const nextTheme = themes[nextIndex]
      if (nextTheme) {
        this.setTheme(nextTheme)
      }
    },

    applyTheme() {
      const isDark = this.isDark
      if (isDark) {
        document.documentElement.classList.add('dark')
      } else {
        document.documentElement.classList.remove('dark')
      }
    },

    initTheme() {
      this.applyTheme()
      // Listen for system theme changes
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (this.currentTheme === 'system') {
          this.applyTheme()
        }
      })
    }
  }
})
