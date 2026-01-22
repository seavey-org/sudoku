<script setup lang="ts">
import { ref, onMounted } from 'vue'

const emit = defineEmits(['back-to-menu'])

const stats = ref({ totalSolved: 0, details: {} as Record<string, Record<string, Record<string, number>>> })
const loading = ref(true)

const formatType = (type: string) => {
    if (type === 'standard') return 'Classic Sudoku'
    if (type === 'killer') return 'Killer Sudoku'
    return type.charAt(0).toUpperCase() + type.slice(1)
}

const formatSize = (size: string) => {
    return `${size}x${size}`
}

const formatDifficulty = (diff: string) => {
    return diff.charAt(0).toUpperCase() + diff.slice(1)
}

onMounted(async () => {
    try {
        const res = await fetch('/api/stats')
        stats.value = await res.json()
    } catch (e) {
        console.error('Failed to load stats', e)
    } finally {
        loading.value = false
    }
})
</script>

<template>
  <div class="flex justify-center items-start w-full px-2 box-border mt-4">
    <div class="bg-white dark:bg-gray-800 p-4 sm:p-6 rounded-lg shadow-md text-gray-800 dark:text-gray-100 flex flex-col gap-3 sm:gap-4 w-full max-w-md box-border">
      <h2 class="m-0 text-center text-gray-700 dark:text-gray-200 text-xl sm:text-2xl font-semibold">Global Stats</h2>
      <div v-if="loading" class="text-center text-gray-500 dark:text-gray-400">Loading...</div>
      <div v-else>
        <div class="text-base sm:text-lg font-bold flex justify-between py-2 border-b border-gray-200 dark:border-gray-700 mb-3">
          <span class="font-medium">Total Puzzles Solved:</span>
          <span>{{ stats.totalSolved }}</span>
        </div>

        <div v-for="(sizes, type) in stats.details" :key="type">
          <h3 class="mt-4 mb-1 text-base sm:text-lg text-gray-600 dark:text-gray-300 border-b-2 border-purple-300 dark:border-purple-500 pb-1 text-left">
            {{ formatType(type) }}
          </h3>
          <div v-for="(difficulties, size) in sizes" :key="size">
            <template v-if="typeof difficulties === 'object' && !isNaN(parseInt(size))">
              <h4 class="mt-2 mb-1 ml-3 text-sm text-gray-500 dark:text-gray-400 text-left uppercase tracking-wide">
                {{ formatSize(size) }}
              </h4>
              <div v-for="(count, difficulty) in difficulties" :key="difficulty" 
                   class="text-sm flex justify-between py-1 pl-6 border-b border-gray-100 dark:border-gray-700">
                <span class="font-medium">{{ formatDifficulty(difficulty) }}:</span>
                <span>{{ count }}</span>
              </div>
            </template>
          </div>
        </div>
      </div>
      <button 
        class="py-3 mt-3 bg-gray-700 dark:bg-gray-600 text-white border-none rounded cursor-pointer text-sm sm:text-base min-h-11 touch-manipulation hover:bg-gray-800 dark:hover:bg-gray-500"
        @click="emit('back-to-menu')"
      >
        Back to Menu
      </button>
    </div>
  </div>
</template>
