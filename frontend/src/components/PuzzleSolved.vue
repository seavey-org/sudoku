<script setup lang="ts">
import type { PropType } from 'vue'

const props = defineProps({
    board: {
        type: Array as PropType<(number | null)[][]>,
        required: true
    },
    difficulty: {
        type: String,
        required: true
    },
    size: {
        type: Number,
        required: true
    },
    gameType: {
        type: String,
        default: 'standard'
    }
})

const emit = defineEmits(['new-game', 'back-to-menu'])

// Helper to get border classes for sudoku grid lines
const getCellBorderClass = (row: number, col: number): string => {
    const classes: string[] = []
    
    if (props.size === 9) {
        // Right border for columns 2, 5 (0-indexed)
        if (col === 2 || col === 5) {
            classes.push('border-r-3 border-r-black')
        }
        // Bottom border for rows 2, 5 (0-indexed)
        if (row === 2 || row === 5) {
            classes.push('border-b-3 border-b-black')
        }
    } else if (props.size === 6) {
        // Right border for column 2
        if (col === 2) {
            classes.push('border-r-3 border-r-black')
        }
        // Bottom border for rows 1, 3
        if (row === 1 || row === 3) {
            classes.push('border-b-3 border-b-black')
        }
    }
    
    return classes.join(' ')
}
</script>

<template>
  <div class="flex flex-col items-center gap-3 sm:gap-4 mt-4 text-purple-300 dark:text-purple-300 w-full px-2 box-border">
    <div class="text-center">
      <h1 class="text-2xl sm:text-3xl font-bold m-0 mb-2">Puzzle Solved!</h1>
      <p class="text-base m-0 text-gray-300 dark:text-gray-400">Great job completing this {{ difficulty }} puzzle.</p>
    </div>

    <!-- Grid with special border handling -->
    <div class="solved-grid bg-black border-3 border-black w-full max-w-[280px]" :class="`size-${size}`">
      <div v-for="(row, rIndex) in board" :key="rIndex" class="contents">
        <div v-for="(cell, cIndex) in row" :key="cIndex" 
             class="bg-white dark:bg-gray-200 aspect-square border border-gray-300 flex justify-center items-center text-base sm:text-lg text-black font-bold cell"
             :class="getCellBorderClass(rIndex, cIndex)">
          <span>{{ cell }}</span>
        </div>
      </div>
    </div>

    <div class="flex gap-2 sm:gap-3 w-full max-w-[280px]">
      <button 
        class="flex-1 py-3 px-2 text-white border-none rounded cursor-pointer text-sm sm:text-base min-h-11 sm:min-h-12 touch-manipulation bg-gray-700 dark:bg-gray-600 hover:bg-gray-800 dark:hover:bg-gray-500"
        @click="emit('back-to-menu')"
      >
        Back to Menu
      </button>
      <button 
        class="flex-1 py-3 px-2 text-white border-none rounded cursor-pointer text-sm sm:text-base min-h-11 sm:min-h-12 touch-manipulation bg-green-500 hover:bg-green-600"
        @click="emit('new-game')"
      >
        New Game
      </button>
    </div>
  </div>
</template>

<style scoped>
.solved-grid {
  display: grid;
  grid-template-columns: repeat(v-bind(size), 1fr);
}

.border-3 {
  border-width: 3px;
}

.border-r-3 {
  border-right-width: 3px !important;
}

.border-b-3 {
  border-bottom-width: 3px !important;
}
</style>
