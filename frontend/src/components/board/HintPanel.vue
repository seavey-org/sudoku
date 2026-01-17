<script setup lang="ts">
import type { HintResult } from '../../strategies/types'

defineProps<{
    hint: HintResult
}>()

const emit = defineEmits<{
    apply: []
    dismiss: []
}>()
</script>

<template>
    <div class="hint-panel">
        <div class="hint-header">
            <strong>{{ hint.strategyName }}</strong>
            <span class="hint-difficulty">(Difficulty: {{ hint.difficulty.toFixed(1) }})</span>
        </div>
        <p class="hint-description">{{ hint.description }}</p>
        <div class="hint-actions">
            <button
                v-if="hint.eliminations.length > 0"
                @click="emit('apply')"
                class="apply-btn"
            >
                Apply ({{ hint.eliminations.length }} elimination{{ hint.eliminations.length > 1 ? 's' : '' }})
            </button>
            <button @click="emit('dismiss')" class="dismiss-btn">Dismiss</button>
        </div>
    </div>
</template>

<style scoped>
.hint-panel {
    background: linear-gradient(145deg, #2a2050, #1a1040);
    border: 2px solid #6a5a9a;
    border-radius: 8px;
    padding: 15px;
    margin-top: 10px;
    color: #e8e0ff;
    width: 100%;
    max-width: 450px;
    box-sizing: border-box;
}

.hint-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

.hint-header strong {
    color: #c0b0ff;
    font-size: 1.1rem;
}

.hint-difficulty {
    font-size: 0.85rem;
    color: #a090d0;
}

.hint-description {
    margin: 0 0 12px 0;
    font-size: 0.95rem;
    line-height: 1.4;
}

.hint-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.apply-btn {
    background: linear-gradient(145deg, #4a8040, #3a6030);
    border: 2px solid #4a8040;
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.15s ease;
}

.apply-btn:hover {
    background: linear-gradient(145deg, #5a9050, #4a7040);
}

.dismiss-btn {
    background: linear-gradient(145deg, #5a4a7a, #4a3a6a);
    border: 2px solid #6a5a9a;
    color: #d0c0f0;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
}

.dismiss-btn:hover {
    background: linear-gradient(145deg, #6a5a8a, #5a4a7a);
}
</style>
