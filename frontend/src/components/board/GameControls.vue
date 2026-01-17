<script setup lang="ts">
defineProps<{
    isPaused: boolean
    isGameOver: boolean
    isNoteMode: boolean
    candidatesPopulated: boolean
}>()

const emit = defineEmits<{
    newGame: []
    togglePause: []
    undo: []
    share: []
    showSolution: []
    toggleNotes: []
    generateCandidates: []
    getHint: []
}>()
</script>

<template>
    <div class="controls-wrapper">
        <div class="controls">
            <button @click="emit('newGame')" title="New Game">
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 5v14M5 12h14"/>
                </svg>
                <span class="btn-text">New</span>
            </button>
            <button @click="emit('togglePause')" :title="isPaused ? 'Resume' : 'Pause'">
                <svg v-if="isPaused" class="btn-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                </svg>
                <svg v-else class="btn-icon" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
                </svg>
                <span class="btn-text">{{ isPaused ? 'Resume' : 'Pause' }}</span>
            </button>
            <button @click="emit('undo')" :disabled="isPaused || isGameOver" title="Undo">
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 10h10a5 5 0 0 1 5 5v2M3 10l5-5M3 10l5 5"/>
                </svg>
                <span class="btn-text">Undo</span>
            </button>
            <button @click="emit('share')" title="Share">
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
                    <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
                </svg>
                <span class="btn-text">Share</span>
            </button>
            <button @click="emit('showSolution')" class="secondary" title="Show Solution">
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                </svg>
                <span class="btn-text">Solve</span>
            </button>
        </div>

        <div class="controls secondary-controls">
            <button
                @click="emit('toggleNotes')"
                :class="{ 'active': isNoteMode }"
                title="Toggle Candidate Mode (Pencil)"
            >
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/>
                </svg>
                <span class="btn-text">{{ isNoteMode ? 'Notes ON' : 'Notes' }}</span>
            </button>
            <button
                @click="emit('generateCandidates')"
                :class="['fill-notes-btn', { 'hidden-btn': candidatesPopulated }]"
                title="Populate Candidates"
            >
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
                    <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
                </svg>
                <span class="btn-text">Fill Notes</span>
            </button>
            <button
                @click="emit('getHint')"
                :disabled="isPaused || isGameOver"
                :class="['hint-btn', { 'hidden-btn': !candidatesPopulated }]"
                title="Get Hint"
            >
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/>
                </svg>
                <span class="btn-text">Hint</span>
            </button>
            <button
                @click="emit('generateCandidates')"
                :class="['refresh-btn', { 'hidden-btn': !candidatesPopulated }]"
                title="Refresh Candidates"
            >
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M23 4v6h-6M1 20v-6h6"/>
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                </svg>
                <span class="btn-text">Refresh</span>
            </button>
        </div>
    </div>
</template>

<style scoped>
.controls-wrapper {
    width: 100%;
    max-width: 450px;
}

.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-bottom: 10px;
}

.controls button {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px 12px;
    border: 2px solid #4a3f80;
    background: linear-gradient(145deg, #5a4990, #3a2f70);
    color: white;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
    min-width: 60px;
}

.controls button:hover:not(:disabled) {
    background: linear-gradient(145deg, #6a59a0, #4a3f80);
    transform: translateY(-1px);
}

.controls button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.controls button.active {
    background: linear-gradient(145deg, #7a69b0, #5a4990);
    border-color: #8a79c0;
}

.controls button.secondary {
    background: linear-gradient(145deg, #6a4a5a, #4a2a3a);
    border-color: #7a5a6a;
}

.btn-icon {
    width: 20px;
    height: 20px;
}

.btn-text {
    font-size: 0.75rem;
    font-weight: 600;
}

.secondary-controls {
    margin-top: 5px;
}

.hidden-btn {
    display: none !important;
}

@media (max-width: 480px) {
    .controls button {
        padding: 6px 8px;
        min-width: 50px;
    }

    .btn-icon {
        width: 18px;
        height: 18px;
    }

    .btn-text {
        font-size: 0.65rem;
    }
}
</style>
