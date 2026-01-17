<script setup lang="ts">
defineProps<{
    size: number
}>()

const emit = defineEmits<{
    input: [num: number]
    clear: []
}>()

const onInput = (n: number) => {
    emit('input', n)
}

const onClear = () => {
    emit('clear')
}
</script>

<template>
    <div class="number-pad">
        <button
            v-for="n in size"
            :key="n"
            class="pad-btn"
            @mousedown.prevent
            @click="onInput(n)"
        >
            {{ n }}
        </button>
        <button class="pad-btn clear-btn" @mousedown.prevent @click="onClear">X</button>
    </div>
</template>

<style scoped>
.number-pad {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 15px;
    width: 100%;
    max-width: 300px;
}

.pad-btn {
    padding: 15px;
    font-size: 1.5rem;
    font-weight: bold;
    border: 2px solid #4a3f80;
    background: linear-gradient(145deg, #5a4990, #3a2f70);
    color: white;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
    touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
}

.pad-btn:active {
    transform: scale(0.95);
    background: linear-gradient(145deg, #4a3980, #2a1f60);
}

.clear-btn {
    background: linear-gradient(145deg, #8a4040, #6a2020);
    border-color: #8a4040;
}

.clear-btn:active {
    background: linear-gradient(145deg, #7a3030, #5a1010);
}
</style>
