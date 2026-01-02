* The Gemini prompt for Killer Sudoku specifically instructs the model to recognize dashed lines as cage boundaries and extract cage sums from the top-left corner of cells.
* Unit tests verifying Gemini JSON response parsing, including markdown stripping and error handling, are located in backend/sudoku/gemini_test.go.
* The backend JSON parsing logic for Gemini responses is isolated in ParseGeminiResponse (backend/sudoku/gemini.go) to facilitate unit testing.
* The backend utilizes the gemini-3.0-pro-image model variant for the image extraction functionality, as specified by user requirements.
* The Sudoku image extraction logic uses prompt engineering to explicitly ignore small candidate numbers (pencil marks) when parsing the grid.
* The SudokuBoard.vue component provides a manual cage creation interface in custom mode, featuring a toggleable selection mode (isCageSelectionMode), cage sum input, and deletion capabilities.
* The ExtractSudokuFromImage function and /api/upload endpoint accept a gameType parameter to dynamically adjust the Gemini prompt (e.g., for Killer Sudoku cages) and return a complete Puzzle JSON object.
* The SudokuBoard.vue component accepts initialBoardUpload and initialCages props to pre-fill the grid and cage data from uploaded images or custom input.
* The backend exposes a POST /api/upload endpoint that uses the Google Gemini API (model gemini-3.0-pro, requiring GOOGLE_API_KEY) to extract Sudoku grids from uploaded images.
* The LandingPage.vue component allows users to upload images (JPEG, PNG, WebP) to generate a custom puzzle from an image.
* Navigation between views in App.vue (Landing, Game, Stats, Solved) is managed using a single currentView state variable ('landing', 'game', 'stats', 'solved') rather than boolean flags.
* The backend tracks global puzzle completions in-memory (reset on restart) via the POST /api/complete endpoint and exposes counts via GET /api/stats.
* The Vite configuration (vite.config.ts) includes a server proxy redirecting /api requests to http://localhost:8080.
* Custom UI buttons (like the number pad) use @mousedown.prevent to avoid blurring the active cell focus.
* On mobile devices, Sudoku cell inputs are set to readonly to suppress the native keyboard, utilizing a custom on-screen number pad instead.
* Mobile device detection in SudokuBoard.vue is determined by window.innerWidth < 768.
* The component frontend/src/components/LandingPage.vue handles initial game configuration (difficulty, size, game type) and emits events to start the game.
* Killer Sudoku puzzle generation enforces a unique solution constraint by validating the generated cages with a dedicated backtracking solver (solveCountKiller).
* Puzzle validation logic in SudokuBoard verifies compliance with game rules (row/column/box/cage uniqueness and cage sums) rather than checking against a pre-generated solution array.
* The frontend project uses Playwright for end-to-end testing, while relying on vue-tsc and vite build for build verification.
* Killer Sudoku difficulty levels determine the initial board state: 'Hard' starts with an empty board (0 clues), while 'Easy' and 'Medium' utilize standard Sudoku hole counts to provide initial clues.
* Killer Sudoku cages are visualized using ::after pseudo-elements with absolute inset positioning to render dotted borders without conflicting with the existing grid lines.
* The SudokuBoard component accepts a gameType prop to conditionally render variant-specific UI elements, such as cages for 'killer' mode.
* Backend Go tests must be executed from the backend/ directory (e.g., cd backend && go test ./sudoku/...) to correctly resolve the module root.
* Code changes must include permanent, fully implemented tests (e.g., *_test.go files) rather than temporary verification scripts.
* The backend implements Killer Sudoku generation via GenerateKiller, which creates Cage structures (sums and cell coordinates) over a valid solution.
* The /api/puzzle endpoint accepts a gameType query parameter (e.g., "killer") to specify the puzzle variant.
* The backend is implemented in Go, with the core puzzle logic residing in backend/sudoku.
* The frontend development server can be started by running npm run dev in the frontend directory.
* Game state is persisted to localStorage using keys formatted as sudoku_game_state_${size}.
* The project contains a frontend directory which is a Vue 3 application using Vite and TypeScript.
* The frontend build process requires running npm install followed by npm run build in the frontend directory.
* The primary game component is located at frontend/src/components/SudokuBoard.vue.:wq

