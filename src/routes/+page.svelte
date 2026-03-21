<script lang="ts">
	import { onMount } from 'svelte';
	import { AudioSource } from '$lib/audio/audio-source.js';
	import { MelFeatureExtractor } from '$lib/dsp/mel.js';
	import { DirectFeatureEmbedding } from '$lib/embedding/pca.js';
	import { EmbeddingSmoother } from '$lib/embedding/smoother.js';
	import { PointCloudRenderer } from '$lib/render/point-cloud.js';
	import { SpectrogramRenderer } from '$lib/render/spectrogram.js';

	// ── DOM refs ───────────────────────────────────────────
	let pointCanvas: HTMLCanvasElement;
	let spectroCanvas: HTMLCanvasElement;

	// ── Rendering objects (not reactive) ──────────────────
	let pointCloud: PointCloudRenderer | null = null;
	let spectrogram: SpectrogramRenderer | null = null;

	// ── Audio pipeline (not reactive) ─────────────────────
	let audioSource: AudioSource | null = null;
	let melExtractor: MelFeatureExtractor | null = null;
	let embedding: DirectFeatureEmbedding | null = null;
	let smoother: EmbeddingSmoother | null = null;

	// Plain boolean — NOT $state, zero reactivity in hot loop
	let processing = false;

	let animFrameId = 0;
	let sampleIntervalId = 0;

	// ── UI state (reactive, only touched by user interaction) ──
	let isRunning = $state(false);
	let inputMode = $state<'mic' | 'file'>('mic');
	let smoothing = $state(0.35);
	let outputDim = $state<2 | 3>(3);
	let fps = $state(0);
	let selectedFile = $state<File | null>(null);
	let status = $state('Ready');

	const FFT_SIZE = 2048;
	const NUM_MEL_BANDS = 80;
	const MAX_POINTS = 4000;
	const SAMPLE_INTERVAL_MS = 4; // ~250 samples/sec

	// ── Pre-allocated buffers (ZERO allocation in hot path) ──
	const embeddingBuf = new Float32Array(3);
	const pointData = new Float32Array(5);

	// ── Online normalization for rendering metadata ──────
	let energyEma = 0;
	let energyVar = 0.01;
	let centroidEma = 0;
	let centroidVar = 1;
	let warmupCount = 0;
	const RENDER_DECAY = 0.95;

	function normalizeForRendering(raw: number, ema: number, variance: number): number {
		const std = Math.sqrt(variance);
		if (std < 1e-8) return 0.5;
		return Math.max(0, Math.min(1, 0.5 + (raw - ema) / (4 * std)));
	}

	// ── Adaptive noise gate ──────────────────────────────
	// Tracks the noise floor with a very slow EMA biased toward quiet frames.
	// Gates at GATE_MULT × noise floor. No user-facing slider needed.
	let noiseFloorEma = 0;
	let noiseFloorInitialized = false;
	const NOISE_FLOOR_DECAY = 0.998;  // very slow adaptation
	const NOISE_FLOOR_UP = 0.95;      // rises quickly if floor increases
	const GATE_MULT = 1.8;            // gate threshold = 1.8× noise floor
	const MIN_GATE = 0.0005;          // absolute minimum gate

	// FPS tracking
	let frameCount = 0;
	let lastFpsTime = 0;

	// ── High-frequency audio sampling ────────────────────
	// Runs at ~250Hz via setInterval, independent of rAF.
	// This gives smooth continuous trails instead of 60fps dots.
	function sampleAudio(): void {
		if (!processing || !audioSource || !melExtractor || !embedding || !smoother) return;

		// Read current audio snapshot
		audioSource.read();
		melExtractor.compute(audioSource.freqData, audioSource.timeData);

		const rms = melExtractor.rms;

		// ── Adaptive noise gate ─────────────────────────
		if (!noiseFloorInitialized) {
			noiseFloorEma = rms;
			noiseFloorInitialized = true;
		} else {
			// Update noise floor: slow decay when signal is near/below floor,
			// faster rise if the ambient level increases
			if (rms < noiseFloorEma * 2.0) {
				noiseFloorEma = NOISE_FLOOR_DECAY * noiseFloorEma + (1 - NOISE_FLOOR_DECAY) * rms;
			} else if (rms > noiseFloorEma * 5.0) {
				// Don't let very loud transients pull the floor up
			} else {
				noiseFloorEma = NOISE_FLOOR_UP * noiseFloorEma + (1 - NOISE_FLOOR_UP) * rms;
			}
		}

		const gate = Math.max(noiseFloorEma * GATE_MULT, MIN_GATE);
		if (rms < gate) return;

		// ── Embedding ───────────────────────────────────
		embedding.projectFromExtractor(melExtractor, embeddingBuf);
		smoother.smooth(embeddingBuf);

		// ── Rendering metadata (online normalization) ───
		const logRms = Math.log1p(rms * 1000);
		const logCentroid = Math.log1p(melExtractor.centroid);

		warmupCount++;
		if (warmupCount < 10) {
			energyEma = logRms;
			centroidEma = logCentroid;
		} else {
			energyEma = RENDER_DECAY * energyEma + (1 - RENDER_DECAY) * logRms;
			const eDiff = logRms - energyEma;
			energyVar = RENDER_DECAY * energyVar + (1 - RENDER_DECAY) * eDiff * eDiff;
			centroidEma = RENDER_DECAY * centroidEma + (1 - RENDER_DECAY) * logCentroid;
			const cDiff = logCentroid - centroidEma;
			centroidVar = RENDER_DECAY * centroidVar + (1 - RENDER_DECAY) * cDiff * cDiff;
		}

		const normEnergy = normalizeForRendering(logRms, energyEma, energyVar);
		const normCentroid = normalizeForRendering(logCentroid, centroidEma, centroidVar);

		pointData[0] = embeddingBuf[0];
		pointData[1] = embeddingBuf[1];
		pointData[2] = outputDim >= 3 ? embeddingBuf[2] : 0;
		pointData[3] = normEnergy;
		pointData[4] = normCentroid;
		pointCloud!.addPoints(pointData, 1);
	}

	// ── Lifecycle ─────────────────────────────────────────
	onMount(() => {
		pointCloud = new PointCloudRenderer(pointCanvas, {
			maxPoints: MAX_POINTS,
			outputDim,
			pointSize: 1.8
		});

		spectrogram = new SpectrogramRenderer(spectroCanvas, NUM_MEL_BANDS);

		const onResize = () => {
			pointCloud?.resize();
			spectrogram?.resize();
		};
		window.addEventListener('resize', onResize);

		// ── Render loop (rAF) — only rendering, no audio processing ──
		function loop() {
			// Feed spectrogram from latest mel data (once per display frame)
			if (processing && melExtractor) {
				spectrogram!.addColumn(melExtractor.logMelEnergies);
			}

			// Render point cloud (ages points, draws scene)
			pointCloud?.render();

			// FPS (update Svelte state only once per second)
			frameCount++;
			const now = performance.now();
			if (now - lastFpsTime >= 1000) {
				fps = frameCount;
				frameCount = 0;
				lastFpsTime = now;
			}

			animFrameId = requestAnimationFrame(loop);
		}

		loop();

		return () => {
			cancelAnimationFrame(animFrameId);
			clearInterval(sampleIntervalId);
			window.removeEventListener('resize', onResize);
			stop();
			pointCloud?.dispose();
		};
	});

	// ── Start / Stop ─────────────────────────────────────
	async function start() {
		if (isRunning) return;

		try {
			status = 'Initialising\u2026';

			audioSource = new AudioSource(FFT_SIZE);

			if (inputMode === 'mic') {
				await audioSource.startMic();
			} else if (selectedFile) {
				await audioSource.startFile(selectedFile);
			} else {
				status = 'Select an audio file first';
				return;
			}

			melExtractor = new MelFeatureExtractor({
				sampleRate: audioSource.sampleRate,
				fftSize: FFT_SIZE,
				numMelBands: NUM_MEL_BANDS,
				numMfccs: 13
			});

			embedding = new DirectFeatureEmbedding(outputDim);
			smoother = new EmbeddingSmoother(outputDim, smoothing);

			// Reset rendering normalization
			energyEma = 0;
			energyVar = 0.01;
			centroidEma = 0;
			centroidVar = 1;
			warmupCount = 0;

			// Reset adaptive noise gate
			noiseFloorEma = 0;
			noiseFloorInitialized = false;

			// Start high-frequency audio sampling
			processing = true;
			sampleIntervalId = window.setInterval(sampleAudio, SAMPLE_INTERVAL_MS);

			isRunning = true;
			status = inputMode === 'mic' ? 'Listening' : 'Playing';
		} catch (err) {
			status = `Error: ${err instanceof Error ? err.message : String(err)}`;
		}
	}

	function stop() {
		processing = false;
		clearInterval(sampleIntervalId);
		isRunning = false;
		audioSource?.stop();
		audioSource = null;
		melExtractor = null;
		embedding = null;
		smoother = null;
		status = 'Ready';
	}

	function toggle() {
		if (isRunning) stop();
		else start();
	}

	function onFileChange(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files && input.files.length > 0) {
			selectedFile = input.files[0];
			status = `File: ${input.files[0].name}`;
		}
	}

	$effect(() => {
		if (smoother) smoother.alpha = smoothing;
	});
</script>

<svelte:head>
	<title>SonoMaps</title>
</svelte:head>

<main>
	<!-- Embedding point cloud -->
	<div class="embedding-view">
		<canvas bind:this={pointCanvas}></canvas>
		<div class="overlay-top-left">
			<span class="title">SonoMaps</span>
			<span class="subtitle">x: brightness &middot; y: energy{outputDim === 3 ? ' &middot; z: tonality' : ''}</span>
		</div>
	</div>

	<!-- Scrolling mel spectrogram -->
	<div class="spectrogram-view">
		<canvas bind:this={spectroCanvas}></canvas>
		<div class="spectrogram-label">
			<span>Mel Spectrogram</span>
		</div>
	</div>

	<!-- Controls -->
	<div class="controls">
		<div class="controls-inner">
			<button class="btn" class:active={isRunning} onclick={toggle}>
				{isRunning ? 'Stop' : 'Start'}
			</button>

			<div class="control-group" role="group" aria-label="Input source">
				<span class="label">Source</span>
				<div class="toggle-group">
					<button
						class="toggle-btn"
						class:active={inputMode === 'mic'}
						disabled={isRunning}
						onclick={() => (inputMode = 'mic')}
					>Mic</button>
					<button
						class="toggle-btn"
						class:active={inputMode === 'file'}
						disabled={isRunning}
						onclick={() => (inputMode = 'file')}
					>File</button>
				</div>
			</div>

			{#if inputMode === 'file'}
				<div class="control-group">
					<input
						type="file"
						accept="audio/*"
						onchange={onFileChange}
						disabled={isRunning}
						class="file-input"
					/>
				</div>
			{/if}

			<div class="control-group">
				<label class="label" for="smoothing-slider">
					Smoothing {smoothing.toFixed(2)}
				</label>
				<input
					id="smoothing-slider"
					type="range"
					min="0.02"
					max="0.6"
					step="0.01"
					bind:value={smoothing}
					class="slider"
				/>
			</div>

			<div class="control-group" role="group" aria-label="Dimensions">
				<span class="label">Dim</span>
				<div class="toggle-group">
					<button
						class="toggle-btn"
						class:active={outputDim === 2}
						disabled={isRunning}
						onclick={() => (outputDim = 2)}
					>2D</button>
					<button
						class="toggle-btn"
						class:active={outputDim === 3}
						disabled={isRunning}
						onclick={() => (outputDim = 3)}
					>3D</button>
				</div>
			</div>

			<div class="status-fps">
				<span class="status">{status}</span>
				<span class="fps">{fps} fps</span>
			</div>
		</div>
	</div>
</main>

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		background: #08080f;
		color: #b0b0c8;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
		overflow: hidden;
	}

	main {
		width: 100vw;
		height: 100vh;
		display: flex;
		flex-direction: column;
	}

	.embedding-view {
		flex: 1;
		position: relative;
		min-height: 0;
	}

	.embedding-view canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.overlay-top-left {
		position: absolute;
		top: 14px;
		left: 18px;
		display: flex;
		flex-direction: column;
		gap: 2px;
		pointer-events: none;
	}

	.title {
		font-size: 14px;
		font-weight: 600;
		color: rgba(160, 170, 210, 0.5);
		letter-spacing: 1.5px;
		text-transform: uppercase;
	}

	.subtitle {
		font-size: 10px;
		color: rgba(120, 130, 160, 0.35);
		letter-spacing: 0.3px;
	}

	.spectrogram-view {
		height: 160px;
		position: relative;
		border-top: 1px solid rgba(80, 80, 120, 0.12);
		background: #06060c;
		flex-shrink: 0;
	}

	.spectrogram-view canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.spectrogram-label {
		position: absolute;
		top: 6px;
		left: 12px;
		pointer-events: none;
	}

	.spectrogram-label span {
		font-size: 9px;
		text-transform: uppercase;
		letter-spacing: 0.6px;
		color: rgba(140, 140, 170, 0.3);
	}

	.controls {
		background: rgba(8, 8, 15, 0.94);
		backdrop-filter: blur(16px);
		border-top: 1px solid rgba(80, 80, 120, 0.12);
		padding: 9px 18px;
		flex-shrink: 0;
	}

	.controls-inner {
		max-width: 960px;
		margin: 0 auto;
		display: flex;
		align-items: center;
		gap: 16px;
		flex-wrap: wrap;
	}

	.btn {
		padding: 6px 20px;
		border: 1px solid rgba(100, 160, 255, 0.3);
		background: rgba(100, 160, 255, 0.05);
		color: rgba(140, 180, 255, 0.75);
		border-radius: 4px;
		font-size: 12px;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.12s;
	}
	.btn:hover {
		background: rgba(100, 160, 255, 0.12);
	}
	.btn.active {
		background: rgba(255, 90, 90, 0.08);
		border-color: rgba(255, 90, 90, 0.35);
		color: rgba(255, 130, 130, 0.85);
	}

	.control-group {
		display: flex;
		flex-direction: column;
		gap: 3px;
	}

	.label {
		font-size: 9px;
		text-transform: uppercase;
		letter-spacing: 0.6px;
		color: rgba(140, 140, 170, 0.4);
	}

	.toggle-group { display: flex; }

	.toggle-btn {
		padding: 3px 9px;
		border: 1px solid rgba(80, 80, 120, 0.2);
		background: transparent;
		color: rgba(140, 140, 170, 0.4);
		font-size: 10px;
		cursor: pointer;
		transition: all 0.12s;
	}
	.toggle-btn:first-child { border-radius: 3px 0 0 3px; }
	.toggle-btn:last-child { border-radius: 0 3px 3px 0; }
	.toggle-btn.active {
		background: rgba(100, 160, 255, 0.08);
		color: rgba(140, 180, 255, 0.75);
		border-color: rgba(100, 160, 255, 0.25);
	}
	.toggle-btn:disabled { opacity: 0.3; cursor: not-allowed; }

	.slider { width: 90px; accent-color: rgba(100, 160, 255, 0.5); }

	.file-input { font-size: 10px; color: rgba(140, 140, 170, 0.45); }
	.file-input::file-selector-button {
		padding: 3px 7px;
		border: 1px solid rgba(80, 80, 120, 0.2);
		background: transparent;
		color: rgba(140, 140, 170, 0.45);
		border-radius: 3px;
		cursor: pointer;
		font-size: 10px;
	}

	.status-fps {
		margin-left: auto;
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 1px;
	}

	.status { font-size: 10px; color: rgba(140, 180, 255, 0.4); }
	.fps { font-size: 10px; font-variant-numeric: tabular-nums; color: rgba(100, 100, 130, 0.35); }
</style>
