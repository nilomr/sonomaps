<script lang="ts">
	import { onMount } from 'svelte';
	import { AudioSource } from '$lib/audio/audio-source.js';
	import { MelFeatureExtractor } from '$lib/dsp/mel.js';
	import { DirectFeatureEmbedding, FEATURES, getFeature } from '$lib/embedding/pca.js';
	import { EmbeddingSmoother } from '$lib/embedding/smoother.js';
	import { PointCloudRenderer } from '$lib/render/point-cloud.js';
	import { MelCloudRenderer } from '$lib/render/mel-cloud.js';
	import { SpectrogramRenderer } from '$lib/render/spectrogram.js';

	// ── DOM refs ───────────────────────────────────────────
	let melCanvas: HTMLCanvasElement;
	let spectroCanvas: HTMLCanvasElement;
	let pointCanvas: HTMLCanvasElement;
	let radarCanvas: HTMLCanvasElement;

	// ── Rendering objects (not reactive) ──────────────────
	let melCloud: MelCloudRenderer | null = null;
	let spectrogram: SpectrogramRenderer | null = null;
	let pointCloud: PointCloudRenderer | null = null;
	let radarCtx: CanvasRenderingContext2D | null = null;

	// ── Audio pipeline (not reactive) ─────────────────────
	let audioSource: AudioSource | null = null;
	let melExtractor: MelFeatureExtractor | null = null;
	let embedding: DirectFeatureEmbedding | null = null;
	let smoother: EmbeddingSmoother | null = null;

	let processing = false;
	let animFrameId = 0;
	let sampleIntervalId = 0;

	// ── UI state ──────────────────────────────────────────
	let isRunning = $state(false);
	let inputMode = $state<'mic' | 'file'>('mic');
	let fps = $state(0);
	let selectedFile = $state<File | null>(null);
	let status = $state('READY');
	let volume = $state(1.0);
	let noiseThreshold = $state(1.8);

	// ── Fixed parameters ─────────────────────────────────
	const smoothing = 0.35;
	const outputDim = 3;

	// ── Axis mapping (user-configurable) ─────────────────
	let axisX = $state('centroid');
	let axisY = $state('bandwidth');
	let axisZ = $state('zcr');

	function onAxesChange() {
		if (embedding) {
			embedding.setAxes([axisX, axisY, axisZ]);
		}
		if (smoother) {
			smoother.reset();
		}
		pointCloud?.clear();
	}

	// ── Feature display state (updated ~10Hz) ─────────────
	let featCentroid = $state('—');
	let featRms = $state('—');
	let featZcr = $state('—');
	let featFlat = $state('—');
	let featBw = $state('—');
	let featRol = $state('—');
	let barCentroid = $state(0);
	let barRms = $state(0);
	let barZcr = $state(0);
	let barFlat = $state(0);
	let barBw = $state(0);
	let barRol = $state(0);

	// ── Radar trail history ──────────────────────────────
	const RADAR_TRAIL_LENGTH = 15;
	let radarSnapshots: number[][] = [];

	const FFT_SIZE = 2048;
	const NUM_MEL_BANDS = 80;
	const MAX_POINTS = 4000;
	const SAMPLE_INTERVAL_MS = 4;
	const MIN_GATE = 0.0005;

	// ── Pre-allocated buffers ─────────────────────────────
	const embeddingBuf = new Float32Array(3);
	const pointData = new Float32Array(5);

	// ── Online normalization ──────────────────────────────
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

	// ── Per-feature online normalizer for radar ──────────
	class FeatureNormalizer {
		private mean = 0;
		private variance = 1;
		private count = 0;
		private readonly decay: number;

		constructor(decay = 0.98) { this.decay = decay; }

		update(raw: number): number {
			this.count++;
			if (this.count < 8) {
				this.mean = raw;
				return 0.5;
			}
			const d = this.decay;
			this.mean = d * this.mean + (1 - d) * raw;
			const diff = raw - this.mean;
			this.variance = d * this.variance + (1 - d) * diff * diff;
			const std = Math.sqrt(this.variance);
			if (std < 1e-8) return 0.5;
			return Math.max(0, Math.min(1, 0.5 + diff / (4 * std)));
		}

		reset(): void { this.mean = 0; this.variance = 1; this.count = 0; }
	}

	const radarNorm = {
		centroid: new FeatureNormalizer(),
		rms: new FeatureNormalizer(),
		zcr: new FeatureNormalizer(),
		flatness: new FeatureNormalizer(),
		bandwidth: new FeatureNormalizer(),
		rolloff: new FeatureNormalizer()
	};

	// ── Adaptive noise gate ──────────────────────────────
	let noiseFloorEma = 0;
	let noiseFloorInitialized = false;
	const NOISE_FLOOR_DECAY = 0.998;
	const NOISE_FLOOR_UP = 0.95;

	let frameCount = 0;
	let lastFpsTime = 0;
	let featCounter = 0;

	// ── Formatting helpers ───────────────────────────────
	function fmtHz(hz: number): string {
		if (hz >= 10000) return (hz / 1000).toFixed(0) + 'k';
		if (hz >= 1000) return (hz / 1000).toFixed(1) + 'k';
		return Math.round(hz).toString();
	}

	// ── High-frequency audio sampling (~250Hz) ───────────
	function sampleAudio(): void {
		if (!processing || !audioSource || !melExtractor || !embedding || !smoother) return;

		audioSource.read();
		melExtractor.compute(audioSource.freqData, audioSource.timeData);

		const rms = melExtractor.rms;

		if (!noiseFloorInitialized) {
			noiseFloorEma = rms;
			noiseFloorInitialized = true;
		} else {
			if (rms < noiseFloorEma * 2.0) {
				noiseFloorEma = NOISE_FLOOR_DECAY * noiseFloorEma + (1 - NOISE_FLOOR_DECAY) * rms;
			} else if (rms < noiseFloorEma * 5.0) {
				noiseFloorEma = NOISE_FLOOR_UP * noiseFloorEma + (1 - NOISE_FLOOR_UP) * rms;
			}
		}

		const gate = Math.max(noiseFloorEma * noiseThreshold, MIN_GATE);
		if (rms < gate) return;

		embedding.projectFromExtractor(melExtractor, embeddingBuf);
		smoother.smooth(embeddingBuf);

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
		pointData[2] = embeddingBuf[2];
		pointData[3] = normEnergy;
		pointData[4] = normCentroid;
		pointCloud!.addPoints(pointData, 1);
	}

	// ── HiDPI canvas helper ─────────────────────────────
	function resizeHiDPI(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): void {
		const dpr = window.devicePixelRatio || 1;
		const w = Math.round(canvas.clientWidth);
		const h = Math.round(canvas.clientHeight);
		const pw = Math.round(w * dpr);
		const ph = Math.round(h * dpr);
		if (w > 0 && h > 0 && (canvas.width !== pw || canvas.height !== ph)) {
			canvas.width = pw;
			canvas.height = ph;
			ctx.resetTransform();
			ctx.scale(dpr, dpr);
		}
	}

	// ── Radar chart rendering ───────────────────────────
	function renderRadar(): void {
		if (!radarCtx || !radarCanvas) return;
		resizeHiDPI(radarCanvas, radarCtx);

		const w = radarCanvas.clientWidth;
		const h = radarCanvas.clientHeight;
		if (w === 0 || h === 0) return;

		radarCtx.clearRect(0, 0, w, h);

		const cx = w / 2;
		const cy = h / 2;
		const radius = Math.min(w, h) * 0.30;

		const features = [
			{ label: 'Centroid', val: featCentroid, bar: barCentroid / 100 },
			{ label: 'Energy', val: featRms, bar: barRms / 100 },
			{ label: 'Crossings', val: featZcr, bar: barZcr / 100 },
			{ label: 'Tonality', val: featFlat, bar: barFlat / 100 },
			{ label: 'Spread', val: featBw, bar: barBw / 100 },
			{ label: 'Rolloff', val: featRol, bar: barRol / 100 }
		];

		const n = features.length;
		const step = (Math.PI * 2) / n;
		const start = -Math.PI / 2;

		// Concentric rings
		for (let ring = 1; ring <= 3; ring++) {
			const r = radius * (ring / 3);
			radarCtx.strokeStyle = ring === 3 ? 'rgba(42,42,50,0.14)' : 'rgba(42,42,50,0.06)';
			radarCtx.lineWidth = 0.5;
			radarCtx.beginPath();
			for (let i = 0; i <= n; i++) {
				const a = start + (i % n) * step;
				const x = cx + Math.cos(a) * r;
				const y = cy + Math.sin(a) * r;
				if (i === 0) radarCtx.moveTo(x, y);
				else radarCtx.lineTo(x, y);
			}
			radarCtx.closePath();
			radarCtx.stroke();
		}

		// Axis lines
		radarCtx.strokeStyle = 'rgba(42,42,50,0.1)';
		radarCtx.lineWidth = 0.5;
		for (let i = 0; i < n; i++) {
			const a = start + i * step;
			radarCtx.beginPath();
			radarCtx.moveTo(cx, cy);
			radarCtx.lineTo(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius);
			radarCtx.stroke();
		}

		// ── Trail polygons (oldest first) ───────────────
		for (let t = 0; t < radarSnapshots.length; t++) {
			const snap = radarSnapshots[t];
			const trailAge = (radarSnapshots.length - t) / (radarSnapshots.length + 1);
			const trailAlpha = Math.max(0.008, (1 - trailAge * trailAge) * 0.08);

			radarCtx.fillStyle = `rgba(42,42,50,${trailAlpha.toFixed(4)})`;
			radarCtx.beginPath();
			for (let i = 0; i <= n; i++) {
				const idx = i % n;
				const a = start + idx * step;
				const r = radius * Math.max(0.03, snap[idx]);
				const x = cx + Math.cos(a) * r;
				const y = cy + Math.sin(a) * r;
				if (i === 0) radarCtx.moveTo(x, y);
				else radarCtx.lineTo(x, y);
			}
			radarCtx.closePath();
			radarCtx.fill();
		}

		// ── Current polygon — fill ──────────────────────
		radarCtx.fillStyle = 'rgba(42,42,50,0.045)';
		radarCtx.beginPath();
		for (let i = 0; i <= n; i++) {
			const idx = i % n;
			const a = start + idx * step;
			const r = radius * Math.max(0.03, features[idx].bar);
			const x = cx + Math.cos(a) * r;
			const y = cy + Math.sin(a) * r;
			if (i === 0) radarCtx.moveTo(x, y);
			else radarCtx.lineTo(x, y);
		}
		radarCtx.closePath();
		radarCtx.fill();

		// ── Current polygon — stroke ────────────────────
		radarCtx.strokeStyle = 'rgba(42,42,50,0.38)';
		radarCtx.lineWidth = 1.5;
		radarCtx.beginPath();
		for (let i = 0; i <= n; i++) {
			const idx = i % n;
			const a = start + idx * step;
			const r = radius * Math.max(0.03, features[idx].bar);
			const x = cx + Math.cos(a) * r;
			const y = cy + Math.sin(a) * r;
			if (i === 0) radarCtx.moveTo(x, y);
			else radarCtx.lineTo(x, y);
		}
		radarCtx.closePath();
		radarCtx.stroke();

		// Vertex dots
		for (let i = 0; i < n; i++) {
			const a = start + i * step;
			const r = radius * Math.max(0.03, features[i].bar);
			const x = cx + Math.cos(a) * r;
			const y = cy + Math.sin(a) * r;
			radarCtx.fillStyle = 'rgba(42,42,50,0.5)';
			radarCtx.beginPath();
			radarCtx.arc(x, y, 2.5, 0, Math.PI * 2);
			radarCtx.fill();
		}

		// Labels
		for (let i = 0; i < n; i++) {
			const a = start + i * step;
			const lr = radius + 22;
			const x = cx + Math.cos(a) * lr;
			const y = cy + Math.sin(a) * lr;

			radarCtx.fillStyle = 'rgba(42,42,50,0.38)';
			radarCtx.font = '500 9px "JetBrains Mono"';
			radarCtx.textAlign = 'center';
			radarCtx.textBaseline = 'middle';
			radarCtx.fillText(features[i].label, x, y - 6);

			radarCtx.fillStyle = 'rgba(42,42,50,0.58)';
			radarCtx.font = '400 10px "JetBrains Mono"';
			radarCtx.fillText(features[i].val, x, y + 7);
		}
	}

	// ── Lifecycle ─────────────────────────────────────────
	onMount(() => {
		radarCtx = radarCanvas.getContext('2d');

		melCloud = new MelCloudRenderer(melCanvas, {
			maxFrames: 250,
			numBands: NUM_MEL_BANDS
		});

		spectrogram = new SpectrogramRenderer(spectroCanvas, NUM_MEL_BANDS);

		pointCloud = new PointCloudRenderer(pointCanvas, {
			maxPoints: MAX_POINTS,
			outputDim,
			pointSize: 1.8
		});

		const onResize = () => {
			melCloud?.resize();
			spectrogram?.resize();
			pointCloud?.resize();
		};
		window.addEventListener('resize', onResize);

		function loop() {
			if (processing && melExtractor) {
				melCloud!.addFrame(melExtractor.logMelEnergies);
				spectrogram?.addColumn(melExtractor.logMelEnergies);
			}

			melCloud?.render();
			pointCloud?.render();
			renderRadar();

			featCounter++;
			if (featCounter % 6 === 0 && processing && melExtractor) {
				const m = melExtractor;
				featCentroid = fmtHz(m.centroid);
				barCentroid = radarNorm.centroid.update(Math.log1p(m.centroid)) * 100;
				featRms = m.rms.toFixed(3);
				barRms = radarNorm.rms.update(Math.log1p(m.rms * 1000)) * 100;
				featZcr = m.zcr.toFixed(3);
				barZcr = radarNorm.zcr.update(m.zcr) * 100;
				featFlat = m.flatness.toFixed(3);
				barFlat = radarNorm.flatness.update(Math.log1p(m.flatness * 1000)) * 100;
				featBw = fmtHz(m.bandwidth);
				barBw = radarNorm.bandwidth.update(Math.log1p(m.bandwidth)) * 100;
				featRol = fmtHz(m.rolloff);
				barRol = radarNorm.rolloff.update(Math.log1p(m.rolloff)) * 100;

				// Add radar trail snapshot
				radarSnapshots.push([
					barCentroid / 100, barRms / 100, barZcr / 100,
					barFlat / 100, barBw / 100, barRol / 100
				]);
				if (radarSnapshots.length > RADAR_TRAIL_LENGTH) radarSnapshots.shift();
			}

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
			melCloud?.dispose();
			pointCloud?.dispose();
		};
	});

	async function start() {
		if (isRunning) return;
		try {
			status = 'INIT';
			audioSource = new AudioSource(FFT_SIZE);

			if (inputMode === 'mic') {
				await audioSource.startMic();
			} else if (selectedFile) {
				await audioSource.startFile(selectedFile);
			} else {
				status = 'NO FILE';
				return;
			}

			audioSource.setVolume(volume);

			melExtractor = new MelFeatureExtractor({
				sampleRate: audioSource.sampleRate,
				fftSize: FFT_SIZE,
				numMelBands: NUM_MEL_BANDS,
				numMfccs: 13
			});

			embedding = new DirectFeatureEmbedding([axisX, axisY, axisZ]);
			smoother = new EmbeddingSmoother(outputDim, smoothing);

			energyEma = 0; energyVar = 0.01;
			centroidEma = 0; centroidVar = 1;
			warmupCount = 0;
			noiseFloorEma = 0; noiseFloorInitialized = false;
			radarSnapshots = [];
			for (const n of Object.values(radarNorm)) n.reset();

			melCloud?.clear();
			pointCloud?.clear();

			processing = true;
			sampleIntervalId = window.setInterval(sampleAudio, SAMPLE_INTERVAL_MS);
			isRunning = true;
			status = inputMode === 'mic' ? 'LISTENING' : 'PLAYING';
		} catch (err) {
			status = `ERR: ${err instanceof Error ? err.message : String(err)}`;
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
		status = 'READY';
		featCentroid = '—'; featRms = '—'; featZcr = '—';
		featFlat = '—'; featBw = '—'; featRol = '—';
		barCentroid = 0; barRms = 0; barZcr = 0;
		barFlat = 0; barBw = 0; barRol = 0;
	}

	function toggle() {
		if (isRunning) stop();
		else start();
	}

	function onFileChange(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files && input.files.length > 0) {
			selectedFile = input.files[0];
		}
	}

	function onVolumeInput() {
		audioSource?.setVolume(volume);
	}
</script>

<svelte:head>
	<title>SonoMaps</title>
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
	<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet" />
</svelte:head>

<main>
	<!-- ─── Mel cloud (left) ─── -->
	<section class="panel mel-cloud">
		<div class="mel-3d">
			<canvas bind:this={melCanvas}></canvas>
		</div>
		<div class="mel-2d">
			<canvas bind:this={spectroCanvas}></canvas>
		</div>
		<span class="panel-label">MEL SPECTROGRAM</span>
		<div class="corner tl"></div>
		<div class="corner tr"></div>
		<div class="corner bl"></div>
		<div class="corner br"></div>
	</section>

	<!-- ─── Trajectory (center) ─── -->
	<section class="panel trajectory">
		<canvas bind:this={pointCanvas}></canvas>
		<span class="panel-label">TRAJECTORY</span>
		<div class="corner tl"></div>
		<div class="corner tr"></div>
		<div class="corner bl"></div>
		<div class="corner br"></div>
		<div class="axes-overlay">
			<span class="ax-key">X</span> {getFeature(axisX).axisLabel}
			<span class="ax-sep">/</span>
			<span class="ax-key">Y</span> {getFeature(axisY).axisLabel}
			<span class="ax-sep">/</span>
			<span class="ax-key">Z</span> {getFeature(axisZ).axisLabel}
		</div>
	</section>

	<!-- ─── Radar analysis (right) ─── -->
	<section class="panel analysis">
		<canvas bind:this={radarCanvas} class="fill-canvas"></canvas>
		<span class="panel-label">ANALYSIS</span>
	</section>

	<!-- ─── Controls bar ─── -->
	<section class="panel controls-bar">
		<div class="ctrl-row">
			<div class="ctrl-group">
				<button class="ctrl-btn play" class:active={isRunning} onclick={toggle}
					aria-label={isRunning ? 'Stop' : 'Start'}>
					{#if isRunning}
						<span class="icon-stop"></span>
					{:else}
						<span class="icon-play"></span>
					{/if}
					<span>{isRunning ? 'STOP' : 'START'}</span>
				</button>

				<button class="ctrl-btn" class:active={inputMode === 'mic'} disabled={isRunning}
					onclick={() => (inputMode = 'mic')}>MIC</button>
				<button class="ctrl-btn" class:active={inputMode === 'file'} disabled={isRunning}
					onclick={() => (inputMode = 'file')}>FILE</button>

				{#if inputMode === 'file'}
					<label class="ctrl-btn file-label">
						<span>{selectedFile ? selectedFile.name.slice(0, 14).toUpperCase() : 'CHOOSE FILE'}</span>
						<input type="file" accept="audio/*" onchange={onFileChange}
							disabled={isRunning} class="file-hidden" />
					</label>
				{/if}
			</div>

			<div class="ctrl-group axes-group">
				<div class="axis-inline">
					<span class="ax-tag">X</span>
					<select class="ax-sel" bind:value={axisX} onchange={onAxesChange}>
						{#each FEATURES as f}<option value={f.id}>{f.label}</option>{/each}
					</select>
				</div>
				<div class="axis-inline">
					<span class="ax-tag">Y</span>
					<select class="ax-sel" bind:value={axisY} onchange={onAxesChange}>
						{#each FEATURES as f}<option value={f.id}>{f.label}</option>{/each}
					</select>
				</div>
				<div class="axis-inline">
					<span class="ax-tag">Z</span>
					<select class="ax-sel" bind:value={axisZ} onchange={onAxesChange}>
						{#each FEATURES as f}<option value={f.id}>{f.label}</option>{/each}
					</select>
				</div>
			</div>
		</div>

		<div class="ctrl-row">
			<div class="slider-group">
				<span class="slider-label">VOL</span>
				<input type="range" class="slider" min="0" max="2" step="0.01"
					bind:value={volume} oninput={onVolumeInput} />
			</div>

			<div class="slider-group">
				<span class="slider-label">NOISE THRESHOLD</span>
				<input type="range" class="slider" min="0.5" max="5" step="0.1"
					bind:value={noiseThreshold} />
			</div>

			<div class="ctrl-group status-group">
				<span class="brand-micro">SONOMAPS</span>
				<span class="status-dot" class:active={isRunning}></span>
				<span class="status-text">{status}</span>
				<span class="fps-val">{fps} <span class="fps-unit">FPS</span></span>
			</div>
		</div>
	</section>
</main>

<style>
	/* ── Reset & base ────────────────────────────── */
	:global(body) {
		margin: 0;
		padding: 0;
		background: #f2ede4;
		color: #2a2a32;
		font-family: 'JetBrains Mono', 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
		overflow: hidden;
		-webkit-font-smoothing: antialiased;
	}

	/* ── Grid layout ─────────────────────────────── */
	main {
		width: 100vw;
		height: 100vh;
		display: grid;
		grid-template-columns: 1fr 1fr 0.55fr;
		grid-template-rows: 1fr auto;
		grid-template-areas:
			"mel traj radar"
			"ctrl ctrl ctrl";
		gap: 1px;
		background: rgba(42, 42, 50, 0.12);
	}

	/* ── Panel base ──────────────────────────────── */
	.panel {
		position: relative;
		background: #f2ede4;
		overflow: hidden;
	}

	.panel-label {
		position: absolute;
		top: 14px;
		left: 18px;
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 2.5px;
		color: rgba(42, 42, 50, 0.32);
		pointer-events: none;
		z-index: 2;
	}

	/* ── Panels ──────────────────────────────────── */
	.mel-cloud { grid-area: mel; }
	.trajectory { grid-area: traj; }
	.analysis { grid-area: radar; }
	.controls-bar { grid-area: ctrl; }

	.mel-cloud {
		display: flex;
		flex-direction: column;
	}

	.mel-3d {
		flex: 1;
		min-height: 0;
	}

	.mel-3d canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.mel-2d {
		height: 10%;
		min-height: 28px;
		border-top: 1px solid rgba(42, 42, 50, 0.08);
	}

	.mel-2d canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.trajectory canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.fill-canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	/* ── Corner marks ────────────────────────────── */
	.corner {
		position: absolute;
		width: 22px;
		height: 22px;
		pointer-events: none;
		z-index: 1;
	}
	.corner.tl { top: 10px; left: 10px; border-left: 1px solid rgba(42,42,50,0.16); border-top: 1px solid rgba(42,42,50,0.16); }
	.corner.tr { top: 10px; right: 10px; border-right: 1px solid rgba(42,42,50,0.16); border-top: 1px solid rgba(42,42,50,0.16); }
	.corner.bl { bottom: 10px; left: 10px; border-left: 1px solid rgba(42,42,50,0.16); border-bottom: 1px solid rgba(42,42,50,0.16); }
	.corner.br { bottom: 10px; right: 10px; border-right: 1px solid rgba(42,42,50,0.16); border-bottom: 1px solid rgba(42,42,50,0.16); }

	/* ── Axes overlay ────────────────────────────── */
	.axes-overlay {
		position: absolute;
		bottom: 18px;
		left: 40px;
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 1.5px;
		color: rgba(42, 42, 50, 0.35);
		pointer-events: none;
		z-index: 2;
	}

	.ax-key {
		font-weight: 600;
		color: rgba(42, 42, 50, 0.55);
	}

	.ax-sep {
		margin: 0 5px;
		font-weight: 300;
		color: rgba(42, 42, 50, 0.18);
	}

	/* ── Controls bar ────────────────────────────── */
	.controls-bar {
		display: flex;
		flex-direction: column;
		padding: 10px 20px;
		gap: 8px;
	}

	.ctrl-row {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.ctrl-group {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.ctrl-group + .ctrl-group,
	.ctrl-group + .axes-group {
		margin-left: 20px;
	}

	.axes-group {
		gap: 10px;
	}

	.status-group {
		margin-left: auto;
		gap: 8px;
	}

	/* ── Buttons ──────────────────────────────────── */
	.ctrl-btn {
		height: 30px;
		padding: 0 12px;
		border: 1px solid rgba(42, 42, 50, 0.14);
		border-radius: 4px;
		background: transparent;
		color: rgba(42, 42, 50, 0.45);
		font-family: inherit;
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 2px;
		cursor: pointer;
		transition: all 0.12s;
		display: flex;
		align-items: center;
		gap: 7px;
		white-space: nowrap;
	}

	.ctrl-btn:hover {
		border-color: rgba(42, 42, 50, 0.3);
		color: rgba(42, 42, 50, 0.7);
		background: rgba(42, 42, 50, 0.025);
	}

	.ctrl-btn.active {
		border-color: rgba(42, 42, 50, 0.22);
		color: rgba(42, 42, 50, 0.75);
		background: rgba(42, 42, 50, 0.04);
	}

	.ctrl-btn:disabled {
		opacity: 0.35;
		cursor: not-allowed;
	}

	.ctrl-btn.play.active {
		border-color: rgba(160, 50, 50, 0.3);
		color: rgba(160, 50, 50, 0.65);
		background: rgba(160, 50, 50, 0.03);
	}

	.icon-play {
		width: 0;
		height: 0;
		border-style: solid;
		border-width: 5px 0 5px 8px;
		border-color: transparent transparent transparent rgba(42, 42, 50, 0.5);
		flex-shrink: 0;
	}

	.icon-stop {
		width: 9px;
		height: 9px;
		background: rgba(160, 50, 50, 0.5);
		border-radius: 1.5px;
		flex-shrink: 0;
	}

	.file-label {
		border-style: dashed;
		cursor: pointer;
	}

	.file-hidden {
		position: absolute;
		width: 0;
		height: 0;
		opacity: 0;
		pointer-events: none;
	}

	/* ── Axis selectors ──────────────────────────── */
	.axis-inline {
		display: flex;
		align-items: center;
		gap: 5px;
	}

	.ax-tag {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 1px;
		color: rgba(42, 42, 50, 0.5);
	}

	.ax-sel {
		padding: 3px 6px;
		border: 1px solid rgba(42, 42, 50, 0.1);
		border-radius: 3px;
		background: transparent;
		color: rgba(42, 42, 50, 0.6);
		font-family: inherit;
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 0.3px;
		cursor: pointer;
		outline: none;
		transition: border-color 0.12s;
	}

	.ax-sel:hover { border-color: rgba(42, 42, 50, 0.22); }
	.ax-sel:focus { border-color: rgba(42, 42, 50, 0.3); }

	/* ── Sliders ─────────────────────────────────── */
	.slider-group {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.slider-group + .slider-group {
		margin-left: 24px;
	}

	.slider-label {
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 2px;
		color: rgba(42, 42, 50, 0.38);
		white-space: nowrap;
	}

	.slider {
		-webkit-appearance: none;
		appearance: none;
		width: 130px;
		height: 2px;
		background: rgba(42, 42, 50, 0.15);
		border-radius: 1px;
		outline: none;
		cursor: pointer;
	}

	.slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: #f2ede4;
		border: 1.5px solid rgba(42, 42, 50, 0.35);
		cursor: pointer;
		transition: border-color 0.12s;
	}

	.slider::-webkit-slider-thumb:hover {
		border-color: rgba(42, 42, 50, 0.55);
	}

	.slider::-moz-range-thumb {
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: #f2ede4;
		border: 1.5px solid rgba(42, 42, 50, 0.35);
		cursor: pointer;
	}

	/* ── Status elements ─────────────────────────── */
	.brand-micro {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 3px;
		color: rgba(42, 42, 50, 0.3);
	}

	.status-dot {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: rgba(42, 42, 50, 0.18);
		transition: background 0.2s;
		flex-shrink: 0;
	}

	.status-dot.active {
		background: rgba(160, 50, 50, 0.6);
	}

	.status-text {
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 1.5px;
		color: rgba(42, 42, 50, 0.38);
	}

	.fps-val {
		font-size: 11px;
		font-weight: 400;
		color: rgba(42, 42, 50, 0.4);
		font-variant-numeric: tabular-nums;
		margin-left: 4px;
	}

	.fps-unit {
		font-size: 9px;
		font-weight: 300;
		letter-spacing: 1px;
		color: rgba(42, 42, 50, 0.25);
	}
</style>
