<script lang="ts">
	import { onMount } from 'svelte';
	import { AudioSource } from '$lib/audio/audio-source.js';
	import { MelFeatureExtractor } from '$lib/dsp/mel.js';
	import { DirectFeatureEmbedding, FEATURES, getFeature } from '$lib/embedding/pca.js';
	import { EmbeddingSmoother } from '$lib/embedding/smoother.js';
	import { PointCloudRenderer } from '$lib/render/point-cloud.js';
	import { SpectrogramRenderer } from '$lib/render/spectrogram.js';

	// ── DOM refs ───────────────────────────────────────────
	let pointCanvas: HTMLCanvasElement;
	let spectroCanvas: HTMLCanvasElement;
	let waveformCanvas: HTMLCanvasElement;
	let waveCtx: CanvasRenderingContext2D | null = null;

	// ── Rendering objects (not reactive) ──────────────────
	let pointCloud: PointCloudRenderer | null = null;
	let spectrogram: SpectrogramRenderer | null = null;

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

	const FFT_SIZE = 2048;
	const NUM_MEL_BANDS = 80;
	const MAX_POINTS = 4000;
	const SAMPLE_INTERVAL_MS = 4;

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

	// ── Adaptive noise gate ──────────────────────────────
	let noiseFloorEma = 0;
	let noiseFloorInitialized = false;
	const NOISE_FLOOR_DECAY = 0.998;
	const NOISE_FLOOR_UP = 0.95;
	const GATE_MULT = 1.8;
	const MIN_GATE = 0.0005;

	let frameCount = 0;
	let lastFpsTime = 0;
	let featCounter = 0;

	// ── Formatting helpers ───────────────────────────────
	function fmtHz(hz: number): string {
		if (hz >= 10000) return (hz / 1000).toFixed(0) + 'k';
		if (hz >= 1000) return (hz / 1000).toFixed(1) + 'k';
		return Math.round(hz).toString();
	}

	function normLog(value: number, min: number, max: number): number {
		const v = Math.log1p(value);
		const lo = Math.log1p(min);
		const hi = Math.log1p(max);
		return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
	}

	function normLin(value: number, min: number, max: number): number {
		return Math.max(0, Math.min(1, (value - min) / (max - min)));
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

		const gate = Math.max(noiseFloorEma * GATE_MULT, MIN_GATE);
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

	// ── Waveform rendering ───────────────────────────────
	function resizeWaveform(): void {
		if (!waveformCanvas) return;
		const w = Math.round(waveformCanvas.clientWidth);
		const h = Math.round(waveformCanvas.clientHeight);
		if (w > 0 && h > 0 && (waveformCanvas.width !== w || waveformCanvas.height !== h)) {
			waveformCanvas.width = w;
			waveformCanvas.height = h;
		}
	}

	function renderWaveform(): void {
		if (!waveCtx || !waveformCanvas) return;
		resizeWaveform();
		const w = waveformCanvas.width;
		const h = waveformCanvas.height;
		if (w === 0 || h === 0) return;

		waveCtx.fillStyle = '#f2ede4';
		waveCtx.fillRect(0, 0, w, h);

		waveCtx.strokeStyle = 'rgba(42, 42, 50, 0.08)';
		waveCtx.lineWidth = 1;
		waveCtx.beginPath();
		waveCtx.moveTo(0, h / 2);
		waveCtx.lineTo(w, h / 2);
		waveCtx.stroke();

		if (!processing || !audioSource) return;

		const data = audioSource.timeData;
		if (!data || data.length === 0) return;

		waveCtx.strokeStyle = 'rgba(42, 42, 50, 0.45)';
		waveCtx.lineWidth = 1.5;
		waveCtx.beginPath();
		const step = data.length / w;
		const hHalf = h / 2;
		for (let i = 0; i < w; i++) {
			const idx = Math.floor(i * step);
			const y = hHalf - data[idx] * hHalf * 0.85;
			if (i === 0) waveCtx.moveTo(i, y);
			else waveCtx.lineTo(i, y);
		}
		waveCtx.stroke();
	}

	// ── Lifecycle ─────────────────────────────────────────
	onMount(() => {
		waveCtx = waveformCanvas.getContext('2d');
		resizeWaveform();

		pointCloud = new PointCloudRenderer(pointCanvas, {
			maxPoints: MAX_POINTS,
			outputDim,
			pointSize: 1.8
		});

		spectrogram = new SpectrogramRenderer(spectroCanvas, NUM_MEL_BANDS);

		const onResize = () => {
			pointCloud?.resize();
			spectrogram?.resize();
			resizeWaveform();
		};
		window.addEventListener('resize', onResize);

		function loop() {
			if (processing && melExtractor) {
				spectrogram!.addColumn(melExtractor.logMelEnergies);
			}

			renderWaveform();
			pointCloud?.render();

			featCounter++;
			if (featCounter % 6 === 0 && processing && melExtractor) {
				const m = melExtractor;
				featCentroid = fmtHz(m.centroid);
				barCentroid = normLog(m.centroid, 50, 10000) * 100;
				featRms = m.rms.toFixed(3);
				barRms = normLin(m.rms, 0, 0.15) * 100;
				featZcr = m.zcr.toFixed(3);
				barZcr = normLin(m.zcr, 0, 0.5) * 100;
				featFlat = m.flatness.toFixed(3);
				barFlat = m.flatness * 100;
				featBw = fmtHz(m.bandwidth);
				barBw = normLog(m.bandwidth, 50, 8000) * 100;
				featRol = fmtHz(m.rolloff);
				barRol = normLog(m.rolloff, 100, 15000) * 100;
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
</script>

<svelte:head>
	<title>SonoMaps</title>
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
	<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet" />
</svelte:head>

<main>
	<!-- ─── Sidebar ─── -->
	<aside class="sidebar">
		<div class="brand-block">
			<h1 class="brand">SONO<br/>MAPS</h1>
			<div class="status-badge">
				<span class="status-dot" class:active={isRunning}></span>
				<span class="status-text">{status}</span>
			</div>
		</div>

		<div class="divider"></div>

		<div class="sb-section">
			<span class="sb-label">WAVEFORM</span>
			<div class="waveform-wrap">
				<canvas bind:this={waveformCanvas} class="waveform-canvas"></canvas>
			</div>
		</div>

		<div class="divider"></div>

		<div class="sb-section">
			<span class="sb-label">FEATURES</span>
			<div class="features">
				<div class="feat">
					<div class="feat-head">
						<span class="feat-lbl">CENTROID</span>
						<span class="feat-val">{featCentroid}</span>
					</div>
					<div class="feat-track"><div class="feat-fill" style="width:{barCentroid}%"></div></div>
				</div>
				<div class="feat">
					<div class="feat-head">
						<span class="feat-lbl">RMS</span>
						<span class="feat-val">{featRms}</span>
					</div>
					<div class="feat-track"><div class="feat-fill" style="width:{barRms}%"></div></div>
				</div>
				<div class="feat">
					<div class="feat-head">
						<span class="feat-lbl">ZERO CROSS</span>
						<span class="feat-val">{featZcr}</span>
					</div>
					<div class="feat-track"><div class="feat-fill" style="width:{barZcr}%"></div></div>
				</div>
				<div class="feat">
					<div class="feat-head">
						<span class="feat-lbl">FLATNESS</span>
						<span class="feat-val">{featFlat}</span>
					</div>
					<div class="feat-track"><div class="feat-fill" style="width:{barFlat}%"></div></div>
				</div>
				<div class="feat">
					<div class="feat-head">
						<span class="feat-lbl">BANDWIDTH</span>
						<span class="feat-val">{featBw}</span>
					</div>
					<div class="feat-track"><div class="feat-fill" style="width:{barBw}%"></div></div>
				</div>
				<div class="feat">
					<div class="feat-head">
						<span class="feat-lbl">ROLLOFF</span>
						<span class="feat-val">{featRol}</span>
					</div>
					<div class="feat-track"><div class="feat-fill" style="width:{barRol}%"></div></div>
				</div>
			</div>
		</div>

		<div class="divider"></div>

		<div class="sb-section controls-section">
			<button class="play-btn" class:active={isRunning} onclick={toggle}
				aria-label={isRunning ? 'Stop' : 'Start'}>
				{#if isRunning}
					<span class="icon-stop"></span>
					<span class="play-label">STOP</span>
				{:else}
					<span class="icon-play"></span>
					<span class="play-label">START</span>
				{/if}
			</button>

			<div class="ctl-group">
				<span class="ctl-label">INPUT</span>
				<div class="toggle-row">
					<button class="toggle-btn" class:active={inputMode === 'mic'} disabled={isRunning}
						onclick={() => (inputMode = 'mic')}>MIC</button>
					<button class="toggle-btn" class:active={inputMode === 'file'} disabled={isRunning}
						onclick={() => (inputMode = 'file')}>FILE</button>
				</div>
			</div>

			{#if inputMode === 'file'}
				<label class="file-wrap">
					<span class="file-name" class:has-file={!!selectedFile}>
						{selectedFile ? selectedFile.name.slice(0, 18).toUpperCase() : 'CHOOSE FILE'}
					</span>
					<input type="file" accept="audio/*" onchange={onFileChange} disabled={isRunning}
						class="file-input" />
				</label>
			{/if}

			<div class="ctl-group">
				<span class="ctl-label">AXES</span>
				<div class="axis-row">
					<span class="axis-tag">X</span>
					<select class="axis-select" bind:value={axisX} onchange={onAxesChange}>
						{#each FEATURES as f}
							<option value={f.id}>{f.label}</option>
						{/each}
					</select>
				</div>
				<div class="axis-row">
					<span class="axis-tag">Y</span>
					<select class="axis-select" bind:value={axisY} onchange={onAxesChange}>
						{#each FEATURES as f}
							<option value={f.id}>{f.label}</option>
						{/each}
					</select>
				</div>
				<div class="axis-row">
					<span class="axis-tag">Z</span>
					<select class="axis-select" bind:value={axisZ} onchange={onAxesChange}>
						{#each FEATURES as f}
							<option value={f.id}>{f.label}</option>
						{/each}
					</select>
				</div>
			</div>
		</div>

		<div class="sidebar-footer">
			<span class="fps-display">{fps} <span class="fps-unit">FPS</span></span>
		</div>
	</aside>

	<!-- ─── Main visualization ─── -->
	<div class="viz-area">
		<section class="embedding-view">
			<canvas bind:this={pointCanvas}></canvas>

			<div class="corner tl"></div>
			<div class="corner tr"></div>
			<div class="corner bl"></div>
			<div class="corner br"></div>

			<div class="axes-label">
				<span class="axis-key">X</span> {getFeature(axisX).axisLabel}
				<span class="axes-sep">/</span>
				<span class="axis-key">Y</span> {getFeature(axisY).axisLabel}
				<span class="axes-sep">/</span>
				<span class="axis-key">Z</span> {getFeature(axisZ).axisLabel}
			</div>
		</section>

		<section class="spectrogram-view">
			<canvas bind:this={spectroCanvas}></canvas>
			<span class="spec-label">MEL SPECTROGRAM</span>
			<div class="spec-freq">
				<span>HI</span>
				<span>LO</span>
			</div>
		</section>
	</div>
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

	main {
		width: 100vw;
		height: 100vh;
		display: flex;
		flex-direction: row;
	}

	/* ── Sidebar ──────────────────────────────────── */
	.sidebar {
		width: 260px;
		height: 100vh;
		display: flex;
		flex-direction: column;
		border-right: 1px solid rgba(42, 42, 50, 0.1);
		flex-shrink: 0;
	}

	.brand-block {
		padding: 24px 24px 20px;
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		flex-shrink: 0;
	}

	.brand {
		margin: 0;
		font-size: 20px;
		font-weight: 600;
		letter-spacing: 5px;
		line-height: 1.2;
		color: rgba(42, 42, 50, 0.72);
	}

	.status-badge {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 5px;
	}

	.status-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: rgba(42, 42, 50, 0.2);
		flex-shrink: 0;
		transition: background 0.2s;
	}
	.status-dot.active {
		background: rgba(160, 50, 50, 0.65);
	}

	.status-text {
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 1.5px;
		color: rgba(42, 42, 50, 0.45);
	}

	.divider {
		height: 1px;
		margin: 0 24px;
		background: rgba(42, 42, 50, 0.08);
		flex-shrink: 0;
	}

	/* ── Sidebar sections ─────────────────────────── */
	.sb-section {
		padding: 18px 24px;
		flex-shrink: 0;
	}

	.sb-label {
		display: block;
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 2px;
		color: rgba(42, 42, 50, 0.38);
		margin-bottom: 12px;
	}

	/* ── Waveform ─────────────────────────────────── */
	.waveform-wrap {
		height: 44px;
		border-radius: 4px;
		overflow: hidden;
		border: 1px solid rgba(42, 42, 50, 0.06);
	}

	.waveform-canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	/* ── Features ─────────────────────────────────── */
	.features {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.feat {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.feat-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}

	.feat-lbl {
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 1.5px;
		color: rgba(42, 42, 50, 0.5);
	}

	.feat-val {
		font-size: 12px;
		font-weight: 400;
		color: rgba(42, 42, 50, 0.75);
		font-variant-numeric: tabular-nums;
	}

	.feat-track {
		width: 100%;
		height: 3px;
		background: rgba(42, 42, 50, 0.07);
		border-radius: 2px;
		overflow: hidden;
	}

	.feat-fill {
		height: 100%;
		background: rgba(42, 42, 50, 0.45);
		border-radius: 2px;
		transition: width 0.1s ease-out;
	}

	/* ── Controls ─────────────────────────────────── */
	.controls-section {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.play-btn {
		width: 100%;
		height: 40px;
		border: 1px solid rgba(42, 42, 50, 0.18);
		border-radius: 6px;
		background: transparent;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 10px;
		transition: all 0.15s;
		font-family: inherit;
	}
	.play-btn:hover {
		border-color: rgba(42, 42, 50, 0.35);
		background: rgba(42, 42, 50, 0.03);
	}
	.play-btn.active {
		border-color: rgba(160, 50, 50, 0.35);
		background: rgba(160, 50, 50, 0.04);
	}

	.play-label {
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 2.5px;
		color: rgba(42, 42, 50, 0.65);
	}
	.play-btn.active .play-label {
		color: rgba(160, 50, 50, 0.65);
	}

	.icon-play {
		width: 0;
		height: 0;
		border-style: solid;
		border-width: 6px 0 6px 10px;
		border-color: transparent transparent transparent rgba(42, 42, 50, 0.55);
	}

	.icon-stop {
		width: 10px;
		height: 10px;
		background: rgba(160, 50, 50, 0.55);
		border-radius: 2px;
	}

	.ctl-group {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.ctl-label {
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 1.5px;
		color: rgba(42, 42, 50, 0.42);
	}

	.toggle-row {
		display: flex;
		gap: 0;
	}

	.toggle-btn {
		flex: 1;
		padding: 7px 0;
		border: 1px solid rgba(42, 42, 50, 0.12);
		background: transparent;
		color: rgba(42, 42, 50, 0.38);
		font-family: inherit;
		font-size: 11px;
		font-weight: 400;
		letter-spacing: 1.5px;
		cursor: pointer;
		transition: all 0.12s;
	}
	.toggle-btn:first-child {
		border-radius: 4px 0 0 4px;
		border-right: none;
	}
	.toggle-btn:last-child {
		border-radius: 0 4px 4px 0;
	}
	.toggle-btn:hover {
		color: rgba(42, 42, 50, 0.6);
		background: rgba(42, 42, 50, 0.03);
	}
	.toggle-btn.active {
		color: rgba(42, 42, 50, 0.8);
		background: rgba(42, 42, 50, 0.06);
		border-color: rgba(42, 42, 50, 0.18);
	}
	.toggle-btn:disabled {
		opacity: 0.35;
		cursor: not-allowed;
	}

	.file-wrap {
		cursor: pointer;
		display: block;
	}
	.file-name {
		display: block;
		padding: 8px 12px;
		border: 1px dashed rgba(42, 42, 50, 0.18);
		border-radius: 4px;
		font-family: inherit;
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 1px;
		color: rgba(42, 42, 50, 0.42);
		text-align: center;
		transition: all 0.12s;
	}
	.file-name:hover {
		border-color: rgba(42, 42, 50, 0.35);
		color: rgba(42, 42, 50, 0.65);
	}
	.file-name.has-file {
		border-style: solid;
		color: rgba(42, 42, 50, 0.7);
	}
	.file-input {
		position: absolute;
		width: 0;
		height: 0;
		opacity: 0;
		pointer-events: none;
	}

	/* ── Axis selectors ──────────────────────────── */
	.axis-row {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.axis-row + .axis-row {
		margin-top: 4px;
	}

	.axis-tag {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 1px;
		color: rgba(42, 42, 50, 0.55);
		width: 14px;
		flex-shrink: 0;
	}

	.axis-select {
		flex: 1;
		padding: 5px 8px;
		border: 1px solid rgba(42, 42, 50, 0.12);
		border-radius: 3px;
		background: transparent;
		color: rgba(42, 42, 50, 0.7);
		font-family: inherit;
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 0.5px;
		cursor: pointer;
		outline: none;
	}
	.axis-select:hover {
		border-color: rgba(42, 42, 50, 0.25);
	}
	.axis-select:focus {
		border-color: rgba(42, 42, 50, 0.3);
	}

	/* ── Sidebar footer ───────────────────────────── */
	.sidebar-footer {
		margin-top: auto;
		padding: 16px 24px;
		border-top: 1px solid rgba(42, 42, 50, 0.06);
		flex-shrink: 0;
	}

	.fps-display {
		font-size: 12px;
		font-weight: 400;
		color: rgba(42, 42, 50, 0.5);
		font-variant-numeric: tabular-nums;
	}

	.fps-unit {
		font-size: 9px;
		font-weight: 300;
		letter-spacing: 1px;
		color: rgba(42, 42, 50, 0.3);
	}

	/* ── Main visualization area ──────────────────── */
	.viz-area {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	/* ── Embedding view ──────────────────────────── */
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

	.corner {
		position: absolute;
		width: 28px;
		height: 28px;
		pointer-events: none;
		z-index: 1;
	}
	.corner.tl { top: 16px; left: 16px; border-left: 1.5px solid rgba(42,42,50,0.15); border-top: 1.5px solid rgba(42,42,50,0.15); }
	.corner.tr { top: 16px; right: 16px; border-right: 1.5px solid rgba(42,42,50,0.15); border-top: 1.5px solid rgba(42,42,50,0.15); }
	.corner.bl { bottom: 16px; left: 16px; border-left: 1.5px solid rgba(42,42,50,0.15); border-bottom: 1.5px solid rgba(42,42,50,0.15); }
	.corner.br { bottom: 16px; right: 16px; border-right: 1.5px solid rgba(42,42,50,0.15); border-bottom: 1.5px solid rgba(42,42,50,0.15); }

	.axes-label {
		position: absolute;
		bottom: 24px;
		left: 52px;
		font-size: 11px;
		font-weight: 400;
		letter-spacing: 1.5px;
		color: rgba(42, 42, 50, 0.38);
		pointer-events: none;
	}
	.axis-key {
		font-weight: 600;
		color: rgba(42, 42, 50, 0.6);
	}
	.axes-sep {
		margin: 0 6px;
		font-weight: 300;
		color: rgba(42, 42, 50, 0.2);
	}

	/* ── Spectrogram ─────────────────────────────── */
	.spectrogram-view {
		height: 150px;
		position: relative;
		border-top: 1px solid rgba(42, 42, 50, 0.08);
		flex-shrink: 0;
	}

	.spectrogram-view canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.spec-label {
		position: absolute;
		top: 12px;
		left: 20px;
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 2px;
		color: rgba(42, 42, 50, 0.32);
		pointer-events: none;
	}

	.spec-freq {
		position: absolute;
		right: 16px;
		top: 12px;
		bottom: 8px;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		pointer-events: none;
	}
	.spec-freq span {
		font-size: 9px;
		font-weight: 400;
		letter-spacing: 1px;
		color: rgba(42, 42, 50, 0.28);
	}
</style>
