<script lang="ts">
	import { onMount } from "svelte";
	import { AudioSource } from "$lib/audio/audio-source.js";
	import { MelFeatureExtractor } from "$lib/dsp/mel.js";
	import { OnlinePCAEmbedding } from "$lib/embedding/pca.js";
	import { EmbeddingSmoother } from "$lib/embedding/smoother.js";
	import {
		PointCloudRenderer,
		type TrajectoryMetrics,
	} from "$lib/render/point-cloud.js";
	import { MelCloudRenderer } from "$lib/render/mel-cloud.js";
	import { SpectrogramRenderer } from "$lib/render/spectrogram.js";
	import { OscilloscopeRenderer } from "$lib/render/oscilloscope.js";
	import { PitchGaugeRenderer } from "$lib/render/pitch-gauge.js";
	import demoAudioUrl from "$lib/assets/greti_greatest_hits_2021.mp3?url";
	import socialPreviewImage from "$lib/assets/sonomaps-compressed.jpg";
	import birdHawthorn from "$lib/assets/bird-hawthorn.png";
	import { base } from "$app/paths";

	// ── DOM refs ───────────────────────────────────────────
	let melCanvas: HTMLCanvasElement;
	let spectroCanvas: HTMLCanvasElement;
	let pointCanvas: HTMLCanvasElement;
	let radarCanvas: HTMLCanvasElement;
	let scopeCanvas: HTMLCanvasElement;
	let pitchCanvas: HTMLCanvasElement;
	let cardsViewport: HTMLDivElement;

	// ── Rendering objects (not reactive) ──────────────────
	let melCloud: MelCloudRenderer | null = null;
	let spectrogram: SpectrogramRenderer | null = null;
	let pointCloud: PointCloudRenderer | null = null;
	let radarCtx: CanvasRenderingContext2D | null = null;
	let scope: OscilloscopeRenderer | null = null;
	let pitchGauge: PitchGaugeRenderer | null = null;

	// ── Audio pipeline (not reactive) ─────────────────────
	let audioSource: AudioSource | null = null;
	let melExtractor: MelFeatureExtractor | null = null;
	let embedding: OnlinePCAEmbedding | null = null;
	let smoother: EmbeddingSmoother | null = null;

	let processing = false;
	let animFrameId = 0;
	let sampleIntervalId = 0;

	// ── Landing state ────────────────────────────────────
	let showLanding = $state(true);
	let demoReady = $state(false);
	let landingDismissing = $state(false);
	let landingRevealed = $state(false);

	// Preload the bird illustration so it's ready before reveal
	if (typeof window !== "undefined") {
		const img = new Image();
		img.src = birdHawthorn;
		const reveal = () => { landingRevealed = true; };
		img.complete ? reveal() : img.onload = img.onerror = reveal;
	}

	// ── UI state ──────────────────────────────────────────
	let isRunning = $state(false);
	let inputMode = $state<"mic" | "file">("mic");
	let fps = $state(0);
	let selectedFile = $state<File | null>(null);
	let status = $state("READY");
	let volume = $state(1.0);
	let noiseThreshold = $state(1.8);
	let floorDb = $state(-60);
	let freqLo = $state(2000);
	let freqHi = $state(12000);

	// ── Mobile state ────────────────────────────────────
	let isMobile = $state(false);
	let currentCard = $state(0);
	let touchStartX = 0;
	let touchStartY = 0;
	let touchDeltaX = $state(0);
	let isSwiping = $state(false);
	let touchStartTime = 0;
	let showSwipeHint = $state(false);
	let swipeHintTimeoutId = 0;
	let swipeHintRepeatTimeoutId = 0;
	let hasDiscoveredCardNavigation = $state(false);

	const CARD_NAMES = ["TRAJECTORY", "ANALYSIS", "MEL SPECTROGRAM"];
	const NUM_CARDS = 3;

	// ── Fixed parameters ─────────────────────────────────
	const smoothing = 0.58;
	const outputDim = 3;

	// ── PCA calibration state ────────────────────────────
	let pcaCalibrating = $state(false);

	// ── Trajectory metrics (updated every frame) ─────────
	let trajMetrics = $state<TrajectoryMetrics>({
		spread: 0,
		drift: 0,
		flux: 0,
		segments: 0,
	});

	// ── Feature display state (updated ~10Hz) ─────────────
	let featCentroid = $state("—");
	let featRms = $state("—");
	let featZcr = $state("—");
	let featFlat = $state("—");
	let featBw = $state("—");
	let featRol = $state("—");
	let barCentroid = $state(0);
	let barRms = $state(0);
	let barZcr = $state(0);
	let barFlat = $state(0);
	let barBw = $state(0);
	let barRol = $state(0);

	// ── Radar trail history ──────────────────────────────
	const RADAR_TRAIL_LENGTH = 12;
	let radarSnapshots: number[][] = [];

	// ── Radar smooth animation state ─────────────────────
	const radarSmooth = new Float64Array(6); // smoothed bar values (0–1)
	const RADAR_LERP = 0.18; // interpolation speed per frame

	// ── Radar static grid cache ──────────────────────────
	let radarGridCanvas: OffscreenCanvas | null = null;
	let radarGridW = 0;
	let radarGridH = 0;

	// Pre-computed trig for 6 radar axes
	const RADAR_N = 6;
	const RADAR_STEP = (Math.PI * 2) / RADAR_N;
	const RADAR_START = -Math.PI / 2;
	const radarCos = new Float64Array(RADAR_N);
	const radarSin = new Float64Array(RADAR_N);
	for (let i = 0; i < RADAR_N; i++) {
		const a = RADAR_START + i * RADAR_STEP;
		radarCos[i] = Math.cos(a);
		radarSin[i] = Math.sin(a);
	}
	const RADAR_LABELS = [
		"Centroid",
		"Energy",
		"Cross.",
		"Tonality",
		"Spread",
		"Rolloff",
	];

	const radarTextStyle = {
		labelFont: '500 10px "JetBrains Mono"',
		labelColor: "rgba(42,42,50,0.94)",
		labelRadiusOffset: 24,
		labelYOffset: -8,
		valueFont: '400 10px "JetBrains Mono"',
		valueColor: "rgba(42,42,50,0.40)",
		valueYOffset: 11,
	};

	function readRadarTextStyle(): void {
		if (!radarCanvas) return;
		const host = radarCanvas.parentElement ?? radarCanvas;
		const s = getComputedStyle(host);

		radarTextStyle.labelFont =
			s.getPropertyValue("--radar-label-font").trim() ||
			radarTextStyle.labelFont;
		radarTextStyle.labelColor =
			s.getPropertyValue("--radar-label-color").trim() ||
			radarTextStyle.labelColor;
		radarTextStyle.valueFont =
			s.getPropertyValue("--radar-value-font").trim() ||
			radarTextStyle.valueFont;
		radarTextStyle.valueColor =
			s.getPropertyValue("--radar-value-color").trim() ||
			radarTextStyle.valueColor;

		const labelRadiusOffset = parseFloat(
			s.getPropertyValue("--radar-label-radius-offset"),
		);
		if (!Number.isNaN(labelRadiusOffset))
			radarTextStyle.labelRadiusOffset = labelRadiusOffset;

		const labelYOffset = parseFloat(
			s.getPropertyValue("--radar-label-y-offset"),
		);
		if (!Number.isNaN(labelYOffset))
			radarTextStyle.labelYOffset = labelYOffset;

		const valueYOffset = parseFloat(
			s.getPropertyValue("--radar-value-y-offset"),
		);
		if (!Number.isNaN(valueYOffset))
			radarTextStyle.valueYOffset = valueYOffset;
	}

	const FFT_SIZE = 1024;
	const NUM_MEL_BANDS = 64;
	const MAX_POINTS = 10000;
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

	function normalizeForRendering(
		raw: number,
		ema: number,
		variance: number,
	): number {
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

		constructor(decay = 0.995) {
			this.decay = decay;
		}

		update(raw: number): number {
			this.count++;
			if (this.count < 30) {
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

		reset(): void {
			this.mean = 0;
			this.variance = 1;
			this.count = 0;
		}
	}

	const radarNorm = {
		centroid: new FeatureNormalizer(),
		rms: new FeatureNormalizer(),
		zcr: new FeatureNormalizer(),
		flatness: new FeatureNormalizer(),
		bandwidth: new FeatureNormalizer(),
		rolloff: new FeatureNormalizer(),
	};

	// ── Adaptive noise gate ──────────────────────────────
	let noiseFloorEma = 0;
	let noiseFloorInitialized = false;
	const NOISE_FLOOR_DECAY = 0.998;
	const NOISE_FLOOR_UP = 0.95;

	let frameCount = 0;
	let lastFpsTime = 0;
	let featCounter = 0;
	let isAboveGate = false;

	// ── Formatting helpers ───────────────────────────────
	function fmtHz(hz: number): string {
		if (hz >= 10000) return (hz / 1000).toFixed(0) + "k";
		if (hz >= 1000) return (hz / 1000).toFixed(1) + "k";
		return Math.round(hz).toString();
	}

	function fmtTonality(tonality: number): string {
		const t = Math.max(0, Math.min(1, tonality));
		if (t > 0.99) return t.toFixed(4);
		return t.toFixed(3);
	}

	function clearSwipeHintTimeout(): void {
		if (swipeHintTimeoutId) {
			window.clearTimeout(swipeHintTimeoutId);
			swipeHintTimeoutId = 0;
		}
	}

	function clearSwipeHintRepeatTimeout(): void {
		if (swipeHintRepeatTimeoutId) {
			window.clearTimeout(swipeHintRepeatTimeoutId);
			swipeHintRepeatTimeoutId = 0;
		}
	}

	function dismissSwipeHint(): void {
		showSwipeHint = false;
		clearSwipeHintTimeout();
		clearSwipeHintRepeatTimeout();
	}

	function scheduleSwipeHint(): void {
		if (!isMobile || hasDiscoveredCardNavigation || currentCard >= NUM_CARDS - 1) {
			dismissSwipeHint();
			return;
		}

		showSwipeHint = true;
		clearSwipeHintTimeout();
		clearSwipeHintRepeatTimeout();
		swipeHintTimeoutId = window.setTimeout(() => {
			showSwipeHint = false;
			swipeHintTimeoutId = 0;
			swipeHintRepeatTimeoutId = window.setTimeout(() => {
				swipeHintRepeatTimeoutId = 0;
				scheduleSwipeHint();
			}, 7200);
		}, 2400);
	}

	// ── High-frequency audio sampling (~250Hz) ───────────
	function sampleAudio(): void {
		if (
			!processing ||
			!audioSource ||
			!melExtractor ||
			!embedding ||
			!smoother
		)
			return;

		audioSource.read();
		melExtractor.compute(audioSource.freqData, audioSource.timeData);

		const rms = melExtractor.rms;

		if (!noiseFloorInitialized) {
			noiseFloorEma = rms;
			noiseFloorInitialized = true;
		} else {
			if (rms < noiseFloorEma * 2.0) {
				noiseFloorEma =
					NOISE_FLOOR_DECAY * noiseFloorEma +
					(1 - NOISE_FLOOR_DECAY) * rms;
			} else if (rms < noiseFloorEma * 5.0) {
				noiseFloorEma =
					NOISE_FLOOR_UP * noiseFloorEma + (1 - NOISE_FLOOR_UP) * rms;
			}
		}

		const gate = Math.max(noiseFloorEma * noiseThreshold, MIN_GATE);
		isAboveGate = rms >= gate;
		if (!isAboveGate) return;

		embedding.projectFromExtractor(melExtractor, embeddingBuf);
		pcaCalibrating = embedding.isWarmingUp;
		if (pcaCalibrating) return;
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
			energyVar =
				RENDER_DECAY * energyVar + (1 - RENDER_DECAY) * eDiff * eDiff;
			centroidEma =
				RENDER_DECAY * centroidEma + (1 - RENDER_DECAY) * logCentroid;
			const cDiff = logCentroid - centroidEma;
			centroidVar =
				RENDER_DECAY * centroidVar + (1 - RENDER_DECAY) * cDiff * cDiff;
		}

		const normEnergy = normalizeForRendering(logRms, energyEma, energyVar);
		const normCentroid = normalizeForRendering(
			logCentroid,
			centroidEma,
			centroidVar,
		);

		pointData[0] = embeddingBuf[0];
		pointData[1] = embeddingBuf[1];
		pointData[2] = embeddingBuf[2];
		pointData[3] = normEnergy;
		pointData[4] = normCentroid;
		pointCloud!.addPoints(pointData, 1);
	}

	// ── HiDPI canvas helper ─────────────────────────────
	function resizeHiDPI(
		canvas: HTMLCanvasElement,
		ctx: CanvasRenderingContext2D,
	): void {
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

	// ── Radar: build static grid offscreen ──────────────
	function buildRadarGrid(w: number, h: number): void {
		radarGridW = w;
		radarGridH = h;
		readRadarTextStyle();
		const dpr = window.devicePixelRatio || 1;
		radarGridCanvas = new OffscreenCanvas(
			Math.round(w * dpr),
			Math.round(h * dpr),
		);
		const g = radarGridCanvas.getContext("2d")!;
		g.scale(dpr, dpr);

		const cx = w / 2;
		const cy = h / 2;
		const radius = Math.min(w, h) * 0.3;

		// Concentric rings
		for (let ring = 1; ring <= 3; ring++) {
			const r = radius * (ring / 3);
			g.strokeStyle =
				ring === 3 ? "rgba(42,42,50,0.14)" : "rgba(42,42,50,0.06)";
			g.lineWidth = 0.5;
			g.beginPath();
			for (let i = 0; i <= RADAR_N; i++) {
				const idx = i % RADAR_N;
				const x = cx + radarCos[idx] * r;
				const y = cy + radarSin[idx] * r;
				if (i === 0) g.moveTo(x, y);
				else g.lineTo(x, y);
			}
			g.closePath();
			g.stroke();
		}

		// Axis lines
		g.strokeStyle = "rgba(42,42,50,0.1)";
		g.lineWidth = 0.5;
		for (let i = 0; i < RADAR_N; i++) {
			g.beginPath();
			g.moveTo(cx, cy);
			g.lineTo(cx + radarCos[i] * radius, cy + radarSin[i] * radius);
			g.stroke();
		}

		// Labels (static part — feature names)
		g.textAlign = "center";
		g.textBaseline = "middle";
		g.fillStyle = radarTextStyle.labelColor;
		g.font = radarTextStyle.labelFont;
		const lr = radius + radarTextStyle.labelRadiusOffset;
		for (let i = 0; i < RADAR_N; i++) {
			g.fillText(
				RADAR_LABELS[i],
				cx + radarCos[i] * lr,
				cy + radarSin[i] * lr + radarTextStyle.labelYOffset,
			);
		}
	}

	// ── Radar chart rendering (optimized) ────────────────
	function renderRadar(): void {
		if (!radarCtx || !radarCanvas) return;
		resizeHiDPI(radarCanvas, radarCtx);

		const w = radarCanvas.clientWidth;
		const h = radarCanvas.clientHeight;
		if (w === 0 || h === 0) return;

		// Rebuild static grid cache on size change
		if (w !== radarGridW || h !== radarGridH) buildRadarGrid(w, h);

		const cx = w / 2;
		const cy = h / 2;
		const radius = Math.min(w, h) * 0.3;

		// Target bar values
		const targets = [
			barCentroid / 100,
			barRms / 100,
			barZcr / 100,
			barFlat / 100,
			barBw / 100,
			barRol / 100,
		];
		const vals = [
			featCentroid,
			featRms,
			featZcr,
			featFlat,
			featBw,
			featRol,
		];

		// Smooth interpolation every frame
		for (let i = 0; i < RADAR_N; i++) {
			radarSmooth[i] += (targets[i] - radarSmooth[i]) * RADAR_LERP;
		}

		// Clear and stamp cached grid
		radarCtx.clearRect(0, 0, w, h);
		if (radarGridCanvas) radarCtx.drawImage(radarGridCanvas, 0, 0, w, h);

		// ── Trail polygons (batched: single path per trail) ──
		const sLen = radarSnapshots.length;
		for (let t = 0; t < sLen; t++) {
			const snap = radarSnapshots[t];
			const age = (sLen - t) / (sLen + 1);
			const alpha = Math.max(0.008, (1 - age * age) * 0.08);

			radarCtx.fillStyle = `rgba(42,42,50,${alpha.toFixed(3)})`;
			radarCtx.beginPath();
			for (let i = 0; i <= RADAR_N; i++) {
				const idx = i % RADAR_N;
				const rv = radius * Math.max(0.03, snap[idx]);
				const x = cx + radarCos[idx] * rv;
				const y = cy + radarSin[idx] * rv;
				if (i === 0) radarCtx.moveTo(x, y);
				else radarCtx.lineTo(x, y);
			}
			radarCtx.closePath();
			radarCtx.fill();
		}

		// ── Current polygon (fill + stroke in one path build) ──
		radarCtx.beginPath();
		for (let i = 0; i <= RADAR_N; i++) {
			const idx = i % RADAR_N;
			const rv = radius * Math.max(0.03, radarSmooth[idx]);
			const x = cx + radarCos[idx] * rv;
			const y = cy + radarSin[idx] * rv;
			if (i === 0) radarCtx.moveTo(x, y);
			else radarCtx.lineTo(x, y);
		}
		radarCtx.closePath();
		radarCtx.fillStyle = "rgba(42,42,50,0.045)";
		radarCtx.fill();
		radarCtx.strokeStyle = "rgba(42,42,50,0.38)";
		radarCtx.lineWidth = 1.5;
		radarCtx.stroke();

		// Vertex dots (single fillStyle, batch arcs)
		radarCtx.fillStyle = "rgba(42,42,50,0.5)";
		for (let i = 0; i < RADAR_N; i++) {
			const rv = radius * Math.max(0.03, radarSmooth[i]);
			radarCtx.beginPath();
			radarCtx.arc(
				cx + radarCos[i] * rv,
				cy + radarSin[i] * rv,
				2.5,
				0,
				Math.PI * 2,
			);
			radarCtx.fill();
		}

		// Value labels (dynamic — update every frame for smooth text)
		radarCtx.textAlign = "center";
		radarCtx.textBaseline = "middle";
		radarCtx.fillStyle = radarTextStyle.valueColor;
		radarCtx.font = radarTextStyle.valueFont;
		const lr = radius + radarTextStyle.labelRadiusOffset;
		for (let i = 0; i < RADAR_N; i++) {
			radarCtx.fillText(
				vals[i],
				cx + radarCos[i] * lr,
				cy + radarSin[i] * lr + radarTextStyle.valueYOffset,
			);
		}
	}

	// ── Lifecycle ─────────────────────────────────────────
	onMount(() => {
		// Pre-load the bundled demo file so users can play immediately
		fetch(demoAudioUrl)
			.then((r) => r.blob())
			.then((blob) => {
				selectedFile = new File([blob], "greti_greatest_hits_2021.mp3", {
					type: "audio/mpeg",
				});
				inputMode = "file";
				demoReady = true;
			})
			.catch(() => {
				demoReady = true; /* demo unavailable, still allow mic */
			});

		radarCtx = radarCanvas.getContext("2d");

		melCloud = new MelCloudRenderer(melCanvas, {
			maxFrames: 400,
			numBands: NUM_MEL_BANDS,
		});

		spectrogram = new SpectrogramRenderer(spectroCanvas, NUM_MEL_BANDS);
		scope = new OscilloscopeRenderer(scopeCanvas);
		pitchGauge = new PitchGaugeRenderer(pitchCanvas);

		pointCloud = new PointCloudRenderer(pointCanvas, {
			maxPoints: MAX_POINTS,
			outputDim,
			pointSize: 1,
		});

		// ── Mobile detection ──────────────────────────
		const mql = window.matchMedia("(max-width: 768px)");
		isMobile = mql.matches;
		hasDiscoveredCardNavigation = false;
		if (isMobile) scheduleSwipeHint();
		const onMqlChange = (e: MediaQueryListEvent) => {
			isMobile = e.matches;
			if (!isMobile) {
				currentCard = 0;
				touchDeltaX = 0;
				hasDiscoveredCardNavigation = false;
				dismissSwipeHint();
			} else {
				hasDiscoveredCardNavigation = false;
				scheduleSwipeHint();
			}
		};
		mql.addEventListener("change", onMqlChange);

		// Non-passive touchmove for preventDefault
		if (cardsViewport) {
			cardsViewport.addEventListener("touchmove", onTouchMove, {
				passive: false,
			});
		}

		const onResize = () => {
			melCloud?.resize();
			spectrogram?.resize();
			pointCloud?.resize();
		};
		window.addEventListener("resize", onResize);

		function loop() {
			if (processing && melExtractor) {
				melCloud!.addFrame(melExtractor.logMelEnergies);
				spectrogram?.addColumn(melExtractor.logMelEnergies);
			}

			// Render only visible card on mobile for performance
			// Card order on mobile: 0=Trajectory, 1=Analysis, 2=Mel
			if (!isMobile || currentCard === 1) renderRadar();
			if (!isMobile || currentCard === 2) melCloud?.render();
			if (!isMobile || currentCard === 0) {
				pointCloud?.render();
				if (pointCloud) trajMetrics = pointCloud.getMetrics();
			}

			// Oscilloscope + pitch gauge (Analysis card)
			if (!isMobile || currentCard === 1) {
				if (processing && audioSource) {
					scope?.draw(audioSource.timeData);
					pitchGauge?.draw(
						melExtractor?.peakFreq ?? 0,
						melExtractor?.rms ?? 0,
					);
				} else {
					scope?.draw(new Float32Array(0));
					pitchGauge?.draw(0, 0);
				}
			}

			featCounter++;
			if (featCounter % 6 === 0 && processing && melExtractor) {
				const m = melExtractor;
				// Only update radar normalizers when above noise gate
				const gate = Math.max(noiseFloorEma * noiseThreshold, MIN_GATE);
				if (m.rms >= gate) {
					featCentroid = fmtHz(m.centroid);
					barCentroid =
						radarNorm.centroid.update(Math.log1p(m.centroid)) * 100;
					featRms = m.rms.toFixed(3);
					barRms =
						radarNorm.rms.update(Math.log1p(m.rms * 1000)) * 100;
					featZcr = m.zcr.toFixed(3);
					barZcr = radarNorm.zcr.update(m.zcr) * 100;
					const tonality = m.tonality;
					featFlat = fmtTonality(tonality);
					barFlat = radarNorm.flatness.update(tonality) * 100;
					featBw = fmtHz(m.bandwidth);
					barBw =
						radarNorm.bandwidth.update(Math.log1p(m.bandwidth)) *
						100;
					featRol = fmtHz(m.rolloff);
					barRol =
						radarNorm.rolloff.update(Math.log1p(m.rolloff)) * 100;

					// Add radar trail snapshot
					radarSnapshots.push([
						barCentroid / 100,
						barRms / 100,
						barZcr / 100,
						barFlat / 100,
						barBw / 100,
						barRol / 100,
					]);
					if (radarSnapshots.length > RADAR_TRAIL_LENGTH)
						radarSnapshots.shift();
				}
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
			clearSwipeHintRepeatTimeout();
			clearSwipeHintTimeout();
			window.removeEventListener("resize", onResize);
			mql.removeEventListener("change", onMqlChange);
			cardsViewport?.removeEventListener("touchmove", onTouchMove);
			stop();
			melCloud?.dispose();
			pointCloud?.dispose();
		};
	});

	async function start() {
		if (isRunning) return;
		try {
			status = "INIT";
			audioSource = new AudioSource(FFT_SIZE);

			if (inputMode === "mic") {
				await audioSource.startMic();
			} else if (selectedFile) {
				await audioSource.startFile(selectedFile);
			} else {
				status = "NO FILE";
				return;
			}

			audioSource.setVolume(volume);

			melExtractor = new MelFeatureExtractor({
				sampleRate: audioSource.sampleRate,
				fftSize: FFT_SIZE,
				numMelBands: NUM_MEL_BANDS,
				numMfccs: 13,
			});
			melExtractor.floorDb = floorDb;
			melExtractor.minFreqHz = freqLo;
			melExtractor.maxFreqHz = freqHi;

			embedding = new OnlinePCAEmbedding();
			smoother = new EmbeddingSmoother(outputDim, smoothing);

			energyEma = 0;
			energyVar = 0.01;
			centroidEma = 0;
			centroidVar = 1;
			warmupCount = 0;
			noiseFloorEma = 0;
			noiseFloorInitialized = false;
			radarSnapshots = [];
			for (const n of Object.values(radarNorm)) n.reset();

			melCloud?.clear();
			pointCloud?.clear();
			pitchGauge?.reset();

			processing = true;
			sampleIntervalId = window.setInterval(
				sampleAudio,
				SAMPLE_INTERVAL_MS,
			);
			isRunning = true;
			status = inputMode === "mic" ? "LISTENING" : "PLAYING";
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
		status = "READY";
		featCentroid = "—";
		featRms = "—";
		featZcr = "—";
		featFlat = "—";
		featBw = "—";
		featRol = "—";
		barCentroid = 0;
		barRms = 0;
		barZcr = 0;
		barFlat = 0;
		barBw = 0;
		barRol = 0;
	}

	function toggle() {
		if (isRunning) stop();
		else start();
	}

	function dismissLanding(mode: "file" | "mic") {
		inputMode = mode;
		landingDismissing = true;
		// Start audio immediately (user gesture satisfies browser autoplay policy)
		start();
		setTimeout(() => {
			showLanding = false;
		}, 500);
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

	function onFilterChange() {
		if (!melExtractor) return;
		melExtractor.floorDb = floorDb;
		melExtractor.minFreqHz = freqLo;
		melExtractor.maxFreqHz = freqHi;
	}

	// ── Mobile touch handlers ───────────────────────────
	function onTouchStart(e: TouchEvent) {
		if (!isMobile) return;
		dismissSwipeHint();
		touchStartX = e.touches[0].clientX;
		touchStartY = e.touches[0].clientY;
		touchStartTime = Date.now();
		touchDeltaX = 0;
		isSwiping = false;
	}

	function onTouchMove(e: TouchEvent) {
		if (!isMobile) return;
		const dx = e.touches[0].clientX - touchStartX;
		const dy = e.touches[0].clientY - touchStartY;

		if (!isSwiping) {
			if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 8) {
				isSwiping = true;
			} else {
				return;
			}
		}

		e.preventDefault();
		// Rubber band at edges
		if (
			(currentCard === 0 && dx > 0) ||
			(currentCard === NUM_CARDS - 1 && dx < 0)
		) {
			touchDeltaX = dx * 0.25;
		} else {
			touchDeltaX = dx;
		}
	}

	function onTouchEnd() {
		if (!isMobile || !isSwiping) {
			touchDeltaX = 0;
			if (!hasDiscoveredCardNavigation) scheduleSwipeHint();
			return;
		}
		const elapsed = Date.now() - touchStartTime;
		const velocity = Math.abs(touchDeltaX / Math.max(elapsed, 1));
		// Velocity-based: fast flicks need less distance
		const threshold = velocity > 0.4 ? 30 : window.innerWidth * 0.2;

		if (touchDeltaX < -threshold && currentCard < NUM_CARDS - 1) {
			hasDiscoveredCardNavigation = true;
			currentCard++;
		} else if (touchDeltaX > threshold && currentCard > 0) {
			hasDiscoveredCardNavigation = true;
			currentCard--;
		} else if (!hasDiscoveredCardNavigation) {
			scheduleSwipeHint();
		}
		touchDeltaX = 0;
		isSwiping = false;
		triggerCardResize();
	}

	function triggerCardResize() {
		setTimeout(() => {
			melCloud?.resize();
			spectrogram?.resize();
			pointCloud?.resize();
		}, 60);
	}

	function goToCard(i: number) {
		hasDiscoveredCardNavigation = true;
		dismissSwipeHint();
		currentCard = i;
		triggerCardResize();
	}
</script>

<svelte:head>
	<title>Wytham Tits — Song diversity of great tits in Wytham Woods</title>

	<!-- SEO Meta Tags -->
	<meta
		name="description"
		content="Explore the song diversity of great tits (Parus major) in Wytham Woods through real-time audio analysis and 3D visualization."
	/>
	<meta name="keywords" content="great tit, Parus major, birdsong, Wytham Woods, song diversity, audio analysis, visualization" />
	<meta name="author" content="sedum.studio" />
	<meta name="robots" content="index, follow" />

	<!-- Open Graph Meta Tags (Social Media) -->
	<meta property="og:type" content="website" />
	<meta
		property="og:title"
		content="Wytham Tits — Song diversity of great tits in Wytham Woods"
	/>
	<meta
		property="og:description"
		content="Explore the song diversity of great tits (Parus major) in Wytham Woods through real-time audio analysis and 3D visualization."
	/>
	<meta property="og:image" content={socialPreviewImage} />
	<meta property="og:site_name" content="SonoMaps" />

	<!-- Twitter Card Meta Tags -->
	<meta name="twitter:card" content="summary_large_image" />
	<meta
		name="twitter:title"
		content="Wytham Tits — Song diversity of great tits in Wytham Woods"
	/>
	<meta
		name="twitter:description"
		content="Explore the song diversity of great tits (Parus major) in Wytham Woods through real-time audio analysis and 3D visualization."
	/>
	<meta name="twitter:image" content={socialPreviewImage} />
	<meta name="twitter:site" content="@nilomr" />

	<!-- Additional Meta Tags -->
	<meta
		name="viewport"
		content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover"
	/>
	<meta name="theme-color" content="#f2ede4" />

	<!-- Preconnect for Performance -->
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link
		rel="preconnect"
		href="https://fonts.gstatic.com"
		crossorigin="anonymous"
	/>
	<link
		href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<main class:mobile={isMobile}>
	<!-- ─── Landing overlay ─── -->
	{#if showLanding}
		<div class="landing" class:dismissing={landingDismissing}>
			<div class="landing-content wytham-landing" class:revealed={landingRevealed}>

				<!-- Hawthorn bird illustration -->
				<div class="wytham-illustration stagger" aria-label="Great tit illustration">
					<img src={birdHawthorn} alt="Great tit bird illustration" />
				</div>

				<div class="landing-brand stagger">
					<span class="landing-top">WYTHAM</span>
					<span class="landing-bottom">TITS</span>
				</div>

				<p class="landing-tagline stagger">
					Song diversity of great tits
					<span class="wytham-latin">Parus major</span>
					in Wytham Woods
				</p>

				<p class="wytham-description stagger">
					A representation of the sound space occupied
					by great tit songs recorded across
					a wild population in Wytham Woods, Oxford.
				</p>

				{#if demoReady}
					<div class="landing-actions stagger">
						{#if selectedFile}
							<button
								class="landing-cta"
								onclick={() => dismissLanding("file")}
							>
								<span class="cta-icon"></span>
								<span class="cta-text">LISTEN</span>
							</button>
						{/if}
					</div>
				{:else}
					<div class="landing-loading stagger">
						<span class="loading-dot"></span>
						<span class="loading-dot"></span>
						<span class="loading-dot"></span>
					</div>
				{/if}

				<div class="wytham-links stagger">
					<a
						href="https://www.cell.com/current-biology/fulltext/S0960-9822(25)00150-2"
						class="wytham-link"
						target="_blank"
						rel="noopener noreferrer"
						>PAPER</a
					>
					<span class="wytham-link-sep"></span>
					<a
						href="https://wythamtits.com/"
						class="wytham-link"
						target="_blank"
						rel="noopener noreferrer"
						>PROJECT</a
					>
					<a
						href="https://www.sedum.studio/"
						class="wytham-link"
						target="_blank"
						rel="noopener noreferrer"
						>SEDUM.STUDIO</a
					>
				</div>
			</div>
		</div>
	{/if}

	<!-- ─── Mobile header ─── -->
	<header class="mobile-header">
		<div class="mobile-brand">
			<span class="brand-top">WYTHAM</span>
			<span class="brand-bottom">TITS</span>
		</div>
		<span class="status-badge mobile-status" class:active={isRunning}
			>{status}</span
		>
		<div class="mobile-fps">
			<span class="fps-number">{fps}</span>
			<span class="fps-label">FPS</span>
		</div>
	</header>

	<!-- ─── Card viewport (wraps all viz panels) ─── -->
	<div
		class="cards-viewport"
		bind:this={cardsViewport}
		role="region"
		aria-label="Visualization cards"
		ontouchstart={onTouchStart}
		ontouchend={onTouchEnd}
	>
		<div
			class="cards-track"
			style={isMobile
				? `transform:translateX(calc(${-currentCard} * 100vw + ${touchDeltaX}px));${isSwiping ? "" : "transition:transform 0.35s cubic-bezier(0.22,0.68,0.35,1)"}`
				: ""}
		>
			<!-- ─── Trajectory (card 0 on mobile) ─── -->
			<section class="panel trajectory">
				<canvas bind:this={pointCanvas}></canvas>
				<span class="panel-label">TRAJECTORY</span>
				<div class="axes-overlay">
					{#if isRunning && pcaCalibrating}
						CALIBRATING
					{:else}
						<span class="ax-key">X</span> PC1
						<span class="ax-sep">/</span>
						<span class="ax-key">Y</span> PC2
						<span class="ax-sep">/</span>
						<span class="ax-key">Z</span> PC3
					{/if}
				</div>
				{#if isRunning && !pcaCalibrating && trajMetrics.spread > 0}
					<div class="traj-metrics">
						<div class="tm-row">
							<span class="tm-label">SPREAD</span>
							<span class="tm-value"
								>{trajMetrics.spread.toFixed(2)}</span
							>
						</div>
						<div class="tm-row">
							<span class="tm-label">DRIFT</span>
							<span class="tm-value"
								>{trajMetrics.drift.toFixed(3)}</span
							>
						</div>
						<div class="tm-row">
							<span class="tm-label">FLUX</span>
							<span class="tm-value"
								>{trajMetrics.flux.toFixed(3)}</span
							>
						</div>
						<div class="tm-row">
							<span class="tm-label">LINKS</span>
							<span class="tm-value">{trajMetrics.segments}</span>
						</div>
					</div>
				{/if}
			</section>

			<!-- ─── Analysis (card 1 on mobile) ─── -->
			<section class="panel analysis">
				<div class="analysis-radar">
					<canvas bind:this={radarCanvas} class="fill-canvas"
					></canvas>
				</div>
				<div class="analysis-pitch">
					<canvas bind:this={pitchCanvas}></canvas>
					<span class="panel-label sub">PEAK FREQ</span>
				</div>
				<div class="analysis-scope">
					<canvas bind:this={scopeCanvas}></canvas>
					<span class="panel-label sub">WAVEFORM</span>
				</div>
				<span class="panel-label">ANALYSIS</span>
			</section>

			<!-- ─── Mel cloud (card 2 on mobile) ─── -->
			<section class="panel mel-cloud">
				<div class="mel-3d">
					<canvas bind:this={melCanvas}></canvas>
				</div>
				<div class="mel-2d">
					<canvas bind:this={spectroCanvas}></canvas>
				</div>
				<span class="panel-label">MEL SPECTROGRAM</span>
			</section>
		</div>

		{#if isMobile && showSwipeHint}
			<div class="swipe-hint" aria-hidden="true">
				<span class="swipe-hint-text">SWIPE</span>
				<span class="swipe-hint-arrows">
					<span class="swipe-hint-arrow"></span>
					<span class="swipe-hint-arrow"></span>
				</span>
			</div>
		{/if}
	</div>

	<!-- ─── Card indicator (visible on mobile) ─── -->
	<nav class="card-indicator">
		<span class="card-label">{CARD_NAMES[currentCard]}</span>
		<div class="card-dots">
			{#each CARD_NAMES as _, i}
				<button
					class="card-dot"
					class:active={currentCard === i}
					onclick={() => goToCard(i)}
					aria-label={CARD_NAMES[i]}
				></button>
			{/each}
		</div>
	</nav>

	<!-- ─── Controls bar ─── -->
	<section class="panel controls-bar">
		<!-- Brand -->
		<div class="ctrl-brand">
			<div class="brand-text">
				<span class="brand-top">WYTHAM</span>
				<span class="brand-bottom">TITS</span>
			</div>
			<span class="status-badge" class:active={isRunning}>{status}</span>
		</div>
		<!-- Credit: absolutely positioned so it doesn't affect flex baseline -->
		<a
			class="design-credit"
			href="https://sedum.studio"
			target="_blank"
			rel="noopener noreferrer">designed by <span>sedum.studio</span></a
		>

		<div class="ctrl-sep"></div>

		<!-- Input -->
		<div class="ctrl-section">
			<span class="section-label">INPUT</span>
			<div class="section-body">
				<button
					class="ctrl-btn play"
					class:active={isRunning}
					onclick={toggle}
					aria-label={isRunning ? "Stop" : "Start"}
				>
					{#if isRunning}
						<span class="icon-stop"></span>
					{:else}
						<span class="icon-play"></span>
					{/if}
				</button>
				<div class="input-toggle">
					<button
						class="toggle-btn"
						class:active={inputMode === "mic"}
						disabled={isRunning}
						onclick={() => (inputMode = "mic")}>MIC</button
					>
					<button
						class="toggle-btn"
						class:active={inputMode === "file"}
						disabled={isRunning}
						onclick={() => (inputMode = "file")}>FILE</button
					>
				</div>
				{#if inputMode === "file"}
					<label class="file-btn">
						<span
							>{selectedFile
								? selectedFile.name.slice(0, 12).toUpperCase()
								: "CHOOSE"}</span
						>
						<input
							type="file"
							accept="audio/*"
							onchange={onFileChange}
							disabled={isRunning}
							class="file-hidden"
						/>
					</label>
				{/if}
			</div>
		</div>

		<div class="ctrl-sep"></div>

		<!-- Parameters -->
		<div class="ctrl-section grow">
			<span class="section-label">PARAMETERS</span>
			<div class="section-body params-body">
				<div class="param">
					<span class="param-label">VOL</span>
					<div class="param-wheel">
						<input
							type="range"
							class="wheel-slider"
							min="0"
							max="2"
							step="0.01"
							bind:value={volume}
							oninput={onVolumeInput}
						/>
					</div>
					<span class="param-readout">{volume.toFixed(2)}</span>
				</div>
				<div class="filter-group">
					<div class="filter-sel">
						<span class="param-label">FLOOR</span>
						<select bind:value={floorDb} onchange={onFilterChange}>
							<option value={-100}>OFF</option>
							<option value={-80}>-80 dB</option>
							<option value={-70}>-70 dB</option>
							<option value={-60}>-60 dB</option>
							<option value={-50}>-50 dB</option>
							<option value={-40}>-40 dB</option>
						</select>
					</div>
					<div class="filter-sel">
						<span class="param-label">LO</span>
						<select bind:value={freqLo} onchange={onFilterChange}>
							<option value={0}>OFF</option>
							<option value={50}>50</option>
							<option value={100}>100</option>
							<option value={200}>200</option>
							<option value={500}>500</option>
							<option value={1000}>1k</option>
							<option value={2000}>2k</option>
						</select>
					</div>
					<div class="filter-sel">
						<span class="param-label">HI</span>
						<select bind:value={freqHi} onchange={onFilterChange}>
							<option value={20000}>OFF</option>
							<option value={16000}>16k</option>
							<option value={12000}>12k</option>
							<option value={8000}>8k</option>
							<option value={4000}>4k</option>
							<option value={2000}>2k</option>
						</select>
					</div>
				</div>
			</div>
		</div>

		<!-- FPS -->
		<div class="ctrl-fps">
			<span class="fps-number">{fps}</span>
			<span class="fps-label">FPS</span>
		</div>
	</section>

	<!-- ─── Mobile credit (very bottom) ─── -->
	<a
		class="mobile-credit"
		href="https://sedum.studio"
		target="_blank"
		rel="noopener noreferrer">designed by <span>sedum.studio</span></a
	>
</main>

<style>
	/* ── Reset & base ────────────────────────────── */
	:global(html) {
		height: -webkit-fill-available;
	}

	:global(body) {
		margin: 0;
		padding: 0;
		background: #f2ede4;
		color: #2a2a32;
		font-family: "JetBrains Mono", "SF Mono", "Cascadia Code", "Consolas",
			monospace;
		overflow: hidden;
		-webkit-font-smoothing: antialiased;
		min-height: 100vh;
		min-height: -webkit-fill-available;
	}

	/* ── Landing overlay ─────────────────────────── */
	.landing {
		position: fixed;
		inset: 0;
		z-index: 100;
		background: #f2ede4;
		display: flex;
		align-items: center;
		justify-content: center;
		opacity: 1;
		transition: opacity 0.45s cubic-bezier(0.22, 0.68, 0.35, 1);
	}

	.landing.dismissing {
		opacity: 0;
		pointer-events: none;
	}

	.landing-content {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0;
		user-select: none;
	}

	/* ── Staggered reveal ─────────────────────── */
	.landing-content .stagger {
		opacity: 0;
		transform: translateY(12px);
	}

	.landing-content.revealed .stagger {
		animation: landingReveal 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
	}

	.landing-content.revealed .stagger:nth-child(1) { animation-delay: 0.04s; }
	.landing-content.revealed .stagger:nth-child(2) { animation-delay: 0.14s; }
	.landing-content.revealed .stagger:nth-child(3) { animation-delay: 0.24s; }
	.landing-content.revealed .stagger:nth-child(4) { animation-delay: 0.34s; }
	.landing-content.revealed .stagger:nth-child(5) { animation-delay: 0.46s; }
	.landing-content.revealed .stagger:nth-child(6) { animation-delay: 0.58s; }

	@keyframes landingReveal {
		from {
			opacity: 0;
			transform: translateY(12px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.landing-brand {
		display: flex;
		flex-direction: column;
		align-items: center;
		line-height: 1.05;
		margin-bottom: 22px;
	}

	.landing-top {
		font-size: 42px;
		font-weight: 600;
		letter-spacing: 18px;
		color: rgba(42, 42, 50, 0.52);
		margin-left: 18px; /* optical center for letter-spacing */
	}

	.landing-bottom {
		font-size: 42px;
		font-weight: 300;
		letter-spacing: 18px;
		color: rgba(42, 42, 50, 0.22);
		margin-left: 18px;
	}

	.landing-tagline {
		margin: 0 0 32px;
		font-size: 11px;
		font-weight: 400;
		letter-spacing: 2.5px;
		line-height: 1.7;
		color: rgba(42, 42, 50, 0.3);
		text-align: center;
		text-transform: uppercase;
		max-width: 50ch;
	}

	.landing-actions {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 18px;
	}

	.landing-cta {
		display: flex;
		align-items: center;
		gap: 12px;
		height: 44px;
		padding: 0 32px;
		border: 1.5px solid rgba(42, 42, 50, 0.18);
		background: transparent;
		color: rgba(42, 42, 50, 0.55);
		font-family: inherit;
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 3px;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.landing-cta:hover {
		border-color: rgba(42, 42, 50, 0.35);
		background: rgba(42, 42, 50, 0.03);
		color: rgba(42, 42, 50, 0.72);
	}

	.cta-icon {
		width: 0;
		height: 0;
		border-style: solid;
		border-width: 5px 0 5px 9px;
		border-color: transparent transparent transparent currentColor;
		flex-shrink: 0;
		opacity: 0.7;
	}

	.cta-text {
		margin-left: 2px;
	}

	/* Loading dots */
	.landing-loading {
		display: flex;
		gap: 6px;
		align-items: center;
		height: 44px;
	}

	.loading-dot {
		width: 3px;
		height: 3px;
		border-radius: 50%;
		background: rgba(42, 42, 50, 0.25);
		animation: loadingPulse 1.2s ease-in-out infinite;
	}

	.loading-dot:nth-child(2) {
		animation-delay: 0.15s;
	}
	.loading-dot:nth-child(3) {
		animation-delay: 0.3s;
	}

	@keyframes loadingPulse {
		0%,
		80%,
		100% {
			opacity: 0.2;
			transform: scale(1);
		}
		40% {
			opacity: 1;
			transform: scale(1.3);
		}
	}

	/* ── Grid layout ─────────────────────────────── */
	main {
		width: 100vw;
		height: 100vh;
		height: 100dvh;
		display: grid;
		grid-template-columns: 1fr 1fr 0.55fr;
		grid-template-rows: 1fr auto;
		grid-template-areas:
			"mel traj radar"
			"ctrl ctrl ctrl";
		gap: 1px;
		background: rgba(42, 42, 50, 0.18);
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
		color: rgba(42, 42, 50, 0.42);
		pointer-events: none;
		z-index: 2;
	}

	/* ── Panels ──────────────────────────────────── */
	.mel-cloud {
		grid-area: mel;
	}
	.trajectory {
		grid-area: traj;
	}
	.analysis {
		grid-area: radar;
	}
	.controls-bar {
		grid-area: ctrl;
	}

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
		border-top: 1px solid rgba(42, 42, 50, 0.14);
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

	/* ── Trajectory metrics overlay ──────────────── */
	.traj-metrics {
		position: absolute;
		bottom: 18px;
		right: 18px;
		display: flex;
		flex-direction: column;
		gap: 3px;
		pointer-events: none;
		z-index: 2;
	}

	.tm-row {
		display: flex;
		justify-content: flex-end;
		align-items: baseline;
		gap: 8px;
	}

	.tm-label {
		font-size: 8px;
		font-weight: 500;
		letter-spacing: 2px;
		color: rgba(42, 42, 50, 0.28);
	}

	.tm-value {
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 0.5px;
		color: rgba(42, 42, 50, 0.45);
		font-variant-numeric: tabular-nums;
		min-width: 3.5em;
		text-align: right;
	}

	/* ── Analysis layout ─────────────────────────── */
	.analysis {
		display: flex;
		flex-direction: column;
	}

	.analysis-radar {
		flex: 1;
		min-height: 0;
		position: relative;
		--radar-label-font: 500 9px "JetBrains Mono";
		--radar-label-color: rgba(42, 42, 50, 0.34);
		--radar-label-radius-offset: 24;
		--radar-label-y-offset: -8;
		--radar-value-font: 400 10px "JetBrains Mono";
		--radar-value-color: rgba(42, 42, 50, 0.4);
		--radar-value-y-offset: 11;
	}

	.analysis-pitch {
		height: 30%;
		min-height: 80px;
		border-top: 1px solid rgba(42, 42, 50, 0.14);
		position: relative;
	}

	.analysis-pitch canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.analysis-pitch .panel-label.sub {
		font-size: 10px;
		top: 12px;
		letter-spacing: 2px;
	}

	.analysis-scope {
		height: 22%;
		min-height: 50px;
		border-top: 1px solid rgba(42, 42, 50, 0.14);
		position: relative;
	}

	.analysis-scope canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.panel-label.sub {
		font-size: 9px;
		color: rgba(42, 42, 50, 0.3);
	}

	/* ── Controls bar ────────────────────────────── */
	.controls-bar {
		position: relative;
		display: flex;
		align-items: flex-end;
		--control-height: 28px;
		padding: 32px 32px 36px;
		padding-bottom: max(36px, calc(20px + env(safe-area-inset-bottom)));
		padding-left: max(32px, env(safe-area-inset-left));
		padding-right: max(32px, env(safe-area-inset-right));
		gap: 0.5em;
	}

	/* ── Brand ────────────────────────────────────── */
	.ctrl-brand {
		display: flex;
		align-items: flex-end;
		gap: 14px;
		padding-right: 28px;
		padding-bottom: 15px;
		flex-shrink: 0;
	}

	.brand-text {
		display: flex;
		flex-direction: column;
		line-height: 1.15;
		margin-bottom: -4px;
	}

	.brand-top {
		font-size: 15px;
		font-weight: 600;
		letter-spacing: 6px;
		color: rgba(42, 42, 50, 0.48);
	}

	.brand-bottom {
		font-size: 15px;
		font-weight: 300;
		letter-spacing: 6px;
		color: rgba(42, 42, 50, 0.28);
	}

	/* ── Separator ───────────────────────────────── */
	.ctrl-sep {
		width: 1px;
		height: 100%;
		background: rgba(42, 42, 50, 0.1);
		flex-shrink: 0;
		align-self: flex-end;
	}

	/* ── Sections ────────────────────────────────── */
	.ctrl-section {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 0 24px;
		min-width: 0;
	}

	.ctrl-section.grow {
		flex: 1;
	}

	.section-label {
		display: block;
		height: 10px;
		font-size: 9px;
		font-weight: 500;
		letter-spacing: 2.5px;
		color: rgba(42, 42, 50, 0.28);
		line-height: 1;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.section-body {
		display: flex;
		align-items: flex-end;
		gap: 8px;
		flex-wrap: nowrap;
		min-width: 0;
	}

	/* ── Play button ─────────────────────────────── */
	.ctrl-btn.play {
		width: var(--control-height);
		height: var(--control-height);
		padding: 0;
		border: 1.5px solid rgba(42, 42, 50, 0.18);
		border-radius: 50%;
		background: transparent;
		color: rgba(42, 42, 50, 0.45);
		font-family: inherit;
		cursor: pointer;
		transition: all 0.15s;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
	}

	.ctrl-btn.play:hover {
		border-color: rgba(42, 42, 50, 0.35);
		background: rgba(42, 42, 50, 0.03);
	}

	.ctrl-btn.play.active {
		border-color: rgba(160, 50, 50, 0.35);
		background: rgba(160, 50, 50, 0.04);
	}

	.icon-play {
		width: 0;
		height: 0;
		border-style: solid;
		border-width: 4px 0 4px 7px;
		border-color: transparent transparent transparent rgba(42, 42, 50, 0.5);
		flex-shrink: 0;
		margin-left: 1px;
	}

	.icon-stop {
		width: 8px;
		height: 8px;
		background: rgba(160, 50, 50, 0.5);
		flex-shrink: 0;
	}

	/* ── Input toggle ────────────────────────────── */
	.input-toggle {
		display: flex;
		height: var(--control-height);
		box-sizing: border-box;
		align-items: stretch;
		border: 1px solid rgba(42, 42, 50, 0.12);
		overflow: hidden;
	}

	.toggle-btn {
		height: 100%;
		padding: 0 12px;
		border: none;
		background: transparent;
		color: rgba(42, 42, 50, 0.35);
		font-family: inherit;
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 2px;
		display: flex;
		align-items: center;
		justify-content: center;
		cursor: pointer;
		transition: all 0.12s;
	}

	.toggle-btn:hover {
		color: rgba(42, 42, 50, 0.6);
		background: rgba(42, 42, 50, 0.02);
	}

	.toggle-btn.active {
		color: rgba(42, 42, 50, 0.7);
		background: rgba(42, 42, 50, 0.05);
	}

	.toggle-btn:disabled {
		opacity: 0.35;
		cursor: not-allowed;
	}

	.toggle-btn + .toggle-btn {
		border-left: 1px solid rgba(42, 42, 50, 0.08);
	}

	.file-btn {
		height: var(--control-height);
		box-sizing: border-box;
		padding: 0 12px;
		border: 1px dashed rgba(42, 42, 50, 0.18);
		background: transparent;
		color: rgba(42, 42, 50, 0.4);
		font-family: inherit;
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 1px;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		cursor: pointer;
		transition: border-color 0.12s;
		position: relative;
	}

	.file-btn:hover {
		border-color: rgba(42, 42, 50, 0.3);
	}

	.file-hidden {
		position: absolute;
		width: 0;
		height: 0;
		opacity: 0;
		pointer-events: none;
	}

	/* ── Parameters ──────────────────────────────── */
	.params-body {
		gap: 16px;
		min-width: 0;
	}

	.param {
		display: flex;
		align-items: center;
		gap: 9px;
	}

	.param-label {
		font-size: 10px;
		font-weight: 500;
		letter-spacing: 2px;
		color: rgba(42, 42, 50, 0.38);
		white-space: nowrap;
	}

	.param-wheel {
		position: relative;
		width: 120px;
		min-width: 60px;
		height: var(--control-height);
		display: flex;
		align-items: center;
		flex-shrink: 1;
	}

	.param-wheel::before {
		content: "";
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		height: 14px;
		transform: translateY(-50%);
		background: repeating-linear-gradient(
			to right,
			rgba(42, 42, 50, 0.22) 0px,
			rgba(42, 42, 50, 0.22) 1px,
			transparent 1px,
			transparent 8px
		);
		opacity: 0.8;
		pointer-events: none;
	}

	.param-wheel::after {
		content: "";
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		height: 1px;
		transform: translateY(-50%);
		background: rgba(42, 42, 50, 0.1);
		pointer-events: none;
	}

	.wheel-slider {
		-webkit-appearance: none;
		appearance: none;
		width: 100%;
		height: var(--control-height);
		background: transparent;
		border-radius: 0;
		outline: none;
		cursor: pointer;
		position: relative;
		z-index: 1;
	}

	.wheel-slider::-webkit-slider-runnable-track {
		height: var(--control-height);
		background: transparent;
	}

	.wheel-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 2px;
		height: 20px;
		margin-top: 4px;
		border-radius: 1px;
		background: rgba(42, 42, 50, 0.55);
		border: none;
		cursor: pointer;
		transition: background 0.12s;
	}

	.wheel-slider::-webkit-slider-thumb:hover {
		background: rgba(42, 42, 50, 0.75);
	}

	.wheel-slider::-moz-range-track {
		height: var(--control-height);
		background: transparent;
	}

	.wheel-slider::-moz-range-thumb {
		width: 2px;
		height: 20px;
		border-radius: 1px;
		background: rgba(42, 42, 50, 0.55);
		border: none;
		cursor: pointer;
	}

	.param-readout {
		width: 32px;
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 1px;
		text-align: right;
		color: rgba(42, 42, 50, 0.34);
		font-variant-numeric: tabular-nums;
	}

	/* ── Filter selects ─────────────────────────── */
	.filter-group {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-left: 8px;
		padding-left: 12px;
		border-left: 1px solid rgba(42, 42, 50, 0.08);
		flex-shrink: 1;
		min-width: 0;
	}

	.filter-sel {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.filter-sel select {
		height: var(--control-height);
		box-sizing: border-box;
		padding: 0 5px;
		border: 1px solid rgba(42, 42, 50, 0.1);
		background: transparent;
		color: rgba(42, 42, 50, 0.55);
		font-family: inherit;
		font-size: 10px;
		font-weight: 400;
		cursor: pointer;
		outline: none;
		transition: border-color 0.12s;
	}

	.filter-sel select:hover {
		border-color: rgba(42, 42, 50, 0.22);
	}
	.filter-sel select:focus {
		border-color: rgba(42, 42, 50, 0.3);
	}

	/* ── FPS ──────────────────────────────────────── */
	.ctrl-fps {
		display: flex;
		align-items: baseline;
		gap: 4px;
		margin-left: auto;
		padding-left: 16px;
		flex-shrink: 0;
		align-self: flex-end;
	}

	.fps-number {
		font-size: 14px;
		font-weight: 300;
		color: rgba(42, 42, 50, 0.3);
		font-variant-numeric: tabular-nums;
		line-height: 1;
	}

	.fps-label {
		font-size: 8px;
		font-weight: 500;
		letter-spacing: 2px;
		color: rgba(42, 42, 50, 0.2);
	}

	/* ── Status badge ────────────────────────────── */
	.status-badge {
		display: inline-block;
		width: 72px;
		padding: 4px 0;
		text-align: center;
		font-size: 9px;
		font-weight: 500;
		letter-spacing: 2px;
		color: rgba(42, 42, 50, 0.35);
		border: 1px solid rgba(42, 42, 50, 0.12);
		background: transparent;
		transition:
			color 0.2s,
			border-color 0.2s;
		flex-shrink: 0;
		font-variant-numeric: tabular-nums;
	}

	.status-badge.active {
		color: rgba(160, 50, 50, 0.7);
		border-color: rgba(160, 50, 50, 0.25);
	}

	.design-credit {
		position: absolute;
		bottom: 34px;
		left: 32px;
		font-size: 9px;
		font-weight: 600;
		letter-spacing: 0.35px;
		color: rgba(34, 34, 41, 0.25);
		text-decoration: none;
		text-transform: lowercase;
		transition: color 0.15s ease;
		white-space: nowrap;
	}

	.design-credit span {
		font-weight: 800;
		letter-spacing: 1.2px;
		border-bottom: 1px solid rgba(42, 42, 50, 0.18);
		padding-bottom: 1px;
	}

	.design-credit:hover {
		color: rgba(42, 42, 50, 0.45);
	}

	/* ── Desktop: card wrappers are transparent ─── */
	.cards-viewport {
		display: contents;
	}
	.cards-track {
		display: contents;
	}
	.mobile-header {
		display: none;
	}
	.card-indicator {
		display: none;
	}
	.mobile-credit {
		display: none;
	}

	/* ── Responsive (desktop) ────────────────────── */
	@media (max-width: 1200px) {
		.ctrl-section {
			padding: 0 16px;
		}
		.ctrl-brand {
			padding-right: 18px;
		}
		.param-wheel {
			width: 90px;
		}
		.filter-group {
			gap: 6px;
			padding-left: 8px;
			margin-left: 4px;
		}
	}

	@media (max-width: 1065px) and (min-width: 961px) {
		.ctrl-fps {
			display: none;
		}
		.ctrl-section {
			padding: 0 14px;
		}
		.param-wheel {
			width: 80px;
		}
	}

	@media (max-width: 960px) and (min-width: 769px) {
		.controls-bar {
			padding: 24px 20px 28px;
			padding-bottom: max(28px, calc(12px + env(safe-area-inset-bottom)));
			gap: 0.8em;
		}
		.ctrl-section {
			padding: 0 10px;
		}
		.param-wheel {
			width: 70px;
		}
		.filter-group {
			display: none;
		}
		.ctrl-fps {
			display: none;
		}
		.ctrl-brand {
			gap: 10px;
			padding-right: 14px;
		}
		.brand-top,
		.brand-bottom {
			font-size: 15px;
			letter-spacing: 4px;
		}
		.design-credit {
			font-size: 8px;
			letter-spacing: 0.2px;
			left: 20px;
			bottom: 28px;
		}
	}

	@media (max-width: 860px) and (min-width: 769px) {
		.ctrl-section {
			padding: 0 7px;
		}
		.ctrl-brand {
			padding-right: 8px;
		}
	}

	/* ═══════════════════════════════════════════════
	   MOBILE LAYOUT (≤768px)
	   ═══════════════════════════════════════════════ */
	@media (max-width: 768px) {
		/* ── Main grid → flex column ──────────── */
		main.mobile {
			display: flex;
			flex-direction: column;
			grid-template-columns: unset;
			grid-template-rows: unset;
			grid-template-areas: unset;
			gap: 0;
			background: #f2ede4;
		}

		/* ── Mobile header ────────────────────── */
		.mobile-header {
			display: flex;
			align-items: flex-end;
			padding: 14px 18px 12px;
			padding-top: calc(14px + env(safe-area-inset-top));
			background: #f2ede4;
			flex-shrink: 0;
			z-index: 10;
			border-bottom: 1px solid rgba(42, 42, 50, 0.1);
			gap: 14px;
		}

		.mobile-brand {
			display: flex;
			flex-direction: column;
			line-height: 1.15;
			transform: translateY(3px);
		}

		.mobile-brand .brand-top {
			font-size: 12px;
			font-weight: 600;
			letter-spacing: 5px;
			color: rgba(42, 42, 50, 0.48);
		}

		.mobile-brand .brand-bottom {
			font-size: 12px;
			font-weight: 300;
			letter-spacing: 5px;
			color: rgba(42, 42, 50, 0.28);
		}

		.mobile-status {
			width: auto;
			min-width: 60px;
			padding: 3px 8px;
			font-size: 8px;
			margin-left: auto;
		}

		.mobile-fps {
			display: flex;
			align-items: baseline;
			gap: 3px;
			flex-shrink: 0;
		}

		.mobile-fps .fps-number {
			font-size: 12px;
			font-weight: 300;
			color: rgba(42, 42, 50, 0.25);
			font-variant-numeric: tabular-nums;
			line-height: 1;
		}

		.mobile-fps .fps-label {
			font-size: 7px;
			font-weight: 500;
			letter-spacing: 1.5px;
			color: rgba(42, 42, 50, 0.18);
		}

		/* ── Cards viewport ──────────────────── */
		.cards-viewport {
			display: block;
			flex: 1;
			min-height: 0;
			overflow: hidden;
			position: relative;
		}

		.cards-track {
			display: flex;
			height: 100%;
			will-change: transform;
		}

		.cards-track > .panel {
			width: 100vw;
			flex-shrink: 0;
			height: 100%;
			box-sizing: border-box;
			touch-action: pan-y pinch-zoom;
		}

		.cards-track > .panel:first-child {
			border-left: none;
		}

		.swipe-hint {
			position: absolute;
			top: 18px;
			right: 0;
			transform: none;
			display: flex;
			align-items: center;
			gap: 8px;
			padding: 10px 10px 10px 24px;
			background: linear-gradient(
				90deg,
				rgba(242, 237, 228, 0) 0%,
				rgba(242, 237, 228, 0.68) 34%,
				rgba(242, 237, 228, 0.94) 64%,
				#f2ede4 100%
			);
			pointer-events: none;
			z-index: 3;
			animation: swipeCue 1.45s ease-in-out infinite;
		}

		.swipe-hint-text {
			font-size: 7px;
			font-weight: 600;
			letter-spacing: 2.4px;
			color: rgba(42, 42, 50, 0.28);
		}

		.swipe-hint-arrows {
			display: flex;
			align-items: center;
			gap: 1px;
		}

		.swipe-hint-arrow {
			display: block;
			width: 7px;
			height: 7px;
			border-left: 0.75px solid rgba(42, 42, 50, 0.26);
			border-bottom: 0.75px solid rgba(42, 42, 50, 0.26);
			transform: rotate(45deg);
		}

		.swipe-hint-arrow:last-child {
			opacity: 0.75;
		}


		/* ── Card indicator ───────────────────── */
		.card-indicator {
			display: flex;
			align-items: center;
			justify-content: space-between;
			padding: 9px 18px;
			background: #f2ede4;
			flex-shrink: 0;
			border-top: 1px solid rgba(42, 42, 50, 0.1);
		}

		.card-label {
			font-size: 9px;
			font-weight: 500;
			letter-spacing: 2.5px;
			color: rgba(42, 42, 50, 0.35);
			min-width: 100px;
		}

		.card-dots {
			display: flex;
			gap: 6px;
			align-items: center;
		}

		.card-dot {
			width: 16px;
			height: 2px;
			padding: 0;
			border: none;
			background: rgba(42, 42, 50, 0.12);
			cursor: pointer;
			transition: all 0.3s ease;
		}

		.card-dot.active {
			width: 24px;
			background: rgba(42, 42, 50, 0.4);
		}

		@keyframes swipeCue {
			0%,
			100% {
				transform: translateX(0);
				opacity: 0.35;
			}

			50% {
				transform: translateX(-6px);
				opacity: 1;
			}
		}

		/* ── Panel labels ────────────────────── */
		.panel-label:not(.sub) {
			display: none;
		}

		.panel-label.sub {
			top: 8px;
			left: 14px;
			font-size: 8px;
		}

		.axes-overlay {
			bottom: 12px;
			left: 14px;
			font-size: 9px;
		}

		/* ── Analysis panel ──────────────────── */
		.analysis-radar {
			--radar-label-font: 500 8px "JetBrains Mono";
			--radar-label-radius-offset: 20;
			--radar-value-font: 400 9px "JetBrains Mono";
		}

		/* ── Controls bar (mobile) ───────────── */
		.controls-bar {
			flex-direction: column;
			align-items: stretch;
			gap: 12px;
			padding: 16px 18px 12px;
			padding-bottom: calc(env(safe-area-inset-bottom));
			flex-shrink: 0;
			border-top: 1px solid rgba(42, 42, 50, 0.1);
		}

		/* Hide desktop-only elements */
		.ctrl-brand {
			display: none;
		}
		.ctrl-sep {
			display: none;
		}
		.filter-group {
			display: none;
		}
		.ctrl-fps {
			display: none;
		}

		/* Section labels visible as row headers */
		.section-label {
			font-size: 8px;
			letter-spacing: 2px;
			color: rgba(42, 42, 50, 0.22);
			height: auto;
			margin-bottom: 6px;
		}

		/* Input section */
		.ctrl-section {
			padding: 0;
			gap: 0;
		}

		.controls-bar > .ctrl-section:not(.grow) .section-label {
			display: none;
		}

		.controls-bar > .ctrl-section:first-of-type .section-body {
			padding-right: 132px;
		}

		.section-body {
			gap: 10px;
			align-items: center;
		}

		.ctrl-btn.play {
			width: 30px;
			height: 30px;
		}

		.input-toggle {
			height: 30px;
		}

		.toggle-btn {
			padding: 0 12px;
			font-size: 9px;
			letter-spacing: 1.5px;
		}

		.file-btn {
			height: 30px;
			padding: 0 10px;
			font-size: 9px;
		}

		/* Parameters section: full-width volume */
		.ctrl-section.grow {
			flex: none;
			width: 100%;
		}

		.params-body {
			gap: 8px;
		}

		.param {
			gap: 8px;
			width: 100%;
		}

		.param-label {
			font-size: 9px;
			flex-shrink: 0;
		}

		.param-wheel {
			flex: 1;
			width: auto;
			min-width: 0;
		}

		.param-readout {
			width: 30px;
			font-size: 9px;
		}

		/* ── Credit inside controls on mobile ──────────────── */
		.design-credit {
			display: none;
		}

		/* ── Mobile credit (very bottom of page) ─────────── */
		.mobile-credit {
			display: block;
			text-align: left;
			padding: 8px 18px;
			padding-bottom: calc(15px + env(safe-area-inset-bottom));
			font-size: 8px;
			font-weight: 600;
			letter-spacing: 0.35px;
			color: rgba(34, 34, 41, 0.22);
			text-decoration: none;
			text-transform: lowercase;
			flex-shrink: 0;
			transition: color 0.15s ease;
		}

		.mobile-credit span {
			font-weight: 800;
			letter-spacing: 1px;
			border-bottom: 1px solid rgba(42, 42, 50, 0.15);
			padding-bottom: 1px;
		}

		.mobile-credit:hover {
			color: rgba(42, 42, 50, 0.4);
		}

		/* ── Mel panel mobile adjustments ────── */
		.mel-2d {
			height: 12%;
			min-height: 24px;
		}

		/* ── Landing mobile ──────────────────── */
		.landing {
			padding: env(safe-area-inset-top) 0 env(safe-area-inset-bottom);
		}

		.wytham-landing {
			max-width: 320px;
			padding: 0 28px;
		}

		.landing-brand {
			margin-bottom: 18px;
		}

		.landing-top,
		.landing-bottom {
			font-size: 28px;
			letter-spacing: 12px;
			margin-left: 12px;
		}

		.landing-tagline {
			font-size: 9.5px;
			letter-spacing: 1.8px;
			line-height: 1.8;
			margin-bottom: 24px;
			max-width: 40ch;
		}

		.landing-cta {
			height: 42px;
			padding: 0 30px;
			font-size: 10px;
			letter-spacing: 2.5px;
		}

		.wytham-illustration {
			margin-bottom: 24px;
		}

		.wytham-illustration img {
			width: 100px;
			height: 100px;
			object-fit: contain;
		}

		.wytham-description {
			font-size: 9px;
			font-weight: 400;
			letter-spacing: 1.5px;
			line-height: 2;
			margin-bottom: 24px;
			max-width: 28ch;
		}

		.wytham-links {
			margin-top: 24px;
			flex-wrap: wrap;
			gap: 4px 0;
			max-width: 260px;
		}

		.wytham-link {
			font-size: 8px;
			letter-spacing: 2px;
			padding: 4px 10px;
		}

		.wytham-link-sep {
			height: 8px;
		}
	}

	/* ── Wytham Tits landing ────────────────────── */
	.wytham-landing {
		max-width: 440px;
		padding: 0 32px;
	}

	.wytham-illustration {
		margin-bottom: 32px;
	}

	.wytham-illustration svg,
	.wytham-illustration img {
		width: 140px;
		height: 140px;
		object-fit: contain;
		opacity: 0.85;
	}

	.wytham-latin {
		font-style: italic;
		letter-spacing: 1.5px;
		color: rgba(42, 42, 50, 0.22);
	}

	.wytham-description {
		margin: 0 0 30px;
		font-size: 10px;
		font-weight: 400;
		letter-spacing: 2px;
		line-height: 2;
		color: rgba(42, 42, 50, 0.22);
		text-align: center;
		text-transform: uppercase;
		max-width: 34ch;
	}

	.wytham-links {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0;
		margin-top: 40px;
	}

	.wytham-link {
		font-size: 9px;
		font-weight: 500;
		letter-spacing: 2.5px;
		color: rgba(42, 42, 50, 0.2);
		text-decoration: none;
		text-transform: uppercase;
		transition: color 0.15s;
		padding: 4px 14px;
	}

	.wytham-link:hover {
		color: rgba(42, 42, 50, 0.45);
	}

	.wytham-link-sep {
		width: 1px;
		height: 10px;
		background: rgba(42, 42, 50, 0.1);
		flex-shrink: 0;
	}
</style>
