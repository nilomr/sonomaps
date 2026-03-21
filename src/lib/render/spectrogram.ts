/**
 * Scrolling mel spectrogram rendered on Canvas 2D.
 *
 * Each animation frame, the existing canvas content is shifted left by
 * 1 pixel and a new column is drawn on the right edge using the
 * drawImage self-copy trick.
 *
 * Uses auto-ranging: tracks the actual min/max of log mel energies
 * over time so the colormap always uses the full dynamic range,
 * regardless of mic gain or input level.
 */

// Pre-computed 256-entry colour LUT (inferno-inspired).
const COLORMAP = buildColormap();

function buildColormap(): Uint8Array {
	const lut = new Uint8Array(256 * 3);
	for (let i = 0; i < 256; i++) {
		const t = i / 255;
		let r: number, g: number, b: number;
		if (t < 0.25) {
			const s = t / 0.25;
			r = s * 80;  g = s * 10;  b = s * 120;
		} else if (t < 0.5) {
			const s = (t - 0.25) / 0.25;
			r = 80 + s * 150;  g = 10 + s * 20;  b = 120 - s * 60;
		} else if (t < 0.75) {
			const s = (t - 0.5) / 0.25;
			r = 230 + s * 25;  g = 30 + s * 140;  b = 60 - s * 50;
		} else {
			const s = (t - 0.75) / 0.25;
			r = 255;  g = 170 + s * 75;  b = 10 + s * 200;
		}
		lut[i * 3] = Math.min(255, Math.round(r));
		lut[i * 3 + 1] = Math.min(255, Math.round(g));
		lut[i * 3 + 2] = Math.min(255, Math.round(b));
	}
	return lut;
}

export class SpectrogramRenderer {
	private readonly canvas: HTMLCanvasElement;
	private readonly ctx: CanvasRenderingContext2D;
	private readonly numBands: number;

	// Auto-ranging: tracks actual value range with EMA
	private rangeMin = -30;
	private rangeMax = -5;
	private readonly rangeDecay = 0.995; // slow adaptation

	private columnData: ImageData | null = null;
	private initialized = false;

	constructor(canvas: HTMLCanvasElement, numBands: number) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d')!;
		this.numBands = numBands;
	}

	addColumn(logMelEnergies: Float32Array): void {
		const { canvas, ctx, numBands } = this;

		// Lazy resize on first call (ensures layout is ready)
		if (!this.initialized) {
			this.resize();
			this.initialized = true;
		}

		const w = canvas.width;
		const h = canvas.height;
		if (w === 0 || h === 0) return;

		// ── Auto-range: adapt to actual data ─────────────────
		let frameMin = Infinity;
		let frameMax = -Infinity;
		for (let m = 0; m < numBands; m++) {
			const v = logMelEnergies[m];
			if (v < frameMin) frameMin = v;
			if (v > frameMax) frameMax = v;
		}
		// EMA on the range so it adapts smoothly
		const d = this.rangeDecay;
		this.rangeMin = Math.min(this.rangeMin, frameMin) * d + frameMin * (1 - d);
		this.rangeMax = Math.max(this.rangeMax, frameMax) * d + frameMax * (1 - d);
		// Ensure minimum range so we don't divide by zero in silence
		const range = Math.max(this.rangeMax - this.rangeMin, 5);
		const rMin = this.rangeMin;

		// ── Shift canvas left by 2px (faster scroll) ────────
		const scrollPx = 2;
		if (!this.columnData || this.columnData.height !== h || this.columnData.width !== scrollPx) {
			this.columnData = ctx.createImageData(scrollPx, h);
		}
		ctx.drawImage(canvas, -scrollPx, 0);

		// ── Map mel bands → pixels ───────────────────────────
		const data = this.columnData.data;
		for (let row = 0; row < h; row++) {
			// row 0 = top = highest freq, row h-1 = bottom = lowest freq
			const bandIdx = Math.min(
				numBands - 1,
				Math.floor(((h - 1 - row) / h) * numBands)
			);
			const norm = (logMelEnergies[bandIdx] - rMin) / range;
			const idx = Math.max(0, Math.min(255, Math.round(norm * 255)));

			const r = COLORMAP[idx * 3];
			const g = COLORMAP[idx * 3 + 1];
			const b = COLORMAP[idx * 3 + 2];

			// Fill both pixel columns with the same data
			for (let col = 0; col < scrollPx; col++) {
				const off = (row * scrollPx + col) * 4;
				data[off] = r;
				data[off + 1] = g;
				data[off + 2] = b;
				data[off + 3] = 255;
			}
		}

		ctx.putImageData(this.columnData, w - scrollPx, 0);
	}

	resize(): void {
		const parent = this.canvas.parentElement;
		if (!parent) return;
		const rect = parent.getBoundingClientRect();
		// Use 1:1 pixel ratio for the spectrogram — DPR scaling
		// just wastes fill rate and doesn't improve readability
		const w = Math.round(rect.width);
		const h = Math.round(rect.height);
		if (w > 0 && h > 0 && (this.canvas.width !== w || this.canvas.height !== h)) {
			this.canvas.width = w;
			this.canvas.height = h;
			this.columnData = null;
		}
	}
}
