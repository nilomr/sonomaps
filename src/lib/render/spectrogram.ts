/**
 * Scrolling mel spectrogram on Canvas 2D.
 *
 * Light-background aesthetic: cream → warm gray → dark charcoal.
 * Silence blends with background; loud regions are dark and precise.
 */

// Colormap: cream background → warm gray → charcoal
const COLORMAP = buildColormap();

function buildColormap(): Uint8Array {
	const lut = new Uint8Array(256 * 3);
	// Background cream: rgb(242, 237, 228) = #f2ede4
	// Dark endpoint: rgb(20, 20, 28) = #14141c
	for (let i = 0; i < 256; i++) {
		const t = i / 255;
		// Smooth cubic ramp — subtle at low energy, resolves detail at high
		const s = t * t * (3 - 2 * t); // smoothstep

		const r = Math.round(242 - s * 222);
		const g = Math.round(237 - s * 217);
		const b = Math.round(228 - s * 200);

		lut[i * 3] = r;
		lut[i * 3 + 1] = g;
		lut[i * 3 + 2] = b;
	}
	return lut;
}

export class SpectrogramRenderer {
	private readonly canvas: HTMLCanvasElement;
	private readonly ctx: CanvasRenderingContext2D;
	private readonly numBands: number;

	// Auto-ranging
	private rangeMin = -30;
	private rangeMax = -5;
	private readonly rangeDecay = 0.995;

	private columnData: ImageData | null = null;
	private initialized = false;

	constructor(canvas: HTMLCanvasElement, numBands: number) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d')!;
		this.numBands = numBands;
	}

	addColumn(logMelEnergies: Float32Array): void {
		const { canvas, ctx, numBands } = this;

		if (!this.initialized) {
			this.resize();
			// Fill with background color
			ctx.fillStyle = '#f2ede4';
			ctx.fillRect(0, 0, canvas.width, canvas.height);
			this.initialized = true;
		}

		const w = canvas.width;
		const h = canvas.height;
		if (w === 0 || h === 0) return;

		// Auto-range
		let frameMin = Infinity;
		let frameMax = -Infinity;
		for (let m = 0; m < numBands; m++) {
			const v = logMelEnergies[m];
			if (v < frameMin) frameMin = v;
			if (v > frameMax) frameMax = v;
		}
		const d = this.rangeDecay;
		this.rangeMin = Math.min(this.rangeMin, frameMin) * d + frameMin * (1 - d);
		this.rangeMax = Math.max(this.rangeMax, frameMax) * d + frameMax * (1 - d);
		const range = Math.max(this.rangeMax - this.rangeMin, 5);
		const rMin = this.rangeMin;

		// Scroll left 1px — single-pixel columns for sharper temporal resolution
		const scrollPx = 1;
		if (!this.columnData || this.columnData.height !== h || this.columnData.width !== scrollPx) {
			this.columnData = ctx.createImageData(scrollPx, h);
		}
		ctx.drawImage(canvas, -scrollPx, 0);

		const data = this.columnData.data;
		for (let row = 0; row < h; row++) {
			const bandIdx = Math.min(
				numBands - 1,
				Math.floor(((h - 1 - row) / h) * numBands)
			);
			const norm = (logMelEnergies[bandIdx] - rMin) / range;
			const idx = Math.max(0, Math.min(255, Math.round(norm * 255)));

			const r = COLORMAP[idx * 3];
			const g = COLORMAP[idx * 3 + 1];
			const b = COLORMAP[idx * 3 + 2];

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
		const w = Math.round(rect.width);
		const h = Math.round(rect.height);
		if (w > 0 && h > 0 && (this.canvas.width !== w || this.canvas.height !== h)) {
			this.canvas.width = w;
			this.canvas.height = h;
			this.columnData = null;
		}
	}
}
