/**
 * Oscilloscope renderer — cream/charcoal aesthetic with auto-scaling.
 *
 * Trigger-stabilized waveform with persistence trail,
 * automatic amplitude scaling, and multi-pass glow rendering.
 */

const BG = 'rgba(242, 237, 228, ';
const GRID = 'rgba(42, 42, 50, 0.05)';
const CENTER = 'rgba(42, 42, 50, 0.08)';
const GLOW = 'rgba(42, 42, 50, 0.03)';
const LINE = 'rgba(42, 42, 50, 0.22)';
const BRIGHT = 'rgba(42, 42, 50, 0.45)';
const FADE = 0.35;

export class OscilloscopeRenderer {
	private readonly canvas: HTMLCanvasElement;
	private readonly ctx: CanvasRenderingContext2D;
	private cssW = 0;
	private cssH = 0;
	private ampScale = 1;

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d', { alpha: false })!;
	}

	draw(timeData: Float32Array): void {
		this.resizeIfNeeded();
		const { ctx, cssW: w, cssH: h } = this;
		if (w === 0 || h === 0) return;

		// Phosphor fade
		ctx.fillStyle = BG + FADE + ')';
		ctx.fillRect(0, 0, w, h);

		// Grid
		ctx.strokeStyle = GRID;
		ctx.lineWidth = 0.5;
		for (let i = 1; i < 8; i++) {
			const x = Math.round((i / 8) * w) + 0.5;
			ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
		}
		for (let i = 1; i < 4; i++) {
			const y = Math.round((i / 4) * h) + 0.5;
			ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
		}
		ctx.strokeStyle = CENTER;
		ctx.lineWidth = 0.8;
		const cy = Math.round(h / 2) + 0.5;
		ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();

		if (timeData.length === 0) return;

		// Auto-scale amplitude
		let maxAmp = 0;
		for (let i = 0; i < timeData.length; i++) {
			const a = Math.abs(timeData[i]);
			if (a > maxAmp) maxAmp = a;
		}
		const target = maxAmp > 0.002 ? Math.min(0.8 / maxAmp, 20) : this.ampScale;
		// Asymmetric: fast grow, slow shrink
		const rate = target > this.ampScale ? 0.25 : 0.04;
		this.ampScale += (target - this.ampScale) * rate;

		// Trigger: rising zero-crossing
		let trigger = 0;
		const half = timeData.length >> 1;
		for (let i = 1; i < half; i++) {
			if (timeData[i - 1] < 0 && timeData[i] >= 0) { trigger = i; break; }
		}

		const step = half / w;
		const scale = this.ampScale;
		this.wave(timeData, trigger, step, w, h, scale, GLOW, 5);
		this.wave(timeData, trigger, step, w, h, scale, LINE, 1.8);
		this.wave(timeData, trigger, step, w, h, scale, BRIGHT, 0.7);
	}

	private wave(
		data: Float32Array, trigger: number, step: number,
		w: number, h: number, scale: number, color: string, lw: number
	): void {
		const { ctx } = this;
		ctx.strokeStyle = color;
		ctx.lineWidth = lw;
		ctx.beginPath();
		for (let x = 0; x < w; x++) {
			const i = trigger + Math.floor(x * step);
			const s = (i < data.length ? data[i] : 0) * scale;
			const y = (1 - s) * h / 2;
			if (x === 0) ctx.moveTo(x, y);
			else ctx.lineTo(x, y);
		}
		ctx.stroke();
	}

	private resizeIfNeeded(): void {
		const dpr = window.devicePixelRatio || 1;
		const parent = this.canvas.parentElement;
		if (!parent) return;
		const w = Math.round(parent.clientWidth);
		const h = Math.round(parent.clientHeight);
		if (w <= 0 || h <= 0) return;
		const pw = Math.round(w * dpr);
		const ph = Math.round(h * dpr);
		if (this.canvas.width !== pw || this.canvas.height !== ph) {
			this.canvas.width = pw;
			this.canvas.height = ph;
			this.ctx.resetTransform();
			this.ctx.scale(dpr, dpr);
			this.ctx.fillStyle = '#f2ede4';
			this.ctx.fillRect(0, 0, w, h);
		}
		this.cssW = w;
		this.cssH = h;
	}
}
