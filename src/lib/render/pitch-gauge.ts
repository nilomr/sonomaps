/**
 * Arc pitch gauge — logarithmic frequency display with trail.
 *
 * Renders a precision instrument arc showing detected pitch
 * on a log-frequency scale with octave markers, a persistence
 * trail of recent detections, and Hz/note readout at center.
 */

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const ARC_START = Math.PI * 0.78;
const ARC_END = Math.PI * 2.22;
const LOG_MIN = Math.log2(32.7);   // C1
const LOG_MAX = Math.log2(4186);   // C8
const LOG_RANGE = LOG_MAX - LOG_MIN;

const OCTAVE_LABELS = [
	{ freq: 65.41, label: 'C2' },
	{ freq: 130.81, label: 'C3' },
	{ freq: 261.63, label: 'C4' },
	{ freq: 523.25, label: 'C5' },
	{ freq: 1046.5, label: 'C6' },
	{ freq: 2093, label: 'C7' },
];

export class PitchGaugeRenderer {
	private readonly canvas: HTMLCanvasElement;
	private readonly ctx: CanvasRenderingContext2D;
	private cssW = 0;
	private cssH = 0;

	private smoothFreq = 0;
	private readonly trail: number[] = [];
	private readonly trailMax = 60;

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d')!;
	}

	draw(freqHz: number): void {
		this.resizeIfNeeded();
		const { ctx, cssW: w, cssH: h } = this;
		if (w === 0 || h === 0) return;

		ctx.fillStyle = '#f2ede4';
		ctx.fillRect(0, 0, w, h);

		const cx = w / 2;
		const cy = h * 0.52;
		const r = Math.min(w * 0.42, h * 0.40);

		// Smooth frequency
		if (freqHz > 20) {
			this.smoothFreq = this.smoothFreq > 20
				? this.smoothFreq * 0.82 + freqHz * 0.18
				: freqHz;
			this.trail.push(this.smoothFreq);
			if (this.trail.length > this.trailMax) this.trail.shift();
		}

		this.drawArc(cx, cy, r);
		this.drawTrail(cx, cy, r);
		if (this.smoothFreq > 20) this.drawMarker(cx, cy, r, this.smoothFreq);
		this.drawText(cx, cy, freqHz);
	}

	reset(): void {
		this.trail.length = 0;
		this.smoothFreq = 0;
	}

	private freqToAngle(freq: number): number {
		const t = (Math.log2(Math.max(32.7, Math.min(4186, freq))) - LOG_MIN) / LOG_RANGE;
		return ARC_START + t * (ARC_END - ARC_START);
	}

	private drawArc(cx: number, cy: number, r: number): void {
		const { ctx } = this;

		// Main arc
		ctx.strokeStyle = 'rgba(42, 42, 50, 0.07)';
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.arc(cx, cy, r, ARC_START, ARC_END);
		ctx.stroke();

		// Octave ticks + labels
		for (const oct of OCTAVE_LABELS) {
			const a = this.freqToAngle(oct.freq);
			const cos = Math.cos(a);
			const sin = Math.sin(a);

			ctx.strokeStyle = 'rgba(42, 42, 50, 0.12)';
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(cx + cos * (r - 5), cy + sin * (r - 5));
			ctx.lineTo(cx + cos * (r + 5), cy + sin * (r + 5));
			ctx.stroke();

			ctx.fillStyle = 'rgba(42, 42, 50, 0.2)';
			ctx.font = '500 8px "JetBrains Mono"';
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.fillText(oct.label, cx + cos * (r + 15), cy + sin * (r + 15));
		}

		// Semitone ticks (subtle)
		for (let midi = 36; midi <= 96; midi++) {
			if (midi % 12 === 0) continue;
			const freq = 440 * Math.pow(2, (midi - 69) / 12);
			const a = this.freqToAngle(freq);
			ctx.strokeStyle = 'rgba(42, 42, 50, 0.04)';
			ctx.lineWidth = 0.5;
			ctx.beginPath();
			ctx.moveTo(cx + Math.cos(a) * (r - 2), cy + Math.sin(a) * (r - 2));
			ctx.lineTo(cx + Math.cos(a) * (r + 2), cy + Math.sin(a) * (r + 2));
			ctx.stroke();
		}
	}

	private drawTrail(cx: number, cy: number, r: number): void {
		const { ctx, trail } = this;
		const n = trail.length;
		for (let i = 0; i < n; i++) {
			const t = i / n;
			const alpha = t * t * 0.22;
			const a = this.freqToAngle(trail[i]);
			ctx.fillStyle = `rgba(42, 42, 50, ${alpha.toFixed(3)})`;
			ctx.beginPath();
			ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 1.5 + t * 1, 0, Math.PI * 2);
			ctx.fill();
		}
	}

	private drawMarker(cx: number, cy: number, r: number, freq: number): void {
		const { ctx } = this;
		const a = this.freqToAngle(freq);
		const cos = Math.cos(a);
		const sin = Math.sin(a);

		// Glow
		ctx.fillStyle = 'rgba(42, 42, 50, 0.06)';
		ctx.beginPath();
		ctx.arc(cx + cos * r, cy + sin * r, 8, 0, Math.PI * 2);
		ctx.fill();

		// Marker
		ctx.fillStyle = 'rgba(42, 42, 50, 0.45)';
		ctx.beginPath();
		ctx.arc(cx + cos * r, cy + sin * r, 3, 0, Math.PI * 2);
		ctx.fill();
	}

	private drawText(cx: number, cy: number, freqHz: number): void {
		const { ctx } = this;

		if (freqHz < 20) {
			ctx.fillStyle = 'rgba(42, 42, 50, 0.15)';
			ctx.font = '300 20px "JetBrains Mono"';
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.fillText('—', cx, cy - 2);
			return;
		}

		// Hz value
		const hzStr = freqHz >= 1000
			? (freqHz / 1000).toFixed(1) + 'k'
			: Math.round(freqHz).toString();
		ctx.fillStyle = 'rgba(42, 42, 50, 0.45)';
		ctx.font = '300 22px "JetBrains Mono"';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'alphabetic';
		ctx.fillText(hzStr, cx, cy - 1);

		// Hz unit
		ctx.fillStyle = 'rgba(42, 42, 50, 0.18)';
		ctx.font = '400 9px "JetBrains Mono"';
		ctx.textBaseline = 'top';
		ctx.fillText('Hz', cx, cy + 3);

		// Note name
		const semi = 12 * Math.log2(freqHz / 440) + 69;
		const r = Math.round(semi);
		const note = NOTE_NAMES[((r % 12) + 12) % 12] + (Math.floor(r / 12) - 1);
		ctx.fillStyle = 'rgba(42, 42, 50, 0.28)';
		ctx.font = '500 11px "JetBrains Mono"';
		ctx.fillText(note, cx, cy + 15);
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
		}
		this.cssW = w;
		this.cssH = h;
	}
}
