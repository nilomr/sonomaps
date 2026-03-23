/**
 * Arc peak-frequency gauge with cumulative histogram.
 *
 * Renders a precision instrument arc showing detected peak frequency
 * on a log-frequency scale with octave markers. A rolling histogram
 * accumulates peak frequency detections weighted by amplitude, so the
 * most common and loudest frequencies build up visible bars radiating
 * outward from the arc. The histogram decays over a few seconds.
 */

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const ARC_START = Math.PI * 0.78;
const ARC_END = Math.PI * 2.22;
const LOG_MIN = Math.log2(500);    // 500 Hz
const LOG_MAX = Math.log2(12000);  // 12 kHz
const LOG_RANGE = LOG_MAX - LOG_MIN;

const OCTAVE_LABELS = [
	{ freq: 500, label: '500' },
	{ freq: 1000, label: '1k' },
	{ freq: 2000, label: '2k' },
	{ freq: 4000, label: '4k' },
	{ freq: 8000, label: '8k' },
	{ freq: 12000, label: '12k' },
];

// Histogram configuration
const NUM_BINS = 64;
const DECAY_RATE = 0.97;       // per-frame decay (~3 s half-life at 60fps)
const MAX_BAR_HEIGHT = 28;     // max outward bar height in CSS pixels

export class PitchGaugeRenderer {
	private readonly canvas: HTMLCanvasElement;
	private readonly ctx: CanvasRenderingContext2D;
	private cssW = 0;
	private cssH = 0;

	private smoothFreq = 0;

	// Cumulative histogram: each bin accumulates amplitude-weighted hits
	private readonly bins = new Float32Array(NUM_BINS);
	// Track peak bin for normalization
	private peakBin = 0.001;

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d')!;
	}

	draw(freqHz: number, amplitude = 0): void {
		this.resizeIfNeeded();
		const { ctx, cssW: w, cssH: h } = this;
		if (w === 0 || h === 0) return;

		ctx.fillStyle = '#f2ede4';
		ctx.fillRect(0, 0, w, h);

		const cx = w / 2;
		const cy = h * 0.58;
		const r = Math.min(w * 0.41, h * 0.37);

		// Decay all bins
		for (let i = 0; i < NUM_BINS; i++) {
			this.bins[i] *= DECAY_RATE;
		}
		this.peakBin *= DECAY_RATE;

		// Accumulate raw frequency into histogram (no smoothing — preserves
		// distinct peaks when nearby frequencies alternate in succession)
		if (freqHz > 20 && amplitude > 0) {
			// Light smoothing only for the marker/text display
			this.smoothFreq = this.smoothFreq > 20
				? this.smoothFreq * 0.7 + freqHz * 0.3
				: freqHz;

			// Bin the RAW frequency so alternating tones stay separate
			const binIdx = this.freqToBin(freqHz);
			if (binIdx >= 0 && binIdx < NUM_BINS) {
				const weight = amplitude * amplitude;
				this.bins[binIdx] += weight;
				// Minimal neighbor spread (1 bin each side, low weight)
				if (binIdx > 0) this.bins[binIdx - 1] += weight * 0.15;
				if (binIdx < NUM_BINS - 1) this.bins[binIdx + 1] += weight * 0.15;

				this.peakBin = Math.max(this.peakBin, this.bins[binIdx]);
			}
		}

		this.drawArc(cx, cy, r);
		this.drawHistogram(cx, cy, r);
		if (this.smoothFreq > 20) this.drawMarker(cx, cy, r, this.smoothFreq);
		this.drawText(cx, cy, freqHz);
	}

	reset(): void {
		this.bins.fill(0);
		this.peakBin = 0.001;
		this.smoothFreq = 0;
	}

	private freqToBin(freq: number): number {
		const t = (Math.log2(Math.max(500, Math.min(12000, freq))) - LOG_MIN) / LOG_RANGE;
		return Math.round(t * (NUM_BINS - 1));
	}

	private binToAngle(bin: number): number {
		const t = bin / (NUM_BINS - 1);
		return ARC_START + t * (ARC_END - ARC_START);
	}

	private freqToAngle(freq: number): number {
		const t = (Math.log2(Math.max(500, Math.min(12000, freq))) - LOG_MIN) / LOG_RANGE;
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
		for (let midi = 71; midi <= 119; midi++) {
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

	private drawHistogram(cx: number, cy: number, r: number): void {
		const { ctx, bins } = this;

		// Normalize against peak with a floor so bars are visible
		const norm = Math.max(this.peakBin, 0.001);

		const angStep = (ARC_END - ARC_START) / NUM_BINS;
		const barWidth = angStep * 0.7; // leave gaps between bars

		for (let i = 0; i < NUM_BINS; i++) {
			const val = bins[i] / norm;
			if (val < 0.01) continue;

			const a = this.binToAngle(i);
			const h = val * MAX_BAR_HEIGHT;

			// Draw bar radiating inward from arc
			const innerR = r - h - 2;
			const outerR = r - 2;

			// Bar as a small arc segment
			const a0 = a - barWidth / 2;
			const a1 = a + barWidth / 2;

			const alpha = 0.06 + val * 0.32;
			ctx.fillStyle = `rgba(42, 42, 50, ${alpha.toFixed(3)})`;
			ctx.beginPath();
			ctx.arc(cx, cy, outerR, a0, a1);
			ctx.arc(cx, cy, innerR, a1, a0, true);
			ctx.closePath();
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
