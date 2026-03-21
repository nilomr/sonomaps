/**
 * Embedding modules for projecting audio features to 2D/3D.
 *
 * Uses DirectFeatureEmbedding: maps perceptually meaningful features
 * directly to spatial axes with online z-score normalisation.
 *
 * The EmbeddingModel interface allows swapping in a neural model
 * (TF.js / ONNX Runtime Web) without changing the caller.
 */

import type { MelFeatureExtractor } from '../dsp/mel.js';

/** Implement this interface to swap in a neural embedding model. */
export interface EmbeddingModel {
	readonly outputDim: number;
	/** Project features into low-dimensional space. Writes into `out`. */
	projectFromExtractor(extractor: MelFeatureExtractor, out: Float32Array): void;
}

/**
 * Online statistics tracker using exponential moving averages.
 * Converges in ~30 frames (~0.5 s at 60 fps).
 */
class OnlineStat {
	mean = 0;
	variance = 1;
	private readonly decay: number;
	private warmup = 0;
	private readonly warmupFrames: number;

	constructor(decay = 0.97, warmupFrames = 30) {
		this.decay = decay;
		this.warmupFrames = warmupFrames;
	}

	/** Update with new value and return the z-scored result. */
	update(x: number): number {
		if (this.warmup < this.warmupFrames) {
			this.warmup++;
			const n = this.warmup;
			const oldMean = this.mean;
			this.mean += (x - oldMean) / n;
			this.variance += (x - oldMean) * (x - this.mean);
			if (n > 1) {
				const std = Math.sqrt(this.variance / (n - 1));
				return std > 1e-6 ? (x - this.mean) / std : 0;
			}
			return 0;
		}
		const d = this.decay;
		this.mean = d * this.mean + (1 - d) * x;
		const diff = x - this.mean;
		this.variance = d * this.variance + (1 - d) * diff * diff;
		const std = Math.sqrt(this.variance);
		return std > 1e-6 ? diff / std : 0;
	}

	reset(): void {
		this.mean = 0;
		this.variance = 1;
		this.warmup = 0;
	}
}

/**
 * Maps perceptually meaningful features directly to spatial axes.
 *
 * Axes:
 *   X → log spectral centroid  (dark ↔ bright)
 *   Y → log RMS energy         (quiet ↔ loud)
 *   Z → spectral flatness      (tonal ↔ noisy)
 *
 * Each axis is independently z-scored with online EMA statistics so
 * the embedding auto-scales regardless of mic gain or input level.
 */
export class DirectFeatureEmbedding implements EmbeddingModel {
	readonly outputDim: number;
	private readonly stats: OnlineStat[];
	private readonly scale: number;

	constructor(outputDim: 2 | 3 = 3, scale = 2.5) {
		this.outputDim = outputDim;
		this.scale = scale;
		// At ~250 samples/sec, decay=0.985 gives ~3s effective window.
		// warmup=50 converges in ~0.2s.
		this.stats = Array.from({ length: 3 }, () => new OnlineStat(0.985, 50));
	}

	projectFromExtractor(ext: MelFeatureExtractor, out: Float32Array): void {
		const s = this.scale;

		// X: brightness (log centroid compresses the wide Hz range)
		out[0] = this.stats[0].update(Math.log1p(ext.centroid)) * s;

		// Y: loudness (log RMS for perceptual scaling)
		out[1] = this.stats[1].update(Math.log1p(ext.rms * 1000)) * s;

		if (this.outputDim >= 3) {
			// Z: tonality (spectral flatness, already 0..1)
			out[2] = this.stats[2].update(ext.flatness) * s;
		}
	}

	reset(): void {
		for (const s of this.stats) s.reset();
	}
}
