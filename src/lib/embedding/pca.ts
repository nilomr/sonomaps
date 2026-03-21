/**
 * Configurable feature embedding for projecting audio features to 3D.
 *
 * The user selects which audio feature maps to each spatial axis from
 * a catalogue of scientifically grounded descriptors. Each axis is
 * independently z-scored with online EMA statistics so the embedding
 * auto-scales regardless of mic gain or input level.
 */

import type { MelFeatureExtractor } from '../dsp/mel.js';

/** Implement this interface to swap in a neural embedding model. */
export interface EmbeddingModel {
	readonly outputDim: number;
	projectFromExtractor(extractor: MelFeatureExtractor, out: Float32Array): void;
}

// ── Feature catalogue ────────────────────────────────────

export interface FeatureDef {
	readonly id: string;
	/** Short name for dropdown UI. */
	readonly label: string;
	/** Short label for the 3D axis overlay. */
	readonly axisLabel: string;
	/** Extract a scalar value from the current frame. */
	readonly extract: (ext: MelFeatureExtractor) => number;
}

export const FEATURES: readonly FeatureDef[] = [
	{ id: 'centroid',  label: 'Centroid',   axisLabel: 'BRIGHTNESS', extract: ext => Math.log1p(ext.centroid) },
	{ id: 'bandwidth', label: 'Bandwidth',  axisLabel: 'WIDTH',      extract: ext => Math.log1p(ext.bandwidth) },
	{ id: 'rolloff',   label: 'Rolloff',    axisLabel: 'ROLLOFF',    extract: ext => Math.log1p(ext.rolloff) },
	{ id: 'rms',       label: 'RMS Energy', axisLabel: 'ENERGY',     extract: ext => Math.log1p(ext.rms * 500) },
	{ id: 'zcr',       label: 'Zero Cross', axisLabel: 'NOISINESS',  extract: ext => ext.zcr },
	{ id: 'flatness',  label: 'Flatness',   axisLabel: 'TONALITY',   extract: ext => ext.flatness },
	{ id: 'mfcc1',     label: 'MFCC 1',     axisLabel: 'MFCC 1',     extract: ext => ext.mfccs[1] },
	{ id: 'mfcc2',     label: 'MFCC 2',     axisLabel: 'MFCC 2',     extract: ext => ext.mfccs[2] },
	{ id: 'mfcc3',     label: 'MFCC 3',     axisLabel: 'MFCC 3',     extract: ext => ext.mfccs[3] },
	{ id: 'mfcc4',     label: 'MFCC 4',     axisLabel: 'MFCC 4',     extract: ext => ext.mfccs[4] },
];

const FEATURE_MAP = new Map(FEATURES.map(f => [f.id, f]));

export function getFeature(id: string): FeatureDef {
	return FEATURE_MAP.get(id) ?? FEATURES[0];
}

// ── Online z-score normalisation ─────────────────────────

class OnlineStat {
	mean = 0;
	variance = 1;
	private readonly decay: number;
	private warmup = 0;
	private readonly warmupFrames: number;

	constructor(decay = 0.995, warmupFrames = 50) {
		this.decay = decay;
		this.warmupFrames = warmupFrames;
	}

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

// ── Configurable feature embedding ──────────────────────

export class DirectFeatureEmbedding implements EmbeddingModel {
	readonly outputDim = 3;
	private extractors: ((ext: MelFeatureExtractor) => number)[];
	private readonly stats: OnlineStat[];
	private readonly scale: number;

	constructor(axes: [string, string, string] = ['centroid', 'bandwidth', 'zcr'], scale = 3.0) {
		this.scale = scale;
		this.extractors = axes.map(id => getFeature(id).extract);
		// At ~250 samples/sec, decay=0.995 gives ~0.8s effective window.
		this.stats = Array.from({ length: 3 }, () => new OnlineStat(0.995, 50));
	}

	/** Change which features map to the three spatial axes. Resets statistics. */
	setAxes(axes: [string, string, string]): void {
		this.extractors = axes.map(id => getFeature(id).extract);
		for (const s of this.stats) s.reset();
	}

	projectFromExtractor(ext: MelFeatureExtractor, out: Float32Array): void {
		const s = this.scale;
		out[0] = this.stats[0].update(this.extractors[0](ext)) * s;
		out[1] = this.stats[1].update(this.extractors[1](ext)) * s;
		out[2] = this.stats[2].update(this.extractors[2](ext)) * s;
	}

	reset(): void {
		for (const s of this.stats) s.reset();
	}
}
