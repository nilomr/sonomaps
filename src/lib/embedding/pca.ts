/**
 * Online PCA embedding: projects 13-dim MFCC vectors into 3D space.
 *
 * Uses exponential moving average covariance with deflated power iteration
 * to track the top 3 principal components of the live MFCC distribution.
 * Similar sounds cluster together because they produce similar MFCC patterns;
 * different sounds occupy different regions because PCA finds the axes of
 * maximum acoustic variation.
 *
 * All buffers pre-allocated; hot-path is GC-free.
 */

import type { MelFeatureExtractor } from '../dsp/mel.js';

export interface EmbeddingModel {
	readonly outputDim: number;
	projectFromExtractor(extractor: MelFeatureExtractor, out: Float32Array): void;
}

export class OnlinePCAEmbedding implements EmbeddingModel {
	readonly outputDim = 3;

	private readonly D = 13;
	private frameCount = 0;

	// EMA statistics (hot-path, pre-allocated)
	private readonly mean = new Float32Array(13);
	private readonly cov = new Float32Array(13 * 13); // symmetric, row-major
	private readonly centered = new Float32Array(13);

	// Top-3 eigenvectors stored row-major: V[k*D + i] = component i of PC k
	private readonly V = new Float32Array(13 * 3);
	private readonly sqrtLambda = new Float32Array(3); // sqrt(eigenvalues) for normalisation

	// Pre-allocated scratch for eigenvector computation (not hot-path)
	private readonly _Vwork = new Float32Array(13 * 3);
	private readonly _buf = new Float32Array(13);

	// EMA alpha: ~500-frame window (≈2 s at 250 Hz sampling)
	private readonly alpha = 0.002;
	// Number of frames before first projection (lets EMA stabilise)
	private readonly warmupFrames = 250;
	// Re-compute eigenvectors every N frames (~200 ms)
	private readonly updateInterval = 50;
	// Output scale (visual spread)
	private readonly outputScale = 2.8;

	constructor() {
		// Initialise to first 3 standard basis vectors so early projections
		// degrade gracefully to MFCC 0/1/2 before eigenvectors converge.
		for (let k = 0; k < 3; k++) this.V[k * this.D + k] = 1.0;
		this.sqrtLambda.fill(1.0);
	}

	/** True while the covariance estimate has not yet stabilised. */
	get isWarmingUp(): boolean {
		return this.frameCount < this.warmupFrames;
	}

	projectFromExtractor(ext: MelFeatureExtractor, out: Float32Array): void {
		const { D, alpha } = this;
		const mfccs = ext.mfccs;
		this.frameCount++;

		// 1. Update running mean (EMA)
		for (let i = 0; i < D; i++) {
			this.mean[i] += alpha * (mfccs[i] - this.mean[i]);
		}

		// 2. Centre current frame
		for (let i = 0; i < D; i++) {
			this.centered[i] = mfccs[i] - this.mean[i];
		}

		// 3. Update upper-triangle covariance (EMA outer product), copy to lower
		for (let i = 0; i < D; i++) {
			const ci = this.centered[i];
			for (let j = i; j < D; j++) {
				const idx = i * D + j;
				this.cov[idx] += alpha * (ci * this.centered[j] - this.cov[idx]);
				if (i !== j) this.cov[j * D + i] = this.cov[idx];
			}
		}

		// 4. Periodically re-compute eigenvectors once warm
		if (this.frameCount % this.updateInterval === 0 && this.frameCount >= this.warmupFrames) {
			this._computeEigenvectors();
		}

		// 5. Return zeros during warm-up (caller should suppress point-cloud adds)
		if (this.frameCount < this.warmupFrames) {
			out[0] = out[1] = out[2] = 0;
			return;
		}

		// 6. Project centred MFCC onto top-3 eigenvectors, normalise by √λ
		const s = this.outputScale;
		for (let k = 0; k < 3; k++) {
			let dot = 0;
			const base = k * D;
			for (let i = 0; i < D; i++) dot += this.V[base + i] * this.centered[i];
			const scale = this.sqrtLambda[k] > 0.001 ? s / this.sqrtLambda[k] : s;
			out[k] = dot * scale;
		}
	}

	/**
	 * Deflated power iteration for the top 3 eigenvectors of `this.cov`.
	 * Warm-starts from current eigenvectors so 25 iterations converge well
	 * for a 13×13 symmetric PD matrix.
	 */
	private _computeEigenvectors(): void {
		const D = this.D;
		const nIter = 25;
		const lambda = [0, 0, 0];

		// Work on a copy to avoid half-updated state during projection
		this._Vwork.set(this.V);

		for (let k = 0; k < 3; k++) {
			const vk = this._Vwork.subarray(k * D, (k + 1) * D);

			for (let iter = 0; iter < nIter; iter++) {
				// _buf = C * vk
				for (let i = 0; i < D; i++) {
					let sum = 0;
					const row = i * D;
					for (let j = 0; j < D; j++) sum += this.cov[row + j] * vk[j];
					this._buf[i] = sum;
				}

				// Gram-Schmidt deflation: remove projections onto already-found PCs
				for (let prev = 0; prev < k; prev++) {
					const vp = this._Vwork.subarray(prev * D, (prev + 1) * D);
					let dot = 0;
					for (let i = 0; i < D; i++) dot += this._buf[i] * vp[i];
					for (let i = 0; i < D; i++) this._buf[i] -= dot * vp[i];
				}

				// Normalise in-place
				let norm = 0;
				for (let i = 0; i < D; i++) norm += this._buf[i] * this._buf[i];
				norm = Math.sqrt(norm);
				if (norm > 1e-10) {
					const inv = 1 / norm;
					for (let i = 0; i < D; i++) vk[i] = this._buf[i] * inv;
				}
			}

			// Rayleigh quotient → eigenvalue: λ = vk^T C vk
			for (let i = 0; i < D; i++) {
				let Cv = 0;
				const row = i * D;
				for (let j = 0; j < D; j++) Cv += this.cov[row + j] * vk[j];
				this._buf[i] = Cv;
			}
			let lam = 0;
			for (let i = 0; i < D; i++) lam += vk[i] * this._buf[i];
			lambda[k] = lam;

			// Sign-stabilise: keep the same orientation as before to avoid
			// sudden trajectory flips when the covariance shifts slightly.
			const vkPrev = this.V.subarray(k * D, (k + 1) * D);
			let dot = 0;
			for (let i = 0; i < D; i++) dot += vkPrev[i] * vk[i];
			if (dot < 0) for (let i = 0; i < D; i++) vk[i] = -vk[i];
		}

		// Commit
		this.V.set(this._Vwork);
		for (let k = 0; k < 3; k++) {
			this.sqrtLambda[k] = Math.sqrt(Math.max(lambda[k], 1e-6));
		}
	}

	reset(): void {
		this.mean.fill(0);
		this.cov.fill(0);
		this.frameCount = 0;
		for (let k = 0; k < 3; k++) {
			this.V.fill(0, k * this.D, (k + 1) * this.D);
			this.V[k * this.D + k] = 1.0;
		}
		this.sqrtLambda.fill(1.0);
	}
}
