/**
 * Exponential moving average smoother for embedding vectors.
 *
 * Smooths the trajectory in embedding space to reduce jitter.
 * Factor α ∈ (0, 1]: lower values = more smoothing.
 */

export class EmbeddingSmoother {
	private readonly dim: number;
	private readonly state: Float32Array;
	private initialized = false;
	alpha: number;

	constructor(dim: number, alpha = 0.5) {
		this.dim = dim;
		this.alpha = alpha;
		this.state = new Float32Array(dim);
	}

	/**
	 * Apply EMA smoothing in-place.
	 * @param embedding  The raw embedding — will be overwritten with the smoothed value.
	 */
	smooth(embedding: Float32Array): void {
		if (!this.initialized) {
			// First frame: initialise state directly
			this.state.set(embedding.subarray(0, this.dim));
			this.initialized = true;
			return;
		}
		const a = this.alpha;
		const b = 1 - a;
		for (let i = 0; i < this.dim; i++) {
			this.state[i] = a * embedding[i] + b * this.state[i];
			embedding[i] = this.state[i];
		}
	}

	/** Reset internal state (e.g. when audio source changes). */
	reset(): void {
		this.state.fill(0);
		this.initialized = false;
	}
}
