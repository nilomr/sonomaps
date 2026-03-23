/**
 * Mel filterbank + feature extraction from AnalyserNode frequency data.
 *
 * Replaces fft.ts + mfcc.ts + features.ts. No custom FFT needed —
 * the AnalyserNode provides the power spectrum natively.
 *
 * All buffers are pre-allocated; compute() is GC-free.
 */

export interface MelConfig {
	sampleRate: number;
	fftSize: number;
	numMelBands?: number;   // default 80 (good resolution for spectrogram)
	numMfccs?: number;      // default 13
	lowFreq?: number;       // default 20 Hz
	highFreq?: number;      // default sampleRate/2
}

export class MelFeatureExtractor {
	readonly numMelBands: number;
	readonly numMfccs: number;
	private readonly numBins: number;
	private readonly sampleRate: number;
	private readonly fftSize: number;

	// Pre-computed filterbank: [numMelBands][numBins]
	private readonly filterBank: Float32Array[];
	// Pre-computed DCT-II matrix: [numMfccs][numMelBands]
	private readonly dctMatrix: Float32Array;

	// Pre-allocated scratch buffers
	private readonly powerBuf: Float32Array;       // linear power from dB input
	readonly melEnergies: Float32Array;             // mel band energies (public for spectrogram)
	readonly logMelEnergies: Float32Array;          // log mel energies
	readonly mfccs: Float32Array;                   // MFCC coefficients

	/** Amplitude floor in dB — bins below this are zeroed. -100 = off. */
	floorDb = -100;
	/** Low-frequency cutoff in Hz (bandpass). 0 = off. */
	minFreqHz = 0;
	/** High-frequency cutoff in Hz (bandpass). Infinity = off. */
	maxFreqHz = Infinity;

	// Scalar features computed each frame
	centroid = 0;       // spectral centroid (Hz)
	rms = 0;            // RMS energy
	zcr = 0;            // zero-crossing rate
	flatness = 0;       // spectral flatness (0 = tonal, 1 = noise)
	bandwidth = 0;      // spectral bandwidth (Hz)
	rolloff = 0;        // spectral rolloff (Hz)
	peakFreq = 0;       // peak frequency (Hz) — bin with max power

	constructor(config: MelConfig) {
		const {
			sampleRate,
			fftSize,
			numMelBands = 80,
			numMfccs = 13,
			lowFreq = 20,
			highFreq = sampleRate / 2
		} = config;

		this.sampleRate = sampleRate;
		this.fftSize = fftSize;
		this.numMelBands = numMelBands;
		this.numMfccs = numMfccs;
		this.numBins = fftSize / 2;

		this.powerBuf = new Float32Array(this.numBins);
		this.melEnergies = new Float32Array(numMelBands);
		this.logMelEnergies = new Float32Array(numMelBands);
		this.mfccs = new Float32Array(numMfccs);

		// ── Build mel filterbank ─────────────────────────────
		const melLow = MelFeatureExtractor.hzToMel(lowFreq);
		const melHigh = MelFeatureExtractor.hzToMel(highFreq);
		const melPoints = new Float32Array(numMelBands + 2);
		for (let i = 0; i < numMelBands + 2; i++) {
			melPoints[i] = melLow + (i * (melHigh - melLow)) / (numMelBands + 1);
		}

		const binPoints = new Float32Array(numMelBands + 2);
		for (let i = 0; i < numMelBands + 2; i++) {
			binPoints[i] = Math.floor(
				((fftSize + 1) * MelFeatureExtractor.melToHz(melPoints[i])) / sampleRate
			);
		}

		this.filterBank = new Array(numMelBands);
		for (let m = 0; m < numMelBands; m++) {
			const filter = new Float32Array(this.numBins);
			const start = binPoints[m];
			const center = binPoints[m + 1];
			const end = binPoints[m + 2];
			for (let k = start; k < center; k++) {
				if (center > start) filter[k] = (k - start) / (center - start);
			}
			for (let k = center; k <= end && k < this.numBins; k++) {
				if (end > center) filter[k] = (end - k) / (end - center);
			}
			this.filterBank[m] = filter;
		}

		// ── DCT-II matrix ────────────────────────────────────
		this.dctMatrix = new Float32Array(numMfccs * numMelBands);
		for (let i = 0; i < numMfccs; i++) {
			for (let j = 0; j < numMelBands; j++) {
				this.dctMatrix[i * numMelBands + j] = Math.cos(
					(Math.PI * i * (j + 0.5)) / numMelBands
				);
			}
		}
	}

	/**
	 * Compute all features from AnalyserNode data.
	 *
	 * @param freqData   Float32Array from analyser.getFloatFrequencyData() — dB values
	 * @param timeData   Float32Array from analyser.getFloatTimeDomainData() — waveform
	 */
	compute(freqData: Float32Array, timeData: Float32Array): void {
		const numBins = this.numBins;

		// ── Convert dB to linear power + floor/bandpass ──────
		// AnalyserNode returns dB (typically -100 to -10).
		// Fused loop: dB threshold + bandpass + power conversion.
		const binHz = this.sampleRate / this.fftSize;
		const minBin = Math.max(0, Math.floor(this.minFreqHz / binHz));
		const maxBin = Math.min(numBins - 1, Math.ceil(this.maxFreqHz / binHz));
		const floorDb = this.floorDb;
		for (let k = 0; k < numBins; k++) {
			if (k < minBin || k > maxBin || freqData[k] < floorDb) {
				this.powerBuf[k] = 0;
			} else {
				this.powerBuf[k] = Math.pow(10, freqData[k] / 10);
			}
		}

		// ── Mel filterbank ───────────────────────────────────
		for (let m = 0; m < this.numMelBands; m++) {
			let energy = 0;
			const filter = this.filterBank[m];
			for (let k = 0; k < numBins; k++) {
				energy += filter[k] * this.powerBuf[k];
			}
			this.melEnergies[m] = energy;
			this.logMelEnergies[m] = Math.log(energy + 1e-10);
		}

		// ── MFCCs via DCT ────────────────────────────────────
		for (let i = 0; i < this.numMfccs; i++) {
			let sum = 0;
			const row = i * this.numMelBands;
			for (let j = 0; j < this.numMelBands; j++) {
				sum += this.dctMatrix[row + j] * this.logMelEnergies[j];
			}
			this.mfccs[i] = sum;
		}

		// ── Spectral centroid (Hz) ───────────────────────────
		let weightedSum = 0;
		let totalEnergy = 0;
		for (let k = 0; k < numBins; k++) {
			weightedSum += k * this.powerBuf[k];
			totalEnergy += this.powerBuf[k];
		}
		const centroidBin = totalEnergy > 0 ? weightedSum / totalEnergy : 0;
		this.centroid = centroidBin * (this.sampleRate / this.fftSize);

		// ── Spectral flatness ────────────────────────────────
		let logSum = 0;
		let arithSum = 0;
		for (let k = 0; k < numBins; k++) {
			const p = this.powerBuf[k] + 1e-20;
			logSum += Math.log(p);
			arithSum += p;
		}
		const geoMean = Math.exp(logSum / numBins);
		const arithMean = arithSum / numBins;
		this.flatness = arithMean > 0 ? Math.min(1, geoMean / arithMean) : 0;

		// ── Spectral bandwidth (Hz) ──────────────────────────
		let bwSum = 0;
		for (let k = 0; k < numBins; k++) {
			const freqHz = k * (this.sampleRate / this.fftSize);
			const diff = freqHz - this.centroid;
			bwSum += this.powerBuf[k] * diff * diff;
		}
		this.bandwidth = totalEnergy > 0 ? Math.sqrt(bwSum / totalEnergy) : 0;

		// ── Spectral rolloff (Hz, 85th percentile) ───────────
		const threshold = totalEnergy * 0.85;
		let cumEnergy = 0;
		let rolloffBin = 0;
		for (let k = 0; k < numBins; k++) {
			cumEnergy += this.powerBuf[k];
			if (cumEnergy >= threshold) {
				rolloffBin = k;
				break;
			}
		}
		this.rolloff = rolloffBin * (this.sampleRate / this.fftSize);

		// ── Peak frequency (Hz) ─────────────────────────────
		let maxPow = 0;
		let peakBin = 0;
		for (let k = 1; k < numBins; k++) {
			if (this.powerBuf[k] > maxPow) {
				maxPow = this.powerBuf[k];
				peakBin = k;
			}
		}
		this.peakFreq = peakBin * (this.sampleRate / this.fftSize);

		// ── RMS energy (from time-domain data) ───────────────
		let sumSq = 0;
		const n = timeData.length;
		for (let i = 0; i < n; i++) {
			sumSq += timeData[i] * timeData[i];
		}
		this.rms = Math.sqrt(sumSq / n);

		// ── Zero-crossing rate ───────────────────────────────
		let crossings = 0;
		for (let i = 1; i < n; i++) {
			if ((timeData[i] >= 0) !== (timeData[i - 1] >= 0)) crossings++;
		}
		this.zcr = crossings / (n - 1);
	}

	static hzToMel(hz: number): number {
		return 2595 * Math.log10(1 + hz / 700);
	}

	static melToHz(mel: number): number {
		return 700 * (10 ** (mel / 2595) - 1);
	}
}
