/**
 * Audio source using the Web Audio API's built-in AnalyserNode.
 *
 * AnalyserNode runs native C++ FFT on the audio rendering thread and
 * exposes a snapshot of the current frequency/time-domain data via
 * getFloatFrequencyData() / getFloatTimeDomainData(). These reads
 * return the CURRENT audio state with zero buffering or message passing.
 */

export class AudioSource {
	private ctx: AudioContext | null = null;
	private analyser: AnalyserNode | null = null;
	private stream: MediaStream | null = null;
	private sourceNode: MediaStreamAudioSourceNode | MediaElementAudioSourceNode | null = null;
	private audioElement: HTMLAudioElement | null = null;
	// Silent output shim so the browser actually processes the analyser
	// (some browsers skip nodes not routed to destination)
	private silentGain: GainNode | null = null;

	// Pre-allocated output buffers — caller reads these directly
	freqData: Float32Array<ArrayBuffer> = new Float32Array(0);
	timeData: Float32Array<ArrayBuffer> = new Float32Array(0);

	readonly fftSize: number;
	sampleRate = 48000;

	get frequencyBinCount(): number {
		return this.fftSize / 2;
	}

	constructor(fftSize = 2048) {
		this.fftSize = fftSize;
	}

	async startMic(): Promise<void> {
		await this.initContext();
		// Disable all browser audio processing to get raw mic signal.
		// Use both standard constraints and Chrome-specific ones — browsers
		// treat bare `false` as a "preference" they can ignore; `exact` and
		// the `googXxx` keys are harder to override.
		this.stream = await navigator.mediaDevices.getUserMedia({
			audio: {
				echoCancellation: { exact: false },
				noiseSuppression: { exact: false },
				autoGainControl: { exact: false },
				// Chrome 120+: Voice Isolation is a separate processing
				// layer that aggressively filters non-speech audio
				voiceIsolation: false,
				// Chrome-specific: these are the only way to truly
				// disable Chrome's built-in audio processing pipeline
				googEchoCancellation: false,
				googAutoGainControl: false,
				googNoiseSuppression: false,
				googHighpassFilter: false
			} as MediaTrackConstraints
		});
		this.sourceNode = this.ctx!.createMediaStreamSource(this.stream);
		this.sourceNode.connect(this.analyser!);
		// Route analyser → silent gain → destination so the browser
		// actually processes the analyser node (no audible output).
		this.analyser!.connect(this.silentGain!);
	}

	async startFile(file: File): Promise<void> {
		await this.initContext();
		const url = URL.createObjectURL(file);
		this.audioElement = new Audio(url);
		this.audioElement.crossOrigin = 'anonymous';
		this.audioElement.loop = true;

		const source = this.ctx!.createMediaElementSource(this.audioElement);
		this.sourceNode = source;
		source.connect(this.analyser!);
		// Route through analyser to destination so user hears the file
		this.analyser!.connect(this.ctx!.destination);

		await this.audioElement.play();
	}

	/**
	 * Read the current audio state. Call once per rAF.
	 * Returns the CURRENT audio — no buffering, no latency.
	 */
	read(): void {
		if (!this.analyser) return;
		this.analyser.getFloatFrequencyData(this.freqData);
		this.analyser.getFloatTimeDomainData(this.timeData);
	}

	stop(): void {
		if (this.audioElement) {
			this.audioElement.pause();
			this.audioElement.src = '';
			this.audioElement = null;
		}
		if (this.sourceNode) {
			this.sourceNode.disconnect();
			this.sourceNode = null;
		}
		if (this.stream) {
			this.stream.getTracks().forEach((t) => t.stop());
			this.stream = null;
		}
		if (this.analyser) {
			this.analyser.disconnect();
			this.analyser = null;
		}
		if (this.silentGain) {
			this.silentGain.disconnect();
			this.silentGain = null;
		}
		if (this.ctx) {
			this.ctx.close();
			this.ctx = null;
		}
	}

	private async initContext(): Promise<void> {
		if (this.ctx) return;

		// Use the device's native sample rate — avoids resampling latency
		this.ctx = new AudioContext();
		this.sampleRate = this.ctx.sampleRate;

		// Resume context (browsers require a user gesture)
		if (this.ctx.state === 'suspended') {
			await this.ctx.resume();
		}

		this.analyser = this.ctx.createAnalyser();
		this.analyser.fftSize = this.fftSize;
		// Minimal smoothing — we want maximum temporal detail.
		// The embedding smoother handles trajectory smoothness.
		this.analyser.smoothingTimeConstant = 0.05;
		this.analyser.minDecibels = -100;
		this.analyser.maxDecibels = -10;

		// Silent gain node: routes to destination with zero volume
		// so the browser processes the audio graph even for mic input
		this.silentGain = this.ctx.createGain();
		this.silentGain.gain.value = 0;
		this.silentGain.connect(this.ctx.destination);

		this.freqData = new Float32Array(this.analyser.frequencyBinCount);
		this.timeData = new Float32Array(this.fftSize);
	}
}
