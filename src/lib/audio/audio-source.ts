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
	// User-controlled gain node (before analyser)
	private volumeGain: GainNode | null = null;

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
		// Use simple boolean form (some Chrome versions handle { exact: false } oddly).
		this.stream = await navigator.mediaDevices.getUserMedia({
			audio: {
				echoCancellation: false,
				noiseSuppression: false,
				autoGainControl: false,
				// Chrome / WebRTC-specific flags
				voiceIsolation: false,
				googEchoCancellation: false,
				googAutoGainControl: false,
				googNoiseSuppression: false,
				googHighpassFilter: false
			} as MediaTrackConstraints
		});

		// Re-apply constraints after stream creation (belt-and-suspenders)
		// and log what Chrome actually gave us for diagnostics.
		const track = this.stream.getAudioTracks()[0];
		if (track) {
			try {
				await track.applyConstraints({
					echoCancellation: false,
					noiseSuppression: false,
					autoGainControl: false
				});
			} catch { /* not all browsers support post-hoc applyConstraints */ }

			const s = track.getSettings();
			console.log('[SonoMaps] Mic track settings:', {
				echoCancellation: s.echoCancellation,
				noiseSuppression: s.noiseSuppression,
				autoGainControl: s.autoGainControl,
				sampleRate: s.sampleRate,
				channelCount: s.channelCount,
				deviceId: s.deviceId
			});

			// Warn if Chrome ignored our constraints
			if (s.echoCancellation || s.noiseSuppression || s.autoGainControl) {
				console.warn(
					'[SonoMaps] Browser audio processing is still ON despite requesting it off. ' +
					'This usually means the OS or audio driver is forcing voice filtering. ' +
					'Try: (1) Disable Waves MaxxAudio / Realtek voice enhancement in system settings, ' +
					'(2) Use an external USB mic, (3) Try a different browser.'
				);
			}
		}

		this.sourceNode = this.ctx!.createMediaStreamSource(this.stream);
		this.sourceNode.connect(this.volumeGain!);
		this.volumeGain!.connect(this.analyser!);
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
		source.connect(this.volumeGain!);
		this.volumeGain!.connect(this.analyser!);
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

	/** Set the input gain (0 = silent, 1 = unity, 2 = boost). */
	setVolume(value: number): void {
		if (this.volumeGain && this.ctx) {
			this.volumeGain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.02);
		}
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
		if (this.volumeGain) {
			this.volumeGain.disconnect();
			this.volumeGain = null;
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
		this.analyser.smoothingTimeConstant = 0.05;
		this.analyser.minDecibels = -100;
		this.analyser.maxDecibels = -10;

		// Volume gain node: user-controlled, routes source → analyser
		this.volumeGain = this.ctx.createGain();
		this.volumeGain.gain.value = 1.0;

		// Silent gain node: routes to destination with zero volume
		// so the browser processes the audio graph even for mic input
		this.silentGain = this.ctx.createGain();
		this.silentGain.gain.value = 0;
		this.silentGain.connect(this.ctx.destination);

		this.freqData = new Float32Array(this.analyser.frequencyBinCount);
		this.timeData = new Float32Array(this.fftSize);
	}
}
