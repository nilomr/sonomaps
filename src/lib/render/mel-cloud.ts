/**
 * 3D Mel spectrogram point cloud renderer.
 *
 * Renders mel spectrogram data as a scrolling 3D particle cloud:
 *   X axis: time (rolling window, scrolling left)
 *   Y axis: mel frequency bins
 *   Z axis: amplitude (energy)
 *
 * Particles use gaussian falloff so overlapping points accumulate
 * into density fields. An energy threshold hides the noise floor,
 * and dramatic size scaling makes patterns emerge from particle
 * density rather than individual point colours.
 *
 * Positions are computed in the vertex shader from per-point attributes
 * and a single uniform frame counter, so only 80 attribute writes are
 * needed per new audio frame (not the full point buffer).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { melCloudVertexShader, melCloudFragmentShader } from './shaders.js';

export interface MelCloudConfig {
	maxFrames?: number;
	numBands?: number;
	pointSize?: number;
}

export class MelCloudRenderer {
	private renderer: THREE.WebGLRenderer;
	private scene: THREE.Scene;
	private camera: THREE.OrthographicCamera;
	private controls: OrbitControls;

	private geometry: THREE.BufferGeometry;
	private material: THREE.ShaderMaterial;
	private frameIndexAttr: THREE.BufferAttribute;
	private energyAttr: THREE.BufferAttribute;

	private readonly maxFrames: number;
	private readonly numBands: number;
	private head = 0;
	private frameCounter = 0;

	// Auto-ranging
	private rangeMin = -10;
	private rangeMax = 0;
	private readonly rangeDecay = 0.995;
	private readonly orthoHalfHeight = 4.8;

	constructor(canvas: HTMLCanvasElement, config?: MelCloudConfig) {
		this.maxFrames = config?.maxFrames ?? 250;
		this.numBands = config?.numBands ?? 80;
		const maxPoints = this.maxFrames * this.numBands;
		const pointSize = config?.pointSize ?? 1.2;

		// ── Renderer ───────────────────────────────────────
		this.renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: true,
			alpha: false,
			powerPreference: 'high-performance'
		});
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		this.renderer.setClearColor(0xf2ede4, 1);

		// ── Scene & camera ─────────────────────────────────
		this.scene = new THREE.Scene();
		const hh = this.orthoHalfHeight;
		this.camera = new THREE.OrthographicCamera(-hh, hh, hh, -hh, 0.1, 200);
		this.camera.position.set(8.83, 3.95, 6.33);
		this.camera.zoom = 1.59;

		this.controls = new OrbitControls(this.camera, canvas);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.12;
		this.controls.target.set(1.73, -0.08, 1.17);
		this.controls.enablePan = true;
		this.controls.minZoom = 0.8;
		this.controls.maxZoom = 3.5;

		this.initGuideLines();

		// ── Geometry ───────────────────────────────────────
		this.geometry = new THREE.BufferGeometry();

		// Dummy position attribute (shader computes real positions)
		const positions = new Float32Array(maxPoints * 3);
		this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

		const frameIndices = new Float32Array(maxPoints).fill(-1000);
		const melBands = new Float32Array(maxPoints);
		const energies = new Float32Array(maxPoints);

		// Pre-fill mel band normalized positions (static)
		for (let f = 0; f < this.maxFrames; f++) {
			for (let b = 0; b < this.numBands; b++) {
				melBands[f * this.numBands + b] = b / (this.numBands - 1);
			}
		}

		this.frameIndexAttr = new THREE.BufferAttribute(frameIndices, 1);
		this.frameIndexAttr.setUsage(THREE.DynamicDrawUsage);
		this.energyAttr = new THREE.BufferAttribute(energies, 1);
		this.energyAttr.setUsage(THREE.DynamicDrawUsage);

		this.geometry.setAttribute('aFrameIndex', this.frameIndexAttr);
		this.geometry.setAttribute('aMelBand', new THREE.BufferAttribute(melBands, 1));
		this.geometry.setAttribute('aEnergy', this.energyAttr);

		// Prevent frustum culling (positions are computed in shader)
		this.geometry.boundingSphere = new THREE.Sphere(
			new THREE.Vector3(0, 0, 0.6), 15
		);

		// ── Material ───────────────────────────────────────
		this.material = new THREE.ShaderMaterial({
			vertexShader: melCloudVertexShader,
			fragmentShader: melCloudFragmentShader,
			uniforms: {
				uCurrentFrame: { value: 0 },
				uMaxFrames: { value: this.maxFrames },
				uPointSize: { value: pointSize },
				uPixelRatio: { value: this.renderer.getPixelRatio() }
			},
			transparent: true,
			blending: THREE.NormalBlending,
			depthWrite: false
		});

		const pointsObj = new THREE.Points(this.geometry, this.material);
		pointsObj.frustumCulled = false;
		this.scene.add(pointsObj);

		this.resize();
	}

	private initGuideLines(): void {
		const mat = new THREE.LineBasicMaterial({
			color: 0xc8c3ba, transparent: true, opacity: 0.35
		});

		// Floor rectangle (Z = 0 plane)
		const floor = new Float32Array([
			-4, -2.5, 0,  4, -2.5, 0,
			 4, -2.5, 0,  4,  2.5, 0,
			 4,  2.5, 0, -4,  2.5, 0,
			-4,  2.5, 0, -4, -2.5, 0
		]);
		const floorGeo = new THREE.BufferGeometry();
		floorGeo.setAttribute('position', new THREE.Float32BufferAttribute(floor, 3));
		this.scene.add(new THREE.LineSegments(floorGeo, mat));

		// Back wall edges (X = -4, showing freq × amplitude)
		const back = new Float32Array([
			-4, -2.5, 0,   -4, -2.5, 1.2,
			-4,  2.5, 0,   -4,  2.5, 1.2,
			-4, -2.5, 1.2, -4,  2.5, 1.2
		]);
		const backGeo = new THREE.BufferGeometry();
		backGeo.setAttribute('position', new THREE.Float32BufferAttribute(back, 3));
		this.scene.add(new THREE.LineSegments(backGeo, mat));
	}

	/** Add one mel spectrogram frame (called at animation rate, ~60 fps). */
	addFrame(logMelEnergies: Float32Array): void {
		const slot = this.head % this.maxFrames;
		const offset = slot * this.numBands;

		// Auto-range with EMA
		let fMin = Infinity, fMax = -Infinity;
		for (let i = 0; i < this.numBands; i++) {
			if (logMelEnergies[i] < fMin) fMin = logMelEnergies[i];
			if (logMelEnergies[i] > fMax) fMax = logMelEnergies[i];
		}
		const d = this.rangeDecay;
		this.rangeMin = Math.min(this.rangeMin, fMin) * d + fMin * (1 - d);
		this.rangeMax = Math.max(this.rangeMax, fMax) * d + fMax * (1 - d);
		const range = Math.max(this.rangeMax - this.rangeMin, 5);

		const fArr = this.frameIndexAttr.array as Float32Array;
		const eArr = this.energyAttr.array as Float32Array;

		for (let b = 0; b < this.numBands; b++) {
			const idx = offset + b;
			fArr[idx] = this.frameCounter;
			eArr[idx] = Math.max(0, Math.min(1,
				(logMelEnergies[b] - this.rangeMin) / range
			));
		}

		this.frameIndexAttr.needsUpdate = true;
		this.energyAttr.needsUpdate = true;
		this.head++;
	}

	// Camera logging throttle
	private lastCameraLog = 0;

	/** Render one frame. Also advances the frame counter for age-based fading. */
	render(): void {
		this.frameCounter++;
		this.material.uniforms.uCurrentFrame.value = this.frameCounter;
		this.controls.update();
		this.renderer.render(this.scene, this.camera);

		// Log camera settings every 2 seconds
		const now = performance.now();
		if (now - this.lastCameraLog > 2000) {
			this.lastCameraLog = now;
			const p = this.camera.position;
			const t = this.controls.target;
			console.log(
				`[MelCloud camera] pos=(${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}) ` +
				`target=(${t.x.toFixed(2)}, ${t.y.toFixed(2)}, ${t.z.toFixed(2)}) ` +
				`zoom=${this.camera.zoom.toFixed(2)}`
			);
		}
	}

	clear(): void {
		(this.frameIndexAttr.array as Float32Array).fill(-1000);
		this.frameIndexAttr.needsUpdate = true;
		this.head = 0;
	}

	resize(): void {
		const canvas = this.renderer.domElement;
		const parent = canvas.parentElement;
		if (!parent) return;
		const w = parent.clientWidth;
		const h = parent.clientHeight;
		this.renderer.setSize(w, h);
		const aspect = w / h;
		const hh = this.orthoHalfHeight;
		this.camera.left = -hh * aspect;
		this.camera.right = hh * aspect;
		this.camera.top = hh;
		this.camera.bottom = -hh;
		this.camera.updateProjectionMatrix();
	}

	dispose(): void {
		this.controls.dispose();
		this.geometry.dispose();
		this.material.dispose();
		this.renderer.dispose();
	}
}
