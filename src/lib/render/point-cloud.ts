/**
 * Point cloud + trail renderer.
 *
 * Visual hierarchy:
 *   1. Trail line  — primary. Dark, opaque recent history; quadratic fade.
 *   2. Head dots   — secondary. Soft scatter over the last ~0.4 s as a cursor.
 *
 * Camera auto-follow tracks the centroid and spread of the most recent
 * ~1.2 s of data (a 300-sample ring buffer), so the view stays locked
 * on current activity rather than drifting out to fit historical extent.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
	pointVertexShader,
	pointFragmentShader,
	trailVertexShader,
	trailFragmentShader
} from './shaders.js';

export interface PointCloudOptions {
	maxPoints?: number;
	pointSize?: number;
	outputDim?: 2 | 3;
}

const POINT_STRIDE = 5; // x y z energy centroid

export class PointCloudRenderer {
	private renderer!: THREE.WebGLRenderer;
	private scene!: THREE.Scene;
	private camera!: THREE.OrthographicCamera;
	private controls!: OrbitControls;

	private pointsObj!: THREE.Points;
	private posAttr!: THREE.BufferAttribute;
	private ageAttr!: THREE.BufferAttribute;
	private energyAttr!: THREE.BufferAttribute;
	private centroidAttr!: THREE.BufferAttribute;

	private trailObj!: THREE.Line;
	private trailPosAttr!: THREE.BufferAttribute;
	private trailAgeAttr!: THREE.BufferAttribute;
	private trailCentroidAttr!: THREE.BufferAttribute;
	private trailEnergyAttr!: THREE.BufferAttribute;
	private trailIndices!: Uint16Array;
	private trailIndexAttr!: THREE.BufferAttribute;

	// ── Recent-window ring buffer for camera tracking ────
	// Keeps the last RECENT_WINDOW positions so autoFollow
	// tracks current activity, not the full historical cloud.
	private readonly RECENT_WINDOW = 300; // ~1.2 s at 250 Hz
	private readonly recentPos = new Float32Array(300 * 3);
	private recentHead = 0;
	private recentCount = 0;

	// ── Camera auto-follow ───────────────────────────────
	private userInteracting = false;
	private interactCooldown = 0;
	private readonly _tmpVec = new THREE.Vector3();
	private readonly _tmpTargetPos = new THREE.Vector3();
	private readonly orthoHalfHeight = 4.5;

	private readonly maxPoints: number;
	private head = 0;
	private count = 0;
	private readonly outputDim: number;

	// Age is advanced by elapsed wall-clock time for stable decay across FPS changes.
	private readonly agePerMs = 1.0 / 3200; // full fade in ~3.2 s
	private lastRenderTime = 0;

	constructor(canvas: HTMLCanvasElement, opts?: PointCloudOptions) {
		this.maxPoints = opts?.maxPoints ?? 4000;
		this.outputDim = opts?.outputDim ?? 3;
		const pointSize = opts?.pointSize ?? 1.8;

		this.initRenderer(canvas);
		this.initScene();
		this.initPoints(pointSize);
		this.initTrail();
		this.resize();
	}

	private initRenderer(canvas: HTMLCanvasElement): void {
		this.renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: true,
			alpha: false,
			powerPreference: 'high-performance'
		});
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		this.renderer.setClearColor(0xf2ede4, 1);
	}

	private initScene(): void {
		this.scene = new THREE.Scene();

		const hh = this.orthoHalfHeight;
		this.camera = new THREE.OrthographicCamera(-hh, hh, hh, -hh, 0.1, 200);
		this.camera.position.set(8, 5, 7);

		const canvas = this.renderer.domElement;
		this.controls = new OrbitControls(this.camera, canvas);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.10;
		this.controls.enablePan = true;
		this.controls.minZoom = 0.3;
		this.controls.maxZoom = 3.0;

		this.controls.addEventListener('start', () => { this.userInteracting = true; });
		this.controls.addEventListener('end', () => {
			this.userInteracting = false;
			this.interactCooldown = 60; // ~1 s before auto-follow resumes
		});

		// Subtle reference axes
		const axisLen = 2.0;
		const axisMat = new THREE.LineBasicMaterial({
			color: 0xd0ccc4, transparent: true, opacity: 0.45
		});
		for (const pts of [
			[new THREE.Vector3(-axisLen, 0, 0), new THREE.Vector3(axisLen, 0, 0)],
			[new THREE.Vector3(0, -axisLen, 0), new THREE.Vector3(0, axisLen, 0)],
			...(this.outputDim === 3
				? [[new THREE.Vector3(0, 0, -axisLen), new THREE.Vector3(0, 0, axisLen)]]
				: [])
		]) {
			const g = new THREE.BufferGeometry().setFromPoints(pts as THREE.Vector3[]);
			this.scene.add(new THREE.Line(g, axisMat));
		}
	}

	private initPoints(pointSize: number): void {
		const n = this.maxPoints;
		const geometry = new THREE.BufferGeometry();

		const positions = new Float32Array(n * 3);
		const ages      = new Float32Array(n).fill(1.0);
		const energies  = new Float32Array(n);
		const centroids = new Float32Array(n).fill(0.5);

		this.posAttr      = new THREE.BufferAttribute(positions, 3);
		this.ageAttr      = new THREE.BufferAttribute(ages, 1);
		this.energyAttr   = new THREE.BufferAttribute(energies, 1);
		this.centroidAttr = new THREE.BufferAttribute(centroids, 1);
		for (const a of [this.posAttr, this.ageAttr, this.energyAttr, this.centroidAttr]) {
			a.setUsage(THREE.DynamicDrawUsage);
		}

		geometry.setAttribute('position', this.posAttr);
		geometry.setAttribute('aAge',     this.ageAttr);
		geometry.setAttribute('aEnergy',  this.energyAttr);
		geometry.setAttribute('aCentroid',this.centroidAttr);
		geometry.setDrawRange(0, n);

		const material = new THREE.ShaderMaterial({
			vertexShader:   pointVertexShader,
			fragmentShader: pointFragmentShader,
			uniforms: {
				uPointSize:  { value: pointSize },
				uPixelRatio: { value: this.renderer.getPixelRatio() }
			},
			transparent: true,
			blending:    THREE.NormalBlending,
			depthWrite:  false
		});

		this.pointsObj = new THREE.Points(geometry, material);
		this.scene.add(this.pointsObj);
	}

	private initTrail(): void {
		const n = this.maxPoints;
		const geometry = new THREE.BufferGeometry();

		const positions = new Float32Array(n * 3);
		const ages      = new Float32Array(n).fill(1.0);
		const centroids = new Float32Array(n).fill(0.5);
		const energies  = new Float32Array(n);

		this.trailPosAttr      = new THREE.BufferAttribute(positions, 3);
		this.trailAgeAttr      = new THREE.BufferAttribute(ages, 1);
		this.trailCentroidAttr = new THREE.BufferAttribute(centroids, 1);
		this.trailEnergyAttr   = new THREE.BufferAttribute(energies, 1);
		for (const a of [this.trailPosAttr, this.trailAgeAttr, this.trailCentroidAttr, this.trailEnergyAttr]) {
			a.setUsage(THREE.DynamicDrawUsage);
		}

		geometry.setAttribute('position',  this.trailPosAttr);
		geometry.setAttribute('aAge',      this.trailAgeAttr);
		geometry.setAttribute('aCentroid', this.trailCentroidAttr);
		geometry.setAttribute('aEnergy',   this.trailEnergyAttr);

		this.trailIndices   = new Uint16Array(n);
		this.trailIndexAttr = new THREE.BufferAttribute(this.trailIndices, 1);
		this.trailIndexAttr.setUsage(THREE.DynamicDrawUsage);
		geometry.setIndex(this.trailIndexAttr);
		geometry.setDrawRange(0, 0);

		const material = new THREE.ShaderMaterial({
			vertexShader:   trailVertexShader,
			fragmentShader: trailFragmentShader,
			transparent: true,
			blending:    THREE.NormalBlending,
			depthWrite:  false
		});

		this.trailObj = new THREE.Line(geometry, material);
		this.scene.add(this.trailObj);
	}

	// ── Camera auto-follow ───────────────────────────────
	private autoFollowCamera(): void {
		if (this.interactCooldown > 0) {
			this.interactCooldown--;
			return;
		}
		if (this.userInteracting || this.recentCount < 5) return;

		// Centroid of the recent window
		let cx = 0, cy = 0, cz = 0;
		const n = this.recentCount;
		for (let i = 0; i < n; i++) {
			cx += this.recentPos[i * 3];
			cy += this.recentPos[i * 3 + 1];
			cz += this.recentPos[i * 3 + 2];
		}
		cx /= n; cy /= n; cz /= n;

		// Max L∞ distance from centroid → comfortable framing radius
		let spread = 0;
		for (let i = 0; i < n; i++) {
			spread = Math.max(
				spread,
				Math.abs(this.recentPos[i * 3]     - cx),
				Math.abs(this.recentPos[i * 3 + 1] - cy),
				Math.abs(this.recentPos[i * 3 + 2] - cz)
			);
		}
		spread = Math.max(spread, 1.5); // minimum comfortable framing

		const lerpT = 0.05;

		// Smoothly move target toward recent centroid
		this.controls.target.x += (cx - this.controls.target.x) * lerpT;
		this.controls.target.y += (cy - this.controls.target.y) * lerpT;
		this.controls.target.z += (cz - this.controls.target.z) * lerpT;

		// Zoom to fit the recent spread with a comfortable margin
		const desiredZoom = THREE.MathUtils.clamp(
			this.orthoHalfHeight / (spread * 2.0),
			0.35, 2.5
		);
		this.camera.zoom += (desiredZoom - this.camera.zoom) * lerpT * 0.4;
		this.camera.updateProjectionMatrix();

		// Keep orbit radius stable around the tracked centre
		const dir = this._tmpVec.subVectors(this.camera.position, this.controls.target);
		if (dir.length() > 0.1) {
			dir.normalize().multiplyScalar(10.0);
			const targetPos = this._tmpTargetPos.copy(this.controls.target).add(dir);
			this.camera.position.lerp(targetPos, lerpT);
		}
	}

	addPoints(data: Float32Array, count: number): void {
		const posArr      = this.posAttr.array as Float32Array;
		const ageArr      = this.ageAttr.array as Float32Array;
		const energyArr   = this.energyAttr.array as Float32Array;
		const centroidArr = this.centroidAttr.array as Float32Array;
		const tPosArr     = this.trailPosAttr.array as Float32Array;
		const tAgeArr     = this.trailAgeAttr.array as Float32Array;
		const tCentArr    = this.trailCentroidAttr.array as Float32Array;
		const tEnergyArr  = this.trailEnergyAttr.array as Float32Array;

		for (let p = 0; p < count; p++) {
			const srcOff = p * POINT_STRIDE;
			const idx    = this.head % this.maxPoints;
			const i3     = idx * 3;

			const x = data[srcOff], y = data[srcOff + 1], z = data[srcOff + 2];

			posArr[i3]     = x;
			posArr[i3 + 1] = y;
			posArr[i3 + 2] = z;
			ageArr[idx]      = 0;
			energyArr[idx]   = data[srcOff + 3];
			centroidArr[idx] = data[srcOff + 4];

			tPosArr[i3]     = x;
			tPosArr[i3 + 1] = y;
			tPosArr[i3 + 2] = z;
			tAgeArr[idx]     = 0;
			tCentArr[idx]    = data[srcOff + 4];
			tEnergyArr[idx]  = data[srcOff + 3];

			// Recent-window ring buffer for camera tracking
			const rIdx = this.recentHead % this.RECENT_WINDOW;
			this.recentPos[rIdx * 3]     = x;
			this.recentPos[rIdx * 3 + 1] = y;
			this.recentPos[rIdx * 3 + 2] = z;
			this.recentHead++;
			if (this.recentCount < this.RECENT_WINDOW) this.recentCount++;

			this.head++;
			if (this.count < this.maxPoints) this.count++;
		}
	}

	render(): void {
		const ageArr  = this.ageAttr.array as Float32Array;
		const tAgeArr = this.trailAgeAttr.array as Float32Array;

		const now = performance.now();
		if (this.lastRenderTime === 0) this.lastRenderTime = now;
		const dtMs = Math.min(100, now - this.lastRenderTime);
		this.lastRenderTime = now;
		const step = dtMs * this.agePerMs;

		for (let i = 0; i < this.maxPoints; i++) {
			if (ageArr[i] < 1.0) {
				const a = Math.min(1.0, ageArr[i] + step);
				ageArr[i]  = a;
				tAgeArr[i] = a;
			}
		}

		// Rebuild trail index buffer in chronological order (oldest → newest)
		if (this.count > 1) {
			const n     = this.count;
			const start = (this.head - n + this.maxPoints) % this.maxPoints;
			for (let i = 0; i < n; i++) {
				this.trailIndices[i] = (start + i) % this.maxPoints;
			}
			this.trailIndexAttr.needsUpdate = true;
			this.trailObj.geometry.setDrawRange(0, n);
		}

		this.posAttr.needsUpdate      = true;
		this.ageAttr.needsUpdate      = true;
		this.energyAttr.needsUpdate   = true;
		this.centroidAttr.needsUpdate = true;
		this.trailPosAttr.needsUpdate     = true;
		this.trailAgeAttr.needsUpdate     = true;
		this.trailCentroidAttr.needsUpdate = true;
		this.trailEnergyAttr.needsUpdate  = true;

		this.autoFollowCamera();
		this.controls.update();
		this.renderer.render(this.scene, this.camera);
	}

	resize(): void {
		const canvas = this.renderer.domElement;
		const parent = canvas.parentElement;
		if (!parent) return;
		const w = parent.clientWidth;
		const h = parent.clientHeight;
		this.renderer.setSize(w, h);
		const aspect = w / h;
		const hh     = this.orthoHalfHeight;
		this.camera.left   = -hh * aspect;
		this.camera.right  =  hh * aspect;
		this.camera.top    =  hh;
		this.camera.bottom = -hh;
		this.camera.updateProjectionMatrix();
	}

	clear(): void {
		(this.ageAttr.array as Float32Array).fill(1.0);
		(this.trailAgeAttr.array as Float32Array).fill(1.0);
		this.head = 0;
		this.count = 0;
		this.recentHead  = 0;
		this.recentCount = 0;
		this.lastRenderTime = 0;
		this.ageAttr.needsUpdate      = true;
		this.trailAgeAttr.needsUpdate = true;
		this.trailObj.geometry.setDrawRange(0, 0);
	}

	dispose(): void {
		this.controls.dispose();
		this.pointsObj.geometry.dispose();
		(this.pointsObj.material as THREE.ShaderMaterial).dispose();
		this.trailObj.geometry.dispose();
		(this.trailObj.material as THREE.ShaderMaterial).dispose();
		this.renderer.dispose();
	}
}
