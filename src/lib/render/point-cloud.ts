/**
 * Point cloud + trail renderer with adaptive bounding box.
 *
 * Light cream background, dark monochrome data points.
 * Clean, precise, data-visualization aesthetic.
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

const POINT_STRIDE = 5;

export class PointCloudRenderer {
	private renderer!: THREE.WebGLRenderer;
	private scene!: THREE.Scene;
	private camera!: THREE.PerspectiveCamera;
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

	// ── Bounding box ─────────────────────────────────────
	private boxObj!: THREE.LineSegments;
	private boxPosAttr!: THREE.BufferAttribute;
	private smoothBoxMin = new Float64Array([0, 0, 0]);
	private smoothBoxMax = new Float64Array([0, 0, 0]);
	private boxInitialized = false;

	// ── Camera auto-follow ───────────────────────────────
	private userInteracting = false;
	private interactCooldown = 0;
	private readonly _tmpVec = new THREE.Vector3();

	private readonly maxPoints: number;
	private head = 0;
	private count = 0;
	private readonly outputDim: number;

	private readonly ageStep: number;

	constructor(canvas: HTMLCanvasElement, opts?: PointCloudOptions) {
		this.maxPoints = opts?.maxPoints ?? 4000;
		this.outputDim = opts?.outputDim ?? 3;
		const pointSize = opts?.pointSize ?? 1.8;

		this.ageStep = 1.0 / 600;

		this.initRenderer(canvas);
		this.initScene();
		this.initPoints(pointSize);
		this.initTrail();
		this.initBox();
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
		// Warm cream background
		this.renderer.setClearColor(0xf2ede4, 1);
	}

	private initScene(): void {
		this.scene = new THREE.Scene();

		this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
		this.camera.position.set(8, 5, 7);

		const canvas = this.renderer.domElement;
		this.controls = new OrbitControls(this.camera, canvas);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.12;
		this.controls.enablePan = true;
		this.controls.minDistance = 3;
		this.controls.maxDistance = 60;

		this.controls.addEventListener('start', () => { this.userInteracting = true; });
		this.controls.addEventListener('end', () => {
			this.userInteracting = false;
			this.interactCooldown = 90;
		});

		// Very subtle grid/axis lines — light gray on cream
		const axisLen = 2.0;
		const axisMat = new THREE.LineBasicMaterial({
			color: 0xd0ccc4, transparent: true, opacity: 0.5
		});
		for (const pts of [
			[new THREE.Vector3(-axisLen, 0, 0), new THREE.Vector3(axisLen, 0, 0)],
			[new THREE.Vector3(0, -axisLen, 0), new THREE.Vector3(0, axisLen, 0)],
			...(this.outputDim === 3
				? [[new THREE.Vector3(0, 0, -axisLen), new THREE.Vector3(0, 0, axisLen)]]
				: [])
		]) {
			const g = new THREE.BufferGeometry().setFromPoints(pts);
			this.scene.add(new THREE.Line(g, axisMat));
		}
	}

	private initPoints(pointSize: number): void {
		const n = this.maxPoints;
		const geometry = new THREE.BufferGeometry();

		const positions = new Float32Array(n * 3);
		const ages = new Float32Array(n).fill(1.0);
		const energies = new Float32Array(n);
		const centroids = new Float32Array(n).fill(0.5);

		this.posAttr = new THREE.BufferAttribute(positions, 3);
		this.posAttr.setUsage(THREE.DynamicDrawUsage);
		this.ageAttr = new THREE.BufferAttribute(ages, 1);
		this.ageAttr.setUsage(THREE.DynamicDrawUsage);
		this.energyAttr = new THREE.BufferAttribute(energies, 1);
		this.energyAttr.setUsage(THREE.DynamicDrawUsage);
		this.centroidAttr = new THREE.BufferAttribute(centroids, 1);
		this.centroidAttr.setUsage(THREE.DynamicDrawUsage);

		geometry.setAttribute('position', this.posAttr);
		geometry.setAttribute('aAge', this.ageAttr);
		geometry.setAttribute('aEnergy', this.energyAttr);
		geometry.setAttribute('aCentroid', this.centroidAttr);
		geometry.setDrawRange(0, n);

		const material = new THREE.ShaderMaterial({
			vertexShader: pointVertexShader,
			fragmentShader: pointFragmentShader,
			uniforms: {
				uPointSize: { value: pointSize },
				uPixelRatio: { value: this.renderer.getPixelRatio() }
			},
			transparent: true,
			blending: THREE.NormalBlending,
			depthWrite: false
		});

		this.pointsObj = new THREE.Points(geometry, material);
		this.scene.add(this.pointsObj);
	}

	private initTrail(): void {
		const n = this.maxPoints;
		const geometry = new THREE.BufferGeometry();

		const positions = new Float32Array(n * 3);
		const ages = new Float32Array(n).fill(1.0);
		const centroids = new Float32Array(n).fill(0.5);
		const energies = new Float32Array(n);

		this.trailPosAttr = new THREE.BufferAttribute(positions, 3);
		this.trailPosAttr.setUsage(THREE.DynamicDrawUsage);
		this.trailAgeAttr = new THREE.BufferAttribute(ages, 1);
		this.trailAgeAttr.setUsage(THREE.DynamicDrawUsage);
		this.trailCentroidAttr = new THREE.BufferAttribute(centroids, 1);
		this.trailCentroidAttr.setUsage(THREE.DynamicDrawUsage);
		this.trailEnergyAttr = new THREE.BufferAttribute(energies, 1);
		this.trailEnergyAttr.setUsage(THREE.DynamicDrawUsage);

		geometry.setAttribute('position', this.trailPosAttr);
		geometry.setAttribute('aAge', this.trailAgeAttr);
		geometry.setAttribute('aCentroid', this.trailCentroidAttr);
		geometry.setAttribute('aEnergy', this.trailEnergyAttr);

		this.trailIndices = new Uint16Array(n);
		this.trailIndexAttr = new THREE.BufferAttribute(this.trailIndices, 1);
		this.trailIndexAttr.setUsage(THREE.DynamicDrawUsage);
		geometry.setIndex(this.trailIndexAttr);
		geometry.setDrawRange(0, 0);

		const material = new THREE.ShaderMaterial({
			vertexShader: trailVertexShader,
			fragmentShader: trailFragmentShader,
			transparent: true,
			blending: THREE.NormalBlending,
			depthWrite: false
		});

		this.trailObj = new THREE.Line(geometry, material);
		this.scene.add(this.trailObj);
	}

	// ── Bounding box ─────────────────────────────────────
	private initBox(): void {
		const geo = new THREE.BufferGeometry();
		const positions = new Float32Array(24 * 3); // 12 edges × 2 vertices
		this.boxPosAttr = new THREE.BufferAttribute(positions, 3);
		this.boxPosAttr.setUsage(THREE.DynamicDrawUsage);
		geo.setAttribute('position', this.boxPosAttr);

		const mat = new THREE.LineBasicMaterial({
			color: 0x2a2a32,
			transparent: true,
			opacity: 0.1
		});

		this.boxObj = new THREE.LineSegments(geo, mat);
		this.boxObj.visible = false;
		this.scene.add(this.boxObj);
	}

	private updateBox(): void {
		const posArr = this.posAttr.array as Float32Array;
		const ageArr = this.ageAttr.array as Float32Array;

		let minX = Infinity, maxX = -Infinity;
		let minY = Infinity, maxY = -Infinity;
		let minZ = Infinity, maxZ = -Infinity;
		let activeCount = 0;

		for (let i = 0; i < this.maxPoints; i++) {
			if (ageArr[i] >= 0.9) continue;
			activeCount++;
			const i3 = i * 3;
			const x = posArr[i3], y = posArr[i3 + 1], z = posArr[i3 + 2];
			if (x < minX) minX = x;
			if (x > maxX) maxX = x;
			if (y < minY) minY = y;
			if (y > maxY) maxY = y;
			if (z < minZ) minZ = z;
			if (z > maxZ) maxZ = z;
		}

		if (activeCount < 2) {
			this.boxObj.visible = false;
			return;
		}

		// Margin
		const pad = 0.2;
		minX -= pad; minY -= pad; minZ -= pad;
		maxX += pad; maxY += pad; maxZ += pad;

		if (!this.boxInitialized) {
			this.smoothBoxMin[0] = minX;
			this.smoothBoxMin[1] = minY;
			this.smoothBoxMin[2] = minZ;
			this.smoothBoxMax[0] = maxX;
			this.smoothBoxMax[1] = maxY;
			this.smoothBoxMax[2] = maxZ;
			this.boxInitialized = true;
		} else {
			// Asymmetric smoothing: fast expand, slow contract
			const ed = 0.85, cd = 0.995;
			const rawMins = [minX, minY, minZ];
			const rawMaxs = [maxX, maxY, maxZ];
			for (let d = 0; d < 3; d++) {
				this.smoothBoxMin[d] = rawMins[d] < this.smoothBoxMin[d]
					? ed * this.smoothBoxMin[d] + (1 - ed) * rawMins[d]
					: cd * this.smoothBoxMin[d] + (1 - cd) * rawMins[d];
				this.smoothBoxMax[d] = rawMaxs[d] > this.smoothBoxMax[d]
					? ed * this.smoothBoxMax[d] + (1 - ed) * rawMaxs[d]
					: cd * this.smoothBoxMax[d] + (1 - cd) * rawMaxs[d];
			}
		}

		// Write 12 edges (24 vertices)
		const p = this.boxPosAttr.array as Float32Array;
		const x0 = this.smoothBoxMin[0], y0 = this.smoothBoxMin[1], z0 = this.smoothBoxMin[2];
		const x1 = this.smoothBoxMax[0], y1 = this.smoothBoxMax[1], z1 = this.smoothBoxMax[2];

		let vi = 0;
		const e = (ax: number, ay: number, az: number, bx: number, by: number, bz: number) => {
			p[vi++] = ax; p[vi++] = ay; p[vi++] = az;
			p[vi++] = bx; p[vi++] = by; p[vi++] = bz;
		};

		// Bottom face
		e(x0, y0, z0, x1, y0, z0);
		e(x1, y0, z0, x1, y0, z1);
		e(x1, y0, z1, x0, y0, z1);
		e(x0, y0, z1, x0, y0, z0);
		// Top face
		e(x0, y1, z0, x1, y1, z0);
		e(x1, y1, z0, x1, y1, z1);
		e(x1, y1, z1, x0, y1, z1);
		e(x0, y1, z1, x0, y1, z0);
		// Verticals
		e(x0, y0, z0, x0, y1, z0);
		e(x1, y0, z0, x1, y1, z0);
		e(x1, y0, z1, x1, y1, z1);
		e(x0, y0, z1, x0, y1, z1);

		this.boxPosAttr.needsUpdate = true;
		this.boxObj.visible = true;
	}

	// ── Camera auto-follow ───────────────────────────────
	private autoFollowCamera(): void {
		if (this.interactCooldown > 0) {
			this.interactCooldown--;
			return;
		}
		if (this.userInteracting || !this.boxInitialized || !this.boxObj.visible) return;

		const cx = (this.smoothBoxMin[0] + this.smoothBoxMax[0]) / 2;
		const cy = (this.smoothBoxMin[1] + this.smoothBoxMax[1]) / 2;
		const cz = (this.smoothBoxMin[2] + this.smoothBoxMax[2]) / 2;

		const dx = this.smoothBoxMax[0] - this.smoothBoxMin[0];
		const dy = this.smoothBoxMax[1] - this.smoothBoxMin[1];
		const dz = this.smoothBoxMax[2] - this.smoothBoxMin[2];
		const maxDim = Math.max(dx, dy, dz, 2);

		const fovRad = this.camera.fov * Math.PI / 180;
		const desiredDist = Math.max(4, maxDim / (2 * Math.tan(fovRad / 2)) * 1.6);

		const lerp = 0.018;

		// Lerp target toward box center
		this.controls.target.x += (cx - this.controls.target.x) * lerp;
		this.controls.target.y += (cy - this.controls.target.y) * lerp;
		this.controls.target.z += (cz - this.controls.target.z) * lerp;

		// Lerp camera distance
		const dir = this._tmpVec.subVectors(this.camera.position, this.controls.target);
		const currentDist = dir.length();
		if (currentDist > 0.1) {
			const newDist = currentDist + (desiredDist - currentDist) * lerp;
			dir.normalize().multiplyScalar(newDist);
			this.camera.position.copy(this.controls.target).add(dir);
		}
	}

	addPoints(data: Float32Array, count: number): void {
		const posArr = this.posAttr.array as Float32Array;
		const ageArr = this.ageAttr.array as Float32Array;
		const energyArr = this.energyAttr.array as Float32Array;
		const centroidArr = this.centroidAttr.array as Float32Array;
		const tPosArr = this.trailPosAttr.array as Float32Array;
		const tAgeArr = this.trailAgeAttr.array as Float32Array;
		const tCentArr = this.trailCentroidAttr.array as Float32Array;
		const tEnergyArr = this.trailEnergyAttr.array as Float32Array;

		for (let p = 0; p < count; p++) {
			const srcOff = p * POINT_STRIDE;
			const idx = this.head % this.maxPoints;
			const i3 = idx * 3;

			posArr[i3] = data[srcOff];
			posArr[i3 + 1] = data[srcOff + 1];
			posArr[i3 + 2] = data[srcOff + 2];
			ageArr[idx] = 0;
			energyArr[idx] = data[srcOff + 3];
			centroidArr[idx] = data[srcOff + 4];

			tPosArr[i3] = data[srcOff];
			tPosArr[i3 + 1] = data[srcOff + 1];
			tPosArr[i3 + 2] = data[srcOff + 2];
			tAgeArr[idx] = 0;
			tCentArr[idx] = data[srcOff + 4];
			tEnergyArr[idx] = data[srcOff + 3];

			this.head++;
			if (this.count < this.maxPoints) this.count++;
		}
	}

	render(): void {
		const ageArr = this.ageAttr.array as Float32Array;
		const tAgeArr = this.trailAgeAttr.array as Float32Array;
		const step = this.ageStep;
		for (let i = 0; i < this.maxPoints; i++) {
			if (ageArr[i] < 1.0) {
				const a = Math.min(1.0, ageArr[i] + step);
				ageArr[i] = a;
				tAgeArr[i] = a;
			}
		}

		if (this.count > 1) {
			const n = this.count;
			const start = (this.head - n + this.maxPoints) % this.maxPoints;
			for (let i = 0; i < n; i++) {
				this.trailIndices[i] = (start + i) % this.maxPoints;
			}
			this.trailIndexAttr.needsUpdate = true;
			this.trailObj.geometry.setDrawRange(0, n);
		}

		this.posAttr.needsUpdate = true;
		this.ageAttr.needsUpdate = true;
		this.energyAttr.needsUpdate = true;
		this.centroidAttr.needsUpdate = true;
		this.trailPosAttr.needsUpdate = true;
		this.trailAgeAttr.needsUpdate = true;
		this.trailCentroidAttr.needsUpdate = true;
		this.trailEnergyAttr.needsUpdate = true;

		this.updateBox();
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
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
	}

	clear(): void {
		(this.ageAttr.array as Float32Array).fill(1.0);
		(this.trailAgeAttr.array as Float32Array).fill(1.0);
		this.head = 0;
		this.count = 0;
		this.ageAttr.needsUpdate = true;
		this.trailAgeAttr.needsUpdate = true;
		this.trailObj.geometry.setDrawRange(0, 0);
		this.boxObj.visible = false;
		this.boxInitialized = false;
	}

	dispose(): void {
		this.controls.dispose();
		this.pointsObj.geometry.dispose();
		(this.pointsObj.material as THREE.ShaderMaterial).dispose();
		this.trailObj.geometry.dispose();
		(this.trailObj.material as THREE.ShaderMaterial).dispose();
		this.boxObj.geometry.dispose();
		(this.boxObj.material as THREE.LineBasicMaterial).dispose();
		this.renderer.dispose();
	}
}
