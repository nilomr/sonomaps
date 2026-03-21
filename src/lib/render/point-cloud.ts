/**
 * Point cloud + trail renderer.
 *
 * - Tiny crisp points with NormalBlending
 * - Visible continuous trail line
 * - 4000 point capacity for high-frequency sampling (~250/sec)
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

	private readonly maxPoints: number;
	private head = 0;
	private count = 0;
	private readonly outputDim: number;

	// Age step per render frame. At 60fps render, 1/600 = ~10 sec fade.
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
		this.resize();
	}

	private initRenderer(canvas: HTMLCanvasElement): void {
		this.renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: false,
			alpha: false,
			powerPreference: 'high-performance'
		});
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		this.renderer.setClearColor(0x08080f, 1);
	}

	private initScene(): void {
		this.scene = new THREE.Scene();

		this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
		this.camera.position.set(0, 0, 12);

		const canvas = this.renderer.domElement;
		this.controls = new OrbitControls(this.camera, canvas);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.12;
		this.controls.enablePan = true;
		this.controls.minDistance = 3;
		this.controls.maxDistance = 60;
		this.controls.autoRotate = false;

		// Subtle axis lines
		const axisLen = 1.5;
		const axisMat = new THREE.LineBasicMaterial({
			color: 0x14142a, transparent: true, opacity: 0.3
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

		// Build trail index buffer in temporal order
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

	dispose(): void {
		this.controls.dispose();
		this.pointsObj.geometry.dispose();
		(this.pointsObj.material as THREE.ShaderMaterial).dispose();
		this.trailObj.geometry.dispose();
		(this.trailObj.material as THREE.ShaderMaterial).dispose();
		this.renderer.dispose();
	}
}
