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
	trailFragmentShader,
	networkVertexShader,
	networkFragmentShader
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
	private fluxAttr!: THREE.BufferAttribute;

	private trailObj!: THREE.Line;
	private trailPosAttr!: THREE.BufferAttribute;
	private trailAgeAttr!: THREE.BufferAttribute;
	private trailCentroidAttr!: THREE.BufferAttribute;
	private trailEnergyAttr!: THREE.BufferAttribute;
	private trailFluxAttr!: THREE.BufferAttribute;
	private trailIndices!: Uint16Array;
	private trailIndexAttr!: THREE.BufferAttribute;

	private networkObj!: THREE.LineSegments;
	private networkPosAttr!: THREE.BufferAttribute;
	private networkAgeAttr!: THREE.BufferAttribute;
	private networkWeightAttr!: THREE.BufferAttribute;
	private networkCentroidAttr!: THREE.BufferAttribute;
	private networkTargetPos!: Float32Array;
	private networkTargetAge!: Float32Array;
	private networkTargetWeight!: Float32Array;
	private networkTargetCentroid!: Float32Array;
	private networkTargetVertCount = 0;
	private networkCurrentVertCount = 0;

	// ── Recent-window ring buffer for camera tracking ────
	// Keeps the last RECENT_WINDOW positions so autoFollow
	// tracks current activity, not the full historical cloud.
	private readonly RECENT_WINDOW = 300; // ~1.2 s at 250 Hz
	private readonly recentPos = new Float32Array(300 * 3);
	private readonly recentEnergy = new Float32Array(300);
	private readonly recentCentroid = new Float32Array(300);
	private readonly recentFlux = new Float32Array(300);
	private readonly recentSerial = new Uint32Array(300);
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
	private sampleSerial = 0;
	private prevPointValid = false;
	private prevX = 0;
	private prevY = 0;
	private prevZ = 0;
	private prevEnergy = 0;
	private prevCentroid = 0.5;
	private readonly MAX_NETWORK_SEGMENTS = 1024;
	private readonly TRAIL_WINDOW = 20; // keep point trail concise to avoid spaghetti
	private readonly NETWORK_AGE_WINDOW = 250; // slower network fade than RECENT_WINDOW
	private readonly NETWORK_REBUILD_CADENCE = 10; // samples between topology rebuilds
	private lastNetworkHead = -1;

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
		this.initNetwork();
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
		const fluxes    = new Float32Array(n);

		this.posAttr      = new THREE.BufferAttribute(positions, 3);
		this.ageAttr      = new THREE.BufferAttribute(ages, 1);
		this.energyAttr   = new THREE.BufferAttribute(energies, 1);
		this.centroidAttr = new THREE.BufferAttribute(centroids, 1);
		this.fluxAttr     = new THREE.BufferAttribute(fluxes, 1);
		for (const a of [this.posAttr, this.ageAttr, this.energyAttr, this.centroidAttr, this.fluxAttr]) {
			a.setUsage(THREE.DynamicDrawUsage);
		}

		geometry.setAttribute('position', this.posAttr);
		geometry.setAttribute('aAge',     this.ageAttr);
		geometry.setAttribute('aEnergy',  this.energyAttr);
		geometry.setAttribute('aCentroid',this.centroidAttr);
		geometry.setAttribute('aFlux',    this.fluxAttr);
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
		const fluxes    = new Float32Array(n);

		this.trailPosAttr      = new THREE.BufferAttribute(positions, 3);
		this.trailAgeAttr      = new THREE.BufferAttribute(ages, 1);
		this.trailCentroidAttr = new THREE.BufferAttribute(centroids, 1);
		this.trailEnergyAttr   = new THREE.BufferAttribute(energies, 1);
		this.trailFluxAttr     = new THREE.BufferAttribute(fluxes, 1);
		for (const a of [this.trailPosAttr, this.trailAgeAttr, this.trailCentroidAttr, this.trailEnergyAttr, this.trailFluxAttr]) {
			a.setUsage(THREE.DynamicDrawUsage);
		}

		geometry.setAttribute('position',  this.trailPosAttr);
		geometry.setAttribute('aAge',      this.trailAgeAttr);
		geometry.setAttribute('aCentroid', this.trailCentroidAttr);
		geometry.setAttribute('aEnergy',   this.trailEnergyAttr);
		geometry.setAttribute('aFlux',     this.trailFluxAttr);

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

	private initNetwork(): void {
		const maxVerts = this.MAX_NETWORK_SEGMENTS * 2;
		const geometry = new THREE.BufferGeometry();

		this.networkPosAttr = new THREE.BufferAttribute(new Float32Array(maxVerts * 3), 3);
		this.networkAgeAttr = new THREE.BufferAttribute(new Float32Array(maxVerts).fill(1.0), 1);
		this.networkWeightAttr = new THREE.BufferAttribute(new Float32Array(maxVerts), 1);
		this.networkCentroidAttr = new THREE.BufferAttribute(new Float32Array(maxVerts).fill(0.5), 1);
		this.networkTargetPos = new Float32Array(maxVerts * 3);
		this.networkTargetAge = new Float32Array(maxVerts).fill(1.0);
		this.networkTargetWeight = new Float32Array(maxVerts);
		this.networkTargetCentroid = new Float32Array(maxVerts).fill(0.5);

		for (const a of [this.networkPosAttr, this.networkAgeAttr, this.networkWeightAttr, this.networkCentroidAttr]) {
			a.setUsage(THREE.DynamicDrawUsage);
		}

		geometry.setAttribute('position', this.networkPosAttr);
		geometry.setAttribute('aAge', this.networkAgeAttr);
		geometry.setAttribute('aWeight', this.networkWeightAttr);
		geometry.setAttribute('aCentroid', this.networkCentroidAttr);
		geometry.setDrawRange(0, 0);

		const material = new THREE.ShaderMaterial({
			vertexShader: networkVertexShader,
			fragmentShader: networkFragmentShader,
			uniforms: { uTime: { value: 0.0 } },
			transparent: true,
			blending: THREE.NormalBlending,
			depthWrite: false
		});

		this.networkObj = new THREE.LineSegments(geometry, material);
		this.scene.add(this.networkObj);
	}

	private clamp01(v: number): number {
		return Math.max(0, Math.min(1, v));
	}

	private buildNetwork(): void {
		const geom = this.networkObj.geometry;
		if (this.recentCount < 16) {
			this.networkTargetVertCount = 0;
			return;
		}

		const pos = this.networkTargetPos;
		const age = this.networkTargetAge;
		const weight = this.networkTargetWeight;
		const centroid = this.networkTargetCentroid;

		const anchorCap = 96;
		const stride = this.recentCount >= 120 ? 3 : Math.max(2, Math.floor(this.recentCount / anchorCap));
		const start = (this.recentHead - this.recentCount + this.RECENT_WINDOW) % this.RECENT_WINDOW;
		const latestSerial = this.recentSerial[(this.recentHead - 1 + this.RECENT_WINDOW) % this.RECENT_WINDOW] || 1;

		const anchorIdx: number[] = [];
		for (let i = 0; i < this.recentCount; i += stride) {
			anchorIdx.push((start + i) % this.RECENT_WINDOW);
		}
		if (anchorIdx.length < 8) {
			this.networkTargetVertCount = 0;
			return;
		}

		let cx = 0;
		let cy = 0;
		let cz = 0;
		for (let i = 0; i < this.recentCount; i++) {
			cx += this.recentPos[i * 3];
			cy += this.recentPos[i * 3 + 1];
			cz += this.recentPos[i * 3 + 2];
		}
		cx /= this.recentCount;
		cy /= this.recentCount;
		cz /= this.recentCount;

		let spread = 0;
		for (let i = 0; i < this.recentCount; i++) {
			spread = Math.max(
				spread,
				Math.abs(this.recentPos[i * 3] - cx),
				Math.abs(this.recentPos[i * 3 + 1] - cy),
				Math.abs(this.recentPos[i * 3 + 2] - cz)
			);
		}
		const prox = Math.max(0.8, Math.min(2.6, spread * 0.70));

		let segCount = 0;
		const maxLocalLinks = 4;

		for (let i = 0; i < anchorIdx.length; i++) {
			const ai = anchorIdx[i];
			const ax = this.recentPos[ai * 3];
			const ay = this.recentPos[ai * 3 + 1];
			const az = this.recentPos[ai * 3 + 2];
			const af = this.recentFlux[ai];
			const ac = this.recentCentroid[ai];
			const aserial = this.recentSerial[ai];

			let localLinks = 0;
			for (let j = i + 3; j < anchorIdx.length && localLinks < maxLocalLinks; j++) {
				if (segCount >= this.MAX_NETWORK_SEGMENTS) break;
				const bi = anchorIdx[j];
				const bx = this.recentPos[bi * 3];
				const by = this.recentPos[bi * 3 + 1];
				const bz = this.recentPos[bi * 3 + 2];
				const bf = this.recentFlux[bi];
				const bc = this.recentCentroid[bi];
				const bserial = this.recentSerial[bi];

				const dx = ax - bx;
				const dy = ay - by;
				const dz = az - bz;
				const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
				if (dist > prox) continue;

				const sim = 1.0 - dist / prox;
				const flux = Math.max(af, bf);
				const linkWeight = sim * (0.65 + flux * 1.10);
				if (linkWeight < 0.18) continue;

				const newerSerial = Math.max(aserial, bserial);
				const edgeAge = this.clamp01((latestSerial - newerSerial) / this.NETWORK_AGE_WINDOW);
				const edgeCentroid = (ac + bc) * 0.5;

				const v = segCount * 2;
				const p0 = v * 3;
				const p1 = p0 + 3;
				pos[p0] = ax; pos[p0 + 1] = ay; pos[p0 + 2] = az;
				pos[p1] = bx; pos[p1 + 1] = by; pos[p1 + 2] = bz;

				age[v] = edgeAge;
				age[v + 1] = edgeAge;
				weight[v] = linkWeight;
				weight[v + 1] = linkWeight;
				centroid[v] = edgeCentroid;
				centroid[v + 1] = edgeCentroid;

				segCount++;
				localLinks++;
			}
			if (segCount >= this.MAX_NETWORK_SEGMENTS) break;
		}

		// Fallback chords: ensure a visible web even in sparse/linear trajectories.
		if (segCount < 18) {
			const chordStep = Math.max(5, Math.floor(anchorIdx.length / 8));
			for (let i = 0; i + chordStep < anchorIdx.length; i += 2) {
				if (segCount >= this.MAX_NETWORK_SEGMENTS) break;
				const ai = anchorIdx[i];
				const bi = anchorIdx[i + chordStep];
				const ax = this.recentPos[ai * 3];
				const ay = this.recentPos[ai * 3 + 1];
				const az = this.recentPos[ai * 3 + 2];
				const bx = this.recentPos[bi * 3];
				const by = this.recentPos[bi * 3 + 1];
				const bz = this.recentPos[bi * 3 + 2];

				const aserial = this.recentSerial[ai];
				const bserial = this.recentSerial[bi];
				const newerSerial = Math.max(aserial, bserial);
				const edgeAge = this.clamp01((latestSerial - newerSerial) / this.NETWORK_AGE_WINDOW);
				const edgeCentroid = (this.recentCentroid[ai] + this.recentCentroid[bi]) * 0.5;
				const edgeWeight = 0.38 + Math.max(this.recentFlux[ai], this.recentFlux[bi]) * 0.55;

				const v = segCount * 2;
				const p0 = v * 3;
				const p1 = p0 + 3;
				pos[p0] = ax; pos[p0 + 1] = ay; pos[p0 + 2] = az;
				pos[p1] = bx; pos[p1 + 1] = by; pos[p1 + 2] = bz;
				age[v] = edgeAge; age[v + 1] = edgeAge;
				weight[v] = edgeWeight; weight[v + 1] = edgeWeight;
				centroid[v] = edgeCentroid; centroid[v + 1] = edgeCentroid;
				segCount++;
			}
		}

		this.networkTargetVertCount = segCount * 2;
		if (this.networkCurrentVertCount < this.networkTargetVertCount) {
			this.networkCurrentVertCount = this.networkTargetVertCount;
			geom.setDrawRange(0, this.networkCurrentVertCount);
		}
	}

	private smoothNetwork(dtMs: number): void {
		const geom = this.networkObj.geometry;
		const pos = this.networkPosAttr.array as Float32Array;
		const age = this.networkAgeAttr.array as Float32Array;
		const weight = this.networkWeightAttr.array as Float32Array;
		const centroid = this.networkCentroidAttr.array as Float32Array;

		// Only iterate the live portion — skip silent zeroed-out slots beyond current extent
		const activeVerts = Math.max(this.networkCurrentVertCount, this.networkTargetVertCount);

		// Lerp only the weight (alpha/visibility). Positions are snapped directly —
		// lerping positions makes links slide through space and look like trailing worms.
		const fadeBlend = 1.0 - Math.exp(-dtMs / 140);

		for (let i = 0; i < activeVerts; i++) {
			if (i < this.networkTargetVertCount) {
				// Snap endpoint positions and age immediately (they come from fixed ring-buffer coords)
				const i3 = i * 3;
				pos[i3]     = this.networkTargetPos[i3];
				pos[i3 + 1] = this.networkTargetPos[i3 + 1];
				pos[i3 + 2] = this.networkTargetPos[i3 + 2];
				age[i]      = this.networkTargetAge[i];
				centroid[i] = this.networkTargetCentroid[i];
				// Fade weight in smoothly so links don't pop into existence
				weight[i] += (this.networkTargetWeight[i] - weight[i]) * fadeBlend;
			} else {
				// Fade dead links out
				weight[i] += (0.0 - weight[i]) * fadeBlend;
			}
		}

		if (this.networkCurrentVertCount < this.networkTargetVertCount) {
			this.networkCurrentVertCount = this.networkTargetVertCount;
		}

		while (this.networkCurrentVertCount > this.networkTargetVertCount + 1) {
			const i0 = this.networkCurrentVertCount - 1;
			const i1 = this.networkCurrentVertCount - 2;
			if (weight[i0] > 0.03 || weight[i1] > 0.03) break;
			this.networkCurrentVertCount -= 2;
		}

		// Update uTime so the animated pulse stays in sync
		(this.networkObj.material as THREE.ShaderMaterial).uniforms.uTime.value = this.lastRenderTime / 1000.0;

		this.networkPosAttr.needsUpdate = true;
		this.networkAgeAttr.needsUpdate = true;
		this.networkWeightAttr.needsUpdate = true;
		this.networkCentroidAttr.needsUpdate = true;
		geom.setDrawRange(0, this.networkCurrentVertCount);
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
		const fluxArr     = this.fluxAttr.array as Float32Array;
		const tPosArr     = this.trailPosAttr.array as Float32Array;
		const tAgeArr     = this.trailAgeAttr.array as Float32Array;
		const tCentArr    = this.trailCentroidAttr.array as Float32Array;
		const tEnergyArr  = this.trailEnergyAttr.array as Float32Array;
		const tFluxArr    = this.trailFluxAttr.array as Float32Array;

		for (let p = 0; p < count; p++) {
			const srcOff = p * POINT_STRIDE;
			const idx    = this.head % this.maxPoints;
			const i3     = idx * 3;

			const x = data[srcOff], y = data[srcOff + 1], z = data[srcOff + 2];
			const energy = data[srcOff + 3];
			const centroid = data[srcOff + 4];
			let flux = 0.0;
			if (this.prevPointValid) {
				const dx = x - this.prevX;
				const dy = y - this.prevY;
				const dz = z - this.prevZ;
				const jump = Math.sqrt(dx * dx + dy * dy + dz * dz);
				const eDelta = Math.abs(energy - this.prevEnergy);
				const cDelta = Math.abs(centroid - this.prevCentroid);
				flux = this.clamp01(jump * 0.42 + eDelta * 0.95 + cDelta * 0.70);
			}
			this.prevPointValid = true;
			this.prevX = x;
			this.prevY = y;
			this.prevZ = z;
			this.prevEnergy = energy;
			this.prevCentroid = centroid;

			posArr[i3]     = x;
			posArr[i3 + 1] = y;
			posArr[i3 + 2] = z;
			ageArr[idx]      = 0;
			energyArr[idx]   = energy;
			centroidArr[idx] = centroid;
			fluxArr[idx]     = flux;

			tPosArr[i3]     = x;
			tPosArr[i3 + 1] = y;
			tPosArr[i3 + 2] = z;
			tAgeArr[idx]     = 0;
			tCentArr[idx]    = centroid;
			tEnergyArr[idx]  = energy;
			tFluxArr[idx]    = flux;

			// Recent-window ring buffer for camera tracking
			const rIdx = this.recentHead % this.RECENT_WINDOW;
			this.recentPos[rIdx * 3]     = x;
			this.recentPos[rIdx * 3 + 1] = y;
			this.recentPos[rIdx * 3 + 2] = z;
			this.recentEnergy[rIdx] = energy;
			this.recentCentroid[rIdx] = centroid;
			this.recentFlux[rIdx] = flux;
			this.recentSerial[rIdx] = this.sampleSerial++;
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

		// Only iterate occupied slots — skips the empty tail during warm-up
		const activePoints = Math.min(this.count, this.maxPoints);
		for (let i = 0; i < activePoints; i++) {
			if (ageArr[i] < 1.0) {
				const a = Math.min(1.0, ageArr[i] + step);
				ageArr[i]  = a;
				tAgeArr[i] = a;
			}
		}

		// Rebuild trail index buffer in chronological order (oldest → newest)
		if (this.count > 1) {
			const n     = Math.min(this.count, this.TRAIL_WINDOW);
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
		this.fluxAttr.needsUpdate     = true;
		this.trailPosAttr.needsUpdate     = true;
		this.trailAgeAttr.needsUpdate     = true;
		this.trailCentroidAttr.needsUpdate = true;
		this.trailEnergyAttr.needsUpdate  = true;
		this.trailFluxAttr.needsUpdate    = true;

		if (this.lastNetworkHead < 0 || (this.recentHead - this.lastNetworkHead) >= this.NETWORK_REBUILD_CADENCE) {
			this.buildNetwork();
			this.lastNetworkHead = this.recentHead;
		}
		this.smoothNetwork(dtMs);

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
		this.sampleSerial = 0;
		this.lastNetworkHead = -1;
		this.networkTargetVertCount = 0;
		this.networkCurrentVertCount = 0;
		this.prevPointValid = false;
		this.lastRenderTime = 0;
		this.ageAttr.needsUpdate      = true;
		this.trailAgeAttr.needsUpdate = true;
		this.trailObj.geometry.setDrawRange(0, 0);
		this.networkObj.geometry.setDrawRange(0, 0);
	}

	dispose(): void {
		this.controls.dispose();
		this.pointsObj.geometry.dispose();
		(this.pointsObj.material as THREE.ShaderMaterial).dispose();
		this.trailObj.geometry.dispose();
		(this.trailObj.material as THREE.ShaderMaterial).dispose();
		this.networkObj.geometry.dispose();
		(this.networkObj.material as THREE.ShaderMaterial).dispose();
		this.renderer.dispose();
	}
}
