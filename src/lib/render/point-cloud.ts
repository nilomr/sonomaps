/**
 * Point cloud + trail renderer.
 *
 * Visual hierarchy:
 *   1. Trail line  — primary. Dark, opaque recent history; quadratic fade.
 *   2. Head dots   — secondary. Soft scatter over the last ~0.4 s as a cursor.
 *   3. Indicators  — bounding box brackets, crosshair, velocity, peak markers.
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
	networkFragmentShader,
	wireVertexShader,
	wireFragmentShader,
	markerVertexShader,
	markerFragmentShader
} from './shaders.js';

export interface PointCloudOptions {
	maxPoints?: number;
	pointSize?: number;
	outputDim?: 2 | 3;
}

export interface TrajectoryMetrics {
	spread: number;
	drift: number;
	flux: number;
	segments: number;
}

const POINT_STRIDE = 5; // x y z energy centroid

// ── Bounding-box corner bracket topology ─────────────────
// 8 corners × 3 tick lines × 2 verts = 48 vertices.
// Sign table: for each corner, the direction each tick extends.
const BB_CORNER_SIGNS: ReadonlyArray<[number, number, number, number, number, number]> = [
	// cx  cy  cz  sx  sy  sz
	[0, 0, 0, +1, +1, +1],
	[1, 0, 0, -1, +1, +1],
	[1, 1, 0, -1, -1, +1],
	[0, 1, 0, +1, -1, +1],
	[0, 0, 1, +1, +1, -1],
	[1, 0, 1, -1, +1, -1],
	[1, 1, 1, -1, -1, -1],
	[0, 1, 1, +1, -1, -1],
];
const BB_VERTS = 48;
const BB_TICK_FRAC = 0.18; // each tick is 18 % of the edge length

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

	// ── Indicators ───────────────────────────────────────
	private bbObj!: THREE.LineSegments;
	private bbPosAttr!: THREE.BufferAttribute;
	private bbAlphaAttr!: THREE.BufferAttribute;

	private crosshairObj!: THREE.LineSegments;
	private crosshairPosAttr!: THREE.BufferAttribute;
	private crosshairAlphaAttr!: THREE.BufferAttribute;

	private velocityObj!: THREE.LineSegments;
	private velocityPosAttr!: THREE.BufferAttribute;
	private velocityAlphaAttr!: THREE.BufferAttribute;

	private markersObj!: THREE.Points;
	private markerPosAttr!: THREE.BufferAttribute;
	private markerAgeAttr!: THREE.BufferAttribute;
	private markerIntensityAttr!: THREE.BufferAttribute;
	private readonly MAX_MARKERS = 6;

	// ── Recent-window ring buffer for camera tracking ────
	private readonly RECENT_WINDOW = 300; // ~1.2 s at 250 Hz
	private readonly recentPos = new Float32Array(300 * 3);
	private readonly recentEnergy = new Float32Array(300);
	private readonly recentCentroid = new Float32Array(300);
	private readonly recentFlux = new Float32Array(300);
	private readonly recentSerial = new Uint32Array(300);
	private recentHead = 0;
	private recentCount = 0;

	// ── Shared bounds cache (updated once per frame) ─────
	private bCx = 0;
	private bCy = 0;
	private bCz = 0;
	private bMinX = 0;
	private bMinY = 0;
	private bMinZ = 0;
	private bMaxX = 0;
	private bMaxY = 0;
	private bMaxZ = 0;
	private bSpread = 0;

	// ── Camera auto-follow ───────────────────────────────
	private userInteracting = false;
	private interactCooldown = 0;
	private readonly _tmpVec = new THREE.Vector3();
	private readonly _tmpTargetPos = new THREE.Vector3();
	private readonly orthoHalfHeight = 4.5;

	// ── Velocity tracking ────────────────────────────────
	private smoothVelX = 0;
	private smoothVelY = 0;
	private smoothVelZ = 0;
	private readonly VEL_SMOOTH = 0.82;

	// ── Metrics (public via getMetrics) ──────────────────
	private metricSpread = 0;
	private metricDrift = 0;
	private metricFlux = 0;

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
	private readonly TRAIL_WINDOW = 20;
	private readonly NETWORK_AGE_WINDOW = 250;
	private readonly NETWORK_REBUILD_CADENCE = 10;
	private lastNetworkHead = -1;

	private readonly agePerMs = 1.0 / 3200;
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
		this.initBoundingBox();
		this.initCrosshair();
		this.initVelocity();
		this.initMarkers(pointSize);
		this.resize();
	}

	// ================================================================
	// Initialisation
	// ================================================================

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
			this.interactCooldown = 60;
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

	// ── Indicator init ───────────────────────────────────

	private makeWireMaterial(color: [number, number, number], baseAlpha: number): THREE.ShaderMaterial {
		return new THREE.ShaderMaterial({
			vertexShader: wireVertexShader,
			fragmentShader: wireFragmentShader,
			uniforms: {
				uTime: { value: 0 },
				uBaseAlpha: { value: baseAlpha },
				uColor: { value: new THREE.Vector3(...color) }
			},
			transparent: true,
			blending: THREE.NormalBlending,
			depthWrite: false
		});
	}

	private initBoundingBox(): void {
		const geometry = new THREE.BufferGeometry();
		const positions = new Float32Array(BB_VERTS * 3);
		const alphas = new Float32Array(BB_VERTS).fill(1.0);

		this.bbPosAttr = new THREE.BufferAttribute(positions, 3);
		this.bbAlphaAttr = new THREE.BufferAttribute(alphas, 1);
		this.bbPosAttr.setUsage(THREE.DynamicDrawUsage);
		this.bbAlphaAttr.setUsage(THREE.DynamicDrawUsage);

		geometry.setAttribute('position', this.bbPosAttr);
		geometry.setAttribute('aAlpha', this.bbAlphaAttr);
		geometry.setDrawRange(0, 0);

		this.bbObj = new THREE.LineSegments(geometry, this.makeWireMaterial([0.16, 0.14, 0.12], 0.32));
		this.scene.add(this.bbObj);
	}

	private initCrosshair(): void {
		// 3 axis-aligned lines through current position: 6 verts
		const geometry = new THREE.BufferGeometry();
		const positions = new Float32Array(6 * 3);
		const alphas = new Float32Array(6).fill(1.0);

		this.crosshairPosAttr = new THREE.BufferAttribute(positions, 3);
		this.crosshairAlphaAttr = new THREE.BufferAttribute(alphas, 1);
		this.crosshairPosAttr.setUsage(THREE.DynamicDrawUsage);
		this.crosshairAlphaAttr.setUsage(THREE.DynamicDrawUsage);

		geometry.setAttribute('position', this.crosshairPosAttr);
		geometry.setAttribute('aAlpha', this.crosshairAlphaAttr);
		geometry.setDrawRange(0, 0);

		this.crosshairObj = new THREE.LineSegments(geometry, this.makeWireMaterial([0.12, 0.10, 0.08], 0.48));
		this.scene.add(this.crosshairObj);
	}

	private initVelocity(): void {
		const geometry = new THREE.BufferGeometry();
		const positions = new Float32Array(2 * 3);
		const alphas = new Float32Array(2).fill(1.0);

		this.velocityPosAttr = new THREE.BufferAttribute(positions, 3);
		this.velocityAlphaAttr = new THREE.BufferAttribute(alphas, 1);
		this.velocityPosAttr.setUsage(THREE.DynamicDrawUsage);
		this.velocityAlphaAttr.setUsage(THREE.DynamicDrawUsage);

		geometry.setAttribute('position', this.velocityPosAttr);
		geometry.setAttribute('aAlpha', this.velocityAlphaAttr);
		geometry.setDrawRange(0, 0);

		this.velocityObj = new THREE.LineSegments(geometry, this.makeWireMaterial([0.10, 0.10, 0.18], 0.55));
		this.scene.add(this.velocityObj);
	}

	private initMarkers(pointSize: number): void {
		const n = this.MAX_MARKERS;
		const geometry = new THREE.BufferGeometry();

		this.markerPosAttr = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
		this.markerAgeAttr = new THREE.BufferAttribute(new Float32Array(n).fill(1.0), 1);
		this.markerIntensityAttr = new THREE.BufferAttribute(new Float32Array(n), 1);
		for (const a of [this.markerPosAttr, this.markerAgeAttr, this.markerIntensityAttr]) {
			a.setUsage(THREE.DynamicDrawUsage);
		}

		geometry.setAttribute('position', this.markerPosAttr);
		geometry.setAttribute('aMarkerAge', this.markerAgeAttr);
		geometry.setAttribute('aMarkerIntensity', this.markerIntensityAttr);
		geometry.setDrawRange(0, 0);

		const material = new THREE.ShaderMaterial({
			vertexShader: markerVertexShader,
			fragmentShader: markerFragmentShader,
			uniforms: {
				uPointSize: { value: pointSize },
				uPixelRatio: { value: this.renderer.getPixelRatio() },
				uTime: { value: 0 }
			},
			transparent: true,
			blending: THREE.NormalBlending,
			depthWrite: false
		});

		this.markersObj = new THREE.Points(geometry, material);
		this.scene.add(this.markersObj);
	}

	// ================================================================
	// Shared helpers
	// ================================================================

	private clamp01(v: number): number {
		return v < 0 ? 0 : v > 1 ? 1 : v;
	}

	/** Compute bounding box and centroid of the recent window. */
	private computeRecentBounds(): void {
		if (this.recentCount < 2) return;

		const rp = this.recentPos;
		const n = this.recentCount;
		let cx = 0, cy = 0, cz = 0;
		let minX = Infinity, minY = Infinity, minZ = Infinity;
		let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

		for (let i = 0; i < n; i++) {
			const x = rp[i * 3], y = rp[i * 3 + 1], z = rp[i * 3 + 2];
			cx += x; cy += y; cz += z;
			if (x < minX) minX = x; if (x > maxX) maxX = x;
			if (y < minY) minY = y; if (y > maxY) maxY = y;
			if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
		}

		cx /= n; cy /= n; cz /= n;

		this.bCx = cx; this.bCy = cy; this.bCz = cz;
		this.bMinX = minX; this.bMinY = minY; this.bMinZ = minZ;
		this.bMaxX = maxX; this.bMaxY = maxY; this.bMaxZ = maxZ;
		this.bSpread = Math.max(
			maxX - minX, maxY - minY, maxZ - minZ,
			1.0 // minimum so indicators don't collapse
		) * 0.5;
	}

	/** Write a single edge (2 verts) into the network target buffers. */
	private writeEdge(
		seg: number,
		ax: number, ay: number, az: number,
		bx: number, by: number, bz: number,
		edgeAge: number, edgeWeight: number, edgeCentroid: number
	): void {
		const v = seg * 2;
		const p = v * 3;
		this.networkTargetPos[p] = ax; this.networkTargetPos[p + 1] = ay; this.networkTargetPos[p + 2] = az;
		this.networkTargetPos[p + 3] = bx; this.networkTargetPos[p + 4] = by; this.networkTargetPos[p + 5] = bz;
		this.networkTargetAge[v] = edgeAge; this.networkTargetAge[v + 1] = edgeAge;
		this.networkTargetWeight[v] = edgeWeight; this.networkTargetWeight[v + 1] = edgeWeight;
		this.networkTargetCentroid[v] = edgeCentroid; this.networkTargetCentroid[v + 1] = edgeCentroid;
	}

	// ================================================================
	// Network
	// ================================================================

	private buildNetwork(): void {
		if (this.recentCount < 16) {
			this.networkTargetVertCount = 0;
			return;
		}

		// Anchor selection: evenly spaced samples from the recent window
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

		const prox = Math.max(0.8, Math.min(2.6, this.bSpread * 0.70));
		const maxLocalLinks = 4;
		let segCount = 0;

		// Proximity linking
		for (let i = 0; i < anchorIdx.length; i++) {
			const ai = anchorIdx[i];
			const ax = this.recentPos[ai * 3], ay = this.recentPos[ai * 3 + 1], az = this.recentPos[ai * 3 + 2];
			const af = this.recentFlux[ai], ac = this.recentCentroid[ai];
			const aserial = this.recentSerial[ai];

			let localLinks = 0;
			for (let j = i + 3; j < anchorIdx.length && localLinks < maxLocalLinks; j++) {
				if (segCount >= this.MAX_NETWORK_SEGMENTS) break;

				const bi = anchorIdx[j];
				const bx = this.recentPos[bi * 3], by = this.recentPos[bi * 3 + 1], bz = this.recentPos[bi * 3 + 2];
				const dx = ax - bx, dy = ay - by, dz = az - bz;
				const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
				if (dist > prox) continue;

				const sim = 1.0 - dist / prox;
				const flux = Math.max(af, this.recentFlux[bi]);
				const linkWeight = sim * (0.65 + flux * 1.10);
				if (linkWeight < 0.18) continue;

				const newerSerial = Math.max(aserial, this.recentSerial[bi]);
				const edgeAge = this.clamp01((latestSerial - newerSerial) / this.NETWORK_AGE_WINDOW);

				this.writeEdge(
					segCount,
					ax, ay, az, bx, by, bz,
					edgeAge, linkWeight,
					(ac + this.recentCentroid[bi]) * 0.5
				);
				segCount++;
				localLinks++;
			}
			if (segCount >= this.MAX_NETWORK_SEGMENTS) break;
		}

		// Fallback chords for sparse/linear trajectories
		if (segCount < 18) {
			const chordStep = Math.max(5, Math.floor(anchorIdx.length / 8));
			for (let i = 0; i + chordStep < anchorIdx.length; i += 2) {
				if (segCount >= this.MAX_NETWORK_SEGMENTS) break;
				const ai = anchorIdx[i], bi = anchorIdx[i + chordStep];
				const newerSerial = Math.max(this.recentSerial[ai], this.recentSerial[bi]);

				this.writeEdge(
					segCount,
					this.recentPos[ai * 3], this.recentPos[ai * 3 + 1], this.recentPos[ai * 3 + 2],
					this.recentPos[bi * 3], this.recentPos[bi * 3 + 1], this.recentPos[bi * 3 + 2],
					this.clamp01((latestSerial - newerSerial) / this.NETWORK_AGE_WINDOW),
					0.38 + Math.max(this.recentFlux[ai], this.recentFlux[bi]) * 0.55,
					(this.recentCentroid[ai] + this.recentCentroid[bi]) * 0.5
				);
				segCount++;
			}
		}

		this.networkTargetVertCount = segCount * 2;
		if (this.networkCurrentVertCount < this.networkTargetVertCount) {
			this.networkCurrentVertCount = this.networkTargetVertCount;
			this.networkObj.geometry.setDrawRange(0, this.networkCurrentVertCount);
		}
	}

	private smoothNetwork(dtMs: number): void {
		const pos = this.networkPosAttr.array as Float32Array;
		const age = this.networkAgeAttr.array as Float32Array;
		const weight = this.networkWeightAttr.array as Float32Array;
		const centroid = this.networkCentroidAttr.array as Float32Array;

		const activeVerts = Math.max(this.networkCurrentVertCount, this.networkTargetVertCount);
		const fadeBlend = 1.0 - Math.exp(-dtMs / 140);

		for (let i = 0; i < activeVerts; i++) {
			if (i < this.networkTargetVertCount) {
				const i3 = i * 3;
				pos[i3]     = this.networkTargetPos[i3];
				pos[i3 + 1] = this.networkTargetPos[i3 + 1];
				pos[i3 + 2] = this.networkTargetPos[i3 + 2];
				age[i]      = this.networkTargetAge[i];
				centroid[i] = this.networkTargetCentroid[i];
				weight[i] += (this.networkTargetWeight[i] - weight[i]) * fadeBlend;
			} else {
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

		const timeSec = this.lastRenderTime / 1000.0;
		(this.networkObj.material as THREE.ShaderMaterial).uniforms.uTime.value = timeSec;

		this.networkPosAttr.needsUpdate = true;
		this.networkAgeAttr.needsUpdate = true;
		this.networkWeightAttr.needsUpdate = true;
		this.networkCentroidAttr.needsUpdate = true;
		this.networkObj.geometry.setDrawRange(0, this.networkCurrentVertCount);
	}

	// ================================================================
	// Indicators
	// ================================================================

	private updateIndicators(): void {
		if (this.recentCount < 5) {
			this.bbObj.geometry.setDrawRange(0, 0);
			this.crosshairObj.geometry.setDrawRange(0, 0);
			this.velocityObj.geometry.setDrawRange(0, 0);
			this.markersObj.geometry.setDrawRange(0, 0);
			return;
		}

		const timeSec = this.lastRenderTime / 1000.0;

		// ── Bounding box corner brackets ─────────────────
		this.updateBoundingBox(timeSec);

		// ── Crosshair at current position ────────────────
		this.updateCrosshair(timeSec);

		// ── Velocity vector ──────────────────────────────
		this.updateVelocity(timeSec);

		// ── Peak flux markers ────────────────────────────
		this.updateMarkers(timeSec);
	}

	private updateBoundingBox(timeSec: number): void {
		const pos = this.bbPosAttr.array as Float32Array;
		const alpha = this.bbAlphaAttr.array as Float32Array;

		// Add a small padding so the brackets float just outside the data
		const pad = this.bSpread * 0.06;
		const minX = this.bMinX - pad, minY = this.bMinY - pad, minZ = this.bMinZ - pad;
		const maxX = this.bMaxX + pad, maxY = this.bMaxY + pad, maxZ = this.bMaxZ + pad;
		const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
		const tx = dx * BB_TICK_FRAC, ty = dy * BB_TICK_FRAC, tz = dz * BB_TICK_FRAC;

		let vi = 0;
		for (const [cx01, cy01, cz01, sx, sy, sz] of BB_CORNER_SIGNS) {
			const cx = cx01 === 0 ? minX : maxX;
			const cy = cy01 === 0 ? minY : maxY;
			const cz = cz01 === 0 ? minZ : maxZ;

			// X tick
			pos[vi++] = cx; pos[vi++] = cy; pos[vi++] = cz;
			pos[vi++] = cx + sx * tx; pos[vi++] = cy; pos[vi++] = cz;
			// Y tick
			pos[vi++] = cx; pos[vi++] = cy; pos[vi++] = cz;
			pos[vi++] = cx; pos[vi++] = cy + sy * ty; pos[vi++] = cz;
			// Z tick
			pos[vi++] = cx; pos[vi++] = cy; pos[vi++] = cz;
			pos[vi++] = cx; pos[vi++] = cy; pos[vi++] = cz + sz * tz;
		}
		alpha.fill(1.0);

		(this.bbObj.material as THREE.ShaderMaterial).uniforms.uTime.value = timeSec;
		this.bbPosAttr.needsUpdate = true;
		this.bbAlphaAttr.needsUpdate = true;
		this.bbObj.geometry.setDrawRange(0, BB_VERTS);
	}

	private updateCrosshair(timeSec: number): void {
		const pos = this.crosshairPosAttr.array as Float32Array;
		const latest = (this.recentHead - 1 + this.RECENT_WINDOW) % this.RECENT_WINDOW;
		const px = this.recentPos[latest * 3];
		const py = this.recentPos[latest * 3 + 1];
		const pz = this.recentPos[latest * 3 + 2];

		const arm = this.bSpread * 0.12;
		// X axis
		pos[0] = px - arm; pos[1] = py; pos[2] = pz;
		pos[3] = px + arm; pos[4] = py; pos[5] = pz;
		// Y axis
		pos[6] = px; pos[7] = py - arm; pos[8] = pz;
		pos[9] = px; pos[10] = py + arm; pos[11] = pz;
		// Z axis
		pos[12] = px; pos[13] = py; pos[14] = pz - arm;
		pos[15] = px; pos[16] = py; pos[17] = pz + arm;

		(this.crosshairObj.material as THREE.ShaderMaterial).uniforms.uTime.value = timeSec;
		this.crosshairPosAttr.needsUpdate = true;
		this.crosshairObj.geometry.setDrawRange(0, 6);
	}

	private updateVelocity(timeSec: number): void {
		const pos = this.velocityPosAttr.array as Float32Array;
		const latest = (this.recentHead - 1 + this.RECENT_WINDOW) % this.RECENT_WINDOW;
		const px = this.recentPos[latest * 3];
		const py = this.recentPos[latest * 3 + 1];
		const pz = this.recentPos[latest * 3 + 2];

		const velMag = Math.sqrt(
			this.smoothVelX * this.smoothVelX +
			this.smoothVelY * this.smoothVelY +
			this.smoothVelZ * this.smoothVelZ
		);

		if (velMag < 0.001) {
			this.velocityObj.geometry.setDrawRange(0, 0);
			return;
		}

		// Scale velocity direction to a visible length
		const scale = Math.min(this.bSpread * 0.35, velMag * 8.0) / velMag;
		pos[0] = px; pos[1] = py; pos[2] = pz;
		pos[3] = px + this.smoothVelX * scale;
		pos[4] = py + this.smoothVelY * scale;
		pos[5] = pz + this.smoothVelZ * scale;

		// Fade the tip end
		const alphaArr = this.velocityAlphaAttr.array as Float32Array;
		alphaArr[0] = 1.0;
		alphaArr[1] = 0.35;

		(this.velocityObj.material as THREE.ShaderMaterial).uniforms.uTime.value = timeSec;
		this.velocityPosAttr.needsUpdate = true;
		this.velocityAlphaAttr.needsUpdate = true;
		this.velocityObj.geometry.setDrawRange(0, 2);
	}

	private updateMarkers(timeSec: number): void {
		const n = this.recentCount;
		const latestSerial = this.recentSerial[(this.recentHead - 1 + this.RECENT_WINDOW) % this.RECENT_WINDOW] || 1;

		// Find top-N flux peaks in the recent window
		// Use a simple insertion sort into a small array
		const topN = this.MAX_MARKERS;
		const peakIdx = new Int32Array(topN).fill(-1);
		const peakFlux = new Float32Array(topN);

		for (let i = 0; i < n; i++) {
			const f = this.recentFlux[i];
			if (f <= peakFlux[topN - 1]) continue;
			// Insert in sorted order
			let slot = topN - 1;
			while (slot > 0 && f > peakFlux[slot - 1]) {
				peakFlux[slot] = peakFlux[slot - 1];
				peakIdx[slot] = peakIdx[slot - 1];
				slot--;
			}
			peakFlux[slot] = f;
			peakIdx[slot] = i;
		}

		const mPos = this.markerPosAttr.array as Float32Array;
		const mAge = this.markerAgeAttr.array as Float32Array;
		const mInt = this.markerIntensityAttr.array as Float32Array;
		let count = 0;

		for (let k = 0; k < topN; k++) {
			const idx = peakIdx[k];
			if (idx < 0 || peakFlux[k] < 0.25) break; // minimum flux threshold
			mPos[count * 3]     = this.recentPos[idx * 3];
			mPos[count * 3 + 1] = this.recentPos[idx * 3 + 1];
			mPos[count * 3 + 2] = this.recentPos[idx * 3 + 2];
			mAge[count] = this.clamp01((latestSerial - this.recentSerial[idx]) / this.RECENT_WINDOW);
			mInt[count] = peakFlux[k];
			count++;
		}

		const mat = this.markersObj.material as THREE.ShaderMaterial;
		mat.uniforms.uTime.value = timeSec;
		this.markerPosAttr.needsUpdate = true;
		this.markerAgeAttr.needsUpdate = true;
		this.markerIntensityAttr.needsUpdate = true;
		this.markersObj.geometry.setDrawRange(0, count);
	}

	// ================================================================
	// Metrics
	// ================================================================

	private updateMetrics(): void {
		if (this.recentCount < 5) {
			this.metricSpread = 0;
			this.metricDrift = 0;
			this.metricFlux = 0;
			return;
		}

		// Spread: half-extent of the bounding box (already computed)
		this.metricSpread = this.bSpread;

		// Drift: smoothed velocity magnitude
		this.metricDrift = Math.sqrt(
			this.smoothVelX * this.smoothVelX +
			this.smoothVelY * this.smoothVelY +
			this.smoothVelZ * this.smoothVelZ
		);

		// Flux: average flux over the recent window
		let fSum = 0;
		for (let i = 0; i < this.recentCount; i++) fSum += this.recentFlux[i];
		this.metricFlux = fSum / this.recentCount;
	}

	getMetrics(): TrajectoryMetrics {
		return {
			spread: this.metricSpread,
			drift: this.metricDrift,
			flux: this.metricFlux,
			segments: Math.floor(this.networkCurrentVertCount / 2)
		};
	}

	// ── Camera auto-follow ───────────────────────────────
	private autoFollowCamera(): void {
		if (this.interactCooldown > 0) {
			this.interactCooldown--;
			return;
		}
		if (this.userInteracting || this.recentCount < 5) return;

		const cx = this.bCx, cy = this.bCy, cz = this.bCz;
		const spread = Math.max(this.bSpread, 1.5);
		const lerpT = 0.05;

		this.controls.target.x += (cx - this.controls.target.x) * lerpT;
		this.controls.target.y += (cy - this.controls.target.y) * lerpT;
		this.controls.target.z += (cz - this.controls.target.z) * lerpT;

		const desiredZoom = THREE.MathUtils.clamp(
			this.orthoHalfHeight / (spread * 2.0),
			0.35, 2.5
		);
		this.camera.zoom += (desiredZoom - this.camera.zoom) * lerpT * 0.4;
		this.camera.updateProjectionMatrix();

		const dir = this._tmpVec.subVectors(this.camera.position, this.controls.target);
		if (dir.length() > 0.1) {
			dir.normalize().multiplyScalar(10.0);
			const targetPos = this._tmpTargetPos.copy(this.controls.target).add(dir);
			this.camera.position.lerp(targetPos, lerpT);
		}
	}

	// ================================================================
	// Public API
	// ================================================================

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

				// Update smoothed velocity
				this.smoothVelX = this.smoothVelX * this.VEL_SMOOTH + dx * (1 - this.VEL_SMOOTH);
				this.smoothVelY = this.smoothVelY * this.VEL_SMOOTH + dy * (1 - this.VEL_SMOOTH);
				this.smoothVelZ = this.smoothVelZ * this.VEL_SMOOTH + dz * (1 - this.VEL_SMOOTH);
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

			// Recent-window ring buffer
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

		const activePoints = Math.min(this.count, this.maxPoints);
		for (let i = 0; i < activePoints; i++) {
			if (ageArr[i] < 1.0) {
				const a = Math.min(1.0, ageArr[i] + step);
				ageArr[i]  = a;
				tAgeArr[i] = a;
			}
		}

		// Trail index buffer (oldest → newest)
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

		// Shared bounds used by network, camera, and indicators
		this.computeRecentBounds();

		if (this.lastNetworkHead < 0 || (this.recentHead - this.lastNetworkHead) >= this.NETWORK_REBUILD_CADENCE) {
			this.buildNetwork();
			this.lastNetworkHead = this.recentHead;
		}
		this.smoothNetwork(dtMs);

		this.updateIndicators();
		this.updateMetrics();

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
		this.smoothVelX = 0;
		this.smoothVelY = 0;
		this.smoothVelZ = 0;
		this.metricSpread = 0;
		this.metricDrift = 0;
		this.metricFlux = 0;
		this.ageAttr.needsUpdate      = true;
		this.trailAgeAttr.needsUpdate = true;
		this.trailObj.geometry.setDrawRange(0, 0);
		this.networkObj.geometry.setDrawRange(0, 0);
		this.bbObj.geometry.setDrawRange(0, 0);
		this.crosshairObj.geometry.setDrawRange(0, 0);
		this.velocityObj.geometry.setDrawRange(0, 0);
		this.markersObj.geometry.setDrawRange(0, 0);
	}

	dispose(): void {
		this.controls.dispose();
		for (const obj of [
			this.pointsObj, this.trailObj, this.networkObj,
			this.bbObj, this.crosshairObj, this.velocityObj, this.markersObj
		]) {
			obj.geometry.dispose();
			(obj.material as THREE.ShaderMaterial).dispose();
		}
		this.renderer.dispose();
	}
}
