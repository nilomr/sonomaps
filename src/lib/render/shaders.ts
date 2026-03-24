/**
 * Shaders for the audio point cloud.
 *
 * Aesthetic: dark ink on light cream. Trail is the primary visual — recent
 * segments are dense and dark, older history fades gracefully. Loud sound
 * leaves a darker mark; silence disappears entirely. Spectral centroid adds
 * a warm/cool tonal shift across the monochrome palette.
 */

// ── Head indicator ────────────────────────────────────────
// Scatter points show only the most recent ~0.4 s of trajectory.
// They act as a cursor: a soft glowing cloud at the leading edge of the trail.

export const pointVertexShader = /* glsl */ `
  attribute float aAge;
  attribute float aEnergy;
  attribute float aCentroid;
  attribute float aFlux;

  varying float vAge;
  varying float vEnergy;
  varying float vCentroid;
  varying float vFlux;

  uniform float uPointSize;
  uniform float uPixelRatio;

  void main() {
    vAge      = aAge;
    vEnergy   = aEnergy;
    vCentroid = aCentroid;
    vFlux     = aFlux;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    // Only the freshest ~0.4 s (age < 0.08 with ageStep = 1/300)
    float fresh = max(0.0, 1.0 - vAge / 0.08);
    float energyScale = 0.14 + pow(clamp(vEnergy, 0.0, 1.0), 1.25) * 1.9;
    float fluxScale = 0.75 + clamp(vFlux, 0.0, 1.0) * 1.35;

    gl_PointSize = uPointSize * uPixelRatio * energyScale * fluxScale * fresh * 7.4;
    gl_Position  = projectionMatrix * mvPosition;
  }
`;

export const pointFragmentShader = /* glsl */ `
  varying float vAge;
  varying float vEnergy;
  varying float vCentroid;
  varying float vFlux;

  vec3 palette(float t) {
    vec3 warm = vec3(0.17, 0.12, 0.09);
    vec3 cool = vec3(0.06, 0.10, 0.20);
    return mix(warm, cool, smoothstep(0.0, 1.0, t));
  }

  void main() {
    vec2 coord = gl_PointCoord - 0.5;
    float r = length(coord) * 2.0;
    if (r > 1.0) discard;

    float fresh       = max(0.0, 1.0 - vAge / 0.08);
    float energyGate  = smoothstep(0.12, 0.70, vEnergy);

    vec3 color = palette(vCentroid);
    float fluxGlow = 0.55 + smoothstep(0.12, 0.95, vFlux) * 0.85;
    float alpha = fresh * fresh * energyGate * fluxGlow * 0.86;

    gl_FragColor = vec4(color, alpha);
  }
`;

// ── Trail line ────────────────────────────────────────────
// The trail is the primary visual. Recent = dark and opaque; older history
// fades with a quadratic curve. Quiet segments vanish so gaps in the sound
// read as natural breaks in the ink stroke.

export const trailVertexShader = /* glsl */ `
  attribute float aAge;
  attribute float aCentroid;
  attribute float aEnergy;
  attribute float aFlux;

  varying float vAge;
  varying float vCentroid;
  varying float vEnergy;
  varying float vFlux;

  void main() {
    vAge      = aAge;
    vCentroid = aCentroid;
    vEnergy   = aEnergy;
    vFlux     = aFlux;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const trailFragmentShader = /* glsl */ `
  varying float vAge;
  varying float vCentroid;
  varying float vEnergy;
  varying float vFlux;

  vec3 palette(float t) {
    vec3 warm = vec3(0.17, 0.12, 0.09);
    vec3 cool = vec3(0.06, 0.10, 0.20);
    return mix(warm, cool, smoothstep(0.0, 1.0, t));
  }

  void main() {
    float alive = max(0.0, 1.0 - vAge);

    // Quiet segments disappear — loud segments leave a strong mark
    float energyBoost = smoothstep(0.07, 0.30, vEnergy);

    // Slightly faster than quadratic so the connecting line clears sooner
    float fluxBoost = 0.72 + smoothstep(0.08, 0.95, vFlux) * 1.45;
    float alpha = pow(alive, 2.2) * 0.74 * energyBoost * fluxBoost;

    vec3 color = palette(vCentroid);
    // Loud = darker ink; quiet = lighter (contrast drives attention to amplitude)
    color *= 1.0 - vEnergy * 0.25;

    gl_FragColor = vec4(color, alpha);
  }
`;

// ── Trajectory network links ───────────────────────────
// Lightweight line-segment web that connects nearby states
// from non-adjacent moments in the recent window.

export const networkVertexShader = /* glsl */ `
  attribute float aAge;
  attribute float aWeight;
  attribute float aCentroid;

  varying float vAge;
  varying float vWeight;
  varying float vCentroid;
  varying float vT; // 0.0 = older endpoint, 1.0 = newer endpoint

  void main() {
    vAge      = aAge;
    vWeight   = aWeight;
    vCentroid = aCentroid;
    // gl_VertexID is even for the "source" vertex of every segment, odd for "target"
    vT = mod(float(gl_VertexID), 2.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const networkFragmentShader = /* glsl */ `
  uniform float uTime;

  varying float vAge;
  varying float vWeight;
  varying float vCentroid;
  varying float vT;

  vec3 palette(float t) {
    // Slightly warmer/cooler than trail so network reads as a distinct layer
    vec3 warm = vec3(0.28, 0.17, 0.09);
    vec3 cool = vec3(0.07, 0.17, 0.34);
    return mix(warm, cool, smoothstep(0.0, 1.0, t));
  }

  void main() {
    float ageFade = pow(max(0.0, 1.0 - vAge), 1.2);

    // Directional fade: older end (vT=0) is dim, newer end (vT=1) is bright
    float dirGrad = mix(0.18, 1.0, vT);

    // Slow breath shared across all links
    float pulse = 0.86 + 0.14 * sin(uTime * 1.4);

    // Traveling highlight: a bright point moves from old->new end of each link.
    // Phase is offset by vAge so different-aged links pulse at different times.
    float phase  = mod(uTime * 0.55 + vAge * 4.8, 1.0);
    float travel = max(0.0, 1.0 - abs(vT - phase) * 6.0);

    float base  = ageFade * smoothstep(0.02, 0.75, vWeight) * dirGrad * pulse * 0.66;
    float alpha = base + travel * ageFade * 0.26;

    vec3 color = palette(vCentroid);
    gl_FragColor = vec4(color, alpha);
  }
`;

// ── Wireframe indicators (bounding box, crosshair, velocity) ──
// Shared shader for all line-based HUD elements. A subtle time-based
// pulse keeps them feeling alive; age/intensity drive per-vertex fade.

export const wireVertexShader = /* glsl */ `
  attribute float aAlpha;

  varying float vAlpha;

  void main() {
    vAlpha = aAlpha;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const wireFragmentShader = /* glsl */ `
  uniform float uTime;
  uniform float uBaseAlpha;
  uniform vec3  uColor;

  varying float vAlpha;

  void main() {
    float pulse = 0.88 + 0.12 * sin(uTime * 2.4);
    float alpha = vAlpha * uBaseAlpha * pulse;
    gl_FragColor = vec4(uColor, alpha);
  }
`;

// ── Peak markers ──────────────────────────────────────────
// Hollow diamond glyphs at positions of high spectral flux.
// They mark moments of rapid timbral change in the trajectory.

export const markerVertexShader = /* glsl */ `
  attribute float aMarkerAge;
  attribute float aMarkerIntensity;

  varying float vIntensity;
  varying float vAge;

  uniform float uPointSize;
  uniform float uPixelRatio;
  uniform float uTime;

  void main() {
    vIntensity = aMarkerIntensity;
    vAge = aMarkerAge;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    float pulse = 0.82 + 0.18 * sin(uTime * 2.8 + aMarkerAge * 5.0);
    float sizeScale = 0.5 + aMarkerIntensity * 0.7;
    float ageFade = pow(max(0.0, 1.0 - aMarkerAge), 0.6);

    gl_PointSize = uPointSize * uPixelRatio * sizeScale * pulse * ageFade * 14.0;
    gl_Position  = projectionMatrix * mvPosition;
  }
`;

export const markerFragmentShader = /* glsl */ `
  varying float vIntensity;
  varying float vAge;

  void main() {
    vec2 p = gl_PointCoord - 0.5;
    // Diamond: |x|+|y| < 0.5
    float d = abs(p.x) + abs(p.y);
    if (d > 0.5) discard;
    // Hollow interior
    if (d < 0.30) discard;

    float ageFade = pow(max(0.0, 1.0 - vAge), 0.8);
    float alpha = ageFade * (0.30 + vIntensity * 0.45);

    vec3 color = vec3(0.15, 0.13, 0.11);
    gl_FragColor = vec4(color, alpha);
  }
`;

// ── Mel cloud shaders ─────────────────────────────────────
//
// Particles use gaussian soft falloff so overlapping points
// accumulate into density fields. Patterns emerge from the
// overlap of many semi-transparent blobs rather than from
// individual point colours. An energy threshold hides the
// noise floor entirely.

export const melCloudVertexShader = /* glsl */ `
  attribute float aFrameIndex;
  attribute float aMelBand;
  attribute float aEnergy;

  uniform float uCurrentFrame;
  uniform float uMaxFrames;
  uniform float uPointSize;
  uniform float uPixelRatio;

  varying float vAlpha;
  varying float vMelBand;
  varying float vEnergy;

  void main() {
    float age = uCurrentFrame - aFrameIndex;

    // Hide points outside the visible window
    if (age >= uMaxFrames || aFrameIndex < 0.0) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
      vAlpha = 0.0;
      vMelBand = 0.0;
      vEnergy = 0.0;
      return;
    }

    vMelBand = aMelBand;
    vEnergy = aEnergy;

    // X = time (scrolling), Y = mel freq, Z = subtle height
    float xNorm = 1.0 - age / uMaxFrames;
    float x = (xNorm - 0.5) * 8.0;
    float y = (aMelBand - 0.5) * 5.0;
    float z = aEnergy * 1.2;

    vec4 mvPosition = modelViewMatrix * vec4(x, y, z, 1.0);

    // Energy threshold: below ~0.08, fully invisible
    float visible = smoothstep(0.05, 0.15, aEnergy);

    // Age fade (older data fades)
    float ageFade = 1.0 - smoothstep(0.7, 1.0, age / uMaxFrames);

    // Depth cue
    float depth = clamp((-mvPosition.z - 3.0) / 15.0, 0.0, 1.0);

    // Size: dramatic nonlinear scaling — loud ≈ 25px, threshold ≈ 5px
    float eSize = 0.5 + pow(aEnergy, 0.8) * 10.0;
    gl_PointSize = uPointSize * uPixelRatio * visible * ageFade
                 * eSize * (1.0 - depth * 0.35);

    // Alpha: low per-particle so accumulation creates density
    vAlpha = visible * ageFade * (0.04 + aEnergy * 0.28)
           * (1.0 - depth * 0.25);

    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const melCloudFragmentShader = /* glsl */ `
  varying float vAlpha;
  varying float vMelBand;
  varying float vEnergy;

  void main() {
    // Solid disc (no radial gradient)
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;

    // Dark palette — subtle warm/cool shift by frequency
    vec3 color = vec3(0.06, 0.07, 0.11);
    color += (1.0 - vMelBand) * vec3(0.05, 0.01, -0.03);

    // Energy warms slightly
    color += vEnergy * vec3(0.04, 0.01, -0.01);

    gl_FragColor = vec4(color, vAlpha);
  }
`;
