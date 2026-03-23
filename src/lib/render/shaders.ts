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

  varying float vAge;
  varying float vEnergy;
  varying float vCentroid;

  uniform float uPointSize;
  uniform float uPixelRatio;

  void main() {
    vAge      = aAge;
    vEnergy   = aEnergy;
    vCentroid = aCentroid;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    // Only the freshest ~0.4 s (age < 0.08 with ageStep = 1/300)
    float fresh = max(0.0, 1.0 - vAge / 0.08);
    float energyScale = 0.14 + pow(clamp(vEnergy, 0.0, 1.0), 1.25) * 1.9;

    gl_PointSize = uPointSize * uPixelRatio * energyScale * fresh * 7.4;
    gl_Position  = projectionMatrix * mvPosition;
  }
`;

export const pointFragmentShader = /* glsl */ `
  varying float vAge;
  varying float vEnergy;
  varying float vCentroid;

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
    float alpha = fresh * fresh * energyGate * 0.86;

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

  varying float vAge;
  varying float vCentroid;
  varying float vEnergy;

  void main() {
    vAge      = aAge;
    vCentroid = aCentroid;
    vEnergy   = aEnergy;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const trailFragmentShader = /* glsl */ `
  varying float vAge;
  varying float vCentroid;
  varying float vEnergy;

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
    float alpha = pow(alive, 2.6) * 0.68 * energyBoost;

    vec3 color = palette(vCentroid);
    // Loud = darker ink; quiet = lighter (contrast drives attention to amplitude)
    color *= 1.0 - vEnergy * 0.25;

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
