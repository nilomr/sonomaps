/**
 * Shaders for the audio point cloud.
 *
 * Aesthetic: dark, precise points on light cream background.
 * Monochrome with subtle tonal variation from spectral centroid.
 */

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
    vAge = aAge;
    vEnergy = aEnergy;
    vCentroid = aCentroid;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    float alive = max(0.0, 1.0 - vAge);
    float energyScale = 0.5 + vEnergy * 0.7;

    gl_PointSize = uPointSize * uPixelRatio * energyScale * alive
                   * (120.0 / max(-mvPosition.z, 0.1));

    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const pointFragmentShader = /* glsl */ `
  varying float vAge;
  varying float vEnergy;
  varying float vCentroid;

  vec3 palette(float t) {
    // Dark monochrome with subtle cool/warm shift
    // Low centroid (dark sounds): warm charcoal
    // High centroid (bright sounds): cool dark blue
    vec3 warm = vec3(0.12, 0.11, 0.10);  // warm charcoal
    vec3 mid  = vec3(0.08, 0.10, 0.14);  // neutral dark
    vec3 cool = vec3(0.06, 0.10, 0.18);  // cool dark blue

    vec3 c = mix(warm, mid, smoothstep(0.0, 0.45, t));
    return mix(c, cool, smoothstep(0.35, 1.0, t));
  }

  void main() {
    vec2 coord = gl_PointCoord - 0.5;
    float r2 = dot(coord, coord);
    if (r2 > 0.25) discard;

    vec3 color = palette(vCentroid);

    // Energy darkens the point (more energy = more opaque/dark)
    color *= 1.1 - vEnergy * 0.3;

    float alive = max(0.0, 1.0 - vAge);
    float alpha = alive * alive * 0.85;

    gl_FragColor = vec4(color, alpha);
  }
`;

// ── Trail line shaders ────────────────────────────────────

export const trailVertexShader = /* glsl */ `
  attribute float aAge;
  attribute float aCentroid;
  attribute float aEnergy;

  varying float vAge;
  varying float vCentroid;
  varying float vEnergy;

  void main() {
    vAge = aAge;
    vCentroid = aCentroid;
    vEnergy = aEnergy;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const trailFragmentShader = /* glsl */ `
  varying float vAge;
  varying float vCentroid;
  varying float vEnergy;

  vec3 palette(float t) {
    vec3 warm = vec3(0.12, 0.11, 0.10);
    vec3 mid  = vec3(0.08, 0.10, 0.14);
    vec3 cool = vec3(0.06, 0.10, 0.18);
    vec3 c = mix(warm, mid, smoothstep(0.0, 0.45, t));
    return mix(c, cool, smoothstep(0.35, 1.0, t));
  }

  void main() {
    vec3 color = palette(vCentroid);
    float alive = max(0.0, 1.0 - vAge);
    float alpha = alive * alive * alive * 0.25;

    gl_FragColor = vec4(color, alpha);
  }
`;
