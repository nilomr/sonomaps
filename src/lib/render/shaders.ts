/**
 * Shaders for the audio point cloud.
 *
 * Aesthetic: tiny crisp dots with bright cores, vivid blue→cyan→white palette.
 * Trail line shows trajectory with same coloring.
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
    // Energy scales size subtly: 0.5x–1.2x
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
    // Deep indigo → electric blue → cyan → ice white
    vec3 c0 = vec3(0.10, 0.08, 0.32);
    vec3 c1 = vec3(0.15, 0.35, 0.88);
    vec3 c2 = vec3(0.25, 0.78, 0.92);
    vec3 c3 = vec3(0.82, 0.95, 1.00);

    vec3 c = mix(c0, c1, smoothstep(0.0, 0.30, t));
    c = mix(c, c2, smoothstep(0.20, 0.60, t));
    return mix(c, c3, smoothstep(0.50, 1.0, t));
  }

  void main() {
    vec2 coord = gl_PointCoord - 0.5;
    float r2 = dot(coord, coord);
    if (r2 > 0.25) discard;

    vec3 color = palette(vCentroid);

    // Bright core at center
    float core = 1.0 - smoothstep(0.0, 0.12, sqrt(r2));
    color += core * vec3(0.25, 0.30, 0.35);

    // Energy boosts brightness
    color *= 0.55 + vEnergy * 0.45;

    float alive = max(0.0, 1.0 - vAge);
    // Quadratic fade — crisp disappearance
    float alpha = alive * alive;

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
    vec3 c0 = vec3(0.10, 0.08, 0.32);
    vec3 c1 = vec3(0.15, 0.35, 0.88);
    vec3 c2 = vec3(0.25, 0.78, 0.92);
    vec3 c3 = vec3(0.82, 0.95, 1.00);
    vec3 c = mix(c0, c1, smoothstep(0.0, 0.30, t));
    c = mix(c, c2, smoothstep(0.20, 0.60, t));
    return mix(c, c3, smoothstep(0.50, 1.0, t));
  }

  void main() {
    vec3 color = palette(vCentroid);
    color *= 0.6 + vEnergy * 0.4;

    float alive = max(0.0, 1.0 - vAge);
    // Trail is more visible than before, smooth cubic fade
    float alpha = alive * alive * alive * 0.45;

    gl_FragColor = vec4(color, alpha);
  }
`;
