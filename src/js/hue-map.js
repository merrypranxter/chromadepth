// hue-map.js — depth-to-hue mapping helpers + regime definitions

export const REGIMES = {
  warm_front: {
    hue_near: 0,    // red
    hue_far: 240,   // blue
    edge_strength: 0.0,
    sat_boost: 1.2,
    glasses_sep: 0.0,
    show_depth: 0,
    invert_depth: 0,
  },
  cool_front: {
    hue_near: 240,  // blue near (inverted)
    hue_far: 0,     // red far
    edge_strength: 0.0,
    sat_boost: 1.2,
    glasses_sep: 0.0,
    show_depth: 0,
    invert_depth: 1,
  },
  glasses_mode: {
    hue_near: 0,
    hue_far: 240,
    edge_strength: 0.3,
    sat_boost: 2.5,
    glasses_sep: 1.0,
    show_depth: 0,
    invert_depth: 0,
  },
  natural: {
    hue_near: 0,
    hue_far: 220,
    edge_strength: 0.0,
    sat_boost: 0.8,  // subtle — no-glasses mode uses softer saturation
    glasses_sep: 0.0,
    show_depth: 0,
    invert_depth: 0,
  },
  edge_enhanced: {
    hue_near: 0,
    hue_far: 240,
    edge_strength: 0.85,
    sat_boost: 1.4,
    glasses_sep: 0.0,
    show_depth: 0,
    invert_depth: 0,
  },
  saturation_map: {
    hue_near: 30,   // all orange-gold, depth from saturation only
    hue_far: 40,
    edge_strength: 0.0,
    sat_boost: 2.0,
    glasses_sep: 0.0,
    show_depth: 0,
    invert_depth: 0,
  },
};

/**
 * Convert HSL to RGB (all 0..1).
 * Used in JS to preview palette colours; the GLSL version is in depth-hue.frag.
 */
export function hsl2rgb(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
  };
  return [f(0), f(8), f(4)];
}

/**
 * Build a 256-entry depth→RGB lookup as a DataTexture.
 * Sampled in the fragment shader: texture(u_hue_lut, vec2(depth, 0.5))
 */
export function buildHueLUT(gl, hue_near_deg, hue_far_deg, sat_boost = 1.0) {
  const N = 256;
  const data = new Uint8Array(N * 4);
  for (let i = 0; i < N; i++) {
    const t   = i / (N - 1);
    const h   = hue_near_deg + (hue_far_deg - hue_near_deg) * t;
    const s   = Math.min(1.0, 0.85 * sat_boost);
    const l   = 0.5;
    const [r, g, b] = hsl2rgb(h, s, l);
    data[i*4+0] = Math.round(r * 255);
    data[i*4+1] = Math.round(g * 255);
    data[i*4+2] = Math.round(b * 255);
    data[i*4+3] = 255;
  }
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, N, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}
