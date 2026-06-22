// main.js — chromadepth WebGL2 renderer
// Pipeline:
//   Pass 1: Render 3D scene → depth FBO (colour + depth buffer)
//   Pass 2: depth-hue.frag: read depth tex → apply hue LUT → composite
//   Pass 3: edge.frag: chromatic edge enhancement (optional)

import { buildHueLUT, REGIMES } from './hue-map.js';
import { makeDepthFBO } from './depth-map.js';

const canvas = document.getElementById('gl');
const gl     = canvas.getContext('webgl2');
if (!gl) { document.body.innerHTML = '<p style="color:#fff;padding:2rem">WebGL2 required</p>'; }

gl.enable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

// --- Params ---
const P = {
  hue_near: 0, hue_far: 240,
  edge_strength: 0.0,
  sat_boost: 1.2,
  glasses_sep: 0.0,
  show_depth: 0,
  invert_depth: 0,
  depth_range: 1.0,
  speed: 0.5,
  scene: 0,  // 0=tunnel, 1=landscape, 2=particles
};

let hueLUT = null;

function rebuildLUT() {
  if (hueLUT) gl.deleteTexture(hueLUT);
  hueLUT = buildHueLUT(gl, P.hue_near, P.hue_far, P.sat_boost);
}
rebuildLUT();

// --- Shader sources ---

// Scene vertex: projects 3D points, outputs depth as varying
const SCENE_VS = `#version 300 es
precision highp float;

uniform float u_time;
uniform float u_scene;  // 0=tunnel, 1=landscape, 2=particles
uniform float u_aspect;
uniform float u_speed;

out float v_depth_lin;  // 0=near, 1=far
out vec3  v_color_raw;
out float v_brightness;

const float PI  = 3.14159265;
const float FAR = 20.0;

float rnd(float s)  { return fract(sin(s*127.1+311.7)*43758.5); }
float rnd2(vec2 s)  { return fract(sin(dot(s,vec2(127.1,311.7)))*43758.5); }

void main() {
  float idx = float(gl_VertexID);
  float t   = u_time * u_speed;
  vec3  pos;
  float sz;

  if (u_scene < 0.5) {
    // Tunnel: rings of points flying toward camera
    float ring = mod(idx, 64.0);
    float seg  = floor(idx / 64.0);
    float z    = mod(seg * 1.3 + t * 3.0, FAR);
    float a    = ring / 64.0 * 2.0 * PI;
    float r    = 0.6 + 0.25 * sin(seg * 0.7 + t * 0.5);
    pos = vec3(cos(a)*r, sin(a)*r, -z);
    sz = mix(6.0, 1.0, z/FAR);
    v_brightness = rnd2(vec2(ring, seg));
  } else if (u_scene < 1.5) {
    // Landscape: grid of points with height noise
    float gx = mod(idx, 40.0) - 20.0;
    float gy = floor(idx / 40.0) - 10.0;
    float z  = -abs(gy) * 0.4 - 2.0;
    float height = sin(gx*0.4+t)*0.3 + sin(gy*0.3+t*0.7)*0.25
                 + sin((gx+gy)*0.2+t*0.5)*0.15;
    pos = vec3(gx*0.18, height, z);
    sz = 3.0;
    v_brightness = rnd(idx * 0.013 + 0.5);
  } else {
    // Particles: scattered cloud at varying depths
    float seed = idx * 0.019;
    float x = (rnd(seed)       * 2.0 - 1.0) * 2.5;
    float y = (rnd(seed+0.3)   * 2.0 - 1.0) * 2.0;
    float z = -(rnd(seed+0.7)  * FAR * 0.9 + 0.5);
    float dx = sin(t*0.3 + idx*0.01) * 0.08;
    float dy = cos(t*0.25+ idx*0.013)* 0.06;
    pos = vec3(x+dx, y+dy, z);
    sz = mix(8.0, 1.5, -z/FAR);
    v_brightness = rnd2(vec2(idx, 0.77));
  }

  // Perspective projection
  float fov   = 1.4;
  float proj_x = pos.x * fov / (-pos.z + 0.001);
  float proj_y = pos.y * fov / (-pos.z + 0.001);
  proj_x /= u_aspect;

  // Linear depth 0=near, 1=far
  v_depth_lin = clamp(-pos.z / FAR, 0.0, 1.0);
  v_color_raw = vec3(v_brightness);

  gl_Position  = vec4(proj_x, proj_y, v_depth_lin*2.0-1.0, 1.0);
  gl_PointSize = clamp(sz, 1.0, 20.0);
}`;

const SCENE_FS = `#version 300 es
precision highp float;
in float v_depth_lin;
in float v_brightness;
out vec4 fragColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c,c)*4.0 > 1.0) discard;
  // Store linear depth in alpha, brightness in RGB
  // The depth-hue pass will replace RGB with hue-mapped colour
  fragColor = vec4(vec3(v_brightness), 1.0);
}`;

// Depth-hue fragment: reads scene colour + depth, applies hue LUT
const QUAD_VS = `#version 300 es
in vec2 a_pos;
out vec2 vUv;
void main() { vUv = a_pos*0.5+0.5; gl_Position = vec4(a_pos,0,1); }`;

const DEPTH_HUE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D u_scene_col;   // scene colour render
uniform sampler2D u_depth_tex;   // linear depth (packed in scene R channel via FBO)
uniform sampler2D u_hue_lut;     // 256px depth→colour ramp
uniform float u_show_depth;      // 0=off, 1=show grayscale depth
uniform float u_invert_depth;
uniform float u_depth_range;
uniform float u_glasses_sep;
uniform float u_edge_strength;
uniform vec2  u_texel;
out vec4 fragColor;

void main() {
  vec4  sceneCol = texture(u_scene_col, vUv);
  float d        = texture(u_depth_tex, vUv).r;

  // Linearise WebGL depth buffer (near=0.1, far=20)
  float near = 0.1, far = 20.0;
  float lin  = (2.0*near) / (far + near - d*(far-near));
  lin = clamp(lin * u_depth_range, 0.0, 1.0);
  if (u_invert_depth > 0.5) lin = 1.0 - lin;

  if (u_show_depth > 0.5) {
    fragColor = vec4(vec3(lin), 1.0); return;
  }

  // --- Hue mapping ---
  vec3 hueCol = texture(u_hue_lut, vec2(lin, 0.5)).rgb;

  // --- Chromatic edge enhancement ---
  // Sobel depth gradient; at sharp transitions boost saturation shift
  float edge = 0.0;
  if (u_edge_strength > 0.01) {
    float d00 = texture(u_depth_tex, vUv + vec2(-u_texel.x, -u_texel.y)).r;
    float d10 = texture(u_depth_tex, vUv + vec2( 0.0,       -u_texel.y)).r;
    float d20 = texture(u_depth_tex, vUv + vec2( u_texel.x, -u_texel.y)).r;
    float d01 = texture(u_depth_tex, vUv + vec2(-u_texel.x,  0.0      )).r;
    float d21 = texture(u_depth_tex, vUv + vec2( u_texel.x,  0.0      )).r;
    float d02 = texture(u_depth_tex, vUv + vec2(-u_texel.x,  u_texel.y)).r;
    float d12 = texture(u_depth_tex, vUv + vec2( 0.0,        u_texel.y)).r;
    float d22 = texture(u_depth_tex, vUv + vec2( u_texel.x,  u_texel.y)).r;
    float gx  = -d00 + d20 - 2.0*d01 + 2.0*d21 - d02 + d22;
    float gy  = -d00 - 2.0*d10 - d20 + d02 + 2.0*d12 + d22;
    edge = clamp(sqrt(gx*gx + gy*gy) * 10.0, 0.0, 1.0);
  }

  // Edge: push towards pure red/blue at boundaries for fringing
  vec3 near_col = texture(u_hue_lut, vec2(0.0, 0.5)).rgb;
  vec3 far_col  = texture(u_hue_lut, vec2(1.0, 0.5)).rgb;
  vec3 edgeShift = mix(near_col, far_col, step(0.5, lin));
  hueCol = mix(hueCol, edgeShift, edge * u_edge_strength);

  // --- Glasses separation ---
  // In glasses mode: shift red channel left, blue channel right (prismatic split)
  if (u_glasses_sep > 0.01) {
    float shift = u_glasses_sep * 0.012;
    float r_sep = texture(u_hue_lut, vec2(texture(u_depth_tex, vUv + vec2(-shift, 0)).r, 0.5)).r;
    float b_sep = texture(u_hue_lut, vec2(texture(u_depth_tex, vUv + vec2( shift, 0)).r, 0.5)).b;
    hueCol.r = mix(hueCol.r, r_sep, u_glasses_sep * 0.7);
    hueCol.b = mix(hueCol.b, b_sep, u_glasses_sep * 0.7);
  }

  // Combine: tint scene brightness by hue colour
  float bright = sceneCol.r;
  vec3 col = hueCol * (0.5 + bright * 0.7);

  // Vignette
  float dist = length(vUv - 0.5);
  col *= 1.0 - dist * dist * 0.6;

  fragColor = vec4(col, 1.0);
}`;

// Edge post pass (additional Sobel sharpening on final output)
const EDGE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D u_tex;
uniform float u_strength;
uniform vec2  u_texel;
out vec4 fragColor;
void main() {
  if (u_strength < 0.01) { fragColor = texture(u_tex, vUv); return; }
  vec3 c = texture(u_tex, vUv).rgb;
  // Unsharp mask for chromatic edge pop
  vec3 blur = vec3(0);
  for (int x=-1;x<=1;x++) for (int y=-1;y<=1;y++)
    blur += texture(u_tex, vUv + vec2(float(x),float(y))*u_texel*2.0).rgb / 9.0;
  vec3 sharp = c + (c - blur) * u_strength * 1.5;
  fragColor = vec4(clamp(sharp, vec3(0), vec3(1)), 1.0);
}`;

// --- GL helpers ---
function compile(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
  return s;
}
function link(vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER,   vsSrc));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p));
  return p;
}
function u(prog, name) { return gl.getUniformLocation(prog, name); }
function makeSimpleFBO(w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, tex };
}

// --- Programs ---
const sceneProg    = link(SCENE_VS,    SCENE_FS);
const depthHueProg = link(QUAD_VS,     DEPTH_HUE_FS);
const edgeProg     = link(QUAD_VS,     EDGE_FS);

const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);

const emptyVAO = gl.createVertexArray();

canvas.width  = window.innerWidth;
canvas.height = window.innerHeight;

let W = canvas.width, H = canvas.height;
let sceneFBO = makeDepthFBO(gl, W, H);
let compFBO  = makeSimpleFBO(W, H);

function makeDepthFBO(gl, w, h) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const colTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, colTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colTex, 0);
  const depTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, depTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.NONE);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depTex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, colTex, depTex, w, h };
}

function rebuildFBOs() {
  W = canvas.width; H = canvas.height;
  sceneFBO = makeDepthFBO(gl, W, H);
  compFBO  = makeSimpleFBO(W, H);
}

window.addEventListener('resize', () => {
  canvas.width = window.innerWidth; canvas.height = window.innerHeight;
  rebuildFBOs();
});

// --- UI ---
const PREC = { hue_near:0, hue_far:0, depth_range:2, edge_strength:2, sat_boost:2, glasses_sep:2, show_depth:0, invert_depth:0, speed:2 };
['hue_near','hue_far','depth_range','edge_strength','sat_boost','glasses_sep','show_depth','invert_depth','speed'].forEach(id => {
  const el  = document.getElementById(id);
  const val = document.getElementById('v-'+id);
  if (!el) return;
  el.addEventListener('input', () => {
    P[id] = parseFloat(el.value);
    const labels = { show_depth: ['off','on'], invert_depth: ['off','on'] };
    val.textContent = labels[id] ? labels[id][Math.round(P[id])] : P[id].toFixed(PREC[id]||2);
    if (['hue_near','hue_far','sat_boost'].includes(id)) rebuildLUT();
    updateDepthHint();
  });
});

function updateDepthHint() {
  const el = document.getElementById('depth-hint');
  const near_hue = P.hue_near;
  const far_hue  = P.hue_far;
  const nearName = hueLabel(near_hue);
  const farName  = hueLabel(far_hue);
  el.textContent = `${nearName} near · ${farName} far · ${P.glasses_sep > 0.5 ? 'glasses mode' : 'naked eye'}`;
}
function hueLabel(h) {
  const h360 = ((h % 360) + 360) % 360;
  if (h360 < 20 || h360 >= 340) return 'red';
  if (h360 < 50)  return 'orange';
  if (h360 < 80)  return 'yellow';
  if (h360 < 150) return 'green';
  if (h360 < 200) return 'cyan';
  if (h360 < 260) return 'blue';
  if (h360 < 290) return 'violet';
  return 'magenta';
}

document.querySelectorAll('[data-regime]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-regime]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const r = REGIMES[btn.dataset.regime];
    if (!r) return;
    Object.assign(P, r);
    Object.keys(r).forEach(id => {
      const el  = document.getElementById(id);
      const val = document.getElementById('v-'+id);
      if (!el || !val) return;
      el.value = r[id];
      const labels = { show_depth: ['off','on'], invert_depth: ['off','on'] };
      val.textContent = labels[id] ? labels[id][Math.round(r[id])] : parseFloat(r[id]).toFixed(PREC[id]||2);
    });
    rebuildLUT();
    updateDepthHint();
  });
});

const SCENES = ['tunnel','landscape','particles'];
document.querySelectorAll('[data-scene]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-scene]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    P.scene = SCENES.indexOf(btn.dataset.scene);
  });
});

document.addEventListener('keydown', e => {
  const freq = document.getElementById('frequency');
  switch (e.code) {
    case 'Space': P.scene = (P.scene + 1) % 3;
      document.querySelectorAll('[data-scene]').forEach((b,i) => b.classList.toggle('active', i===P.scene)); break;
    case 'KeyD': P.invert_depth = P.invert_depth > 0.5 ? 0 : 1;
      document.getElementById('invert_depth').value = P.invert_depth;
      document.getElementById('v-invert_depth').textContent = P.invert_depth > 0.5 ? 'on' : 'off'; break;
    case 'KeyE': P.edge_strength = P.edge_strength > 0.4 ? 0 : 0.85;
      document.getElementById('edge_strength').value = P.edge_strength;
      document.getElementById('v-edge_strength').textContent = P.edge_strength.toFixed(2); break;
    case 'KeyG': P.glasses_sep = P.glasses_sep > 0.5 ? 0 : 1.0;
      document.getElementById('glasses_sep').value = P.glasses_sep;
      document.getElementById('v-glasses_sep').textContent = P.glasses_sep.toFixed(2);
      updateDepthHint(); break;
    case 'KeyP': P.show_depth = P.show_depth > 0.5 ? 0 : 1;
      document.getElementById('show_depth').value = P.show_depth;
      document.getElementById('v-show_depth').textContent = P.show_depth > 0.5 ? 'on' : 'off'; break;
    case 'ArrowUp':   P.depth_range = Math.min(2.0, P.depth_range + 0.1);
      document.getElementById('depth_range').value = P.depth_range;
      document.getElementById('v-depth_range').textContent = P.depth_range.toFixed(2); break;
    case 'ArrowDown': P.depth_range = Math.max(0.1, P.depth_range - 0.1);
      document.getElementById('depth_range').value = P.depth_range;
      document.getElementById('v-depth_range').textContent = P.depth_range.toFixed(2); break;
  }
});

function drawQuad(prog) {
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  const loc = gl.getAttribLocation(prog, 'a_pos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

let frameCount = 0, lastFPS = performance.now();
const fpsEl = document.getElementById('fps');
const POINT_COUNTS = [512, 1200, 600];  // per scene

function render(now) {
  requestAnimationFrame(render);
  const t = now * 0.001;
  frameCount++;
  if (now - lastFPS > 1000) { fpsEl.textContent = frameCount+' fps'; frameCount=0; lastFPS=now; }

  const aspect = W / H;

  // Pass 1: scene → FBO with depth
  gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO.fbo);
  gl.viewport(0, 0, W, H);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.02, 0.02, 0.04, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.BLEND);

  gl.useProgram(sceneProg);
  gl.bindVertexArray(emptyVAO);
  gl.uniform1f(u(sceneProg,'u_time'), t);
  gl.uniform1f(u(sceneProg,'u_scene'), P.scene);
  gl.uniform1f(u(sceneProg,'u_aspect'), aspect);
  gl.uniform1f(u(sceneProg,'u_speed'), P.speed);
  gl.drawArrays(gl.POINTS, 0, POINT_COUNTS[P.scene]);

  // Pass 2: depth-hue composite → compFBO
  gl.bindFramebuffer(gl.FRAMEBUFFER, compFBO.fbo);
  gl.viewport(0, 0, W, H);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);

  gl.useProgram(depthHueProg);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, sceneFBO.colTex);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, sceneFBO.depTex);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, hueLUT);
  gl.uniform1i(u(depthHueProg,'u_scene_col'), 0);
  gl.uniform1i(u(depthHueProg,'u_depth_tex'), 1);
  gl.uniform1i(u(depthHueProg,'u_hue_lut'),   2);
  gl.uniform1f(u(depthHueProg,'u_show_depth'),   P.show_depth);
  gl.uniform1f(u(depthHueProg,'u_invert_depth'), P.invert_depth);
  gl.uniform1f(u(depthHueProg,'u_depth_range'),  P.depth_range);
  gl.uniform1f(u(depthHueProg,'u_glasses_sep'),  P.glasses_sep);
  gl.uniform1f(u(depthHueProg,'u_edge_strength'),P.edge_strength);
  gl.uniform2f(u(depthHueProg,'u_texel'), 1.0/W, 1.0/H);
  drawQuad(depthHueProg);

  // Pass 3: edge sharpening → screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);
  gl.useProgram(edgeProg);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, compFBO.tex);
  gl.uniform1i(u(edgeProg,'u_tex'), 0);
  gl.uniform1f(u(edgeProg,'u_strength'), P.edge_strength);
  gl.uniform2f(u(edgeProg,'u_texel'), 1.0/W, 1.0/H);
  drawQuad(edgeProg);
}

requestAnimationFrame(render);
