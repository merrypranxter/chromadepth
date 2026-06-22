// depth-map.js — depth FBO management and range utilities

/**
 * Create a depth+colour FBO pair.
 * colour0 = RGBA scene render
 * depth   = DEPTH_COMPONENT24 (samplable in WebGL2)
 */
export function makeDepthFBO(gl, w, h) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

  // Colour attachment
  const colTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, colTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colTex, 0);

  // Depth attachment (samplable)
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

/**
 * Linearise a depth buffer value (0..1 NDC) to a 0..1 linear range.
 * Done in GLSL; this is the reference formula:
 *   linear = (2 * near) / (far + near - d * (far - near))
 */
export function linearDepthFormula(near = 0.1, far = 50.0) {
  return { near, far };
}
