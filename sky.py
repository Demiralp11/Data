"""
sky.py - shader stars + sun + moon for Panda3D, meant to replace a skybox model.

    from sky import Sky
    self.sky = Sky(self)                          # once, where the skybox was loaded
    self.sky.update(angle, night, moon_brightness)  # every frame, in the update task

angle   : 0..360. Sun is above the horizon for 0..180 (rises at 0, noon at 90,
          sets at 180), below it for 180..360.
night   : 0 (full day) .. 1 (full night). Controls star and moon visibility.
moon_brightness : 0..1, scales the moon disc colour (phase brightness).
"""
import math

from direct.showbase.DirectObject import DirectObject
from panda3d.core import (
    Vec2, Vec3, Shader, GeomVertexData, GeomVertexFormat, Geom, GeomTriangles,
    GeomNode, OmniBoundingVolume, ColorBlendAttrib, ClockObject, GeomVertexWriter,
)

STAR_VERT = r"""
#version 300 es
precision highp float;

in vec4 p3d_Vertex;     // x = vertex index (see _dummy_geom)

uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform float osg_FrameTime;
uniform vec2  ndc_per_px;     // 1/width, 1/height
uniform float night;          // 0 = day, 1 = full night
uniform float star_seed;
uniform float star_size;      // pixel scale, ~1.0

out vec2 coord;
out vec3 color;

float h(float id, float k) {
  return fract(sin(dot(vec2(id, k), vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  int vid = int(p3d_Vertex.x + 0.5);
  int sid = vid / 3;
  int vtx = vid % 3;
  float d = float(sid) + star_seed;

  float theta = h(d, 1.0) * 6.2831853;
  float z     = 2.0 * h(d, 2.0) - 1.0;
  float r     = sqrt(1.0 - z * z);
  vec3  dir   = vec3(r * cos(theta), r * sin(theta), z);

  float mag = pow(h(d, 3.0), 3.0);
  float px  = (1.2 + 2.5 * mag) * star_size;
  float tw  = 0.8 + 0.2 * sin(osg_FrameTime * (1.0 + 3.0 * h(d, 4.0))
                              + 6.2831853 * h(d, 5.0));
  vec3 tint = mix(vec3(1.0, 0.78, 0.55), vec3(0.65, 0.78, 1.0), h(d, 6.0));
  tint = mix(tint, vec3(1.0), 0.5);
  color = tint * mix(0.35, 1.0, mag) * tw * night;

  // w = 0: rotation only, so the stars sit at infinity
  vec4 clip = p3d_ProjectionMatrix * (p3d_ModelViewMatrix * vec4(dir, 0.0));

  vec2 corner = (vtx == 0) ? vec2(0.0, 2.0)
              : (vtx == 1) ? vec2(-1.7320508, -1.0)
                           : vec2( 1.7320508, -1.0);
  coord = corner;

  clip.xy += corner * px * 2.0 * ndc_per_px * clip.w;
  clip.z = clip.w * 0.9999;
  gl_Position = (night <= 0.001) ? vec4(2.0, 2.0, 2.0, 1.0) : clip;
}
"""

STAR_FRAG = r"""
#version 300 es
precision highp float;

out vec4 p3d_FragColor;
in vec2 coord;
in vec3 color;

void main() {
  float d = length(coord);
  float a = smoothstep(1.0, 0.0, d);
  a *= a;
  p3d_FragColor = vec4(color * a, a);
}
"""

DISC_VERT = r"""
#version 300 es
precision highp float;

in vec4 p3d_Vertex;     // x = vertex index (see _dummy_geom)

uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform vec2  ndc_per_px;
uniform vec3  disc_dir;       // direction in the sky pivot's local space
uniform float disc_px;        // radius in pixels (including glow)
uniform float disc_alpha;     // overall fade, 0..1

out vec2 coord;

void main() {
  int vid = int(p3d_Vertex.x + 0.5);
  int vtx = vid % 3;
  vec4 clip = p3d_ProjectionMatrix * (p3d_ModelViewMatrix * vec4(disc_dir, 0.0));

  vec2 corner = (vtx == 0) ? vec2(0.0, 2.0)
              : (vtx == 1) ? vec2(-1.7320508, -1.0)
                           : vec2( 1.7320508, -1.0);
  coord = corner;

  clip.xy += corner * disc_px * 2.0 * ndc_per_px * clip.w;
  clip.z = clip.w * 0.9999;
  // fully faded out: push the triangle off screen so it costs nothing
  gl_Position = (disc_alpha <= 0.001) ? vec4(2.0, 2.0, 2.0, 1.0) : clip;
}
"""

DISC_FRAG = r"""
#version 300 es
precision highp float;

uniform vec3  disc_color;
uniform float core;
uniform float glow_amt;
uniform float disc_alpha;     // overall fade, 0..1

in vec2 coord;
out vec4 p3d_FragColor;

void main() {
  float d = length(coord);
  if (d > 1.0) discard;
  float disc = 1.0 - smoothstep(core * 0.9, core, d);
  float glow = glow_amt * pow(1.0 - d, 3.0);
  float a = clamp(disc + glow, 0.0, 1.0);
  // premultiplied: scaling colour AND alpha is what makes it fade out properly
  p3d_FragColor = vec4(disc_color * a, a) * disc_alpha;
}
"""


# Full-screen triangle (no geometry needed). The fragment shader rebuilds each
# pixel's world-space view ray and intersects it with a horizontal cloud layer.
CLOUD_VERT = r"""
#version 300 es
precision highp float;

in vec4 p3d_Vertex;     // x = vertex index (see _dummy_geom)

out vec2 ndc;

void main() {
  int vid = int(p3d_Vertex.x + 0.5);
  float x = (vid == 1) ? 3.0 : -1.0;
  float y = (vid == 2) ? 3.0 : -1.0;
  ndc = vec2(x, y);
  gl_Position = vec4(x, y, 0.9999, 1.0);
}
"""

CLOUD_FRAG = r"""
#version 300 es
precision highp float;

uniform vec3  cam_pos;
uniform vec3  cam_fwd;
uniform vec3  cam_right;
uniform vec3  cam_up;
uniform vec2  tan_half_fov;
uniform float cloud_alt;      // Z of the layer in render space (rebase already subtracted)
uniform float cloud_scale;    // noise cells per world unit
uniform vec2  cloud_offset;   // noise-space offset: floating origin + wind, done on the CPU
uniform float coverage;       // 0 = clear, 1 = overcast
uniform vec3  sun_dir;        // world-space direction towards the sun
uniform vec3  cloud_color;
uniform vec3  horizon_color;
uniform float overcast;       // 0..1 storm deck: denser, softer, flatter lighting
uniform float flash;          // 0..1 lightning flash

in vec2 ndc;
out vec4 p3d_FragColor;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
  float s = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    s += a * vnoise(p);
    p = p * 2.03 + vec2(17.1, 9.2);
    a *= 0.5;
  }
  return s;
}

void main() {
  vec3 dir = normalize(cam_fwd + ndc.x * tan_half_fov.x * cam_right
                               + ndc.y * tan_half_fov.y * cam_up);
  if (dir.z < 0.02 || cloud_alt <= cam_pos.z) discard;   // below horizon / above layer

  float t = (cloud_alt - cam_pos.z) / dir.z;
  vec2 p = (cam_pos.xy + dir.xy * t) * cloud_scale + cloud_offset;

  float n = fbm(p);
  float thr  = mix(0.72, 0.38, coverage);
  float soft = 0.22;
  thr  = mix(thr, 0.08, overcast);            // storm: nearly everything becomes cloud
  soft = mix(soft, 0.45, overcast);           // ...with soft, smeared edges
  float c = smoothstep(thr, thr + soft, n);
  if (c <= 0.0) discard;

  // second tap towards the sun: thicker towards the light = darker underside
  vec2 toSun = normalize(sun_dir.xy + vec2(1e-4)) * 0.06;
  float n2 = fbm(p + toSun);
  float light = clamp(0.65 + (n - n2) * 4.0, 0.0, 1.0);
  light = mix(light, 0.35, overcast * 0.6);   // storm deck is flat and grey
  float sunGlow = pow(max(dot(dir, sun_dir), 0.0), 6.0) * (1.0 - overcast);

  vec3 col = cloud_color * mix(0.55, 1.1, light) + cloud_color * sunGlow * 0.6;
  col += vec3(0.70, 0.75, 1.0) * flash * smoothstep(0.3, 0.8, n);   // lightning inside the clouds

  // fade into the horizon colour so distant clouds melt into your fog
  float up = smoothstep(0.02, 0.25, dir.z);
  col = mix(horizon_color, col, up);
  float a = c * smoothstep(0.02, 0.15, dir.z);

  p3d_FragColor = vec4(col * a, a);      // premultiplied
}
"""


def _lerp(a, b, t):
    return a + (b - a) * t


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _dummy_geom(name, num_tris):
    """The vertex data only stores each vertex's index (in x): the shaders build the shape from it.
    A real attribute is used, not gl_VertexID, because attribute-less draws can be skipped by the web renderer."""
    vdata = GeomVertexData(name, GeomVertexFormat.getV3(), Geom.UHStatic)
    vdata.setNumRows(num_tris * 3)
    w = GeomVertexWriter(vdata, "vertex")
    for i in range(num_tris * 3):
        w.addData3(float(i), 0.0, 0.0)
    tris = GeomTriangles(Geom.UHStatic)
    tris.addNextVertices(num_tris * 3)
    tris.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    node.setBounds(OmniBoundingVolume())
    node.setFinal(True)
    return node


class Sky(DirectObject):
    def __init__(self, base, num_stars=3000, flip=False, set_background=True):
        """
        flip: rotate the whole sky 180 degrees about the up axis, so the sun
              rises on the opposite side (use it to line the sun disc up with
              your directional light / shadows).
        set_background: let Sky.update() set the window clear colour.
        """
        super().__init__()
        self.base = base
        self.set_background = set_background
        # moon visibility = night ** moon_fade_power. 1 = linear fade with night,
        # higher = the moon stays invisible until it is properly dark.
        self.moon_fade_power = 2.0

        self.pivot = base.render.attachNewNode("sky_pivot")
        self.pivot.setH(180 if flip else 0)

        star_shader = Shader.make(Shader.SL_GLSL, vertex=STAR_VERT, fragment=STAR_FRAG)
        disc_shader = Shader.make(Shader.SL_GLSL, vertex=DISC_VERT, fragment=DISC_FRAG)

        self.stars = self.pivot.attachNewNode(_dummy_geom("stars", num_stars))
        self.stars.setShader(star_shader)
        self.stars.setShaderInput("star_seed", 0.0)
        self.stars.setShaderInput("star_size", 1.0)
        self.stars.setShaderInput("night", 0.0)
        self.stars.setAttrib(ColorBlendAttrib.make(
            ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne, ColorBlendAttrib.OOne))
        self.stars.setBin("background", 1)

        self.sun = self.pivot.attachNewNode(_dummy_geom("sun", 1))
        self.sun.setShader(disc_shader)
        self.sun.setShaderInput("disc_dir", Vec3(0, 1, 0))
        self.sun.setShaderInput("disc_px", 170.0)
        self.sun.setShaderInput("core", 0.13)
        self.sun.setShaderInput("glow_amt", 0.9)
        self.sun.setShaderInput("disc_color", Vec3(1, 0.93, 0.75))
        self.sun.setShaderInput("disc_alpha", 1.0)
        self.sun.setAttrib(ColorBlendAttrib.make(
            ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne, ColorBlendAttrib.OOne))
        self.sun.setBin("background", 2)

        self.moon = self.pivot.attachNewNode(_dummy_geom("moon", 1))
        self.moon.setShader(disc_shader)
        self.moon.setShaderInput("disc_dir", Vec3(0, -1, 0))
        self.moon.setShaderInput("disc_px", 30.0)
        self.moon.setShaderInput("core", 0.9)
        self.moon.setShaderInput("glow_amt", 0.15)
        self.moon.setShaderInput("disc_color", Vec3(0.85, 0.88, 0.95))
        self.moon.setShaderInput("disc_alpha", 0.0)
        self.moon.setAttrib(ColorBlendAttrib.make(
            ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne,
            ColorBlendAttrib.OOneMinusIncomingAlpha))
        self.moon.setBin("background", 3)

        # --- cloud layer: one full-screen triangle, drawn after stars/sun/moon ---
        self.cloud_alt = 260.0                 # world Z of the layer
        self.cloud_scale = 1.0 / 700.0         # smaller = bigger clouds
        self.cloud_wind = Vec2(6.0, 2.0)       # world units per second
        self.coverage = 0.5                    # 0 clear .. 1 overcast (fair-weather value)

        # --- weather look (blended in by update(weather=...)) ---
        self.storm_alt = 150.0                 # cloud base drops in storms
        self.storm_wind_boost = 1.5            # extra wind speed multiplier at full storm
        self.storm_cloud_dark = 0.55           # how much the clouds darken at full storm
        self.storm_sky_day = Vec3(0.30, 0.32, 0.36)
        self.storm_sky_night = Vec3(0.008, 0.010, 0.016)
        self._drift_x = 0.0                    # extra wind distance accumulated in storms
        self._drift_y = 0.0
        cloud_shader = Shader.make(Shader.SL_GLSL, vertex=CLOUD_VERT, fragment=CLOUD_FRAG)
        self.cloud_layer = base.render.attachNewNode(_dummy_geom("clouds", 1))
        self.cloud_layer.setShader(cloud_shader)
        self.cloud_layer.setAttrib(ColorBlendAttrib.make(
            ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne,
            ColorBlendAttrib.OOneMinusIncomingAlpha))
        self.cloud_layer.setBin("background", 4)
        self.cloud_layer.setDepthTest(False)
        self.cloud_layer.setDepthWrite(False)
        self.cloud_layer.setLightOff()
        self.cloud_layer.setTwoSided(True)

        self.nodes = (self.stars, self.sun, self.moon)
        for n in self.nodes:
            n.setDepthTest(False)
            n.setDepthWrite(False)
            n.setLightOff()
            n.setTwoSided(True)

        # Own DirectObject, so this never replaces a handler your game set on
        # the ShowBase object for the same event.
        self.accept("aspectRatioChanged", self._on_resize)
        self._on_resize()

        # Give the cloud shader every input right away. Panda asserts if a frame is
        # drawn (e.g. while a menu is up) before update() has ever been called.
        self._update_clouds(Vec3(1, 1, 1), Vec3(0.5, 0.6, 0.9), 0.0)

    def _on_resize(self):
        """Sun, moon and stars are sized in pixels, so they need the real window size. In the browser the canvas
        size is not final at start-up (and may change without an event), so update() calls this every frame.
        A bogus tiny size would make the discs huge (a white screen), hence the fallback."""
        w, h = self.base.win.getXSize(), self.base.win.getYSize()
        if w < 64 or h < 64:
            w, h = 1280, 720
        if (w, h) == getattr(self, "_size", None):
            return
        self._size = (w, h)
        v = Vec2(1.0 / w, 1.0 / h)
        for n in self.nodes:
            n.setShaderInput("ndc_per_px", v)

    def hide_from(self, mask):
        """Hide the sky from cameras with this camera mask (shadow/reflection cams)."""
        self.pivot.hide(mask)
        self.cloud_layer.hide(mask)

    def _update_clouds(self, cloud_color, horizon_color, cloud_time, origin=None,
                       overcast=0.0, flash=0.0, alt=None, coverage=None):
        base = self.base
        cam = base.cam
        q = cam.getQuat(base.render)
        fov = base.camLens.getFov()
        sun_dir = base.render.getRelativeVector(self.pivot, Vec3(0, 1, 0))
        sun_dir.normalize()

        n = self.cloud_layer
        n.setShaderInput("cam_pos", cam.getPos(base.render))
        n.setShaderInput("cam_fwd", q.getForward())
        n.setShaderInput("cam_right", q.getRight())
        n.setShaderInput("cam_up", q.getUp())
        n.setShaderInput("tan_half_fov", Vec2(math.tan(math.radians(fov[0] * 0.5)),
                                              math.tan(math.radians(fov[1] * 0.5))))

        # Floating origin: the camera is near (0,0,0) in render space, so add the
        # accumulated rebase offset back in (in Python doubles) to get the true
        # world position the clouds are anchored to.
        if origin is None:
            ox = oy = oz = 0.0
        else:
            ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        s = float(self.cloud_scale)
        t = float(cloud_time)
        n.setShaderInput("cloud_alt", float(self.cloud_alt if alt is None else alt) - oz)
        n.setShaderInput("cloud_scale", s)
        n.setShaderInput("cloud_offset",
                         Vec2((ox + self.cloud_wind.x * t + self._drift_x) * s,
                              (oy + self.cloud_wind.y * t + self._drift_y) * s))
        n.setShaderInput("coverage", float(self.coverage if coverage is None else coverage))
        n.setShaderInput("overcast", float(overcast))
        n.setShaderInput("flash", float(flash))
        n.setShaderInput("sun_dir", sun_dir)
        n.setShaderInput("cloud_color", cloud_color)
        n.setShaderInput("horizon_color", horizon_color)

    def update(self, angle, night, moon_brightness=1.0, sky_color=None,
               cloud_color=None, horizon_color=None, cloud_time=None, origin=None,
               weather=0.0, flash=0.0):
        """
        angle: 0..360 (sun above the horizon for 0..180)
        night: 0..1
        moon_brightness: 0..1 (phase brightness; scales the moon colour)
        sky_color: optional colour for the window clear colour; if None a simple
                   day / dusk / night blend is used.
        cloud_color: colour of the clouds (e.g. your self.current_color); if None
                     a white-by-day, dark-by-night default is used.
        horizon_color: colour clouds fade into near the horizon (e.g. your
                       self.horizonfogcurrent_color); if None the sky colour is used.
        cloud_time: seconds driving the wind. Pass your synced custom_time so
                    every client sees the same clouds.
        origin: your floating-origin offset (self.render_origin). Keeps the clouds
                anchored to the real world when you rebase.
        weather: 0 = clear .. 1 = full storm (your WeatherSystem.intensity).
        flash: 0..1 lightning flash.
        """
        self._on_resize()
        night = _clamp01(night)
        w = _clamp01(weather)
        oc = _clamp01(w * 1.5)       # overcast is complete by w ~ 0.67, so clouds lead the rain

        self.pivot.setP(angle)                      # rotate around the X axis
        sun_h = math.sin(math.radians(angle))       # +1 noon, -1 midnight

        # sun is orange near the horizon, and fades out under a storm deck
        sun_c = _lerp(Vec3(1.0, 0.45, 0.15), Vec3(1.0, 0.93, 0.75), _clamp01(sun_h * 4.0))
        self.sun.setShaderInput("disc_color", sun_c)
        self.sun.setShaderInput("disc_alpha", 1.0 - 0.9 * oc)

        # moon: colour follows the phase brightness, visibility fades with night
        m = 0.2 + 0.8 * _clamp01(moon_brightness)
        self.moon.setShaderInput("disc_color", Vec3(0.85 * m, 0.88 * m, 0.95 * m))
        self.moon.setShaderInput("disc_alpha",
                                 (night ** self.moon_fade_power) * (1.0 - 0.9 * oc))

        self.stars.setShaderInput("night", float(night) * (1.0 - oc))

        # sky colour (also used as the default horizon colour for the clouds)
        if sky_color is not None:
            c = Vec3(sky_color[0], sky_color[1], sky_color[2])
        else:
            day_c = Vec3(0.42, 0.62, 0.95)
            night_c = Vec3(0.004, 0.007, 0.02)
            dusk_c = Vec3(0.95, 0.45, 0.2)
            c = _lerp(day_c, night_c, night)
            c = _lerp(c, dusk_c, math.exp(-((sun_h / 0.18) ** 2)) * 0.65)
        storm_c = _lerp(self.storm_sky_day, self.storm_sky_night, night)
        c = _lerp(c, storm_c, oc * 0.85)                      # storms grey the sky
        c = c + Vec3(0.55, 0.60, 0.80) * (flash * oc * 0.4)   # lightning lights the sky
        if self.set_background:
            self.base.setBackgroundColor(c.x, c.y, c.z)

        # clouds
        if cloud_color is None:
            cc = _lerp(Vec3(0.95, 0.95, 1.0), Vec3(0.03, 0.035, 0.06), night)
        else:
            cc = Vec3(cloud_color[0], cloud_color[1], cloud_color[2])
        cc = cc * (1.0 - self.storm_cloud_dark * oc)          # darker storm clouds
        if horizon_color is None:
            hc = c
        else:
            hc = Vec3(horizon_color[0], horizon_color[1], horizon_color[2])
            hc = _lerp(hc, storm_c, oc * 0.8)
        if cloud_time is None:
            cloud_time = ClockObject.getGlobalClock().getFrameTime()

        # Faster wind in storms. It is accumulated rather than multiplied into the
        # time, so changing the weather never makes the clouds jump.
        dt = ClockObject.getGlobalClock().getDt()
        boost = self.storm_wind_boost * oc
        self._drift_x += self.cloud_wind.x * boost * dt
        self._drift_y += self.cloud_wind.y * boost * dt

        self._update_clouds(
            cc, hc, cloud_time, origin,
            overcast=oc, flash=flash,
            alt=_lerp(self.cloud_alt, self.storm_alt, oc),
            coverage=_lerp(self.coverage, 1.0, oc),
        )