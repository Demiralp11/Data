"""
terrain.py  --  procedural SDF terrain for Panda3D (exported from main.py)

What is in here
  * numpy signed-distance-field terrain (mountains, ridges, pits) + dig / fill edits
  * surface-nets meshing, 3 LOD rings, time-sliced chunk streaming
  * triplanar texture-array shader with fog (GLSL ES 3.00, works on WebGL2)
  * ChunkManager.ground_batch / ray_hit / los_clear queries (used by firstperson.py)

Quick use
    import terrain
    terrain.set_view_distance(220)          # optional, call BEFORE ChunkManager()
    chunks = terrain.ChunkManager(base)     # base = your ShowBase
    ...every frame...
    chunks.update(base.camera.get_pos(base.render))

Textures: put grass.png / rock.png / sand.png / snow.png in TEXTURE_DIR.
Missing files fall back to flat speckled colours, so nothing else is required.
Physics is optional: if `physics3d` is not importable, terrain colliders are skipped.
"""
import math
import sys
import time
import itertools
import collections

import numpy as np
from panda3d.core import (
    Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
    GeomVertexArrayFormat, Shader, Vec3, Vec4, LVector3f,
    PNMImage, Filename, Texture, SamplerState,
)

try:                                    # optional rigid-body / collision backend
    import physics3d as b3
except Exception:                       # pragma: no cover
    b3 = None


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------


CELL = 4.0
COL_BUILD_RADIUS = 20.0
COL_DROP_RADIUS = 30.0
COLLISION_BUDGET_MS = 4.0
CS = 16
VOXEL = 2.0
RADIUS = 4
Z_LAYERS = (-1, 0, 1)
WORKERS = 3
USE_PROCESSES = False
INSTALLS_PER_FRAME = 1
EDIT_RADIUS = 2.0
FOG_END = 310
EDIT_LODS = 2
CHUNK_WORLD = CS * VOXEL
# SPAWN_CTX = mp.get_context("spawn")
# terrain / cave parameters (same scene as the ember tests)
CAVE_SCALE, CAVE_R = 15.0, 2.0
MACRO_SCALE, MACRO_R = 10.0, 7.0

CAVE_R, MACRO_R = 0.0, 0.0   # no caves
MOUNTAIN_H = 140.0       # extra height at the highest peaks
MOUNTAIN_SCALE = 0.004   # lower = wider ranges
RIDGE_SCALE = 0.012      # lower = broader peaks
Z_MIN, Z_MAX = -32.0, 200.0
TREE_MAX_PIT = 1.0        # skip points where the pit has dug down more than this
TREE_MIN_Z = -3.0 + 1.0   # sea level + margin: no trees at or under the waterline
FOG_END = 310
LOD_RANGES = [140.0, 220.0, FOG_END + 40]

REMOTE_Z_CHUNKS = 1
MISS = object()

PIT_CELL = 320.0
PIT_CHANCE = 0.05
PIT_R = (40.0, 90.0)
PIT_DEPTH = (25.0, 60.0)
GRASS, ROCK, SAND, SNOW, COAL, IRON = range(6)
MAT_NAMES = ("grass", "rock", "sand", "snow", "coal", "iron")
SNOW_START = 26.0
SAND_TOP   = 1.0

_VDT = np.dtype([("v", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 4), ("t", "<f4", 2)])   # 36 bytes
_FMT = None


LAYER_PX = 32
TEXTURE_DIR = "data/level"
_FALLBACK_RGB = {"grass": (72, 118, 48), "rock": (112, 108, 102),
                 "sand": (206, 188, 132), "snow": (238, 243, 250)}


def flat_layer(r, g, b, size=LAYER_PX):
    img = PNMImage(size, size, 3)
    img.fill(r / 255, g / 255, b / 255)
    return img


def speckle_layer(r, g, b, size=LAYER_PX):
    """Flat colour with a little deterministic noise, used when a texture file is missing."""
    img = PNMImage(size, size, 3)
    jit = np.random.default_rng(r * 65536 + g * 256 + b).uniform(0.88, 1.08, (size, size))
    for y in range(size):
        for x in range(size):
            k = float(jit[y, x])
            img.set_xel(x, y, min(1.0, r / 255 * k), min(1.0, g / 255 * k), min(1.0, b / 255 * k))
    return img


def load_layer(path, fallback, pixels=LAYER_PX):
    src = PNMImage(Filename(path))
    if not src.is_valid():
        return speckle_layer(*fallback, size=pixels)
    out = PNMImage(pixels, pixels, 3)
    out.unfiltered_stretch_from(src)
    return out


def build_terrain_textures():
    tex = Texture("terrain_array")
    tex.setup_2d_texture_array(6)
    for i, name in enumerate(("grass", "rock", "sand", "snow")):
        tex.load(load_layer(f"{TEXTURE_DIR}/{name}.png", _FALLBACK_RGB[name]), i, 0)
    tex.load(flat_layer(20, 20, 22), 4, 0)         # coal
    tex.load(flat_layer(170, 110, 75), 5, 0)       # iron
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_minfilter(SamplerState.FT_nearest_mipmap_nearest)
    return tex


def set_view_distance(fog_end):
    """Scale fog and LOD rings together. Call before creating ChunkManager (phones: ~200)."""
    global FOG_END
    FOG_END = float(fog_end)
    LOD_RANGES[:] = [0.45 * FOG_END, 0.71 * FOG_END, FOG_END + 40.0]


TERRAIN_VERT = """#version 300 es
precision highp float;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;

uniform vec3  camera_pos;
uniform float push_radius;
uniform float push_strength;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;

out vec4 fragPos;
out vec4 terrainWeights;
out vec2 oreWeights;
out vec3 tpos;
out vec3 tnormal;

void main() {
    vec3 modelPos = p3d_Vertex.xyz;
    oreWeights     = p3d_MultiTexCoord0;
    terrainWeights = p3d_Color;
    tpos           = p3d_Vertex.xyz;     // before the push displacement
    tnormal        = p3d_Normal;

    vec3 camLocal = camera_pos - p3d_ModelMatrix[3].xyz;
    float dist = distance(modelPos, camLocal);
    if (dist < push_radius) {
        modelPos.z -= (1.0 - dist / push_radius) * push_strength;
    }

    gl_Position = p3d_ModelViewProjectionMatrix * vec4(modelPos, 1.0);
    fragPos     = p3d_ModelViewMatrix * vec4(modelPos, 1.0);
}
"""


TERRAIN_FRAG = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2DArray;

uniform sampler2DArray terrainTextures;
uniform float terrainTexScale;

uniform vec3  sun_dir;            // world space, pointing toward the sun (normalized)
uniform vec4  sun_color;
uniform vec4  ambientLightColor;
uniform vec4  fogColor;
uniform vec4  horizonColorb;
uniform float fogStart;
uniform float fogEnd;
uniform vec4  modelColor;

in vec4 fragPos;
in vec4 terrainWeights;           // r=grass g=rock b=sand a=snow
in vec2 oreWeights;               // x=coal y=iron
in vec3 tpos;
in vec3 tnormal;

out vec4 color;

vec4 layerTri(int l, vec3 p, vec3 b) {
    return texture(terrainTextures, vec3(p.yz * terrainTexScale, float(l))) * b.x
         + texture(terrainTextures, vec3(p.xz * terrainTexScale, float(l))) * b.y
         + texture(terrainTextures, vec3(p.xy * terrainTexScale, float(l))) * b.z;
}

vec4 splatTerrain(vec3 p, vec3 n, vec4 m, vec2 ore) {
    n = normalize(n);
    float w[6] = float[6](m.r, m.g, m.b, m.a, ore.x, ore.y);
    int best = 0;
    for (int i = 1; i < 6; i++)
        if (w[i] > w[best]) best = i;

    vec3 b = pow(abs(n), vec3(4.0));
    b /= (b.x + b.y + b.z);
    vec4 c = layerTri(best, p, b);
    c.a = 1.0;
    return c;
}

void main() {
    vec4 tex = splatTerrain(tpos, tnormal, terrainWeights, oreWeights);

    // chunk nodes are translation-only, so the model-space normal is also the world normal
    vec3 n = normalize(tnormal);
    vec3 light = sun_color.rgb * max(dot(n, sun_dir), 0.0) + ambientLightColor.rgb;

    float heightFog = clamp((fogEnd - abs(fragPos.y))   / (fogEnd - fogStart), 0.0, 1.0);
    float depthFog  = clamp((fogEnd - length(fragPos.xyz)) / (fogEnd - fogStart), 0.0, 1.0);
    float fogFactor = min(heightFog, depthFog);

    vec4 horizon = mix(fogColor, horizonColorb, 0.5);
    vec3 rgb = mix(horizon.rgb, tex.rgb * light, fogFactor);
    color = vec4(rgb * modelColor.rgb, modelColor.a);
}
"""

def _chunk_format():
    arr = GeomVertexArrayFormat()
    arr.add_column("vertex",   3, Geom.NT_float32, Geom.C_point)
    arr.add_column("normal",   3, Geom.NT_float32, Geom.C_normal)
    arr.add_column("color",    4, Geom.NT_uint8,   Geom.C_color)
    arr.add_column("texcoord", 2, Geom.NT_float32, Geom.C_texcoord)
    fmt = GeomVertexFormat()
    fmt.add_array(arr)
    fmt = GeomVertexFormat.register_format(fmt)
    assert fmt.get_array(0).get_stride() == _VDT.itemsize, fmt.get_array(0).get_stride()
    return fmt



def _filled_mask(P, edits, margin):
    if isinstance(edits, np.ndarray):
        A = edits
    elif edits:
        A = np.array([(0.0 if k == "dig" else 1.0, x, y, z, r) for k, x, y, z, r in edits], np.float32)
    else:
        return np.zeros(len(P), bool)
    f = A[A[:, 0] == 1]
    if not len(f):
        return np.zeros(len(P), bool)
    d2 = ((P[:, None, :] - f[None, :, 1:4]) ** 2).sum(-1)       # (verts, fills)
    return (d2 < (f[:, 4] + margin) ** 2).any(1)
def material_id(P, N, lod=0, edits=()):
    """P (n,3) world points, N (n,3) normals -> (n,) int8 material, exactly one each."""
    P = np.asarray(P, np.float32); N = np.asarray(N, np.float32)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    depth = terrain_h(x, y) - z
    breakup = vnoise(x * 0.015, y * 0.015, np.zeros_like(x)) - 0.5

    mat = np.full(len(P), GRASS, np.int8)
    mat[z < SAND_TOP + breakup * 2.0] = SAND
    mat[z > SNOW_START + 3.0 + breakup * 8.0] = SNOW
    steep = (1.0 - N[:, 2] + breakup * 0.15) > 0.35
    mat[steep | (depth > 2.0)] = ROCK

    if lod <= 1:
        filled = _filled_mask(P, edits, margin=lod_voxel(lod))
        c = vnoise(x * 0.15, y * 0.15, z * 0.15) * 0.7 + vnoise(x * 0.40 + 37, y * 0.40, z * 0.40) * 0.3
        f = vnoise(x * 0.11 + 517, y * 0.11, z * 0.11) * 0.7 + vnoise(x * 0.35, y * 0.35 + 91, z * 0.35) * 0.3
        mat[(c > 0.64) & (depth > 1.5) & ~filled] = COAL
        mat[(f > 0.68) & (depth > 6.0) & ~filled] = IRON
    return mat
def material_weights(P, N, lod=0, edits=()):
    m = material_id(P, N, lod, edits)

    w = np.zeros((len(m), 6), np.float32)
    w[np.arange(len(m)), m] = 1.0
    return w


def pits(x, y):
    """Smooth, sparse depressions. Returns how much to subtract from the height."""
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    gx = np.floor(x / PIT_CELL).astype(np.int64)
    gy = np.floor(y / PIT_CELL).astype(np.int64)
    out = np.zeros_like(x)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            ix, iy = gx + dx, gy + dy
            r_exist = _hash3(ix, iy, np.full_like(ix, 101))
            px = (ix + 0.2 + 0.6 * _hash3(ix, iy, np.full_like(ix, 102))) * PIT_CELL
            py = (iy + 0.2 + 0.6 * _hash3(ix, iy, np.full_like(ix, 103))) * PIT_CELL
            rad = PIT_R[0] + (PIT_R[1] - PIT_R[0]) * _hash3(ix, iy, np.full_like(ix, 104))
            dep = PIT_DEPTH[0] + (PIT_DEPTH[1] - PIT_DEPTH[0]) * _hash3(ix, iy, np.full_like(ix, 105))
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            t = _sstep(1.0, 0.0, dist / rad)
            t = t * t * (3.0 - 2.0 * t)
            out = np.maximum(out, np.where(r_exist < PIT_CHANCE, dep * t, 0.0))
    return out

 
def lod_voxel(lod):  return VOXEL * (1 << lod)
def lod_world(lod):  return CS * lod_voxel(lod)
# ----------------------------------------------------------------------------
# Numpy SDF (module level so worker processes can import it)
# ----------------------------------------------------------------------------
_SEED_MIX = np.uint32(0)
_SEED_CUR = None

def set_seed(seed):
    """Call in every process that evaluates terrain."""
    global _SEED_MIX, _SEED_CUR
    if seed == _SEED_CUR:
        return
    _SEED_CUR = seed
    _SEED_MIX = np.uint32((((int(seed) ^ 29) & 0x7FFFFFFF) * 2654435761) & 0xFFFFFFFF)

def _hash3(ix, iy, iz):
    with np.errstate(over="ignore"):
        n = ((ix.astype(np.uint32) * np.uint32(73856093))
             ^ (iy.astype(np.uint32) * np.uint32(19349663))
             ^ (iz.astype(np.uint32) * np.uint32(83492791))
             ^ _SEED_MIX)                       # <- new
        n = (n ^ (n >> np.uint32(13))) * np.uint32(1274126177)
        n = n ^ (n >> np.uint32(16))
    return (n & np.uint32(0xFFFFFF)).astype(np.float32) * np.float32(1.0 / 16777216.0)
def vnoise(x, y, z):
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32); z = np.asarray(z, np.float32)
    fx0, fy0, fz0 = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - fx0, y - fy0, z - fz0
    ix, iy, iz = fx0.astype(np.int64), fy0.astype(np.int64), fz0.astype(np.int64)
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fz = fz * fz * (3.0 - 2.0 * fz)

    def lerp(a, b, t):
        return a + (b - a) * t

    c = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c[dx, dy, dz] = _hash3(ix + dx, iy + dy, iz + dz)
    return lerp(lerp(lerp(c[0, 0, 0], c[1, 0, 0], fx), lerp(c[0, 1, 0], c[1, 1, 0], fx), fy),
                lerp(lerp(c[0, 0, 1], c[1, 0, 1], fx), lerp(c[0, 1, 1], c[1, 1, 1], fx), fy), fz)


def fbm(x, y, z):
    a, s = 0.5, 0.0
    for _ in range(4):
        s = s + a * (vnoise(x, y, z) - 0.5)
        x, y, z = x * 2.0, y * 2.0, z * 2.0
        a *= 0.5
    return s


def _sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

class HeightCache:
    T = 32.0
    S = 0.5
    def __init__(self):
        self.tiles = {}
        self.n = int(self.T / self.S) + 1

    def _tile(self, tx, ty):
        t = self.tiles.get((tx, ty))
        if t is None:
            xs = (tx * self.T + np.arange(self.n, dtype=np.float32) * self.S)
            ys = (ty * self.T + np.arange(self.n, dtype=np.float32) * self.S)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            t = terrain_h(X.ravel(), Y.ravel()).reshape(self.n, self.n).astype(np.float32)
            if len(self.tiles) > 400:
                self.tiles.clear()
            self.tiles[(tx, ty)] = t
        return t

    def sample(self, x, y):
        tx = np.floor(x / self.T).astype(np.int64)
        ty = np.floor(y / self.T).astype(np.int64)
        key = tx * 1000003 + ty
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        out = np.empty(len(x), np.float32)
        for g, f in enumerate(first):
            idx = np.nonzero(inv == g)[0]
            tile = self._tile(int(tx[f]), int(ty[f]))
            u = (x[idx] - tx[f] * self.T) / self.S
            v = (y[idx] - ty[f] * self.T) / self.S
            i = np.minimum(u.astype(np.int32), self.n - 2)
            j = np.minimum(v.astype(np.int32), self.n - 2)
            fx, fy = u - i, v - j
            out[idx] = ((tile[i, j] * (1 - fx) + tile[i + 1, j] * fx) * (1 - fy)
                        + (tile[i, j + 1] * (1 - fx) + tile[i + 1, j + 1] * fx) * fy)
        return out

def terrain_h(x, y):
    z0 = np.float32(0.0)
    hills = 25.0 * (fbm(x * 0.012, y * 0.012, z0) + 0.47)
    m = fbm(x * MOUNTAIN_SCALE + 31.7, y * MOUNTAIN_SCALE + 17.3, z0) + 0.47
    mask = _sstep(0.45, 0.75, m)
    r = 1.0 - np.abs(fbm(x * RIDGE_SCALE + 5.1, y * RIDGE_SCALE + 9.7, z0) * 2.0)
    r = r * r
    return hills + mask * MOUNTAIN_H * (0.35 + 0.65 * r) - pits(x, y)

def _sd_sphere(x, y, z, c, r):
    return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - r

def _sd_capsule(x, y, z, a, b, r):
    pa = np.stack([x - a[0], y - a[1], z - a[2]], -1)
    ba = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], np.float32)
    hh = np.clip((pa @ ba) / float(ba @ ba), 0.0, 1.0)
    q = pa - hh[:, None] * ba
    return np.sqrt((q ** 2).sum(-1)) - r

def scene_flat_base(x, y, z, h):
    """x,y,z,h: 1D float32 arrays (h = terrain height under each point). negative = solid."""
    d = (0.6 * (z - h)).astype(np.float32)

    sel = np.nonzero(np.abs(d) < 3.0)[0]
    if sel.size:
        d[sel] -= 0.8 * (vnoise(x[sel] * 0.35, y[sel] * 0.35, z[sel] * 0.35) - 0.5)

    sel = np.nonzero(d < MACRO_R)[0]
    if sel.size:
        xs, ys, zs = x[sel] * 0.006, y[sel] * 0.006, z[sel] * 0.006
        m1 = fbm(xs, ys, zs)
        m2 = fbm(xs + 17.3, ys + 17.3, zs + 17.3)
        d[sel] = np.maximum(d[sel], -(np.hypot(m1, m2) * MACRO_SCALE - MACRO_R))

    sel = np.nonzero(d < CAVE_R)[0]                              # small caves
    if sel.size:
        xs, ys, zs = x[sel] * 0.03, y[sel] * 0.03, z[sel] * 0.03
        n1 = fbm(xs, ys, zs)
        n2 = fbm(xs + 31.7, ys + 31.7, zs + 31.7)
        d[sel] = np.maximum(d[sel], -(np.hypot(n1, n2) * CAVE_SCALE - CAVE_R))

    sel = np.nonzero(d < 7.0)[0]
    if sel.size:
        xs, ys, zs = x[sel], y[sel], z[sel]
        ds = d[sel]
        ds = np.maximum(ds, -_sd_capsule(xs, ys, zs, (-60, -40, 12), (-30, -10, 10), 2.5))
        ds = np.maximum(ds, -_sd_sphere(xs, ys, zs, (-30, -10, 10), 7.0))
        d[sel] = ds

    return d

def _as_edit_array(E):
    if isinstance(E, np.ndarray):
        return E
    if not E:
        return np.zeros((0, 5), np.float32)
    return np.array([(0.0 if k == "dig" else 1.0, x, y, z, r) for k, x, y, z, r in E], np.float32)

def apply_edits_near(d, x, y, z, E, margin=4.0):
    """E: (M,5) rows (0=dig/1=fill, cx, cy, cz, r) in application order; tuples also accepted."""
    E = _as_edit_array(E)
    if len(E) == 0:
        return d
    lo = np.array([x.min(), y.min(), z.min()], np.float32)
    hi = np.array([x.max(), y.max(), z.max()], np.float32)
    c, rr = E[:, 1:4], E[:, 4] + margin
    keep = np.flatnonzero(np.all((c + rr[:, None] >= lo) & (c - rr[:, None] <= hi), axis=1))
    if not keep.size:
        return d
    Ek = E[keep]
    if len(x) <= 512:
        dx = x[:, None] - Ek[None, :, 1]
        dy = y[:, None] - Ek[None, :, 2]
        dz = z[:, None] - Ek[None, :, 3]
        near = ((dx * dx + dy * dy + dz * dz) < (Ek[:, 4] + margin) ** 2).any(0)
        Ek = Ek[near]
    for k, cx, cy, cz, r in Ek.tolist():
        rm = r + margin
        dx, dy, dz = x - cx, y - cy, z - cz
        d2 = dx * dx + dy * dy + dz * dz
        sel = np.flatnonzero(d2 < rm * rm)
        if not sel.size:
            continue
        s = np.sqrt(d2[sel]) - r
        d[sel] = np.maximum(d[sel], -s) if k == 0 else np.minimum(d[sel], s)
    return d

def scene_flat(x, y, z, h, edits, margin=4.0):
    d = scene_flat_base(x, y, z, h)
    return apply_edits_near(d, x, y, z, edits, margin).astype(np.float32)
def scene_points(P, edits):
    P = np.asarray(P, np.float32)
    x, y, z = np.ascontiguousarray(P[:, 0]), np.ascontiguousarray(P[:, 1]), np.ascontiguousarray(P[:, 2])
    return scene_flat(x, y, z, terrain_h(x, y), edits)

def scene_points_phys(P, E, hc):
    P = np.asarray(P, np.float32)
    x, y, z = (np.ascontiguousarray(P[:, i]) for i in range(3))
    d = scene_flat_base(x, y, z, hc.sample(x, y))
    return apply_edits_near(d, x, y, z, E).astype(np.float32)
def scene_grid(xs, ys, zs, edits):
    shape = (len(xs), len(ys), len(zs))
    h2 = terrain_h(xs[:, None], ys[None, :])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    H = np.broadcast_to(h2[:, :, None], shape)
    margin = max(4.0, 2.5 * float(xs[1] - xs[0]))
    d = scene_flat(X.ravel(), Y.ravel(), Z.ravel(), np.ascontiguousarray(H).ravel(), edits, margin)
    return d.reshape(shape)
def scene_normals(P, edits, e):
    k = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], np.float32)
    g = np.zeros_like(P)
    for kk in k:
        g += kk * scene_points(P + kk * e, edits)[:, None]
    ln = np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    return (g / ln).astype(np.float32)

def surface_points(xs, ys, edits=(), margin=12.0, depth=40.0, step=1.0, iters=8):
    """Topmost surface under (xs, ys). Returns (idx, pos[M,3], normal[M,3]);
    idx says which input points hit. Pass edits=() to ignore digs."""
    xs = np.asarray(xs, np.float32); ys = np.asarray(ys, np.float32)
    top = terrain_h(xs, ys) + margin
    offs = np.arange(0.0, depth, step, np.float32)
    N, K = len(xs), len(offs)
    X = np.repeat(xs[:, None], K, 1); Y = np.repeat(ys[:, None], K, 1)
    Z = top[:, None] - offs[None, :]
    d = scene_points(np.stack([X, Y, Z], -1).reshape(-1, 3), edits).reshape(N, K)

    solid = d < 0
    first = solid.argmax(1)
    ok = solid.any(1) & (first > 0)
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return idx, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    hi = Z[idx, first[idx] - 1]
    lo = Z[idx, first[idx]]
    px, py = xs[idx], ys[idx]
    for _ in range(iters):
        mid = (hi + lo) * 0.5
        s = scene_points(np.stack([px, py, mid], -1), edits) < 0
        lo = np.where(s, mid, lo)
        hi = np.where(s, hi, mid)
    pos = np.stack([px, py, (hi + lo) * 0.5], -1).astype(np.float32)
    nrm = scene_normals(pos, edits, 0.15)
    return idx, pos, nrm

def ground_near(xs, ys, zs, edits=(), up=4.0, down=8.0, step=0.5, iters=8):
    xs = np.asarray(xs, np.float32); ys = np.asarray(ys, np.float32); zs = np.asarray(zs, np.float32)
    N = len(xs)
    offs = np.arange(0.0, up + down + step, step, np.float32)
    K = len(offs)
    Z = (zs + up)[:, None] - offs[None, :]
    X = np.repeat(xs[:, None], K, 1); Y = np.repeat(ys[:, None], K, 1)
    d = scene_points(np.stack([X, Y, Z], -1).reshape(-1, 3), edits).reshape(N, K)

    solid = d < 0
    cross = solid[:, 1:] & ~solid[:, :-1]
    ok = cross.any(1)
    first = cross.argmax(1) + 1
    idx = np.nonzero(ok)[0]

    z = np.full(N, np.nan, np.float32)
    nrm = np.zeros((N, 3), np.float32)
    if len(idx):
        hi = Z[idx, first[idx] - 1]
        lo = Z[idx, first[idx]]
        px, py = xs[idx], ys[idx]
        for _ in range(iters):
            mid = (hi + lo) * 0.5
            s = scene_points(np.stack([px, py, mid], -1), edits) < 0
            lo = np.where(s, mid, lo)
            hi = np.where(s, hi, mid)
        zz = (hi + lo) * 0.5
        z[idx] = zz
        nrm[idx] = scene_normals(np.stack([px, py, zz], -1), edits, 0.15)
    return ok, z, nrm

def sample_positions_sdf(origin, size_x, size_y, spacing, jitter=1.0,
                         max_slope_deg=40, edits=(), seed=0,
                         density_scale=0.012, density_bias=0.0):
    gx, gy = np.meshgrid(np.arange(0, size_x, spacing), np.arange(0, size_y, spacing))
    wx = (origin.x + gx).ravel().astype(np.float32)
    wy = (origin.y + gy).ravel().astype(np.float32)

    ix = np.round(wx / spacing).astype(np.int64)
    iy = np.round(wy / spacing).astype(np.int64)
    r1 = _hash3(ix, iy, np.full_like(ix, seed + 1))
    r2 = _hash3(ix, iy, np.full_like(ix, seed + 2))
    r3 = _hash3(ix, iy, np.full_like(ix, seed + 3))

    wx = wx + (r1 - 0.5) * spacing * jitter
    wy = wy + (r2 - 0.5) * spacing * jitter

    dens = fbm(wx * density_scale, wy * density_scale, np.float32(seed + 7.0)) + 0.5 + density_bias
    keep = r3 < np.clip(dens * 1.6, 0.0, 1.0)
    wx, wy = wx[keep], wy[keep]

    _, pos, nrm = surface_points(wx, wy, edits)
    ok = nrm[:, 2] > math.cos(math.radians(max_slope_deg))
    return [(LVector3f(*p), LVector3f(*n)) for p, n in zip(pos[ok], nrm[ok])]

# ----------------------------------------------------------------------------
# Surface nets
# ----------------------------------------------------------------------------
def surface_nets(d):
    n = np.array(d.shape)
    c = n - 1
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    cv = np.stack([d[i:i + c[0], j:j + c[1], k:k + c[2]] for i, j, k in corners], -1)
    ins = cv < 0
    mixed = ins.any(-1) & ~ins.all(-1)
    if not mixed.any():
        return None

    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    cp = np.array(corners, np.float32)
    acc = np.zeros(tuple(c) + (3,), np.float32)
    cnt = np.zeros(tuple(c), np.float32)
    for a, b in edges:
        da, db = cv[..., a], cv[..., b]
        cross = (da < 0) != (db < 0)
        t = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
        acc += np.where(cross[..., None], cp[a] + t[..., None] * (cp[b] - cp[a]), 0.0)
        cnt += cross
    idx = -np.ones(tuple(c), np.int64)
    idx[mixed] = np.arange(mixed.sum())
    verts = acc[mixed] / cnt[mixed][:, None] + np.argwhere(mixed)

    s = d < 0
    tris = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a, b = s[tuple(lo)], s[tuple(hi)]
        pts = np.argwhere(a != b)
        ok = np.ones(len(pts), bool)
        for ax in range(3):
            ok &= (pts[:, ax] >= 1) & (pts[:, ax] <= n[ax] - 2)
        pts = pts[ok]
        if not len(pts):
            continue
        flip = ~a[tuple(pts.T)]
        q = []
        for du, dv in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            cc = pts.copy()
            cc[:, u] += du
            cc[:, v] += dv
            q.append(idx[cc[:, 0], cc[:, 1], cc[:, 2]])
        q = np.stack(q, 1)
        q[flip] = q[flip][:, ::-1]
        tris.append(q[:, [0, 1, 2]])
        tris.append(q[:, [0, 2, 3]])
    if not tris:
        return None
    tris = np.concatenate(tris)

    used, inv = np.unique(tris, return_inverse=True)
    return verts[used].astype(np.float32), inv.reshape(-1, 3).astype(np.int32)
def ground_fast(xs, ys, zs, edits=(), up=3.0, down=3.0, step=0.25):
    xs = np.asarray(xs, np.float32); ys = np.asarray(ys, np.float32); zs = np.asarray(zs, np.float32)
    N = len(xs)
    offs = np.arange(0.0, up + down + step, step, np.float32)
    Z = (zs + up)[:, None] - offs[None, :]
    X = np.broadcast_to(xs[:, None], Z.shape)
    Y = np.broadcast_to(ys[:, None], Z.shape)
    d = scene_points(np.stack([X, Y, Z], -1).reshape(-1, 3), edits).reshape(Z.shape)

    solid = d < 0
    cross = solid[:, 1:] & ~solid[:, :-1]
    ok = cross.any(1)
    i = cross.argmax(1)
    r = np.arange(N)
    d0, d1 = d[r, i], d[r, i + 1]
    t = d0 / np.where(ok, d0 - d1, 1.0)
    z = Z[r, i] + t * (Z[r, i + 1] - Z[r, i])
    return ok, np.where(ok, z, np.nan).astype(np.float32)
def column_tops(d, org_z, vox):
    col = d[1:CS + 2, 1:CS + 2, :]
    s = col < 0
    cross = s[:, :, :-1] & ~s[:, :, 1:]
    ok = cross.any(2)
    k = cross.shape[2] - 1 - cross[:, :, ::-1].argmax(2)
    d0 = np.take_along_axis(col, k[..., None], 2)[..., 0]
    d1 = np.take_along_axis(col, (k + 1)[..., None], 2)[..., 0]
    f = d0 / np.where(ok, d0 - d1, 1.0)
    z = org_z + (k - 1 + f) * vox
    return np.where(ok, z, np.nan).astype(np.float32)

def scatter_cell(ox, oy, size, spacing, min_spacing, max_slope_deg=40.0,
                 jitter=1.0, seed=0, density_scale=0.012, density_bias=0.0,
                 world_seed=29):
    set_seed(world_seed)

    gx, gy = np.meshgrid(np.arange(0, size, spacing), np.arange(0, size, spacing))
    wx = (ox + gx).ravel().astype(np.float32)
    wy = (oy + gy).ravel().astype(np.float32)
    ix = np.round(wx / spacing).astype(np.int64)
    iy = np.round(wy / spacing).astype(np.int64)
    r1 = _hash3(ix, iy, np.full_like(ix, seed + 1))
    r2 = _hash3(ix, iy, np.full_like(ix, seed + 2))
    r3 = _hash3(ix, iy, np.full_like(ix, seed + 3))
    wx = wx + (r1 - 0.5) * spacing * jitter
    wy = wy + (r2 - 0.5) * spacing * jitter
    dens = fbm(wx * density_scale, wy * density_scale, np.float32(seed + 7.0)) + 0.5 + density_bias
    keep = r3 < np.clip(dens * 1.6, 0.0, 1.0)

    wx, wy = wx[keep], wy[keep]


    dry = pits(wx, wy) < TREE_MAX_PIT
    wx, wy = wx[dry], wy[dry]

    _, pos, nrm = surface_points(wx, wy, ())
    ok = nrm[:, 2] > math.cos(math.radians(max_slope_deg))
    ok &= pos[:, 2] > TREE_MIN_Z
    pos, nrm = pos[ok], nrm[ok]


    _, pos, nrm = surface_points(wx, wy, ())
    ok = nrm[:, 2] > math.cos(math.radians(max_slope_deg))
    pos, nrm = pos[ok], nrm[ok]

    inv, ms2 = 1.0 / min_spacing, min_spacing * min_spacing
    grid, sel = {}, []
    for i in range(len(pos)):
        x, y, z = float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])
        gx_, gy_ = math.floor(x * inv), math.floor(y * inv)
        near = False
        for a in (gx_ - 1, gx_, gx_ + 1):
            for b in (gy_ - 1, gy_, gy_ + 1):
                for qx, qy, qz in grid.get((a, b), ()):
                    if (x - qx) ** 2 + (y - qy) ** 2 + (z - qz) ** 2 <= ms2:
                        near = True
                        break
                if near: break
            if near: break
        if not near:
            sel.append(i)
            grid.setdefault((gx_, gy_), []).append((x, y, z))
    return pos[sel], nrm[sel]



def mesh_chunk_gen(key, version, edits, seed):
    set_seed(seed)
    lod, cx, cy, cz = key
    vox, cw = lod_voxel(lod), lod_world(lod)
    org = np.array([cx, cy, cz], np.float32) * cw
    EMPTY = (key, version, None, None, None, None, None, None, 0.0)

    cs = [org[i] + np.linspace(-1, CS, 9, dtype=np.float32) * vox for i in range(3)]
    dc = scene_grid(cs[0], cs[1], cs[2], edits)
    margin = 6.0 * (1 << lod)
    if dc.min() > margin or dc.max() < -margin:
        return EMPTY
    yield

    ax = [org[i] + np.arange(-1, CS + 1, dtype=np.float32) * vox for i in range(3)]
    zs = ax[2]
    slabs = []
    for a in range(0, len(zs), 6):
        slabs.append(scene_grid(ax[0], ax[1], zs[a:a + 6], edits))
        yield
    d = np.concatenate(slabs, axis=2)

    res = surface_nets(d)
    if res is None:
        return EMPTY
    verts, tris = res
    yield

    world = (org + (verts - 1.0) * vox).astype(np.float32)
    normals = scene_normals(world, edits, 0.15 * vox)
    yield

    hm = None
    sdf_grid = None
    if lod == 0:
        hm = column_tops(d, float(org[2]), vox)
        if np.isnan(hm).all():
            hm = None
        sdf_grid = d.astype(np.float16)

    mats = material_weights(world, normals, lod, edits)
    return (key, version, world, normals, mats, tris, hm, sdf_grid, 0.0)

MESH_BUDGET_MS = 5.0
class ChunkManager:
    def __init__(self, base):
        self._todo = collections.deque()
        self._job = None
        self._job_ms = 0.0

        self._patterns = {}
        self.seed = 29
        set_seed(self.seed)

        self.base = base
        self.root = base.render.attach_new_node("chunks")

        r = self.root
        r.set_shader(Shader.make(Shader.SL_GLSL, TERRAIN_VERT, TERRAIN_FRAG))
        self.terrain_textures = build_terrain_textures()
        self.root.set_shader_input("terrainTextures", self.terrain_textures)
        r.set_shader_input("terrainTexScale", 0.1)
        r.set_shader_input("push_radius", 0.0)
        r.set_shader_input("push_strength", 0.0)
        sun = Vec3(0.4, 0.3, 0.85); sun.normalize()
        r.set_shader_input("sun_dir", sun)
        r.set_shader_input("sun_color", Vec4(0.9, 0.85, 0.75, 1))
        r.set_shader_input("ambientLightColor", Vec4(0.35, 0.38, 0.42, 1))
        bg = Vec4(0.55, 0.70, 0.85, 1)
        r.set_shader_input("fogColor", bg)
        r.set_shader_input("horizonColorb", bg)
        r.set_shader_input("fogStart", FOG_END * 0.6)
        r.set_shader_input("fogEnd", float(FOG_END))
        r.set_shader_input("modelColor", Vec4(1, 1, 1, 1))

        self.root.set_shader_input("fog_end", float(FOG_END))
        self.root.set_shader_input("camera_pos", Vec3(0, 0, 0))
        self.root.set_two_sided(False)
        self.col_ms = 0.0

        self.edits = []
        self.versions = {}
        self.chunks = {}
        self.pending = {}
        self.mesh_ms = []
        self.last_scan = -1e9
        self.show_col = False
        self._z0_set = set(self._z_layers(0))
        self._zl = [set(self._z_layers(l)) for l in range(len(LOD_RANGES))]
        self._want, self._want_set = [], set()
        self._lvl_cache = {}
        self._desired_key = None
        self.last_op = "idle"
        self._col_z = {}
        self._dead = collections.deque()
        self._edits_t = ()
        self._hc = HeightCache()
        self._edits_np = np.zeros((0, 5), np.float32)
        self._want_cols = {}

    def load_edits(self, edits):
        """Replace the edit history in one go (world load). Call after set_world_seed."""
        self.edits = [(k, float(x), float(y), float(z), float(r)) for k, x, y, z, r in edits]
        self._edits_t = tuple(self.edits)
        self._edits_np = np.array([(0 if k == "dig" else 1, x, y, z, r)
                                for k, x, y, z, r in self.edits], np.float32).reshape(-1, 5)
        self.versions.clear()
        self.last_scan = -1e9
        self._col_t = -1e9

    def prewarm_height(self, wx, wy):
        T = self._hc.T
        tx, ty = math.floor(wx / T), math.floor(wy / T)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                self._hc._tile(tx + dx, ty + dy)
    def set_world_seed(self, seed):
        set_seed(seed)
        self.seed = seed
        self.reset()
    # ---- streaming ---------------------------------------------------------
    def reset(self):
        self._todo.clear()
        self._job = None
        self._job_ms = 0.0
        for fut, _ in self.pending.values():
            fut.cancel()
        self.pending.clear()


        for c in self.chunks.values():
            self._drop_col(c)
            if c["np"] is not None:
                c["np"].remove_node()
        self.chunks.clear()
        self._edits_np = np.zeros((0, 5), np.float32)
        self._hc.tiles.clear()

        self.edits.clear()
        self._edits_t = ()
        self.versions.clear()
        self._want_cols.clear()

        self._want, self._want_set = [], set()
        self._lvl_cache.clear()
        self._desired_key = None
        self._col_z.clear()
        self.last_scan = -1e9
        self._col_t = -1e9
        self.mesh_ms.clear()
        self.last_op = "idle"
    def _child_z(self, lod, cz):
        valid = self._zl[lod - 1]
        return [z for z in (2 * cz, 2 * cz + 1) if z in valid]

    def sdf_dist(self, P, E=None):
        P = np.asarray(P, np.float32)
        out = np.full(len(P), 9999.0, np.float32)
        cw = CHUNK_WORLD
        vox = VOXEL
        cx = np.floor(P[:, 0] / cw).astype(np.int32)
        cy = np.floor(P[:, 1] / cw).astype(np.int32)
        cz = np.floor(P[:, 2] / cw).astype(np.int32)

        for key in np.unique(
            np.column_stack([cx, cy, cz]),
            axis=0
        ):
            kx, ky, kz = key

            c = self.chunks.get((0, kx, ky, kz))
            if c is None:
                continue

            sdf = c.get("sdf")
            if sdf is None:
                continue

            m = (
                (cx == kx) &
                (cy == ky) &
                (cz == kz)
            )

            pts = P[m]

            lx = (pts[:, 0] - kx * cw) / vox + 1
            ly = (pts[:, 1] - ky * cw) / vox + 1
            lz = (pts[:, 2] - kz * cw) / vox + 1

            ix = np.clip(lx.astype(np.int32), 0, sdf.shape[0]-2)
            iy = np.clip(ly.astype(np.int32), 0, sdf.shape[1]-2)
            iz = np.clip(lz.astype(np.int32), 0, sdf.shape[2]-2)

            fx = lx - ix
            fy = ly - iy
            fz = lz - iz

            c000 = sdf[ix    , iy    , iz    ]
            c100 = sdf[ix + 1, iy    , iz    ]
            c010 = sdf[ix    , iy + 1, iz    ]
            c110 = sdf[ix + 1, iy + 1, iz    ]

            c001 = sdf[ix    , iy    , iz + 1]
            c101 = sdf[ix + 1, iy    , iz + 1]
            c011 = sdf[ix    , iy + 1, iz + 1]
            c111 = sdf[ix + 1, iy + 1, iz + 1]

            x00 = c000 * (1-fx) + c100 * fx
            x10 = c010 * (1-fx) + c110 * fx
            x01 = c001 * (1-fx) + c101 * fx
            x11 = c011 * (1-fx) + c111 * fx

            y0 = x00 * (1-fy) + x10 * fy
            y1 = x01 * (1-fy) + x11 * fy

            out[m] = y0 * (1-fz) + y1 * fz

        return out
    def sdf_normal(self, P, E=None):
        E = self._edits_np if E is None else E
        e = 0.25 * VOXEL
        D = np.array([[e, 0, 0], [-e, 0, 0], [0, e, 0], [0, -e, 0], [0, 0, e], [0, 0, -e]], np.float32)
        Q = (P[:, None, :] + D[None]).reshape(-1, 3)
        s = scene_points_phys(Q, E, self._hc).reshape(-1, 6)
        g = np.stack([s[:, 0] - s[:, 1], s[:, 2] - s[:, 3], s[:, 4] - s[:, 5]], 1)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)

    def _col_ready(self, l, cx, cy):
        """True if every wanted chunk covering this column footprint is installed."""
        cols = self._want_cols
        top = len(LOD_RANGES) - 1
        a, x, y = l, cx, cy
        while True:
            ks = cols.get((a, x, y))
            if ks is not None:
                return all(k in self.chunks for k in ks)
            if a >= top:
                break
            a, x, y = a + 1, x >> 1, y >> 1
        if l == 0:
            return True
        return all(self._col_ready(l - 1, cx * 2 + i, cy * 2 + j)
                for i in (0, 1) for j in (0, 1))
    def _fill_cols(self, lod, cols):
        cw = lod_world(lod)
        cols = [c for c in cols if (lod, c[0], c[1]) not in self._col_z]
        if not cols:
            return
        a = np.array(cols, np.float32)
        t = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        xs = (a[:, 0, None] + t[None, :]) * cw
        ys = (a[:, 1, None] + t[None, :]) * cw
        X = np.repeat(xs[:, :, None], 5, 2).ravel()
        Y = np.repeat(ys[:, None, :], 5, 1).ravel()
        h = terrain_h(X, Y).reshape(len(cols), 25)
        margin = 0.3 * cw + 10.0
        lo, hi = h.min(1) - margin, h.max(1) + margin
        allz = tuple(self._z_layers(lod))
        if len(self._col_z) > 50000:
            self._col_z.clear()
        for i, (cx, cy) in enumerate(cols):
            zl = tuple(z for z in allz if z * cw < hi[i] and (z + 1) * cw > lo[i])
            self._col_z[(lod, cx, cy)] = (zl, frozenset(zl))

    def _wanted_at(self, cam, lod):
        cw = lod_world(lod)
        ccx, ccy = math.floor(cam.x / cw), math.floor(cam.y / cw)
        ck = (ccx, ccy)
        hit = self._lvl_cache.get(lod)
        if hit is not None and hit[0] == ck:
            return hit[1]

        r = LOD_RANGES[lod]
        n = int(r / cw) + 1

        r2 = r * r
        out = set()
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                if (dx * cw) ** 2 + (dy * cw) ** 2 < r2:
                    for cz in self._col_layers(lod, ccx + dx, ccy + dy)[0]:
                        out.add((lod, ccx + dx, ccy + dy, cz))
        self._lvl_cache[lod] = (ck, out)
        return out


    def _desired(self, cam):
        cw0 = lod_world(0)
        key = (math.floor(cam.x / cw0), math.floor(cam.y / cw0))
        if key == self._desired_key:
            return self._want
        leaves = self._leaves_for(key)
        self._fill_cols_batch(leaves)

        out, cols = [], {}
        for l, cx, cy in leaves:
            ks = [(l, cx, cy, cz) for cz in self._col_z[(l, cx, cy)][0]]
            cols[(l, cx, cy)] = ks
            out.extend(ks)

        if out:
            arr = np.array(out, np.int64)
            cw = CS * VOXEL * (1 << arr[:, 0])
            d = ((arr[:, 1] + 0.5) * cw - cam.x) ** 2 + ((arr[:, 2] + 0.5) * cw - cam.y) ** 2
            out = [out[i] for i in np.lexsort((d, arr[:, 0]))]

        self._new_set, self._new_cols = set(out), cols
        self._desired_key = key
        return out

    def _refresh_want(self, cam):
        new = self._desired(cam)
        if new is self._want:
            return False
        self._want, self._want_set, self._want_cols = new, self._new_set, self._new_cols
        return True

    def _fill_cols_batch(self, leaves):
        if len(self._col_z) > 50000:
            self._col_z.clear()
        miss = [(l, cx, cy) for l, cx, cy in leaves if (l, cx, cy) not in self._col_z]
        if not miss:
            return
        a = np.array(miss, np.int64)
        cw = (CS * VOXEL * (1 << a[:, 0])).astype(np.float32)
        t = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        xs = (a[:, 1, None] + t[None, :]) * cw[:, None]
        ys = (a[:, 2, None] + t[None, :]) * cw[:, None]
        X = np.repeat(xs[:, :, None], 5, 2).ravel()
        Y = np.repeat(ys[:, None, :], 5, 1).ravel()
        h = terrain_h(X, Y).reshape(len(miss), 25)
        margin = 0.3 * cw + 10.0
        lo, hi = h.min(1) - margin, h.max(1) + margin
        allz = {l: tuple(self._z_layers(l)) for l in set(a[:, 0].tolist())}
        for i, (l, cx, cy) in enumerate(miss):
            c = float(cw[i])
            zl = tuple(z for z in allz[l] if z * c < hi[i] and (z + 1) * c > lo[i])
            self._col_z[(l, cx, cy)] = (zl, frozenset(zl))

    def _leaves_for(self, key):
        S = 1 << (len(LOD_RANGES) - 1)
        phase = (key[0] % S, key[1] % S)
        pat = self._patterns.get(phase)
        if pat is None:
            pat = self._patterns[phase] = self._build_pattern(phase)
        bx, by = key[0] - phase[0], key[1] - phase[1]
        return [(l, cx + (bx >> l), cy + (by >> l)) for l, cx, cy in pat]

    def _build_pattern(self, phase):
        cw0 = lod_world(0)
        px, py = (phase[0] + 0.5) * cw0, (phase[1] + 0.5) * cw0
        top = len(LOD_RANGES) - 1
        leaves = []

        def box_d(l, cx, cy):
            cw = lod_world(l)
            dx = max(cx * cw - px, 0.0, px - (cx + 1) * cw)
            dy = max(cy * cw - py, 0.0, py - (cy + 1) * cw)
            return (dx * dx + dy * dy) ** 0.5

        def visit(l, cx, cy):
            d = box_d(l, cx, cy)
            if l == top and d >= LOD_RANGES[top]:
                return
            if l > 0 and d < LOD_RANGES[l - 1]:
                for i in (0, 1):
                    for j in (0, 1):
                        visit(l - 1, cx * 2 + i, cy * 2 + j)
                return
            leaves.append((l, cx, cy))

        cwt = lod_world(top)
        n = int(LOD_RANGES[top] / cwt) + 1
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                visit(top, dx, dy)
        return leaves

    def _z_layers(self, lod):
        cw = lod_world(lod)
        return range(math.floor(Z_MIN / cw), math.floor(Z_MAX / cw) + 1)

    def _col_layers(self, lod, cx, cy):
        """(tuple, frozenset) of z layers in this column that can contain surface."""
        k = (lod, cx, cy)
        hit = self._col_z.get(k)
        if hit is None:
            cw = lod_world(lod)
            xs = (cx + np.linspace(0.0, 1.0, 5, dtype=np.float32)) * cw
            ys = (cy + np.linspace(0.0, 1.0, 5, dtype=np.float32)) * cw
            h = terrain_h(xs[:, None], ys[None, :])
            margin = 0.3 * cw + 10.0
            lo, hi = float(h.min()) - margin, float(h.max()) + margin
            zl = tuple(z for z in self._z_layers(lod) if z * cw < hi and (z + 1) * cw > lo)
            if len(self._col_z) > 50000:
                self._col_z.clear()
            hit = self._col_z[k] = (zl, frozenset(zl))
        return hit



    def _drop_col(self, c):
        if b3 is None:
            c["col"].clear()
            return
        for h, _v, _i, gen in c["col"].values():
            if gen == b3._gen:
                b3.remove_body(h)
        c["col"].clear()
    
    def _build_chunk_col(self, c, key):
        verts, tris = c["mesh"]
        tv = verts[tris]
        area2 = np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]), axis=1)
        tris = tris[area2 > 1e-8]
        if not len(tris):
            return
        lo = np.array([key[1], key[2], key[3]], dtype=np.float64) * CHUNK_WORLD
        local = np.ascontiguousarray((verts - lo).astype(np.float32))
        idx = np.ascontiguousarray(tris.astype(np.int32).reshape(-1))
        h = b3.add_static_mesh_indexed(lo, local, idx)
  
        if h is not None:
            c["col"][0] = (h, local, idx, b3._gen)
            
            if c["np"] is not None:
                b3.body_nodes[h] = c["np"]
            else:
                b3.body_info[h] = {"name": f"chunk_{key}", "terrain": True}
    def update_collision(self, pts):
        """pts: list of Vec3 in the same frame as the old `cam` (world_root)."""
        if b3 is None:
            return
        now = time.perf_counter()
        if now - getattr(self, "_col_t", -1e9) < 0.1:
            return
        self._col_t = now
        todo = []
        for key, c in self.chunks.items():
            if key[0] != 0:
                continue
            if c["mesh"] is None:
                if c["col"]:
                    self._drop_col(c)
                continue
            lo = (key[1] * CHUNK_WORLD, key[2] * CHUNK_WORLD, key[3] * CHUNK_WORLD)
            hi = (lo[0] + CHUNK_WORLD, lo[1] + CHUNK_WORLD, lo[2] + CHUNK_WORLD)
            d = min(self._box_dist(p, lo, hi) for p in pts)
            if d > COL_DROP_RADIUS:
                if c["col"]:
                    self._drop_col(c)
            elif not c["col"] and d < COL_BUILD_RADIUS:
                todo.append((d, key))

        todo.sort()
        t0 = time.perf_counter()
        for _, key in todo:
            c = self.chunks.get(key)
            if c is not None and c["mesh"] is not None:
                self.last_op = f"build_col {key}"
                self._build_chunk_col(c, key)
            if (time.perf_counter() - t0) * 1000 > COLLISION_BUDGET_MS:
                break
        self.col_ms = (time.perf_counter() - t0) * 1000

    @staticmethod
    def _box_dist(p, lo, hi):
        dx = max(lo[0] - p.x, 0.0, p.x - hi[0])
        dy = max(lo[1] - p.y, 0.0, p.y - hi[1])
        dz = max(lo[2] - p.z, 0.0, p.z - hi[2])
        return (dx * dx + dy * dy + dz * dz) ** 0.5



    def _refresh_want_remote(self, pts):
        """LOD 0 chunk keys within collision range of any remote player (host only)."""
        cells = tuple((int(p.x // CHUNK_WORLD), int(p.y // CHUNK_WORLD), int(p.z // CHUNK_WORLD))
                    for p in pts)
        if cells == getattr(self, "_remote_cells", None):
            return False, self._remote_set
        self._remote_cells = cells
        r = int(COL_BUILD_RADIUS // CHUNK_WORLD) + 1
        keys = set()
        for p, (cx, cy, cz) in zip(pts, cells):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-REMOTE_Z_CHUNKS, REMOTE_Z_CHUNKS + 1):
                        key = (0, cx + dx, cy + dy, cz + dz)
                        lo = (key[1] * CHUNK_WORLD, key[2] * CHUNK_WORLD, key[3] * CHUNK_WORLD)
                        hi = (lo[0] + CHUNK_WORLD, lo[1] + CHUNK_WORLD, lo[2] + CHUNK_WORLD)
                        if self._box_dist(p, lo, hi) < COL_BUILD_RADIUS + CHUNK_WORLD:
                            keys.add(key)
        self._remote_set = keys
        return True, keys


    def update(self, cam, remote=()):
        self.root.set_shader_input("camera_pos", Vec3(cam))

        t0 = time.perf_counter()
        while self._dead and time.perf_counter() - t0 < 0.002:
            c = self._dead.popleft()
            self._drop_col(c)
            if c["np"] is not None:
                c["np"].remove_node()
        now = time.perf_counter()
        if now - self.last_scan >= 0.30:
            self.last_scan = now
            self._scan(cam, remote)

        self._mesh_budget(MESH_BUDGET_MS)

    def _scan(self, cam, remote):
        self._refresh_want(cam)
        _, rkeys = self._refresh_want_remote(remote)
        want, want_set = self._want, self._want_set
        keep = want_set | rkeys

        for key in want_set:
            c = self.chunks.get(key)
            if c is not None and c["np"] is None and c.get("normals") is not None:
                verts, tris = c["mesh"]
                c["np"] = self._make_node(key, verts, c["normals"], c["colors"], tris)
                c["normals"] = None
                c["colors"] = None
                col = c["col"].get(0)
                if col is not None and b3 is not None:
                    b3.body_nodes[col[0]] = c["np"]
                    b3.body_info.pop(col[0], None)
        for key in list(self.chunks):
            if key not in keep and self._col_ready(key[0], key[1], key[2]):
                self._dead.append(self.chunks.pop(key))

        self._todo = collections.deque(
            k for k in itertools.chain(want, sorted(rkeys - want_set))
            if self.chunks.get(k) is None
            or self.chunks[k]["version"] != self.versions.get(k, 0))

    def _mesh_budget(self, ms=5.0):
        t_start = time.perf_counter()
        while (time.perf_counter() - t_start) * 1000.0 < ms:
            if self._job is None:
                if not self._todo:
                    return
                key = self._todo.popleft()
                ver = self.versions.get(key, 0)
                have = self.chunks.get(key)
                if have is not None and have["version"] == ver:
                    continue
                self.last_op = f"mesh {key}"
                self._job = (key, ver, mesh_chunk_gen(key, ver, self._edits_np, self.seed))
                self._job_ms = 0.0
            key, ver, gen = self._job
            t0 = time.perf_counter()
            try:
                next(gen)
                self._job_ms += (time.perf_counter() - t0) * 1000.0
            except StopIteration as e:
                self._job_ms += (time.perf_counter() - t0) * 1000.0
                self._job = None
                self.mesh_ms = (self.mesh_ms + [self._job_ms])[-50:]
                if ver == self.versions.get(key, 0):
                    self.last_op = f"install {key}"
                    self._install(*e.value[:8])

    def ground_lookup(self, x, y, z, up=3.0, down=3.0):
        cw = CHUNK_WORLD
        cx, cy = math.floor(x / cw), math.floor(y / cw)
        cz = math.floor(z / cw)
        layers = self._col_layers(0, cx, cy)[1]
        u, v = (x - cx * cw) / VOXEL, (y - cy * cw) / VOXEL
        ix, iy = min(int(u), CS - 1), min(int(v), CS - 1)
        fx, fy = u - ix, v - iy
        best = None
        for kz in (cz + 1, cz, cz - 1):
            if kz not in layers:
                continue
            c = self.chunks.get((0, cx, cy, kz))
            if c is None:
                return MISS
            hm = c.get("hm")
            if hm is None:
                continue
            h = hm[ix:ix + 2, iy:iy + 2]
            if np.isnan(h).any():
                continue
            g = float((h[0, 0] * (1 - fx) + h[1, 0] * fx) * (1 - fy)
                    + (h[0, 1] * (1 - fx) + h[1, 1] * fx) * fy)
            if z - down <= g <= z + up and (best is None or g > best):
                best = g
        return best

    def material_at(self, p):
        P = np.array([[p.x, p.y, p.z]], np.float32)

        E = self._near_edits(P[0], 0.0)
        N = self.sdf_normal(P, E)
        return MAT_NAMES[int(material_id(P, N, 0, E)[0])]
    def _make_node(self, key, verts, normals, colors, tris):
        global _FMT
        if _FMT is None:
            _FMT = _chunk_format()
        n = len(verts)
        data = np.empty(n, _VDT)
        data["v"], data["n"] = verts, normals
        data["c"] = (colors[:, :4] * 255).astype(np.uint8)    # grass, rock, sand, snow
        data["t"] = colors[:, 4:]      
        vdata = GeomVertexData("chunk", _FMT, Geom.UH_static)
        vdata.unclean_set_num_rows(n)
        memoryview(vdata.modify_array(0)).cast("B")[:] = data.tobytes()
        prim = GeomTriangles(Geom.UH_static)
        prim.set_index_type(Geom.NT_uint16)
        idx_arr = prim.modify_vertices()
        idx_arr.unclean_set_num_rows(tris.size)
        memoryview(idx_arr).cast("B")[:] = tris.astype(np.uint16).tobytes()
        geom = Geom(vdata)
        geom.add_primitive(prim)
        gnode = GeomNode(f"chunk_{key}")
        gnode.add_geom(geom)
        np_ = self.root.attach_new_node(gnode)
        np_.set_bin("fixed", -5 - key[0])
        return np_

    def _install(self, key, version,
             verts, normals, colors,
             tris, hm=None, sdf_grid=None):
        old = self.chunks.get(key)
        if old is not None:
            self._drop_col(old)
            if old["np"] is not None:
                old["np"].remove_node()
        if verts is None:
            self.chunks[key] = {"np": None, "version": version, "tris": 0,
                    "mesh": None, "normals": None, "colors": None, "cells": None, "col": {}, "hm": None}
            return

        collision_only = key[0] == 0 and key not in self._want_set
        np_ = None if collision_only else self._make_node(key, verts, normals, colors, tris)

        self.chunks[key] = {
            "np": np_,
            "version": version,
            "tris": int(len(tris)),
            "mesh": (verts, tris) if key[0] == 0 else None,
            "normals": normals if collision_only else None,
            "colors": colors if collision_only else None,
            "cells": None,
            "col": {},
            "hm": hm,
            "sdf": sdf_grid,
        }
    # ---- editing -----------------------------------------------------------
    def edit(self, kind, pos, r):
        e = (kind, float(pos.x), float(pos.y), float(pos.z), float(r))
        if self.edits and self.edits[-1] == e:
            return
        self.edits.append(e)
        self._edits_t = tuple(self.edits)
        for lod in range(EDIT_LODS):
            cw, vox = lod_world(lod), lod_voxel(lod)
            zs = set(self._z_layers(lod))
            lo = [math.floor((pos[i] - r - 2 * vox) / cw) for i in range(3)]
            hi = [math.floor((pos[i] + r + 2 * vox) / cw) for i in range(3)]
            for cx in range(lo[0], hi[0] + 1):
                for cy in range(lo[1], hi[1] + 1):
                    for cz in range(lo[2], hi[2] + 1):
                        if cz in zs:
                            k = (lod, cx, cy, cz)
                            self.versions[k] = self.versions.get(k, 0) + 1
        self.last_scan = -1e9
        self._col_t = -1e9

        self._edits_np = np.array([(0 if k == "dig" else 1, x, y, z, r)
                           for k, x, y, z, r in self.edits], np.float32).reshape(-1, 5)
    def _near_edits(self, center, reach):
        A = self._edits_np
        if not len(A):
            return A
        c = np.asarray(center, np.float32)
        m = np.linalg.norm(A[:, 1:4] - c, axis=1) < reach + A[:, 4] + 4.0 * VOXEL
        return A[np.flatnonzero(m)]
    def ray_hit(self, origin, direction, max_dist=160.0):
        o = np.array([origin.x, origin.y, origin.z], np.float32)
        dv = np.array([direction.x, direction.y, direction.z], np.float32)
        step = 0.5 * VOXEL
        BATCH = 48
        seg = BATCH * step
        t = 0.0
        while t < max_dist:
            edits = self._near_edits(o + dv * (t + seg * 0.5), seg * 0.5 + step)
            ts = t + np.arange(BATCH, dtype=np.float32) * step
            d = scene_points(o + dv * ts[:, None], edits)
            hit = np.flatnonzero(d < 0.05)
            if len(hit):
                i = int(hit[0])
                if i == 0 and t == 0.0:
                    return Vec3(*o)
                t0 = float(ts[i - 1]) if i > 0 else max(t - step, 0.0)
                t1 = float(ts[i])
                ts2 = np.linspace(t0, t1, 17, dtype=np.float32)
                d2 = scene_points(o + dv * ts2[:, None], edits)
                nz = np.flatnonzero(d2 < 0.05)
                j = int(nz[0]) if len(nz) else 16
                return Vec3(*(o + dv * float(ts2[j])))
            t += seg
        return None
    # ---- queries used by the first-person controller ----------------------
    def height_at(self, x, y):
        """Base heightfield (ignores digs/fills). Good for spawning."""
        return float(terrain_h(np.array([x], np.float32), np.array([y], np.float32))[0])

    def ground_batch(self, xs, ys, zs, up=2.5, down=6.0, step=0.4, with_top=False):
        """Surface height (digs and fills included) near each (x, y, z).
        Returns (ok[N], z[N]); with_top=True adds top_solid[N] = True where the point
        z+up is already inside terrain (a wall higher than `up`)."""
        xs = np.asarray(xs, np.float32)
        ys = np.asarray(ys, np.float32)
        zs = np.asarray(zs, np.float32)
        h = self._hc.sample(xs, ys)
        offs = np.arange(0.0, up + down + step, step, dtype=np.float32)
        Z = (zs + np.float32(up))[:, None] - offs[None, :]
        shape = Z.shape
        X = np.ascontiguousarray(np.broadcast_to(xs[:, None], shape)).ravel()
        Y = np.ascontiguousarray(np.broadcast_to(ys[:, None], shape)).ravel()
        H = np.ascontiguousarray(np.broadcast_to(h[:, None], shape)).ravel()
        Zf = np.ascontiguousarray(Z).ravel()
        d = scene_flat_base(X, Y, Zf, H)
        E = self._edits_np
        if len(E):
            dx = xs[None, :] - E[:, 1:2]
            dy = ys[None, :] - E[:, 2:3]
            E = E[((dx * dx + dy * dy) < ((E[:, 4] + 4.0) ** 2)[:, None]).any(1)]
            d = apply_edits_near(d, X, Y, Zf, E)
        d = d.reshape(shape)
        solid = d < 0
        cross = solid[:, 1:] & ~solid[:, :-1]
        ok = cross.any(1)
        i = cross.argmax(1)
        r = np.arange(len(xs))
        d0, d1 = d[r, i], d[r, i + 1]
        t = d0 / np.where(ok, d0 - d1, 1.0)
        z = Z[r, i] + t * (Z[r, i + 1] - Z[r, i])
        z = np.where(ok, z, np.nan).astype(np.float32)
        if with_top:
            return ok, z, solid[:, 0]
        return ok, z

    def los_clear(self, a, b):
        """True if the straight line a -> b does not pass through terrain."""
        pa = np.array([a.x, a.y, a.z], np.float32)
        pb = np.array([b.x, b.y, b.z], np.float32)
        length = float(np.linalg.norm(pb - pa))
        n = int(min(64, max(8, length / 1.5)))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        P = pa * (1 - t) + pb * t
        E = self._near_edits((pa + pb) * 0.5, length * 0.5)
        return float(scene_points(P, E).min()) > 0.0

    def set_lighting(self, sun_dir, sun_color, ambient, fog_color, horizon_color=None):
        """Drive the terrain shader from your own sky / day-night code.
        sun_dir points toward the sun (world space); colours are (r, g, b[, a])."""
        def v4(c):
            return Vec4(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)
        r = self.root
        sd = Vec3(*sun_dir[:3])
        sd.normalize()
        r.set_shader_input("sun_dir", sd)
        r.set_shader_input("sun_color", v4(sun_color))
        r.set_shader_input("ambientLightColor", v4(ambient))
        r.set_shader_input("fogColor", v4(fog_color))
        r.set_shader_input("horizonColorb", v4(horizon_color if horizon_color is not None else fog_color))

    def stats(self):
        tris = sum(c["tris"] for c in self.chunks.values())
        meshed = sum(1 for c in self.chunks.values() if c["np"] is not None)
        avg = sum(self.mesh_ms) / len(self.mesh_ms) if self.mesh_ms else 0.0
        return len(self.chunks), meshed, len(self.pending), tris, avg


