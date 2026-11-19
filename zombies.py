"""
Zombie horde defense layer for the terrain toy.  Everything is built from boxes and spheres.

main.py creates ONE ZombieDefense(app, terrain_h, ground, los) and calls update(dt) every frame.
  terrain_h(x, y)                    numpy height function (only used to pick the base spot)
  ground(xs, ys, zs, up, down)       -> (ok[N], z[N]) surface height near each point (includes digs/fills)
  los(a, b)                          -> True if the straight line a -> b is not blocked by terrain

Controls (set up in main.py):  WASD walk, Shift sprint, Space jump, click = shoot, drag = look,
X throw bomb cube (digs a crater that hurts zombies), F dig, G fill, Tab walk/fly, Enter restart.
"""
import math
import random
import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
                          GeomVertexWriter, NodePath, Point3, Shader, TextNode, Vec3)
from direct.gui.OnscreenText import OnscreenText
from sound import Sound

# ---------------------------------------------------------------- tuning
GRAV = 30.0
GROUPS = 3                 # zombies are simulated 1/GROUPS per frame (positions are extrapolated between)
MAX_ALIVE = 50             # waves spawn more only when fewer than this are alive
SLOPE_OK = 1.1             # rise/run a zombie can walk up at full speed
SLOPE_FRAC = 0.2           # fraction of normal speed when the slope is steeper (trenches slow them)
EYE = 1.7
BASE_R = 5.0
START_BOMBS, MAX_BOMBS = 4, 12
TURRET_COST, TURRET_COST_STEP, MAX_TURRETS = 100, 40, 8     # price of turret n = COST + STEP * (turrets owned)
TURRET_RANGE, TURRET_RATE, TURRET_DMG, TURRET_HP = 32.0, 0.45, 1.0, 40.0
BOMBER_RADIUS = 6.0                                          # explosion radius of an exploding zombie
BOMBER_TURRET_DMG, BOMBER_BASE_DMG, BOMBER_PLAYER_DMG = 26.0, 18.0, 30.0
SPIT_RANGE, SPIT_COOLDOWN, SPIT_SPEED, SPIT_G = 18.0, 2.4, 14.0, 18.0    # spitters lob acid from a distance
SPIT_TURRET_DMG, SPIT_BASE_DMG, SPIT_PLAYER_DMG, SPIT_SPLASH = 9.0, 5.0, 12.0, 3.0
WAVE_TIME_BASE, WAVE_TIME_PER_WAVE, WAVE_TIME_MAX = 45.0, 4.0, 110.0   # seconds before a wave is force-ended
BOSS_EVERY, EVAC_EVERY = 5, 10          # boss wave every 5 waves (brute, UFO, brute, ...); helicopter relocates the beacon every 10
UFO_HP, UFO_HP_PER_LEVEL = 280.0, 120.0
UFO_BEAM_TURRET, UFO_BEAM_BASE, UFO_BEAM_PLAYER = 14.0, 5.0, 28.0   # damage per SECOND while the beam is on
PLASMA_SPEED, PLASMA_R = 15.0, 4.5
PLASMA_TURRET_DMG, PLASMA_BASE_DMG, PLASMA_PLAYER_DMG = 10.0, 6.0, 18.0
SLAM_R, SLAM_COOLDOWN = 10.0, 6.0        # the giant brute's ground slam (jump to dodge it)
ALLY_COST, ALLY_DURATION, ALLY_RANGE = 250, 40.0, 60.0   # H calls the gunship for a while; free on boss waves
ALLY_MG_DMG, ALLY_MG_BOSS, ALLY_MISSILE_DMG = 0.7, 0.9, 8.0
BOSS_AIR_TURRET_DMG = 1.3                                    # turret damage per shot against the UFO
HANG = 18.0                                                  # how far below the evac helicopter the beacon hangs
BOMBER_CRATER = 3.5                                          # radius of the hole its blast digs
Z_KILL = -60.0
FOOT_LIFT = 0.10           # draw zombies this far above the sampled surface (the SDF crossing is interpolated)
TERM_VZ = 22.0             # terminal fall speed: keeps one update step shorter than the ground search window

# ---- digger zombies: tunnel under the ground (invisible, unshootable, a dirt mound shows the way) and burst
# out next to the nearest turret (or the beacon). Explosions flush them out early.
BURROW_SPEED, BURROW_DEPTH, SURFACE_R = 5.5, 2.8, 6.0
DIGGER_CRATER, DIGGER_BLAST = 2.6, 5.0
DIGGER_TURRET_DMG, DIGGER_BASE_DMG, DIGGER_PLAYER_DMG = 10.0, 8.0, 8.0

# ---- perks: offered (3 random) after every wave. id: (name, description, max stacks)
PERKS = {
    "rapid":   ("Rapid Fire",      "weapons fire 25% faster", 3),
    "heavy":   ("Heavy Rounds",    "+25% weapon damage", 4),
    "blast":   ("Big Bang",        "+25% bomb and launcher blast size", 3),
    "cache":   ("Bomb Cache",      "+4 bomb capacity and +4 bombs now", 3),
    "turret":  ("Turret Tuning",   "turrets fire 33% faster", 3),
    "cheap":   ("Bulk Order",      "turrets cost 20% less", 2),
    "plating": ("Beacon Plating",  "+60 max beacon HP (and repaired)", 4),
    "skin":    ("Thick Skin",      "you take 20% less damage", 3),
    "boots":   ("Light Boots",     "move 12% faster", 3),
    "medic":   ("Field Medic",     "regenerate 1.5 HP per second", 3),
    "loot":    ("Scavenger",       "+25% score from kills", 3),
}
WEAPONS = {
    "rifle": {
        "damage": 1.0,
        "cooldown": 0.18,
        "auto": True,            # hold the mouse button to keep firing
    },

    "shotgun": {
        "damage": 1.25,
        "cooldown": 0.75,
        "pellets": 8,
        "spread": 0.10,
    },

    "smg": {
        "damage": 0.45,
        "cooldown": 0.07,
        "auto": True,
    },

    "sniper": {
        "damage": 8.0,
        "cooldown": 1.20,
        "head_mult": 4.0,
    },

    "launcher": {
        "damage": 10.0,
        "cooldown": 1.40,
    },
}
WEAPON_COSTS = {
    "rifle": 0,
    "shotgun": 150,
    "smg": 200,
    "sniper": 350,
    "launcher": 500,
}
# ---- shop: number keys 1..9 while the shop menu is open (items 1-5 double as weapon-switch keys anywhere)
SHOP_RADIUS, SHOP_SIDE = 5.0, 6.0               # how close you must be / where the stall stands (beside the beacon)
BOMB_PACK, BOMB_PACK_COST = 3, 60
MEDKIT_COST = 40
REPAIR_COST, REPAIR_FRAC = 80, 0.5              # beacon repair restores this fraction of max HP
ARMOR_COST, ARMOR_BONUS, ARMOR_MAX = 150, 50.0, 4   # permanent beacon max-HP upgrades
SHOP_ITEMS = ["rifle", "shotgun", "smg", "sniper", "launcher", "bombs", "medkit", "repair", "armor"]
SHOP_NAMES = {"rifle": "Rifle", "shotgun": "Shotgun", "smg": "SMG", "sniper": "Sniper", "launcher": "Launcher",
              "bombs": f"Bombs x{BOMB_PACK}", "medkit": "Medkit (full HP)",
              "repair": f"Repair beacon (+{int(REPAIR_FRAC * 100)}%)", "armor": f"Beacon armor (+{int(ARMOR_BONUS)} max HP)"}



TYPES = {
    "walker": dict(hp=3,  speed=3.2, dmg=10, scale=1.0, score=10, gait=7.0,
                   skin=(0.35, 0.58, 0.30, 1), shirt=(0.25, 0.30, 0.48, 1), pants=(0.20, 0.20, 0.26, 1)),
    "runner": dict(hp=2,  speed=6.5, dmg=6,  scale=0.9, score=15, gait=13.0,
                   skin=(0.55, 0.62, 0.30, 1), shirt=(0.70, 0.18, 0.15, 1), pants=(0.25, 0.15, 0.10, 1)),
    "bomber": dict(hp=2,  speed=5.0, dmg=0,  scale=1.0, score=25, gait=11.0,
                   skin=(0.62, 0.48, 0.32, 1), shirt=(0.90, 0.42, 0.08, 1), pants=(0.18, 0.14, 0.12, 1)),
    "spitter": dict(hp=4, speed=3.0, dmg=0,  scale=1.0, score=30, gait=6.0,
                   skin=(0.45, 0.60, 0.25, 1), shirt=(0.28, 0.35, 0.20, 1), pants=(0.15, 0.18, 0.12, 1)),
    "titan":  dict(hp=240, speed=2.4, dmg=30, scale=3.4, score=400, gait=3.0,
                   skin=(0.55, 0.22, 0.18, 1), shirt=(0.28, 0.10, 0.14, 1), pants=(0.16, 0.10, 0.10, 1)),
    "brute":  dict(hp=14, speed=2.2, dmg=25, scale=1.7, score=50, gait=4.5,
                   skin=(0.22, 0.42, 0.22, 1), shirt=(0.30, 0.20, 0.30, 1), pants=(0.12, 0.12, 0.14, 1)),
    "digger": dict(hp=4,  speed=3.4, dmg=12, scale=1.0, score=35, gait=8.0,
                   skin=(0.52, 0.42, 0.30, 1), shirt=(0.40, 0.28, 0.14, 1), pants=(0.20, 0.16, 0.12, 1)),
}

# ---------------------------------------------------------------- shaders
VERT = """#version 300 es
precision highp float;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
out vec3 n;
out vec4 c;
void main() {
    n = mat3(p3d_ModelMatrix) * p3d_Normal;
    c = p3d_Color;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""
FRAG = """#version 300 es
precision highp float;
uniform float flash;
in vec3 n;
in vec4 c;
out vec4 color;
void main() {
    float l = max(dot(normalize(n), normalize(vec3(0.4, 0.3, 0.85))), 0.0) * 0.65 + 0.35;
    vec3 rgb = mix(c.rgb * l, vec3(1.0, 0.95, 0.8), flash);
    color = vec4(rgb, 1.0);
}
"""

# ---------------------------------------------------------------- simple-shape mesh builder
_E = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


class Mesh:
    """Collects boxes and spheres (with vertex colours) into ONE Geom, so a whole zombie is one draw call."""

    def __init__(self):
        self.v, self.n, self.c, self.t = [], [], [], []

    def box(self, center, size, color):
        cx, cy, cz = center
        h = (size[0] / 2, size[1] / 2, size[2] / 2)
        for i in range(3):
            for s in (1, -1):
                n = tuple(s * c for c in _E[i])
                u, v = (_E[(i + 1) % 3], _E[(i + 2) % 3]) if s == 1 else (_E[(i + 2) % 3], _E[(i + 1) % 3])
                base = len(self.v)
                for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    self.v.append((cx + h[0] * (n[0] + su * u[0] + sv * v[0]),
                                   cy + h[1] * (n[1] + su * u[1] + sv * v[1]),
                                   cz + h[2] * (n[2] + su * u[2] + sv * v[2])))
                    self.n.append(n)
                    self.c.append(color)
                self.t += [(base, base + 1, base + 2), (base, base + 2, base + 3)]

    def sphere(self, center, r, color, rings=6, segs=10, zs=1.0):
        cx, cy, cz = center
        base = len(self.v)
        for i in range(rings + 1):
            phi = math.pi * i / rings
            for j in range(segs + 1):
                th = 2 * math.pi * j / segs
                n = (math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), math.cos(phi))
                self.v.append((cx + r * n[0], cy + r * n[1], cz + r * n[2] * zs))
                self.n.append(n)
                self.c.append(color)
        w = segs + 1
        for i in range(rings):
            for j in range(segs):
                a = base + i * w + j
                b = a + w
                self.t += [(a, b, a + 1), (b, b + 1, a + 1)]

    def node(self, name):
        vdata = GeomVertexData(name, GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        cw = GeomVertexWriter(vdata, "color")
        for p, n, c in zip(self.v, self.n, self.c):
            vw.add_data3(*p)
            nw.add_data3(*n)
            cw.add_data4(*c)
        tris = GeomTriangles(Geom.UH_static)
        for a, b, c in self.t:
            tris.add_vertices(a, b, c)
        geom = Geom(vdata)
        geom.add_primitive(tris)
        gn = GeomNode(name)
        gn.add_geom(geom)
        return NodePath(gn)


def humanoid(skin, shirt, pants):
    m = Mesh()
    eye = (1.0, 0.1, 0.1, 1)
    m.box((-0.2, 0, 0.45), (0.3, 0.35, 0.9), pants)          # legs
    m.box((0.2, 0, 0.45), (0.3, 0.35, 0.9), pants)
    m.box((0, 0, 1.25), (0.8, 0.4, 0.8), shirt)              # torso
    m.box((0, 0.05, 1.85), (0.5, 0.5, 0.5), skin)            # head
    m.box((-0.12, 0.31, 1.9), (0.1, 0.04, 0.1), eye)         # eyes
    m.box((0.12, 0.31, 1.9), (0.1, 0.04, 0.1), eye)
    m.box((-0.55, 0.45, 1.35), (0.22, 0.8, 0.22), skin)      # arms stretched forward
    m.box((0.55, 0.45, 1.35), (0.22, 0.8, 0.22), skin)
    return m


class Zombie:
    def __init__(self, kind, node, x, y, z, wave):
        d = TYPES[kind]
        self.kind, self.node = kind, node
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.hp = float(d["hp"]) * (1.0 + 0.08 * max(0, wave - 1))      # tougher every wave
        self.speed = d["speed"] * (1.0 + 0.05 * wave)
        self.dmg, self.scale, self.score, self.gait = d["dmg"], d["scale"], d["score"], d["gait"]
        self.t_last = 0.0
        self.gz = None            # ground height under the last position (for slope handling)
        self.vgz = 0.0            # vertical speed used to extrapolate between updates
        self.burrowed, self.mound, self.dust = False, None, 0.0     # digger zombies tunnel underground
        self.flash_t = 0.0
        self.atk_t = random.random()
        self.heading = 0.0
        self.phase = random.random() * 6.28
        self.maxhp = self.hp
        self.slam_t = 3.0 + random.random() * 2.0
        self.slamming = False
        self.dead = False


class Boss:
    """The UFO.  Looks like a Zombie to turrets and the gunship (x, y, z, scale, dead)."""

    def __init__(self, node, beam, x, y, z, hp, level):
        self.node, self.beam = node, beam
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = 0.0
        self.hp = self.maxhp = float(hp)
        self.level = level
        self.scale = 0.0
        self.dead = False
        self.crash_vz = 0.0
        self.heading = 0.0
        self.angle = 0.0
        self.radius = 34.0
        self.radius_t = 5.0
        self.alt_t = 0.0
        self.ground = 0.0
        self.pick_t = 0.0
        self.target = ("base", None)
        self.mode = "idle"                 # idle -> lock (thin warning beam) -> beam (damaging)
        self.gap_t = 3.0
        self.lock_t = self.beam_t = 0.0
        self.locked = ("base", None)
        self.beam_end = Vec3(0, 0, 0)
        self.plasma_cd = 5.0
        self.flash_t = 0.0


class Ally:
    """Your attack helicopter: orbits the beacon and shoots zombies and bosses."""

    def __init__(self, node, rotor, x, y, z, time_left):
        self.node, self.rotor = node, rotor
        self.x, self.y, self.z = x, y, z
        self.time_left = time_left        # None = stays until the boss wave is over
        self.leaving = False
        self.t_leave = 0.0
        self.spin = self.angle = self.heading = 0.0
        self.radius, self.radius_t = 24.0, 6.0
        self.alt_t, self.ground = 0.0, 0.0
        self.target = None
        self.pick_t = 0.0
        self.fire_cd = 0.0
        self.missile_cd = 3.0


def build_ufo():
    m = Mesh()
    metal, dark = (0.62, 0.65, 0.72, 1), (0.22, 0.24, 0.30, 1)
    glow, glass = (0.35, 1.0, 0.65, 1), (0.45, 0.85, 1.0, 1)
    m.sphere((0, 0, 0), 5.5, metal, rings=8, segs=16, zs=0.3)        # flattened saucer hull
    m.sphere((0, 0, 0.6), 2.4, glass, rings=6, segs=12, zs=0.9)      # dome
    m.box((0, 0, -1.2), (1.8, 1.8, 0.6), dark)                       # emitter
    for i in range(8):                                               # rim lights
        a = i * math.pi / 4
        m.box((4.8 * math.cos(a), 4.8 * math.sin(a), -0.1), (0.7, 0.7, 0.35), glow)
    return m


def build_heli(attack, ally=False):
    hull = (0.20, 0.38, 0.62, 1) if ally else ((0.27, 0.31, 0.25, 1) if attack else (0.80, 0.82, 0.78, 1))
    dark = (0.10, 0.14, 0.22, 1) if ally else ((0.14, 0.16, 0.14, 1) if attack else (0.30, 0.34, 0.30, 1))
    glass = (0.45, 0.75, 0.95, 1)
    red = (0.95, 0.80, 0.15, 1) if ally else (0.85, 0.15, 0.10, 1)
    m = Mesh()
    m.box((0, 0, 0), (1.7, 4.4, 1.7), hull)                  # fuselage
    m.box((0, 2.5, -0.25), (1.3, 1.2, 1.1), dark if attack else hull)
    m.box((0, 1.6, 0.55), (1.3, 1.6, 0.9), glass)            # cockpit
    m.box((0, -3.8, 0.35), (0.5, 3.8, 0.5), hull)            # tail boom
    m.box((0, -5.5, 1.0), (0.14, 0.9, 1.5), red if not attack else dark)   # tail fin
    m.box((0, -5.2, 0.45), (1.8, 0.6, 0.1), dark)
    m.box((0, 0, 1.05), (0.3, 0.3, 0.5), dark)               # rotor mast
    for sx in (-1, 1):
        m.box((sx * 1.0, 0.2, -1.25), (0.15, 3.8, 0.15), dark)        # skids
        m.box((sx * 0.8, 0.9, -0.8), (0.1, 0.1, 0.8), dark)
        m.box((sx * 0.8, -0.9, -0.8), (0.1, 0.1, 0.8), dark)
        if attack:
            m.box((sx * 1.9, 0.3, -0.2), (2.0, 0.9, 0.15), dark)      # stub wings
            m.box((sx * 2.7, 0.9, -0.45), (0.4, 1.4, 0.4), red)       # rocket pods
        else:
            m.box((sx * 0.86, 0, 0.1), (0.05, 1.2, 0.5), red)         # red cross panels
    return m

def build_rifle():
    m = Mesh()

    dark = (0.15,0.15,0.15,1)
    wood = (0.45,0.25,0.10,1)

    m.box((0,0.9,0), (0.15,1.8,0.15), dark)
    m.box((0,0,0), (0.35,1.0,0.25), dark)
    m.box((0,-0.25,-0.25), (0.18,0.5,0.25), wood)

    return m.node("rifle")
def build_shotgun():
    m = Mesh()

    dark=(0.1,0.1,0.1,1)
    wood=(0.55,0.35,0.10,1)

    m.box((-0.08,1.1,0),(0.12,2.2,0.12),dark)
    m.box(( 0.08,1.1,0),(0.12,2.2,0.12),dark)

    m.box((0,0,0),(0.4,1.0,0.3),dark)
    m.box((0,-0.35,-0.28),(0.25,0.7,0.5),wood)

    return m.node("shotgun")
def build_smg():
    m = Mesh()

    dark=(0.18,0.18,0.20,1)

    m.box((0,0.55,0),(0.18,1.2,0.18),dark)
    m.box((0,-0.15,0),(0.4,0.75,0.3),dark)

    m.box((0,-0.45,-0.25),(0.15,0.30,0.45),dark)

    m.box((0,-0.15,-0.40),(0.12,0.65,0.20),(0.3,0.3,0.35,1))

    return m.node("smg")
def build_sniper():
    m = Mesh()

    dark=(0.10,0.10,0.12,1)

    m.box((0,1.5,0),(0.12,3.0,0.12),dark)

    m.box((0,0.1,0),(0.38,1.0,0.28),dark)

    m.box((0,0.45,0.24),(0.18,0.8,0.18),(0.35,0.8,1.0,1))

    return m.node("sniper")
def build_launcher():
    m = Mesh()

    body=(0.20,0.55,0.20,1)

    m.box((0,0.75,0),(0.42,2.0,0.42),body)

    m.box((0,-0.2,0),(0.45,0.8,0.35),(0.15,0.15,0.15,1))

    return m.node("launcher")
class Turret:
    def __init__(self, base, head, x, y, z):
        self.base, self.head = base, head
        self.x, self.y, self.z = x, y, z
        self.hp = TURRET_HP
        self.cd = self.scan = self.flash_t = 0.0
        self.target = None
        self.dead = False


class ZombieDefense:
    def __init__(self, app, terrain_h, ground, los):
        self.app, self.terrain_h, self.ground, self.los = app, terrain_h, ground, los
        self.root = app.render.attach_new_node("zombie_game")
        self.root.set_shader(Shader.make(Shader.SL_GLSL, VERT, FRAG))
        self.root.set_shader_input("flash", 0.0)
        self.snd = Sound(app)
        self.weapon_np = None
        # shape templates (copy_to shares the geometry, so copies are cheap)
        self.tmpl = {}

        self.tmpl["rifle"] = build_rifle()
        self.tmpl["shotgun"] = build_shotgun()
        self.tmpl["smg"] = build_smg()
        self.tmpl["sniper"] = build_sniper()
        self.tmpl["launcher"] = build_launcher()

        for kind, d in TYPES.items():
            hm = humanoid(d["skin"], d["shirt"], d["pants"])
            if kind == "bomber":                                  # dynamite sticks + a glowing fuse
                for sx in (-0.26, 0.0, 0.26):
                    hm.box((sx, 0.25, 1.2), (0.17, 0.14, 0.62), (0.85, 0.12, 0.10, 1))
                hm.box((0, 0.25, 1.1), (0.8, 0.16, 0.10), (0.15, 0.12, 0.10, 1))
                hm.box((0, 0.1, 2.25), (0.12, 0.12, 0.3), (1.0, 0.85, 0.2, 1))
            if kind == "spitter":                                 # swollen acid sac + glowing throat and mouth
                hm.box((0, 0.3, 1.1), (0.95, 0.5, 0.75), (0.70, 0.90, 0.15, 1))
                hm.box((0, 0.25, 1.68), (0.35, 0.3, 0.3), (0.70, 0.90, 0.15, 1))
                hm.box((0, 0.32, 1.8), (0.25, 0.05, 0.12), (0.85, 1.0, 0.25, 1))
            if kind == "digger":                                  # hard hat with a lamp + big steel claws
                hm.box((0, 0, 2.12), (0.52, 0.52, 0.2), (0.95, 0.75, 0.10, 1))
                hm.box((0, 0.27, 2.12), (0.14, 0.10, 0.10), (1.0, 0.95, 0.6, 1))
                for sx in (-0.55, 0.55):
                    hm.box((sx, 0.45, 0.95), (0.2, 0.7, 0.2), (0.75, 0.75, 0.78, 1))
            tn = hm.node(kind)
            tn.set_scale(d["scale"])
            self.tmpl[kind] = tn
        m = Mesh(); m.box((0, 0, 0), (0.3, 0.3, 0.3), (0.35, 0.58, 0.30, 1))
        self.tmpl["debris"] = m.node("debris")
        m = Mesh(); m.sphere((0, 0, 0), 0.9, (0.42, 0.31, 0.18, 1), zs=0.45)
        self.tmpl["mound"] = m.node("mound")                  # the dirt hump above a tunnelling digger
        m = Mesh(); m.box((0, 0, 0), (0.28, 0.28, 0.28), (0.40, 0.30, 0.18, 1))
        self.tmpl["dirt"] = m.node("dirt")
        m = Mesh(); m.box((0, 0.5, 0), (0.05, 1.0, 0.05), (1.0, 0.9, 0.4, 1))
        self.tmpl["tracer"] = m.node("tracer")

        m = Mesh()
        m.box((0, 0, 0.4), (1.8, 1.8, 0.8), (0.35, 0.35, 0.40, 1))
        m.box((0, 0, 1.1), (0.7, 0.7, 0.8), (0.50, 0.50, 0.55, 1))
        self.tmpl["turret_base"] = m.node("turret_base")
        m = Mesh()
        m.box((0, 0, 0), (1.1, 1.1, 0.7), (0.20, 0.45, 0.85, 1))
        m.box((0, 0.9, 0), (0.25, 1.4, 0.25), (0.15, 0.15, 0.18, 1))
        m.box((0, 0, 0.5), (0.5, 0.5, 0.3), (1.0, 0.80, 0.20, 1))
        self.tmpl["turret_head"] = m.node("turret_head")

        m = Mesh(); m.sphere((0, 0, 0), 1.0, (1.0, 0.55, 0.10, 1))
        self.tmpl["fireball"] = m.node("fireball")
        m = Mesh(); m.sphere((0, 0, 0), 0.35, (0.70, 0.95, 0.15, 1), rings=4, segs=6)
        self.tmpl["acid"] = m.node("acid")
        m = Mesh(); m.sphere((0, 0, 0), 1.0, (0.55, 0.85, 0.10, 1))
        self.tmpl["acid_burst"] = m.node("acid_burst")
        self.tmpl["heli_attack"] = build_heli(True, ally=True).node("heli_attack")
        self.tmpl["heli_transport"] = build_heli(False).node("heli_transport")
        m = Mesh(); dk = (0.12, 0.12, 0.12, 1)
        m.box((0, 0, 0), (8.0, 0.35, 0.06), dk); m.box((0, 0, 0), (0.35, 8.0, 0.06), dk)
        m.box((0, 0, 0.05), (0.5, 0.5, 0.15), (0.3, 0.3, 0.3, 1))
        self.tmpl["heli_rotor"] = m.node("heli_rotor")
        m = Mesh(); m.box((0, 0, -0.5), (0.12, 0.12, 1.0), (0.1, 0.1, 0.1, 1))
        self.tmpl["cable"] = m.node("cable")
        m = Mesh(); m.box((0, 0.4, 0), (0.28, 1.2, 0.28), (0.85, 0.15, 0.10, 1))
        m.box((0, -0.3, 0), (0.18, 0.3, 0.18), (1.0, 0.8, 0.2, 1))
        self.tmpl["missile"] = m.node("missile")
        m = Mesh(); gold = (1.0, 0.92, 0.3, 1)
        m.box((0, 0, 0.15), (9, 9, 0.3), (0.9, 0.85, 0.2, 1)); m.box((0, 0, 16), (0.7, 0.7, 32), gold)
        for sx in (-4, 4):
            for sy in (-4, 4):
                m.box((sx, sy, 1.2), (0.4, 0.4, 2.4), gold)
        self.tmpl["marker"] = m.node("marker")
        self.tmpl["ufo"] = build_ufo().node("ufo")
        m = Mesh(); m.box((0, 0.5, 0), (1.0, 1.0, 1.0), (0.80, 0.35, 1.0, 1))
        self.tmpl["beam"] = m.node("beam")
        m = Mesh(); m.sphere((0, 0, 0), 0.7, (0.45, 1.0, 0.95, 1), rings=5, segs=8)
        self.tmpl["plasma"] = m.node("plasma")
        self.fireballs = []
        self.spits = []
        self.boss, self.missiles, self.evac = None, [], None
        self.ally, self.titan = None, None
        self.lz, self.lz_node, self.riding = None, None, False
        self.zombies, self.debris, self.tracers, self.queue = [], [], [], []
        self.turrets = []
        self.t = 0.0
        self.frame = 0
        self.next_shot = 0.0
        self.walking = True
        self.px = self.py = self.pz = 0.0
        self.pvz = 0.0
        self.on_ground = False
        self.hurt_t = 0.0
        self.banner_t = 0.0
        self.hud_t = 0.0

        bx, by, bh = self._pick_base()
        self.bx, self.by, self.bz = bx, by, bh - 0.3
        self._build_base()

        m = Mesh()

        m.box(
            (0,0,1),
            (2,2,2),
            (0.2,0.8,1.0,1)
        )

        m.box(
            (0,0,2.4),
            (1.4,0.1,0.8),
            (1.0,1.0,0.3,1)
        )

        self.shop = m.node("shop")
        self.shop.reparent_to(self.root)
        self.shop_open, self.armor_level = False, 0
        self.shop_pos = (self.bx + SHOP_SIDE, self.by, self.bz)
        self.perks, self.perk_choices = {}, []
        self.perk_text = OnscreenText(text="", pos=(-1.3, 0.28), scale=0.05, fg=(1, 0.9, 0.3, 1), shadow=(0, 0, 0, 1),
                                      align=TextNode.ALeft, mayChange=True)
        self.shop_text = OnscreenText(text="", pos=(-1.3, 0.28), scale=0.045, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1),
                                      align=TextNode.ALeft, mayChange=True)

        self.status = OnscreenText(text="", pos=(-1.3, -0.93), scale=0.055, fg=(1, 1, 1, 1),
                                   shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True)
        self.banner = OnscreenText(text="", pos=(0, 0.55), scale=0.13, fg=(1, 0.9, 0.3, 1),
                                   shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True)
        self.info = OnscreenText(text="", pos=(0, 0.42), scale=0.055, fg=(1, 1, 1, 1),
                                 shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True)
        self.cross = OnscreenText(text="+", pos=(0, -0.02), scale=0.08, fg=(1, 1, 1, 0.9),
                                  shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True)
        self.restart()
    def _place_shop(self):
        """Stand the stall on the ground beside the beacon (call after the beacon moves)."""
        sx, sy = self.bx + SHOP_SIDE, self.by
        ok, gz = self.ground(np.array([sx], np.float32), np.array([sy], np.float32),
                             np.array([self.bz + 4.0], np.float32), 30.0, 40.0)
        sz = float(gz[0]) if ok[0] else self.bz
        self.shop_pos = (sx, sy, sz)
        self.shop.set_pos(sx, sy, sz)

    def shop_near(self):
        return self.walking and not self.riding and \
            math.hypot(self.px - self.shop_pos[0], self.py - self.shop_pos[1]) < SHOP_RADIUS

    # ------------------------------------------------------------ perks
    def _recalc_perks(self):
        p = self.perks.get
        self.perk_rate = 0.8 ** p("rapid", 0)              # weapon cooldown multiplier
        self.perk_dmg = 1.0 + 0.25 * p("heavy", 0)
        self.perk_blast = 1.0 + 0.25 * p("blast", 0)
        self.perk_turret = 0.75 ** p("turret", 0)          # turret cooldown multiplier
        self.perk_cheap = 0.8 ** p("cheap", 0)             # turret cost multiplier
        self.perk_taken = 0.8 ** p("skin", 0)              # damage the player takes
        self.perk_speed = 1.0 + 0.12 * p("boots", 0)
        self.perk_regen = 1.5 * p("medic", 0)
        self.perk_loot = 1.0 + 0.25 * p("loot", 0)
        self.max_bombs = MAX_BOMBS + 4 * p("cache", 0)

    def _offer_perks(self):
        avail = [k for k, v in PERKS.items() if self.perks.get(k, 0) < v[2]]
        random.shuffle(avail)
        self.perk_choices = avail[:3]
        if self.perk_choices:
            self._close_shop()
            self._perk_refresh()

    def _perk_refresh(self):
        lines = ["CHOOSE A PERK  (press 1-3)", ""]
        for i, k in enumerate(self.perk_choices):
            name, desc, mx = PERKS[k]
            lvl = f"   (level {self.perks.get(k, 0) + 1}/{mx})" if mx > 1 else ""
            lines.append(f"[{i + 1}] {name}: {desc}{lvl}")
        self.perk_text.setText("\n".join(lines))

    def pick_perk(self, i):
        k = self.perk_choices[i]
        self.perks[k] = self.perks.get(k, 0) + 1
        if k == "plating":
            self.base_max += 60.0
            self.base_hp += 60.0
        self._recalc_perks()
        if k == "cache":
            self.bombs = min(self.max_bombs, self.bombs + 4)
        self.perk_choices = []
        self.perk_text.setText("")
        if self.state == "build":
            self.timer = max(self.timer, 10.0)             # the build timer was paused while choosing
        self._say(f"PERK: {PERKS[k][0].upper()}", 2.0)

    # ------------------------------------------------------------ digger zombies
    def _surface(self, z, boom=True):
        """A burrowed digger comes up: pops out of the ground, digs a small crater and hurts what is nearby."""
        if not z.burrowed:
            return
        z.burrowed = False
        if z.mound is not None:
            z.mound.remove_node()
            z.mound = None
        g = z.gz if z.gz is not None else z.z + BURROW_DEPTH
        p = Vec3(z.x, z.y, g)
        z.z, z.vz, z.gz, z.vgz = g + 0.1, 8.5, None, 0.0
        if not boom:
            return
        self.sfx("explosion", p, 0.6)
        try:
            self.app.chunks.edit("dig", p, DIGGER_CRATER)
        except Exception as e:
            print("digger crater failed:", e)
        for t in list(self.turrets):
            if (Vec3(t.x, t.y, t.z + 0.8) - p).length() < DIGGER_BLAST:
                self.hit_turret(t, DIGGER_TURRET_DMG)
        if math.hypot(self.bx - p.x, self.by - p.y) - 3.5 < DIGGER_BLAST:
            self.base_hp -= DIGGER_BASE_DMG
            self.orb_flash = 0.2
        if self.walking and (Vec3(self.px, self.py, self.pz + 1.0) - p).length() < DIGGER_BLAST:
            self._hurt_player(DIGGER_PLAYER_DMG)
        for _ in range(8):
            node = self.tmpl["dirt"].copy_to(self.root)
            node.set_pos(p)
            self.debris.append([node, Vec3(random.uniform(-6, 6), random.uniform(-6, 6), random.uniform(5, 11)), 0.8])

    def is_auto(self):
        return WEAPONS[self.weapon].get("auto", False)

    def toggle_shop(self):
        """E: open/close the shop menu (only between waves, and only next to the stall)."""
        if self.shop_open:
            return self._close_shop()
        if self.perk_choices:
            return self._say("PICK A PERK FIRST (1-3)", 1.5)
        if self.state not in ("build", "wave"):
            return self._say("SHOP CLOSED", 1.5)
        if not self.shop_near():
            return self._say("MOVE TO THE SHOP", 1.5)
        self.shop_open = True
        self._shop_refresh()

    def _close_shop(self):
        self.shop_open = False
        self.shop_text.setText("")

    def _shop_tick(self):
        if self.shop_open and (self.state not in ("build", "wave") or not self.shop_near()):
            self._close_shop()

    def _shop_cost(self, item):
        return {"bombs": BOMB_PACK_COST, "medkit": MEDKIT_COST, "repair": REPAIR_COST,
                "armor": ARMOR_COST}.get(item, WEAPON_COSTS.get(item, 0))

    def _shop_refresh(self):
        lines = [f"SHOP    Score {self.score}    (number = buy, E = close)", ""]
        for i, item in enumerate(SHOP_ITEMS):
            if item in WEAPONS:
                tag = "EQUIPPED" if item == self.weapon else ("owned - equip" if item in self.owned_weapons
                                                               else f"{WEAPON_COSTS[item]}")
            elif item == "armor":
                tag = "MAX" if self.armor_level >= ARMOR_MAX else f"{ARMOR_COST}  ({self.armor_level}/{ARMOR_MAX})"
            elif item == "bombs":
                tag = f"{BOMB_PACK_COST}  (have {self.bombs}/{self.max_bombs})"
            else:
                tag = f"{self._shop_cost(item)}"
            lines.append(f"[{i + 1}] {SHOP_NAMES[item]}  -  {tag}")
        lines += ["", f"Turret: T ({self.turret_cost()})   Gunship: H ({ALLY_COST})"]
        self.shop_text.setText("\n".join(lines))

    def number_key(self, n):
        """1-9. Shop open: buy item n. Shop closed: 1-5 switch to a weapon you own (works anywhere)."""
        if self.perk_choices:                              # a perk offer takes the number keys first
            if 1 <= n <= len(self.perk_choices):
                self.pick_perk(n - 1)
            return
        if self.state == "over" or n < 1 or n > len(SHOP_ITEMS):
            return
        item = SHOP_ITEMS[n - 1]
        if self.shop_open:
            return self.buy_item(item)
        if item in WEAPONS:
            if item in self.owned_weapons:
                self.equip_weapon(item)
            else:
                self._say(f"{SHOP_NAMES[item].upper()} NOT OWNED - BUY AT SHOP (E)", 1.5)

    def buy_item(self, item):
        if self.state not in ("build", "wave") or not self.shop_near():
            return self._close_shop()
        if item in WEAPONS and item in self.owned_weapons:
            self.equip_weapon(item)
            return self._shop_refresh()
        cost = self._shop_cost(item)
        if item == "bombs" and self.bombs >= self.max_bombs:
            return self._say("ALREADY AT MAX BOMBS", 1.5)
        if item == "medkit" and self.hp >= 100.0:
            return self._say("ALREADY AT FULL HP", 1.5)
        if item == "repair" and self.base_hp >= self.base_max:
            return self._say("ALREADY BEACON FULL HP", 1.5)
        if item == "armor" and self.armor_level >= ARMOR_MAX:
            return self._say("ALREADY MAX ARMOR", 1.5)
        if self.score < cost:
            return self._say(f"NEED {cost} SCORE", 1.5)
        self.score -= cost
        if item in WEAPONS:
            self.owned_weapons.add(item)
            self.equip_weapon(item)
        elif item == "bombs":
            self.bombs = min(self.max_bombs, self.bombs + BOMB_PACK)
        elif item == "medkit":
            self.hp = 100.0
        elif item == "repair":
            self.base_hp = min(self.base_max, self.base_hp + self.base_max * REPAIR_FRAC)
        elif item == "armor":
            self.armor_level += 1
            self.base_max += ARMOR_BONUS
            self.base_hp += ARMOR_BONUS
        self._say(f"BOUGHT {SHOP_NAMES[item].upper()}", 1.5)
        self._shop_refresh()
    def equip_weapon(self,name):

        self.weapon=name

        if self.weapon_np:
            self.weapon_np.remove_node()

        self.weapon_np=self.tmpl[name].copy_to(self.app.camera)

        self.weapon_np.set_pos(0.35,0.8,-0.35)

        self.weapon_np.set_hpr(-8,-2,0)

        self.weapon_np.set_scale(0.35)
    # ------------------------------------------------------------ setup
    def _pick_base(self):
        g = np.arange(-80, 81, 8, dtype=np.float32)
        X, Y = np.meshgrid(g, g, indexing="ij")
        X, Y = X.ravel(), Y.ravel()
        h = self.terrain_h(X, Y)
        e = np.float32(6.0)
        slope = (np.abs(self.terrain_h(X + e, Y) - h) + np.abs(self.terrain_h(X - e, Y) - h)
                 + np.abs(self.terrain_h(X, Y + e) - h) + np.abs(self.terrain_h(X, Y - e) - h)) / e
        score = slope * 10.0 + np.hypot(X, Y) * 0.05 + np.where((h < 2) | (h > 30), 100.0, 0.0)
        i = int(np.argmin(score))
        return float(X[i]), float(Y[i]), float(h[i])

    def _build_base(self):
        stone, dark = (0.55, 0.55, 0.58, 1), (0.32, 0.32, 0.38, 1)
        m = Mesh()
        m.box((0, 0, 0.75), (7, 7, 1.5), stone)
        m.box((0, 0, 4.5), (3, 3, 6), dark)
        m.box((0, 0, 7.9), (4.2, 4.2, 0.7), stone)
        for sx in (-1, 1):
            for sy in (-1, 1):
                m.box((sx * 3.1, sy * 3.1, 2.2), (0.7, 0.7, 2.8), stone)
        self.base = m.node("base")
        self.base.reparent_to(self.root)
        self.base.set_pos(self.bx, self.by, self.bz)
        o = Mesh(); o.sphere((0, 0, 0), 1.3, (1.0, 0.85, 0.2, 1))
        self.orb = o.node("orb")
        self.orb.reparent_to(self.root)
        self.orb.set_pos(self.bx, self.by, self.bz + 9.4)
        self.orb_flash = 0.0
    def restart(self):
        self.snd.stop_all()
        self.weapon = "rifle"
        self.equip_weapon("rifle")
        self.owned_weapons = {"rifle"}
        for z in self.zombies:
            z.node.remove_node()
        for d in self.debris:
            d[0].remove_node()
        for tr in self.tracers:
            tr[0].remove_node()
        for t in self.turrets:
            t.base.remove_node()
            t.head.remove_node()
        self.turrets = []
        for f in self.fireballs:
            f[0].remove_node()
        self.fireballs = []
        for sp in self.spits:
            sp[0].remove_node()
        self.spits = []
        self._boss_remove()
        if self.ally is not None:
            self.ally.node.remove_node()
        self.ally, self.titan = None, None
        for m in self.missiles:
            m[0].remove_node()
        self.missiles = []
        if self.evac is not None:
            self.evac["node"].remove_node()
            self.evac = None
        if self.lz_node is not None:
            self.lz_node.remove_node()
        self.lz, self.lz_node, self.riding = None, None, False
        self.base.set_pos(self.bx, self.by, self.bz)
        self.orb.set_pos(self.bx, self.by, self.bz + 9.4)
        self.wave_t = 0.0
        self.zombies, self.debris, self.tracers, self.queue = [], [], [], []
        self.wave, self.kills, self.score = 0, 0, 0
        self.base_hp, self.base_max = 200.0, 200.0
        self.armor_level = 0
        self._close_shop()
        self.perks, self.perk_choices = {}, []
        self.perk_text.setText("")
        self._recalc_perks()
        self._place_shop()
        self.hp = 100.0
        self.bombs = START_BOMBS
        self.state, self.timer = "build", 15.0
        self.spawn_cd = 0.0
        self.walking = True
        self._respawn_player()
        self._say("DEFEND THE BEACON", 3.0)

    def _respawn_player(self):
        self.px, self.py = self.bx, self.by - 11.0
        ok, gz = self.ground(np.array([self.px], np.float32), np.array([self.py], np.float32),
                             np.array([self.bz + 3.0], np.float32), 30.0, 40.0)
        self.pz = float(gz[0]) if ok[0] else self.bz
        self.pvz = 0.0
        self.hp = 100.0
        self.app.heading, self.app.pitch = 0.0, -6.0
        self.app.camera.set_pos(self.px, self.py, self.pz + EYE)

    # ------------------------------------------------------------ helpers
    SAY_SFX = (("NEED", "deny"), ("TURRET LIMIT", "deny"), ("NO GROUND", "deny"), ("TOO ", "deny"),
               ("TURRET READY", "beep"), ("TURRET LOST", "explosion"), ("WAVE CLEARED", "wave_clear"),
               ("WAVE TIMED", "wave_clear"), ("WAVE ", "wave_start"), ("DEFEND", "wave_start"),
               ("GUNSHIP", "gunship"), ("EVAC", "gunship"), ("LIFTING", "gunship"), ("YOU DIED", "hurt"),
               ("BEACON DESTROYED", "explosion"), ("UFO DESTROYED", "explosion"), ("GIANT BRUTE DOWN", "wave_clear"),
               ("BEACON RELOCATED", "beep"), ("BOUGHT", "beep"), ("ALREADY", "deny"), ("PERK", "beep"))

    def sfx(self, name, pos=None, vol=1.0):
        self.snd.play(name, pos, vol)

    def toggle_mute(self):
        self._say("SOUND OFF" if self.snd.toggle_mute() else "SOUND ON", 1.0)

    def _say(self, text, secs):
        self.banner.setText(text)
        self.banner_t = secs
        for prefix, name in self.SAY_SFX:
            if text.startswith(prefix):
                self.sfx(name)
                break

    def toggle_walk(self):
        self.walking = not self.walking
        if self.walking:
            c = self.app.camera.get_pos(self.app.render)
            ok, gz = self.ground(np.array([c.x], np.float32), np.array([c.y], np.float32),
                                 np.array([c.z], np.float32), 3.0, 250.0)
            if ok[0]:
                self.px, self.py, self.pz, self.pvz = c.x, c.y, float(gz[0]), 0.0
            else:
                self._respawn_player()

    def use_bomb(self):
        if self.state == "over" or self.bombs <= 0:
            return False
        self.bombs -= 1
        return True

    # ------------------------------------------------------------ main update
    def update(self, dt):
        dt = min(dt, 0.1)
        self.t += dt
        self.frame += 1
        if self.state != "over":
            self._waves(dt)
            self._player(dt)
            self._shop_tick()
            self._zombies(dt)
            self._turrets(dt)
            self._spits(dt)
            self._boss(dt)
            self._ally(dt)
            self._missiles(dt)
        self._effects(dt)
        self._hud(dt)

    def _waves(self, dt):
        if self.state == "evac":
            self._evac(dt)
        elif self.state == "build":
            if not self.perk_choices:
                self.timer -= dt
            self.info.setText(f"Wave {self.wave + 1} in {max(0, int(self.timer) + 1)}s  -  dig trenches (F), "
                              f"fill ramps (G), bombs (X), turret (T, {self.turret_cost()}), gunship (H, {ALLY_COST}), skip (N)"
                              + ("  -  SHOP: press E" if self.shop_near() else ""))
            if self.timer <= 0:
                self.wave += 1
                n = 5 + 4 * self.wave
                self.queue = []
                for _ in range(n):
                    r = random.random()
                    w = self.wave
                    p_brute = 0.07
                    p_bomber = p_brute + (min(0.18, 0.04 * (w - 2)) if w >= 3 else 0.0)
                    p_spitter = p_bomber + (min(0.15, 0.03 * (w - 3)) if w >= 4 else 0.0)
                    if w >= 3 and r < p_brute:
                        self.queue.append("brute")
                    elif w >= 3 and r < p_bomber:
                        self.queue.append("bomber")
                    elif w >= 4 and r < p_spitter:
                        self.queue.append("spitter")
                    elif w >= 2 and r < p_spitter + 0.25:
                        self.queue.append("runner")
                    else:
                        self.queue.append("walker")
                p_dig = min(0.22, 0.05 * (self.wave - 3)) if self.wave >= 4 else 0.0
                self.queue = [("digger" if k == "walker" and random.random() < p_dig else k) for k in self.queue]
                if self.wave % EVAC_EVERY == 0:
                    self._pick_lz()                             # mark where the evac helicopter will take the beacon
                if self.wave % BOSS_EVERY == 0:                 # boss wave: fewer zombies, plus a boss and your gunship
                    self.queue = self.queue[: max(6, int(len(self.queue) * 0.6))]
                    if (self.wave // BOSS_EVERY) % 2 == 1:
                        self.queue.append("titan")              # spawned first (the queue is popped from the end)
                    else:
                        self._spawn_boss()
                    self.call_ally(free=True)
                self.state = "wave"
                self.wave_t = 0.0
                self.info.setText("")
                extra = [k.upper() + "S" for k in ("bomber", "spitter", "digger") if k in self.queue]
                if self.wave % BOSS_EVERY == 0:
                    extra.append("GIANT BRUTE" if "titan" in self.queue else "UFO BOSS")
                self._say(f"WAVE {self.wave}" + ("  -  " + " & ".join(extra) + "!" if extra else ""), 2.5)
        elif self.state == "wave":
            self.spawn_cd -= dt
            if self.queue and self.spawn_cd <= 0:
                for _ in range(2 if self.wave >= 6 else 1):         # later waves pour in two at a time
                    if self.queue and len(self.zombies) < MAX_ALIVE:
                        self._spawn(self.queue.pop())
                self.spawn_cd = max(0.1, 0.30 - 0.012 * self.wave)
            self.wave_t += dt
            if not self.queue and not self.zombies and self.boss is None:
                self._finish_wave(False)
            elif self.wave_t >= self.wave_limit():              # stuck zombies: end the wave anyway
                for z in list(self.zombies):
                    self.kill(z, scored=False, explode=False)
                self.queue = []
                self._boss_remove()                             # the helicopter retreats
                self._finish_wave(True)

    def wave_limit(self):
        lim = min(WAVE_TIME_MAX, WAVE_TIME_BASE + WAVE_TIME_PER_WAVE * self.wave)
        return lim * 2.0 if self.wave % BOSS_EVERY == 0 else lim            # boss waves get twice as long

    def _finish_wave(self, timed_out):
        self.score += int(50 * self.wave * (0.5 if timed_out else 1.0))
        self.bombs = min(self.max_bombs, self.bombs + 2)
        self.base_hp = min(self.base_max, self.base_hp + 15)
        self.state, self.timer = "build", 8.0
        self._say("WAVE TIMED OUT" if timed_out else "WAVE CLEARED", 2.5)
        if self.ally is not None and self.ally.time_left is None:   # free boss-wave support leaves with the wave
            self.ally.time_left = 0.0
        if self.wave % EVAC_EVERY == 0:                         # every 10th wave: helicopter relocates the base
            self._start_evac()
        self._offer_perks()

    def _spawn(self, kind):
        for _ in range(8):
            a = random.random() * 6.2832
            d = random.uniform(60, 85)
            x, y = self.bx + math.cos(a) * d, self.by + math.sin(a) * d
            if math.hypot(x - self.px, y - self.py) < 35:
                continue
            h = float(self.terrain_h(np.array([x], np.float32), np.array([y], np.float32))[0])
            ok, gz = self.ground(np.array([x], np.float32), np.array([y], np.float32),
                                 np.array([h + 2.0], np.float32), 6.0, 10.0)
            if not ok[0] or gz[0] < -2.5:
                continue
            node = self.tmpl[kind].copy_to(self.root)
            z = Zombie(kind, node, x, y, float(gz[0]), self.wave)
            z.t_last = self.t
            if kind == "digger":                                 # starts underground, with a dirt mound on top
                z.burrowed, z.gz = True, float(gz[0])
                z.z = z.gz - BURROW_DEPTH
                z.mound = self.tmpl["mound"].copy_to(node)
                z.mound.set_pos(0, 0, BURROW_DEPTH - 0.15)
            if kind == "titan":
                z.hp *= 1.0 + 0.35 * (self.wave // BOSS_EVERY - 1) / 2.0
                z.maxhp = z.hp
                self.titan = z
                self.sfx("boss")
            node.set_pos(x, y, z.z)
            self.zombies.append(z)
            return

    # ------------------------------------------------------------ player
    def _player(self, dt):
        app, k = self.app, self.app.keys
        if self.riding or not self.walking:
            return
        if self.perk_regen:
            self.hp = min(100.0, self.hp + self.perk_regen * dt)
        hd = math.radians(app.heading)
        fx, fy = -math.sin(hd), math.cos(hd)
        rx, ry = math.cos(hd), math.sin(hd)
        mx = fx * (k["w"] - k["s"]) + rx * (k["d"] - k["a"])
        my = fy * (k["w"] - k["s"]) + ry * (k["d"] - k["a"])
        n = math.hypot(mx, my)
        sp = (15.0 if k["shift"] else 9.0) * self.perk_speed
        cx, cy = self.px, self.py
        step = 0.0
        if n > 1e-6:
            step = sp * dt
            cx, cy = self.px + mx / n * step, self.py + my / n * step
        g = self._player_ground(cx, cy, self.pz)
        if step > 0 and g is not None:
            rise = g - self.pz
            if rise > 1.6 or (rise > 0.4 and rise / step > 1.6):   # wall or too steep: stay put
                cx, cy = self.px, self.py
                g = self._player_ground(cx, cy, self.pz)
        self.px, self.py = cx, cy
        if k["space"] and self.on_ground:
            self.pvz = 9.5
            self.on_ground = False
        self.pvz -= GRAV * dt
        nz = self.pz + self.pvz * dt
        if g is not None and self.pvz <= 0 and (nz <= g or (self.on_ground and self.pz - g < 0.6)):
            nz, self.pvz, self.on_ground = g, 0.0, True
        else:
            self.on_ground = False
        self.pz = nz
        if self.pz < Z_KILL:
            self._respawn_player()
        app.camera.set_pos(self.px, self.py, self.pz + EYE)

    def _player_ground(self, x, y, z):
        """Surface height under the player. If nothing is found in the normal window the player is
        buried (a fill landed on them) or facing a tall wall, so look for the nearest surface ABOVE."""
        xs, ys, zs = (np.array([v], np.float32) for v in (x, y, z))
        ok, g = self.ground(xs, ys, zs, 1.6, 8.0)
        if ok[0]:
            return float(g[0])
        ok, g = self.ground(xs, ys, zs, 40.0, 0.5)
        return float(g[0]) if ok[0] else None

    def _hurt_player(self, dmg):
        if not self.walking:
            return
        dmg *= self.perk_taken
        self.hp -= dmg
        self.hurt_t = 0.25
        self.sfx("hurt")
        if self.hp <= 0:
            self.score = max(0, self.score - 50)
            self._respawn_player()
            self._say("YOU DIED", 2.0)

    # ------------------------------------------------------------ zombies
    def _zombies(self, dt):
        zs = self.zombies
        if not zs:
            return
        N = len(zs)
        group = [z for i, z in enumerate(zs) if (i + self.frame) % GROUPS == 0]
        if group:
            P = np.array([(z.x, z.y) for z in zs], np.float32)
            G = np.array([(z.x, z.y) for z in group], np.float32)
            diff = G[:, None, :] - P[None, :, :]
            d2 = (diff ** 2).sum(-1)
            m = (d2 < 1.6 ** 2) & (d2 > 1e-6)
            push = (diff / np.maximum(d2, 0.05)[..., None] * m[..., None]).sum(1)

            cxs, cys, szs, moved = [], [], [], []
            boom, surf = [], []
            for gi, z in enumerate(group):
                dtz = max(self.t - z.t_last, 1e-3)
                z.t_last = self.t
                z.dtz = dtz
                if z.kind == "titan":
                    self._titan_special(z, dtz)
                tx, ty, reach, kind, tt = self.bx, self.by, BASE_R, "base", None
                if z.burrowed:                              # diggers tunnel to the closest turret (else the beacon)
                    bd = 1e9
                    for t in self.turrets:
                        d = math.hypot(t.x - z.x, t.y - z.y)
                        if d < bd:
                            bd, tt = d, t
                    if tt is not None:
                        tx, ty, reach, kind = tt.x, tt.y, SURFACE_R, "turret"
                    else:
                        reach = BASE_R + 1.0
                elif z.kind == "bomber":                      # bombers run at the closest turret, from anywhere
                    bd = 1e9
                    for t in self.turrets:
                        d = math.hypot(t.x - z.x, t.y - z.y)
                        if d < bd:
                            bd, tt = d, t
                    if tt is not None:
                        tx, ty, reach, kind = tt.x, tt.y, 2.6, "turret"
                    elif self.walking and math.hypot(self.px - z.x, self.py - z.y) < 8.0:
                        tx, ty, reach, kind = self.px, self.py, 2.6, "player"
                    else:
                        reach = BASE_R - 1.0
                elif z.kind == "spitter":                   # stop at range and lob acid at turrets first
                    bd = 35.0
                    for t in self.turrets:
                        d = math.hypot(t.x - z.x, t.y - z.y)
                        if d < bd:
                            bd, tt = d, t
                    if tt is not None:
                        tx, ty, reach, kind = tt.x, tt.y, SPIT_RANGE, "turret"
                    elif self.walking and math.hypot(self.px - z.x, self.py - z.y) < 24.0:
                        tx, ty, reach, kind = self.px, self.py, SPIT_RANGE, "player"
                    else:
                        reach = SPIT_RANGE - 3.0            # the beacon is wide, so stand a bit closer
                elif self.walking and math.hypot(self.px - z.x, self.py - z.y) < 22.0:
                    tx, ty, reach, kind = self.px, self.py, 1.6 * z.scale, "player"
                elif self.turrets:                          # a turret close by is the first thing they smash
                    bd = 10.0
                    for t in self.turrets:
                        d = math.hypot(t.x - z.x, t.y - z.y)
                        if d < bd:
                            bd, tt = d, t
                    if tt is not None:
                        tx, ty, reach, kind = tt.x, tt.y, 2.2 * z.scale, "turret"
                dx, dy = tx - z.x, ty - z.y
                dist = math.hypot(dx, dy)
                z.heading = math.degrees(math.atan2(-dx, dy)) if dist > 1e-3 else z.heading
                if z.burrowed and dist <= reach:
                    surf.append(z)                          # close enough: burst out after this pass
                    cx, cy, mv = z.x, z.y, False
                elif dist <= reach and z.kind == "bomber":
                    boom.append(z)                          # arrived: detonate after this pass
                    cx, cy, mv = z.x, z.y, False
                elif dist <= reach:
                    z.atk_t -= dtz
                    if z.atk_t <= 0:
                        if z.kind == "spitter":
                            z.atk_t = SPIT_COOLDOWN
                            self._spit(z, tx, ty, kind, tt)
                        else:
                            z.atk_t = 1.1
                            if kind == "base":
                                self.base_hp -= z.dmg * 0.5
                                self.orb_flash = 0.2
                            elif kind == "turret":
                                self.hit_turret(tt, z.dmg)
                            elif abs(self.pz - z.z) < 2.5:
                                self._hurt_player(z.dmg)
                    cx, cy, mv = z.x, z.y, False
                else:
                    st = (BURROW_SPEED if z.burrowed else z.speed) * dtz
                    ux = dx / dist + push[gi, 0] * 0.6
                    uy = dy / dist + push[gi, 1] * 0.6
                    un = math.hypot(ux, uy) or 1.0
                    cx, cy, mv = z.x + ux / un * st, z.y + uy / un * st, True
                cxs.append(cx); cys.append(cy); szs.append(z.z + BURROW_DEPTH if z.burrowed else z.z); moved.append(mv)

            ax, ay, az = (np.array(v, np.float32) for v in (cxs, cys, szs))
            ok, gz = self.ground(ax, ay, az, 2.5, 6.0)
            ok, gz = ok.copy(), gz.copy()
            fb = np.zeros(len(group), bool)
            bad = np.flatnonzero(~ok)
            if len(bad):                                    # buried or facing a wall: surface above?
                ok2, gz2 = self.ground(ax[bad], ay[bad], az[bad], 12.0, 0.5)
                ok[bad], gz[bad], fb[bad] = ok2, gz2, ok2
                bad = np.flatnonzero(~ok)
                if len(bad):                                # still nothing: buried deep (terrain was filled in)?
                    ok3, gz3 = self.ground(ax[bad], ay[bad], az[bad], 40.0, 0.5)
                    ok[bad], gz[bad], fb[bad] = ok3, gz3, ok3
            dead = []
            for gi, z in enumerate(group):
                dtz = z.dtz
                fdt = min(dtz, 0.08)                        # integrate falling with a bounded step (no tunnelling)
                cx, cy = cxs[gi], cys[gi]
                zold = z.z
                if z.burrowed:                              # tunnelling: stay a fixed depth under the surface, ignore walls
                    if ok[gi]:
                        z.gz = float(gz[gi])
                        z.z = z.gz - BURROW_DEPTH
                    z.vx, z.vy, z.vgz = (cx - z.x) / dtz, (cy - z.y) / dtz, 0.0
                    z.x, z.y = cx, cy
                    continue
                if ok[gi]:
                    g = float(gz[gi])
                    if fb[gi] and g - z.z > 4.0:
                        # a tall wall ahead (stay put) - or this zombie was buried by an edit (pop it out)
                        o_ok, o_g = self.ground(np.array([z.x], np.float32), np.array([z.y], np.float32),
                                                np.array([z.z], np.float32), 2.5, 6.0)
                        if not o_ok[0]:
                            o_ok, o_g = self.ground(np.array([z.x], np.float32), np.array([z.y], np.float32),
                                                    np.array([z.z], np.float32), 40.0, 0.5)
                            if o_ok[0]:
                                z.z, z.vz, z.gz = float(o_g[0]), 0.0, float(o_g[0])
                        z.vx = z.vy = z.vgz = 0.0
                        continue
                    gprev = z.gz if z.gz is not None else z.z
                    run = math.hypot(cx - z.x, cy - z.y)
                    if moved[gi] and run > 1e-4 and (g - gprev) / run > SLOPE_OK and z.z <= g + 0.1:
                        cx, cy = z.x + (cx - z.x) * SLOPE_FRAC, z.y + (cy - z.y) * SLOPE_FRAC
                        g = gprev + (g - gprev) * SLOPE_FRAC     # ground height at the slowed position
                        z.z, z.vz = g, 0.0                       # ride the slope instead of sinking into it
                    elif z.z > g + 0.08:                    # airborne: falling into a trench or crater
                        z.vz = max(z.vz - GRAV * fdt, -TERM_VZ)
                        nz = z.z + z.vz * fdt
                        if nz <= g:
                            if z.vz < -12.0:
                                self.damage(z, 1.0 + (-z.vz - 12.0) * 0.25)
                            z.vz, nz = 0.0, g
                        z.z = nz
                    else:
                        z.z, z.vz = g, 0.0
                    z.gz = g
                else:
                    z.vz = max(z.vz - GRAV * fdt, -TERM_VZ)
                    z.z += z.vz * fdt
                    z.gz = None
                z.vgz = max(-6.0, min(6.0, (z.z - zold) / dtz))
                z.vx, z.vy = (cx - z.x) / dtz, (cy - z.y) / dtz
                z.x, z.y = cx, cy
                if z.z < Z_KILL:
                    dead.append(z)
            for z in dead:
                if z in self.zombies:
                    self.kill(z, scored=False, explode=False)
            for z in boom:
                if not z.dead:
                    self.kill(z, scored=False)              # bombers blow up when they die, however it happens
            for z in surf:
                if not z.dead and z.burrowed:
                    self._surface(z)

        # every frame: extrapolate, bob and sway so movement looks smooth even though only a group is simulated
        for z in self.zombies:
            e = min(self.t - z.t_last, 0.12)
            ph = self.t * z.gait + z.phase
            z.node.set_pos(z.x + z.vx * e, z.y + z.vy * e,
                           z.z + z.vgz * e + FOOT_LIFT + 0.06 * z.scale * abs(math.sin(ph)))
            z.node.set_hpr(z.heading, 0, math.sin(ph) * 6.0)
            if z.burrowed:                                  # dust kicked up along the tunnel
                z.dust -= dt
                if z.dust <= 0 and z.gz is not None:
                    z.dust = 0.1
                    n = self.tmpl["dirt"].copy_to(self.root)
                    n.set_pos(z.x, z.y, z.gz + 0.2)
                    self.debris.append([n, Vec3(random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(3, 6)), 0.5])
            if z.flash_t > 0:
                z.flash_t -= dt
                if z.flash_t <= 0:
                    z.node.set_shader_input("flash", 0.0)

        if self.base_hp <= 0:
            self.state = "over"
            self._say("BEACON DESTROYED", 100.0)
            self.info.setText(f"Survived {self.wave - 1 if self.wave else 0} waves, {self.kills} kills, "
                              f"score {self.score}  -  press Enter to restart")

    # ------------------------------------------------------------ combat
    def damage(self, z, dmg):
        if z.burrowed:                                      # a blast flushes a digger out of the ground
            self._surface(z, boom=False)
        z.hp -= dmg
        z.flash_t = 0.1
        z.node.set_shader_input("flash", 1.0)
        if z.hp <= 0 and z in self.zombies:
            self.kill(z)

    def _spit(self, z, tx, ty, kind, tt):
        """Lob a blob of acid on an arc; it splashes where the target stood when it was fired."""
        tz = tt.z + 1.0 if kind == "turret" else (self.pz + 1.0 if kind == "player" else self.bz + 2.0)
        sx, sy, sz = z.x, z.y, z.z + 1.9 * z.scale
        lx, ly = tx + random.uniform(-1.0, 1.0), ty + random.uniform(-1.0, 1.0)
        T = max(0.5, math.hypot(lx - sx, ly - sy) / SPIT_SPEED)
        vel = Vec3((lx - sx) / T, (ly - sy) / T, (tz - sz + 0.5 * SPIT_G * T * T) / T)
        node = self.tmpl["acid"].copy_to(self.root)
        node.set_pos(sx, sy, sz)
        self.spits.append([node, Vec3(sx, sy, sz), vel, T])
        self.sfx("spit", Vec3(sx, sy, sz))

    def _spits(self, dt):
        for sp in self.spits[:]:
            node, pos, vel, T = sp
            vel.z -= SPIT_G * dt
            pos.x += vel.x * dt
            pos.y += vel.y * dt
            pos.z += vel.z * dt
            sp[3] = T - dt
            node.set_pos(pos)
            if sp[3] > 0:
                continue
            self.spits.remove(sp)                            # landed: splash whatever is nearby
            node.remove_node()
            self.sfx("acid", pos)
            for t in list(self.turrets):
                if (Vec3(t.x, t.y, t.z + 1.0) - pos).length() < SPIT_SPLASH:
                    self.hit_turret(t, SPIT_TURRET_DMG)
            if math.hypot(self.bx - pos.x, self.by - pos.y) < 5.5 and abs(pos.z - self.bz) < 8.0:
                self.base_hp -= SPIT_BASE_DMG
                self.orb_flash = 0.2
            if self.walking and (Vec3(self.px, self.py, self.pz + 1.0) - pos).length() < SPIT_SPLASH * 0.85:
                self._hurt_player(SPIT_PLAYER_DMG)
            ab = self.tmpl["acid_burst"].copy_to(self.root)
            ab.set_pos(pos)
            self.fireballs.append([ab, 0.0, SPIT_SPLASH * 0.8])

    # ================================================================ helicopters
    def _air_h(self, x, y):
        return float(self.terrain_h(np.array([x], np.float32), np.array([y], np.float32))[0])

    def _make_heli(self, attack):
        root = self.root.attach_new_node("heli")
        self.tmpl["heli_attack" if attack else "heli_transport"].copy_to(root)
        rotor = self.tmpl["heli_rotor"].copy_to(root)
        rotor.set_pos(0, 0, 1.4)
        root.set_scale(1.4)
        return root, rotor

    def _tpos(self, kind, obj):
        if kind == "zombie":
            return None if obj is None or obj.dead else Vec3(obj.x, obj.y, obj.z + 1.2 * obj.scale)
        if kind == "turret" and obj is not None and not obj.dead:
            return Vec3(obj.x, obj.y, obj.z + 1.2)
        if kind == "player":
            return Vec3(self.px, self.py, self.pz + 1.0)
        return Vec3(self.bx, self.by, self.bz + 3.0)

    # ---------------------------------------------------------------- boss: UFO
    def _spawn_boss(self):
        level = self.wave // BOSS_EVERY
        node = self.tmpl["ufo"].copy_to(self.root)
        node.set_scale(1.2)
        beam = self.tmpl["beam"].copy_to(self.root)
        beam.hide()
        a = random.random() * 6.2832
        x, y = self.bx + math.cos(a) * 130.0, self.by + math.sin(a) * 130.0
        g = self._air_h(x, y)
        b = Boss(node, beam, x, y, g + 30.0, UFO_HP + UFO_HP_PER_LEVEL * max(0, level // 2 - 1), level)
        b.angle, b.ground = a, g
        self.boss = b
        self.sfx("boss")

    def _boss_remove(self):
        b = self.boss
        if b is not None:
            b.node.remove_node()
            b.beam.remove_node()
            self.boss = None
            self.snd.loop("ufo", False)

    def damage_boss(self, dmg):
        b = self.boss
        if b is None or b.dead:
            return
        b.hp -= dmg
        b.flash_t = 0.06
        b.node.set_shader_input("flash", 1.0)
        if b.hp <= 0:
            b.dead = True
            b.crash_vz = 0.0
            b.beam.hide()
            self.score += 300 * max(1, b.level // 2)
            self.bombs = min(self.max_bombs, self.bombs + 4)
            self._say("UFO DESTROYED", 3.0)

    def _ray_boss(self, o, d):
        b = self.boss
        if b is None or b.dead:
            return None
        c = Vec3(b.x, b.y, b.z)
        best = None
        for center, r, mult in ((c, 4.6, 1.0), (c + Vec3(0, 0, 1.2), 2.6, 1.6)):     # hull, dome (weak spot)
            v = center - o
            tca = v.dot(d)
            if tca < 0 or tca > 130.0:
                continue
            d2 = v.dot(v) - tca * tca
            if d2 > r * r:
                continue
            t = tca - math.sqrt(r * r - d2)
            if best is None or t < best[0]:
                best = (max(t, 0.0), mult)
        return best

    def _pick_boss_target(self, b):
        cands = [("turret", t) for t in self.turrets if math.hypot(t.x - b.x, t.y - b.y) < 90.0]
        if self.walking and math.hypot(self.px - b.x, self.py - b.y) < 80.0:
            cands += [("player", None)] * 2
        cands.append(("base", None))
        return random.choice(cands)

    def _boss(self, dt):
        b = self.boss
        if b is None:
            return
        b.heading += 40.0 * dt
        if b.flash_t > 0:
            b.flash_t -= dt
            if b.flash_t <= 0:
                b.node.set_shader_input("flash", 0.0)
        if b.dead:                                           # crash: fall, spin, blow up on the ground
            b.crash_vz -= 25.0 * dt
            b.z += b.crash_vz * dt
            b.x += b.vx * dt
            b.y += b.vy * dt
            b.node.set_pos(b.x, b.y, b.z)
            b.node.set_hpr(b.node.get_h() + 260.0 * dt, b.node.get_p() + 40.0 * dt, b.node.get_r() + 120.0 * dt)
            g = self._air_h(b.x, b.y)
            if b.z <= g + 1.5:
                self.sfx("explosion", Vec3(b.x, b.y, g + 1.0))
                self._blast(Vec3(b.x, b.y, g + 1.0), 10.0, 30.0, 25.0, 40.0, 6.0, zdmg=10.0)
                self._boss_remove()
            return
        enrage = b.hp < 0.35 * b.maxhp
        b.radius_t -= dt
        if b.radius_t <= 0:
            b.radius_t, b.radius = random.uniform(5, 9), random.uniform(20, 42)
        speed = (13.0 + 1.5 * (b.level // 2)) * (1.35 if enrage else 1.0)
        b.angle += speed / b.radius * dt
        wx, wy = self.bx + math.cos(b.angle) * b.radius, self.by + math.sin(b.angle) * b.radius
        ox, oy = b.x, b.y
        dx, dy = wx - b.x, wy - b.y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            step = min(dist, min(45.0, speed + dist * 0.8) * dt)
            b.x += dx / dist * step
            b.y += dy / dist * step
        b.alt_t -= dt
        if b.alt_t <= 0:
            b.alt_t, b.ground = 0.4, self._air_h(b.x, b.y)
        b.z += (b.ground + 26.0 + 2.0 * math.sin(self.t * 1.3) - b.z) * min(1.0, 2.0 * dt)
        b.vx, b.vy = (b.x - ox) / max(dt, 1e-3), (b.y - oy) / max(dt, 1e-3)
        b.node.set_pos(b.x, b.y, b.z)
        b.node.set_hpr(b.heading, 6.0 * math.sin(self.t * 1.1), 6.0 * math.cos(self.t * 0.8))
        self.snd.loop("ufo", True, 0.7, Vec3(b.x, b.y, b.z))

        b.pick_t -= dt
        kind, obj = b.target
        if b.pick_t <= 0 or (kind == "turret" and (obj is None or obj.dead)):
            b.pick_t, b.target = 2.0, self._pick_boss_target(b)
            kind, obj = b.target

        # tractor/death beam: thin warning line, then a damaging beam that slowly tracks its target
        if b.mode == "idle":
            b.gap_t -= dt
            if b.gap_t <= 0:
                b.mode, b.lock_t, b.locked = "lock", 1.2, b.target
                self.sfx("lock", Vec3(b.x, b.y, b.z))
                b.beam_end = Vec3(self._tpos(*b.locked))
        else:
            tp = self._tpos(*b.locked)
            v = tp - b.beam_end
            d = v.length()
            if d > 1e-3:
                k = min(d, (20.0 if b.mode == "lock" else 10.0) * dt) / d
                b.beam_end = b.beam_end + v * k
            start = Vec3(b.x, b.y, b.z - 1.5)
            w = 0.15 if b.mode == "lock" else 1.1
            b.beam.show()
            b.beam.set_pos(start)
            b.beam.look_at(Point3(b.beam_end))
            b.beam.set_scale(w, max((b.beam_end - start).length(), 0.1), w)
            b.beam.set_shader_input("flash", 0.5 if int(self.t * 12) % 2 else 0.1)
            if b.mode == "lock":
                b.lock_t -= dt
                if b.lock_t <= 0:
                    b.mode, b.beam_t = "beam", 1.3
                    self.sfx("beam", Vec3(b.x, b.y, b.z))
            else:
                e = b.beam_end
                for t in list(self.turrets):
                    if math.hypot(t.x - e.x, t.y - e.y) < 2.5:
                        self.hit_turret(t, UFO_BEAM_TURRET * dt)
                if math.hypot(self.bx - e.x, self.by - e.y) < 5.0:
                    self.base_hp -= UFO_BEAM_BASE * dt
                    self.orb_flash = 0.1
                if self.walking and math.hypot(self.px - e.x, self.py - e.y) < 2.2 and abs(self.pz - e.z) < 5.0:
                    self._hurt_player(UFO_BEAM_PLAYER * dt)
                b.beam_t -= dt
                if b.beam_t <= 0:
                    b.mode, b.gap_t = "idle", (2.5 if enrage else 4.0)
                    b.beam.hide()

        b.plasma_cd -= dt                                    # slow plasma orbs, three at a time
        if b.plasma_cd <= 0:
            b.plasma_cd = 6.0 if enrage else 9.0
            self.sfx("plasma", Vec3(b.x, b.y, b.z))
            for _ in range(3):
                k2, o2 = self._pick_boss_target(b)
                node = self.tmpl["plasma"].copy_to(self.root)
                pos = Vec3(b.x, b.y, b.z - 2.0)
                node.set_pos(pos)
                self.missiles.append([node, pos, k2, o2, 8.0, PLASMA_SPEED, PLASMA_R,
                                      PLASMA_TURRET_DMG, PLASMA_BASE_DMG, PLASMA_PLAYER_DMG, 0.0, 1.5])

    def _missiles(self, dt):
        for m in self.missiles[:]:
            node, pos, kind, obj, life, speed, R, td, bd, pd, zd, crater = m
            if kind == "turret" and (obj is None or obj.dead):
                m[2], m[3] = kind, obj = "base", None
            tp = self._tpos(kind, obj)
            gone = tp is None                                # a zombie target died: explode where we are
            v = Vec3(0, 0, 0) if gone else tp - pos
            dist = v.length()
            m[4] = life - dt
            if gone or dist < 2.0 or m[4] <= 0:
                self.missiles.remove(m)
                node.remove_node()
                if kind == "zombie" and obj is self.boss and not gone:
                    self.damage_boss(12.0)
                self._blast(Vec3(pos), R, td, bd, pd, crater, zdmg=zd)
                continue
            step = speed * dt
            pos.x += v.x / dist * step
            pos.y += v.y / dist * step
            pos.z += v.z / dist * step
            node.set_pos(pos)
            node.look_at(Point3(tp))

    # ---------------------------------------------------------------- giant brute (boss)
    def _titan_special(self, z, dtz):
        """Ground slam: it winds up (flashing) when something is near, then shakes the ground."""
        z.slam_t -= dtz
        if z.slamming:
            z.node.set_shader_input("flash", 0.6 if int(z.slam_t * 12) % 2 else 0.0)
            if z.slam_t <= 0:
                z.slamming, z.slam_t = False, SLAM_COOLDOWN
                z.node.set_shader_input("flash", 0.0)
                p = Vec3(z.x, z.y, z.z)
                self.sfx("slam", p)
                self._blast(p, SLAM_R, 14.0, 10.0, 0.0, 3.2)
                if self.walking and self.on_ground and math.hypot(self.px - z.x, self.py - z.y) < SLAM_R:
                    self._hurt_player(25.0 * (1.0 - math.hypot(self.px - z.x, self.py - z.y) / SLAM_R))
        elif z.slam_t <= 0:
            near = (math.hypot(self.bx - z.x, self.by - z.y) < 16.0
                    or any(math.hypot(t.x - z.x, t.y - z.y) < 14.0 for t in self.turrets)
                    or (self.walking and math.hypot(self.px - z.x, self.py - z.y) < 14.0))
            if near:
                z.slamming, z.slam_t = True, 0.9
                self.sfx("growl", Vec3(z.x, z.y, z.z))
            else:
                z.slam_t = 0.4

    # ---------------------------------------------------------------- your attack helicopter
    def call_ally(self, free=False):
        """H: call in the gunship for 40 s (costs score).  Boss waves bring it for free."""
        if self.ally is not None:
            if free:
                self.ally.time_left, self.ally.leaving = None, False
            return
        if not free:
            if self.state in ("over", "evac"):
                return
            if self.score < ALLY_COST:
                return self._say(f"NEED {ALLY_COST} SCORE", 1.5)
            self.score -= ALLY_COST
        root, rotor = self._make_heli(True)
        a = random.random() * 6.2832
        x, y = self.bx + math.cos(a) * 120.0, self.by + math.sin(a) * 120.0
        self.ally = Ally(root, rotor, x, y, self._air_h(x, y) + 25.0, None if free else ALLY_DURATION)
        self.ally.angle = a
        self._say("GUNSHIP INBOUND", 2.0)

    def skip_build(self):
        """N: start the next wave now (small score bonus for the time you skip)."""
        if self.state == "build" and self.timer > 1.0:
            self.score += int(self.timer) * 2
            self.timer = 0.0

    def _ally_pick(self):
        b = self.boss
        if b is not None and not b.dead:
            return b
        best, bd = None, 140.0
        for z in self.zombies:
            if z.burrowed:
                continue
            d = math.hypot(z.x - self.bx, z.y - self.by) - (30.0 if z.kind == "titan" else 0.0)
            if d < bd:
                best, bd = z, d
        return best

    def _ally(self, dt):
        a = self.ally
        if a is None:
            return
        a.spin += 1500.0 * dt
        a.rotor.set_h(a.spin)
        self.snd.loop("rotor", True, 0.5, Vec3(a.x, a.y, a.z))
        if a.time_left is not None:
            a.time_left -= dt
            if a.time_left <= 0:
                a.leaving = True
        if a.leaving:                                        # fly away from the beacon and disappear
            dx, dy = a.x - self.bx, a.y - self.by
            d = math.hypot(dx, dy) or 1.0
            a.x += dx / d * 35.0 * dt
            a.y += dy / d * 35.0 * dt
            a.z += 8.0 * dt
            a.t_leave += dt
            a.node.set_pos(a.x, a.y, a.z)
            a.node.set_hpr(math.degrees(math.atan2(-dx, dy)), -8.0, 0)
            if a.t_leave > 5.0:
                a.node.remove_node()
                self.ally = None
                self.snd.loop("rotor", False)
            return
        a.pick_t -= dt
        if a.pick_t <= 0 or a.target is None or a.target.dead:
            a.pick_t, a.target = 0.5, self._ally_pick()
        tgt = a.target
        cx, cy = self.bx, self.by
        if tgt is not None:                                  # drift toward the fight
            cx += (tgt.x - self.bx) * 0.45
            cy += (tgt.y - self.by) * 0.45
        a.radius_t -= dt
        if a.radius_t <= 0:
            a.radius_t, a.radius = random.uniform(6, 9), random.uniform(18, 30)
        a.angle += 16.0 / a.radius * dt
        wx, wy = cx + math.cos(a.angle) * a.radius, cy + math.sin(a.angle) * a.radius
        dx, dy = wx - a.x, wy - a.y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            step = min(dist, min(40.0, 16.0 + dist * 0.8) * dt)
            a.x += dx / dist * step
            a.y += dy / dist * step
        a.alt_t -= dt
        if a.alt_t <= 0:
            a.alt_t, a.ground = 0.4, self._air_h(a.x, a.y)
        a.z += (a.ground + 18.0 + 1.5 * math.sin(self.t * 1.7) - a.z) * min(1.0, 2.0 * dt)
        if tgt is not None:
            th = math.degrees(math.atan2(-(tgt.x - a.x), tgt.y - a.y))
        else:
            th = math.degrees(math.atan2(-dx, dy)) if dist > 1e-3 else a.heading
        a.heading += (((th - a.heading + 180.0) % 360.0) - 180.0) * min(1.0, 3.0 * dt)
        a.node.set_pos(a.x, a.y, a.z)
        a.node.set_hpr(a.heading, -8.0, 6.0 * math.sin(self.t))
        if tgt is not None and math.hypot(tgt.x - a.x, tgt.y - a.y) < ALLY_RANGE:
            a.fire_cd -= dt
            if a.fire_cd <= 0:
                a.fire_cd = 0.1
                self._ally_shoot(a, tgt)
            a.missile_cd -= dt
            if a.missile_cd <= 0:
                a.missile_cd = 5.0
                node = self.tmpl["missile"].copy_to(self.root)
                pos = Vec3(a.x, a.y, a.z - 1.5)
                node.set_pos(pos)
                self.sfx("missile", pos)
                self.missiles.append([node, pos, "zombie", tgt, 5.0, 32.0, 5.0, 0.0, 0.0, 0.0, ALLY_MISSILE_DMG, 0.0])

    def _ally_shoot(self, a, t):
        h = math.radians(a.heading)
        muzzle = Vec3(a.x - math.sin(h) * 3.2, a.y + math.cos(h) * 3.2, a.z - 1.2)
        aim = Vec3(t.x, t.y, t.z + 1.0 * t.scale) + Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.5, 0.5))
        self._tracer(muzzle, aim)
        self.sfx("mg", muzzle)
        if random.random() < 0.85:
            if t is self.boss:
                self.damage_boss(ALLY_MG_BOSS)
            else:
                self.damage(t, ALLY_MG_DMG)

    def _blast(self, p, R, tdmg, bdmg, pdmg, crater, zdmg=0.0):
        """A general explosion: hurts turrets, beacon, player (and zombies if zdmg), digs a crater, shows a fireball."""
        self.sfx("blast", p)
        for t in list(self.turrets):
            d = (Vec3(t.x, t.y, t.z + 0.8) - p).length()
            if tdmg > 0 and d < R:
                self.hit_turret(t, tdmg * (1.0 - 0.5 * d / R))
        d = math.hypot(self.bx - p.x, self.by - p.y) - 3.5
        if bdmg > 0 and d < R and abs(p.z - self.bz) < R + 8.0:
            self.base_hp -= bdmg * (1.0 - 0.5 * max(d, 0.0) / R)
            self.orb_flash = 0.2
        if pdmg > 0 and self.walking:
            d = (Vec3(self.px, self.py, self.pz + 1.0) - p).length()
            if d < R:
                self._hurt_player(pdmg * (1.0 - d / R))
        if zdmg:
            for o in list(self.zombies):
                d = (Vec3(o.x, o.y, o.z + o.scale) - p).length()
                if d < R:
                    self.damage(o, zdmg * (1.0 - d / R))
        if crater > 0:
            try:
                self.app.chunks.edit("dig", p, crater)
            except Exception as e:
                print("blast crater failed:", e)
        fb = self.tmpl["fireball"].copy_to(self.root)
        fb.set_pos(p)
        self.fireballs.append([fb, 0.0, R])
        for _ in range(10):
            node = self.tmpl["debris"].copy_to(self.root)
            node.set_pos(p)
            node.set_shader_input("flash", 0.5)
            v = Vec3(random.uniform(-9, 9), random.uniform(-9, 9), random.uniform(5, 14))
            self.debris.append([node, v, 0.8])

    # ---------------------------------------------------------------- landing zone + evacuation
    def _pick_lz(self):
        a = np.random.uniform(0.0, 6.2832, 48).astype(np.float32)
        r = np.random.uniform(150.0, 230.0, 48).astype(np.float32)
        X, Y = self.bx + np.cos(a) * r, self.by + np.sin(a) * r
        h = self.terrain_h(X, Y)
        e = np.float32(6.0)
        slope = (np.abs(self.terrain_h(X + e, Y) - h) + np.abs(self.terrain_h(X - e, Y) - h)
                 + np.abs(self.terrain_h(X, Y + e) - h) + np.abs(self.terrain_h(X, Y - e) - h)) / e
        score = slope * 10.0 + np.where((h < 2) | (h > 30), 100.0, 0.0)
        i = int(np.argmin(score))
        self.lz = (float(X[i]), float(Y[i]), float(h[i]))
        if self.lz_node is not None:
            self.lz_node.remove_node()
        self.lz_node = self.tmpl["marker"].copy_to(self.root)
        self.lz_node.set_pos(self.lz[0], self.lz[1], self.lz[2] - 0.2)
        self.lz_node.set_shader_input("flash", 0.45)

    def _start_evac(self):
        self.state = "evac"
        root, rotor = self._make_heli(False)
        cable = self.tmpl["cable"].copy_to(root)
        cable.hide()
        a = random.random() * 6.2832
        x, y = self.bx + math.cos(a) * 150.0, self.by + math.sin(a) * 150.0
        self.evac = dict(phase="arrive", t=0.0, node=root, rotor=rotor, cable=cable, spin=0.0,
                         x=x, y=y, z=self._air_h(x, y) + 30.0, heading=0.0, offsets=[])
        if self.lz is None:
            self._pick_lz()
        self._say("EVAC HELICOPTER INBOUND", 3.0)

    def _fly(self, e, tx, ty, tz, speed, dt, pitch=-6.0):
        dx, dy, dz = tx - e["x"], ty - e["y"], tz - e["z"]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > 1e-3:
            step = min(dist, speed * dt)
            e["x"] += dx / dist * step
            e["y"] += dy / dist * step
            e["z"] += dz / dist * step
        if math.hypot(dx, dy) > 3.0:
            th = math.degrees(math.atan2(-dx, dy))
            e["heading"] += (((th - e["heading"] + 180.0) % 360.0) - 180.0) * min(1.0, 2.5 * dt)
        e["node"].set_pos(e["x"], e["y"], e["z"])
        e["node"].set_hpr(e["heading"], pitch, 0)
        return dist

    def _evac(self, dt):
        e = self.evac
        if e is None:
            self.state, self.timer = "build", 20.0
            return
        e["spin"] += 1500.0 * dt
        e["rotor"].set_h(e["spin"])
        self.snd.loop("rotor2", True, 0.6, Vec3(e["x"], e["y"], e["z"]))
        ph = e["phase"]
        lx, ly, lzh = self.lz if self.lz else (self.bx, self.by, self.bz)
        if ph == "arrive":
            self.info.setText("Evac helicopter inbound: it will carry the beacon and turrets to the landing zone")
            if self._fly(e, self.bx, self.by, self.bz + HANG + 8.0, 28.0, dt) < 1.5:
                e["phase"], e["t"] = "hook", 0.0
        elif ph == "hook":
            e["t"] += dt
            self.info.setText("Hooking the beacon...")
            self._fly(e, self.bx, self.by, self.bz + HANG + 8.0, 5.0, dt, pitch=0.0)
            if e["t"] > 1.8:
                e["offsets"] = [(t.x - self.bx, t.y - self.by) for t in self.turrets]
                for t in self.turrets:
                    t.base.hide()
                    t.head.hide()
                e["cable"].show()
                e["cable"].set_scale(1, 1, (HANG - 1.6) / 1.4)
                self.riding = True
                self._say("LIFTING OFF", 2.0)
                e["phase"] = "carry"
        elif ph in ("carry", "lower"):
            landed = False
            if ph == "carry":
                d = math.hypot(lx - e["x"], ly - e["y"])
                self.info.setText(f"Carrying the beacon to the landing zone: {int(d)} units to go")
                self._fly(e, lx, ly, max(self.bz, lzh) + HANG + 8.0, 24.0, dt)
                if d < 2.0:
                    e["phase"] = "lower"
            else:
                self.info.setText("Setting the beacon down...")
                e["z"] = max(lzh + HANG, e["z"] - 6.0 * dt)
                e["node"].set_pos(e["x"], e["y"], e["z"])
                landed = e["z"] <= lzh + HANG + 0.01
            self.base.set_pos(e["x"], e["y"], e["z"] - HANG)
            self.orb.set_pos(e["x"], e["y"], e["z"] - HANG + 9.4)
            self.px, self.py, self.pz = e["x"], e["y"], e["z"] - 3.0
            self.app.camera.set_pos(self.px, self.py, self.pz)
            if landed:
                self._land_beacon(e)
        elif ph == "depart":
            e["t"] += dt
            self._fly(e, e["x"] + 120.0, e["y"] + 120.0, e["z"] + 25.0, 30.0, dt)
            self.info.setText("")
            if e["t"] > 6.0:
                e["node"].remove_node()
                self.evac = None
                self.snd.loop("rotor2", False)
                self.state, self.timer = "build", 15.0
                self._say("BEACON RELOCATED", 3.0)

    def _land_beacon(self, e):
        lx, ly, lzh = self.lz
        self.bx, self.by, self.bz = lx, ly, lzh - 0.3
        self.base.set_pos(self.bx, self.by, self.bz)
        self.orb.set_pos(self.bx, self.by, self.bz + 9.4)
        self._place_shop()
        if self.turrets:                                     # turrets keep their layout around the beacon
            xs = np.array([lx + o[0] for o in e["offsets"]], np.float32)
            ys = np.array([ly + o[1] for o in e["offsets"]], np.float32)
            ok, gz = self.ground(xs, ys, np.full(len(xs), lzh, np.float32), 30.0, 40.0)
            for i, t in enumerate(self.turrets):
                t.x, t.y = float(xs[i]), float(ys[i])
                t.z = float(gz[i]) - 0.1 if ok[i] else lzh - 0.1
                t.base.set_pos(t.x, t.y, t.z)
                t.head.set_pos(t.x, t.y, t.z + 1.9)
                t.base.show()
                t.head.show()
                t.target = None
        e["cable"].hide()
        self.riding = False
        self.base_hp = self.base_max
        if self.lz_node is not None:
            self.lz_node.remove_node()
            self.lz_node = None
        self.lz = None
        self._respawn_player()
        e["phase"], e["t"] = "depart", 0.0

    def _hud_extra(self):
        s = ""
        b = self.boss
        if b is not None and not b.dead:
            f = max(0, min(10, int(round(10 * b.hp / b.maxhp))))
            s += f"   UFO [{'#' * f}{'-' * (10 - f)}]"
        t = self.titan
        if t is not None and not t.dead:
            f = max(0, min(10, int(round(10 * t.hp / t.maxhp))))
            s += f"   BRUTE [{'#' * f}{'-' * (10 - f)}]"
        a = self.ally
        if a is not None and not a.leaving:
            s += f"   GUNSHIP {int(a.time_left) + 1}s" if a.time_left is not None else "   GUNSHIP"
        if self.lz is not None:
            c = self.app.camera.get_pos(self.app.render)
            dx, dy = self.lz[0] - c.x, self.lz[1] - c.y
            name = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[int(((math.degrees(math.atan2(dx, dy)) % 360.0) + 22.5) // 45) % 8]
            s += f"   LZ {int(math.hypot(dx, dy))} {name}"
        return s

    def _explode(self, z):
        p = Vec3(z.x, z.y, z.z + 0.5)
        self.sfx("explosion", p)
        R = BOMBER_RADIUS
        for t in list(self.turrets):
            d = (Vec3(t.x, t.y, t.z + 0.8) - p).length()
            if d < R:
                self.hit_turret(t, BOMBER_TURRET_DMG * (1.0 - 0.5 * d / R))
        d = math.hypot(self.bx - p.x, self.by - p.y) - 3.5
        if d < R:
            self.base_hp -= BOMBER_BASE_DMG * (1.0 - 0.5 * max(d, 0.0) / R)
            self.orb_flash = 0.2
        if self.walking:
            d = (Vec3(self.px, self.py, self.pz + 1.0) - p).length()
            if d < R:
                self._hurt_player(BOMBER_PLAYER_DMG * (1.0 - d / R))
        for o in list(self.zombies):                         # friendly fire: shoot a bomber in a crowd
            d = (Vec3(o.x, o.y, o.z + o.scale) - p).length()
            if d < R:
                self.damage(o, 5.0 * (1.0 - d / R))
        try:
            self.app.chunks.edit("dig", p, BOMBER_CRATER)    # the blast leaves a real hole in the terrain
        except Exception as e:
            print("explosion crater failed:", e)
        fb = self.tmpl["fireball"].copy_to(self.root)
        fb.set_pos(p)
        self.fireballs.append([fb, 0.0, BOMBER_RADIUS])
        for _ in range(10):
            node = self.tmpl["debris"].copy_to(self.root)
            node.set_pos(p)
            node.set_shader_input("flash", 0.5)
            v = Vec3(random.uniform(-9, 9), random.uniform(-9, 9), random.uniform(5, 14))
            self.debris.append([node, v, 0.8])

    def kill(self, z, scored=True, explode=True):
        z.dead = True
        self.zombies.remove(z)
        if z.kind == "titan" and scored:
            self.bombs = min(self.max_bombs, self.bombs + 3)
            self._say("GIANT BRUTE DOWN", 2.5)
        if explode and z.kind == "bomber":
            self._explode(z)
        if scored:
            self.sfx("zombie_die", Vec3(z.x, z.y, z.z))
            self.kills += 1
            self.score += int(z.score * self.perk_loot)
            if self.kills % 8 == 0:
                self.bombs = min(self.max_bombs, self.bombs + 1)
            for _ in range(5):
                node = self.tmpl["debris"].copy_to(self.root)
                node.set_pos(z.x, z.y, z.z + 1.0 * z.scale)
                node.set_scale(z.scale)
                v = Vec3(random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(4, 9))
                self.debris.append([node, v, 0.9])
        z.node.remove_node()

    def crater(self, pos, r):
        """Called by main.py when a bomb cube digs a crater: hurt zombies in the blast."""
        self.sfx("explosion", pos, min(1.0, r / 5.0))
        reach = r * 1.5
        for z in list(self.zombies):
            c = Vec3(z.x, z.y, z.z + z.scale) - pos
            d = c.length()
            if d < reach:
                self.damage(z, max(2.0, 14.0 * (1.0 - d / reach)))

    def _weapon_hitscan(self, origin, direction, damage,
                        tracer=True,
                        headshot_mult=2.0):
        """
        Generic hitscan weapon.

        origin          Vec3
        direction       normalized Vec3
        damage          base damage
        tracer          draw tracer beam
        headshot_mult   multiplier for head hits

        Returns:
            True if something was hit.
        """

        damage *= self.perk_dmg
        best = None
        bt = 1e9
        bmult = 1.0

        #
        # Find closest zombie hit
        #

        for z in self.zombies:
            if z.burrowed:
                continue

            s = z.scale

            for h, r, mult in (
                (1.2, 0.65, 1.0),   # body
                (1.9, 0.4, 2.0),    # head
                (0.55, 0.5, 1.0),   # legs
            ):

                c = Vec3(z.x, z.y, z.z + h * s) - origin

                tca = c.dot(direction)

                if tca < 0:
                    continue

                d2 = c.dot(c) - tca * tca

                rr = r * s

                if d2 > rr * rr:
                    continue

                t = tca - math.sqrt(rr * rr - d2)

                if t < bt:
                    best = z
                    bt = max(t, 0.0)
                    bmult = mult

        #
        # Boss hit test
        #

        boss_hit = self._ray_boss(origin, direction)

        end = origin + direction * 70.0

        if boss_hit is not None and boss_hit[0] < bt:

            end = origin + direction * boss_hit[0]

            self.damage_boss(damage * boss_hit[1])

            if tracer:
                self._tracer(
                    Vec3(origin),
                    Vec3(end)
                )

            return True

        #
        # Zombie damage
        #

        if best is not None:

            hit = origin + direction * bt

            if self.los(origin, hit):

                end = hit

                mult = bmult

                #
                # Convert head marker into weapon-specific multiplier
                #

                if bmult > 1.2:
                    mult = headshot_mult
                else:
                    mult = 1.0

                self.damage(
                    best,
                    damage * mult
                )

                if tracer:
                    self._tracer(
                        origin,
                        end
                    )

                return True

        #
        # Miss shot
        #

        if tracer:
            self._tracer(
                origin,
                end
            )

        return False

    def shoot(self):

        if self.state == "over" or self.t < self.next_shot:
            return

        cfg = WEAPONS[self.weapon]

        self.next_shot = self.t + cfg["cooldown"] * self.perk_rate

        self.sfx("shoot")

        app = self.app

        muzzle = app.render.get_relative_point(
            app.camera,
            Point3(0.35, 0.9, -0.3)
        )

        d = app.render.get_relative_vector(
            app.camera,
            Vec3(0, 1, 0)
        )

        d.normalize()

        #
        # Grenade Launcher
        #

        if self.weapon == "launcher":

            hit = self.app.chunks.ray_hit(
                muzzle,
                d,
                70.0
            )

            if hit:

                p = Vec3(hit)

                self._tracer(
                    muzzle,
                    p
                )

                self._blast(
                    p,
                    6.0 * self.perk_blast,   # radius
                    0.0,   # turret dmg
                    0.0,   # base dmg
                    0.0,   # player dmg
                    5.0 * self.perk_blast,   # crater size
                    zdmg=10.0 * self.perk_dmg
                )

            return

        #
        # Shotgun
        #

        if self.weapon == "shotgun":

            for _ in range(cfg["pellets"]):

                spread = cfg["spread"]

                d2 = Vec3(
                    d.x + random.uniform(-spread, spread),
                    d.y + random.uniform(-spread, spread),
                    d.z + random.uniform(-spread, spread)
                )

                d2.normalize()

                self._weapon_hitscan(
                    muzzle,
                    d2,
                    cfg["damage"],
                    tracer=True,
                    headshot_mult=2.0
                )

            return

        #
        # Sniper
        #

        if self.weapon == "sniper":

            self._weapon_hitscan(
                muzzle,
                d,
                cfg["damage"],
                tracer=True,
                headshot_mult=4.0
            )

            return

        #
        # Rifle and SMG
        #

        self._weapon_hitscan(
            muzzle,
            d,
            cfg["damage"],
            tracer=True,
            headshot_mult=2.0
        )

    def _tracer(self, start, end):
        tr = self.tmpl["tracer"].copy_to(self.root)
        tr.set_pos(start)
        tr.look_at(Point3(end))
        tr.set_scale(1, max((Vec3(end) - Vec3(start)).length(), 0.1), 1)
        tr.set_shader_input("flash", 1.0)
        self.tracers.append([tr, 0.06])

    # ------------------------------------------------------------ turrets
    def turret_cost(self):
        return int((TURRET_COST + TURRET_COST_STEP * len(self.turrets)) * self.perk_cheap)

    def buy_turret(self):
        """T: spend score on a turret placed where you are looking."""
        if self.state in ("over", "evac"):
            return
        cost = self.turret_cost()
        if len(self.turrets) >= MAX_TURRETS:
            return self._say("TURRET LIMIT", 1.5)
        if self.score < cost:
            return self._say(f"NEED {cost} SCORE", 1.5)
        app = self.app
        o = Vec3(app.camera.get_pos(app.render))
        d = app.render.get_relative_vector(app.camera, Vec3(0, 1, 0))
        d.normalize()
        hit = app.chunks.ray_hit(o, d, 80.0)
        if hit is None:
            return self._say("NO GROUND THERE", 1.5)
        if math.hypot(hit.x - self.bx, hit.y - self.by) > 75.0:
            return self._say("TOO FAR FROM BEACON", 1.5)
        if any(math.hypot(hit.x - t.x, hit.y - t.y) < 3.5 for t in self.turrets):
            return self._say("TOO CLOSE", 1.5)
        self.score -= cost
        base = self.tmpl["turret_base"].copy_to(self.root)
        head = self.tmpl["turret_head"].copy_to(self.root)
        base.set_pos(hit.x, hit.y, hit.z - 0.1)
        head.set_pos(hit.x, hit.y, hit.z + 1.8)
        self.turrets.append(Turret(base, head, hit.x, hit.y, hit.z - 0.1))
        self._say("TURRET READY", 1.2)

    def hit_turret(self, t, dmg):
        if t.dead:
            return
        t.hp -= dmg
        t.flash_t = 0.1
        t.head.set_shader_input("flash", 1.0)
        self.sfx("hit", Vec3(t.x, t.y, t.z + 1.0))
        if t.hp <= 0:
            t.dead = True
            self.turrets.remove(t)
            t.base.remove_node()
            t.head.remove_node()
            self._say("TURRET LOST", 1.5)

    def _turrets(self, dt):
        for t in list(self.turrets):
            if t.flash_t > 0:
                t.flash_t -= dt
                if t.flash_t <= 0:
                    t.head.set_shader_input("flash", 0.0)
            t.cd -= dt
            t.scan -= dt
            if t.target is not None and t.target.dead:
                t.target = None
            if t.scan <= 0:                                  # pick the nearest zombie in range
                t.scan = 0.15
                best, bd = None, TURRET_RANGE ** 2
                for z in self.zombies:
                    if z.burrowed:
                        continue
                    d2 = (z.x - t.x) ** 2 + (z.y - t.y) ** 2 + (z.z - t.z) ** 2
                    if d2 < bd:
                        best, bd = z, d2
                b = self.boss
                if b is not None and not b.dead and (b.x - t.x) ** 2 + (b.y - t.y) ** 2 + (b.z - t.z) ** 2 < (TURRET_RANGE * 1.4) ** 2:
                    best = b                                 # anti-air takes priority
                t.target = best
            z = t.target
            if z is None:
                continue
            aim = Vec3(z.x, z.y, z.z + 1.2 * z.scale)
            d = aim - Vec3(t.x, t.y, t.z + 1.8)
            t.head.set_hpr(math.degrees(math.atan2(-d.x, d.y)),
                           math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))), 0)
            if t.cd <= 0:
                muzzle = self.root.get_relative_point(t.head, Point3(0, 1.7, 0))
                if self.los(muzzle, aim):
                    self._tracer(muzzle, aim)
                    self.sfx("turret", muzzle)
                    t.cd = TURRET_RATE * self.perk_turret
                    if z is self.boss:
                        self.damage_boss(BOSS_AIR_TURRET_DMG)
                    else:
                        self.damage(z, TURRET_DMG)
                else:                                        # terrain in the way: look for another target
                    t.cd, t.scan, t.target = 0.2, 0.0, None

    # ------------------------------------------------------------ effects + hud
    def _effects(self, dt):
        if self.weapon_np:

            sway=math.sin(self.t*7)*0.015

            self.weapon_np.set_z(
                -0.35+sway
            )
        for f in self.fireballs[:]:
            f[1] += dt
            k = min(f[1] / 0.25, 1.0)
            f[0].set_scale(0.6 + f[2] * 0.9 * k)
            f[0].set_shader_input("flash", 0.6 * k)
            if f[1] > 0.3:
                f[0].remove_node()
                self.fireballs.remove(f)
        for d in self.debris[:]:
            d[2] -= dt
            if d[2] <= 0:
                d[0].remove_node()
                self.debris.remove(d)
                continue
            d[1].z -= 22.0 * dt
            d[0].set_pos(d[0].get_pos() + d[1] * dt)
            d[0].set_hpr(d[0].get_h() + 400 * dt, d[0].get_p() + 300 * dt, 0)
        for tr in self.tracers[:]:
            tr[1] -= dt
            if tr[1] <= 0:
                tr[0].remove_node()
                self.tracers.remove(tr)
        if self.orb_flash > 0:
            self.orb_flash -= dt
            self.orb.set_shader_input("flash", 1.0 if self.orb_flash > 0 else 0.0)
        if self.banner_t > 0:
            self.banner_t -= dt
            if self.banner_t <= 0:
                self.banner.setText("")
        if self.hurt_t > 0:
            self.hurt_t -= dt
            self.cross.setFg((1, 0.2, 0.2, 1) if self.hurt_t > 0 else (1, 1, 1, 0.9))

    def _hud(self, dt):
        self.hud_t -= dt
        if self.hud_t > 0:
            return
        self.hud_t = 0.1
        if self.shop_open:
            self._shop_refresh()
        filled = max(0, min(10, int(round(10 * self.base_hp / self.base_max))))
        left = len(self.zombies) + len(self.queue)
        self.status.setText(f"WAVE {self.wave}   Zombies {left}   Beacon [{'#' * filled}{'-' * (10 - filled)}] "
                            f"{max(0, int(self.base_hp))}   HP {max(0, int(self.hp))}   "
                            f"Bombs {self.bombs}   Turrets {len(self.turrets)}/{MAX_TURRETS}   Score {self.score}   Perks {sum(self.perks.values())}"
                            + self._hud_extra())
        if self.state == "wave":
            left = self.wave_limit() - self.wave_t
            msg = f"Wave ends in {int(left) + 1}s - remaining zombies will be removed" if left < 15 else ""
            if self.shop_near() and not self.shop_open:
                msg = (msg + "   " if msg else "") + "SHOP: press E"
            self.info.setText(msg)
