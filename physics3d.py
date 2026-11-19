
import ctypes, os, sys
import numpy as np

def _find_shim():
    if sys.platform == "emscripten":
        name = "shim.so"
    elif sys.platform == "win32":
        name = "shim.dll"
    elif sys.platform == "darwin":
        name = "libshim.dylib"
    else:
        name = "libshim.so"

    roots = []

    try:
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass

    roots.append(os.path.dirname(os.path.abspath(sys.executable)))

    roots.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    try:
        from panda3d.core import ExecutionEnvironment
        roots.append(ExecutionEnvironment.getEnvironmentVariable("MAIN_DIR"))
    except Exception:
        pass

    roots.append(os.getcwd())

    tried = []
    for r in roots:
        if not r:
            continue
        for p in (os.path.join(r, "box3d", name), os.path.join(r, name)):
            tried.append(p)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError("Box3D shim not found. Tried:\n  " + "\n  ".join(tried))

_shim_path = _find_shim()
lib = ctypes.CDLL(_shim_path)
lib.shim_create_world.argtypes = [ctypes.c_float] * 3

lib.shim_destroy_world.argtypes = []
lib.shim_step.argtypes = [ctypes.c_float, ctypes.c_int]
lib.shim_add_static_box.argtypes = [ctypes.c_float] * 6
lib.shim_add_static_box.restype = ctypes.c_int
lib.shim_add_dynamic_cube.argtypes = [ctypes.c_float] * 4
lib.shim_add_dynamic_cube.restype = ctypes.c_int
lib.shim_get_position.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

lib.shim_get_rotation.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
lib.shim_set_angular_velocity.argtypes = [ctypes.c_int] + [ctypes.c_float] * 3

f = ctypes.c_float
lib.shim_add_dynamic_sphere.argtypes = [f, f, f, f]
lib.shim_add_dynamic_sphere.restype = ctypes.c_int
lib.shim_add_dynamic_capsule.argtypes = [f, f, f, f, f]
lib.shim_add_dynamic_capsule.restype = ctypes.c_int
lib.shim_add_dynamic_hull.argtypes = [f, f, f, ctypes.POINTER(f), ctypes.c_int]
lib.shim_add_dynamic_hull.restype = ctypes.c_int
lib.shim_add_static_mesh.argtypes = [ctypes.POINTER(f), ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.shim_add_static_mesh.restype = ctypes.c_int

f = ctypes.c_float
lib.shim_add_static_point.argtypes = [f, f, f]
lib.shim_add_static_point.restype = ctypes.c_int
lib.shim_add_spherical_joint.argtypes = [ctypes.c_int, ctypes.c_int, f, f, f]
lib.shim_add_spherical_joint.restype = ctypes.c_int
lib.shim_add_weld_joint.argtypes = [ctypes.c_int, ctypes.c_int, f, f, f]
lib.shim_add_weld_joint.restype = ctypes.c_int
lib.shim_add_revolute_joint.argtypes = [ctypes.c_int, ctypes.c_int, f, f, f, f, f, f]
lib.shim_add_revolute_joint.restype = ctypes.c_int

f, i = ctypes.c_float, ctypes.c_int
lib.shim_add_static_mesh_at.argtypes = [f, f, f, ctypes.POINTER(f), i, ctypes.POINTER(i), i]
lib.shim_add_static_mesh_at.restype = i
lib.shim_remove_body.argtypes = [i]
_raw_remove_body = lib.shim_remove_body

lib.shim_shift_origin.argtypes = [f, f, f]

f, i = ctypes.c_float, ctypes.c_int
PF = ctypes.POINTER(ctypes.c_float)
PI = ctypes.POINTER(ctypes.c_int)

lib.shim_add_dynamic_box.argtypes = [f]*10
lib.shim_add_dynamic_box.restype = i
lib.shim_get_transforms.argtypes = [PI, i, PF]

lib.shim_set_static.argtypes = [i, i]
lib.shim_set_damping.argtypes = [i, f, f]
lib.shim_get_speed.argtypes = [i]
lib.shim_get_speed.restype = f
lib.shim_set_position.argtypes = [i, f, f, f]
lib.shim_set_linear_velocity.argtypes = [i, f, f, f]

lib.shim_teleport.argtypes = [i, f, f, f]

lib.shim_set_enabled.argtypes = [i, i]

lib.shim_spring_to.argtypes = [i, f, f, f, f, f]
lib.shim_set_gravity_scale.argtypes = [i, f]

lib.shim_raycast.argtypes = [f, f, f, f, f, f, PF]
lib.shim_raycast.restype = i
_ray_out = (ctypes.c_float * 3)()
lib.shim_set_filter.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64]
lib.shim_raycast_mask.argtypes = [f, f, f, f, f, f, ctypes.c_uint64, PF]
lib.shim_raycast_mask.restype = i

lib.shim_add_revolute_joint_limited.argtypes = [i, i, f, f, f, f, f, f, f, f]
lib.shim_add_revolute_joint_limited.restype = i
lib.shim_add_spherical_joint_limited.argtypes = [i, i, f, f, f, f, f, f, f, f, f]
lib.shim_add_spherical_joint_limited.restype = i

from panda3d.core import Quat, Vec3

lib.shim_get_velocities.argtypes = [i, PF]
lib.shim_set_velocities.argtypes = [i] + [f] * 6
lib.shim_set_transform.argtypes = [i] + [f] * 7
lib.shim_add_weld_joint_frames.argtypes = [i, i, PF, PF]
lib.shim_add_weld_joint_frames.restype = i

lib.shim_set_material.argtypes = [i, f, f]
lib.shim_set_sleep_threshold.argtypes = [i, f]

_vel_buf = (ctypes.c_float * 6)()


lib.shim_get_speeds.argtypes = [i, PF]
lib.shim_clamp_velocities.argtypes = [i, f, f]
_speed_buf = (ctypes.c_float * 2)()


lib.shim_remove_joint.argtypes = [i]

lib.shim_get_contacts.argtypes = [i, PI, i]
lib.shim_get_contacts.restype = i
_contact_buf = (ctypes.c_int * 32)()

def _routed_remove(h):
    remove_body(h)

lib.shim_remove_body = _routed_remove
def contacts(h):
    """Handles of the bodies currently touching body h."""
    n = lib.shim_get_contacts(h, _contact_buf, 32)
    return list(_contact_buf[:n])

_live = set()
_live_joints = set()
body_nodes = {}
def _track(fn, store):
    def wrapper(*a):
        h = fn(*a)
        if h >= 0:
            store.add(h)
        return h
    return wrapper

for _name in ("shim_add_static_box", "shim_add_dynamic_cube", "shim_add_dynamic_sphere",
              "shim_add_dynamic_capsule", "shim_add_dynamic_hull", "shim_add_static_mesh",
              "shim_add_static_mesh_at", "shim_add_static_point", "shim_add_dynamic_box"):
    setattr(lib, _name, _track(getattr(lib, _name), _live))

for _name in ("shim_add_spherical_joint", "shim_add_weld_joint", "shim_add_revolute_joint",
              "shim_add_weld_joint_frames"):
    setattr(lib, _name, _track(getattr(lib, _name), _live_joints))

def is_alive(h):
    return _world_ready and h is not None and h in _live
from panda3d.core import NodePath

body_info = {}
CAT_RAGDOLL = 8
CAT_WORLD, CAT_PLAYER, CAT_DEBRIS = 1, 2, 4
ALL = 0xFFFFFFFFFFFFFFFF
GROUND_MASK = ALL & ~(CAT_PLAYER | CAT_DEBRIS)


def set_filter(h, category, mask):
    lib.shim_set_filter(h, category, mask)

def node_for_body(h):
    """NodePath for a body: registered one, or a tag-only dummy built from body_info."""
    np = body_nodes.get(h)
    if np is not None:
        return np
    info = body_info.get(h)
    if info is None:
        return None
    np = NodePath(str(info.get("name", "body")))
    for k, v in info.items():
        np.setTag(str(k), str(v).lower() if isinstance(v, bool) else str(v))
    if info.get("climbable"):
        if not np.hasTag("Type"):
            np.setTag("Type", "climbable")
        if not np.hasTag("Direction"):
            np.setTag("Direction", "vertical")
    body_nodes[h] = np
    return np
_pending_remove = []

def remove_body(h):
    if not is_alive(h):
        return False
    _live.discard(h)
    body_nodes.pop(h, None)
    body_info.pop(h, None)
    _pending_remove.append(h)
    return True
def speeds(h):
    lib.shim_get_speeds(h, _speed_buf)
    return _speed_buf[0], _speed_buf[1]

def clamp_velocities(h, max_lin, max_ang):
    lib.shim_clamp_velocities(h, max_lin, max_ang)

def get_velocities(h):
    lib.shim_get_velocities(h, _vel_buf)
    return list(_vel_buf[0:3]), list(_vel_buf[3:6])

def set_velocities(h, lin, ang):
    lib.shim_set_velocities(h, *lin, *ang)

def set_transform(h, pos, quat):
    lib.shim_set_transform(h, pos.x, pos.y, pos.z, quat.getR(), quat.getI(), quat.getJ(), quat.getK())

def set_damping(h, lin, ang):
    lib.shim_set_damping(h, lin, ang)

def frame_from_dict(d):
    """{'pos': [...], 'hpr': [...]} -> 7 floats: px, py, pz, qw, qx, qy, qz."""
    q = Quat()
    q.setHpr(Vec3(*d["hpr"]))
    p = d["pos"]
    return (ctypes.c_float * 7)(p[0], p[1], p[2], q.getR(), q.getI(), q.getJ(), q.getK())

def add_weld(h_a, h_b, frame_a_dict, frame_b_dict):
    return lib.shim_add_weld_joint_frames(h_a, h_b,
                                          frame_from_dict(frame_a_dict),
                                          frame_from_dict(frame_b_dict))

PHYSICS_DT = 1.0 / 60.0
SUBSTEPS = 4
_world_ready = False
_accum = 0.0

origin = [0.0, 0.0, 0.0]

def set_origin(x, y, z):
    origin[0], origin[1], origin[2] = x, y, z

def shift_origin(dx, dy, dz):
    origin[0] += dx
    origin[1] += dy
    origin[2] += dz
    if _world_ready:
        lib.shim_shift_origin(dx, dy, dz)

_ray_out = (ctypes.c_float * 6)()

def raycast(origin, direction, max_distance, mask=ALL & ~CAT_PLAYER):
    """Returns (body_handle, point, normal) or None. All in render space."""
    d = direction.normalized()
    h = lib.shim_raycast_mask(origin.x, origin.y, origin.z,
                              d.x * max_distance, d.y * max_distance, d.z * max_distance,
                              mask, _ray_out)
    if h < 0:
        return None
    return h, tuple(_ray_out[0:3]), tuple(_ray_out[3:6])
lib.shim_wake.argtypes = [i]
def wake(h):
    if _world_ready:
        lib.shim_wake(h)

KILL_Z = -500.0

def sync_nodes(nps):

    nps = [n for n in nps if not n.getPythonTag('disabled')]
    if not _world_ready or not nps:
        return []
    handles = np.array([n.getPythonTag('box3d') for n in nps], dtype=np.int32)
    out = np.empty(len(nps) * 7, dtype=np.float32)
    lib.shim_get_transforms(handles.ctypes.data_as(PI), len(nps), out.ctypes.data_as(PF))
    out = out.reshape(-1, 7)

    doomed = []
    for n, o in zip(nps, out):
        if o[2] < KILL_Z:
            doomed.append(n)
            continue
        n.setPos(float(o[0]), float(o[1]), float(o[2]))
        n.setQuat(Quat(float(o[3]), float(o[4]), float(o[5]), float(o[6])))
    return doomed

_gen = 0

def create_world(gx=0.0, gy=0.0, gz=-10.0):
    global _world_ready, _gen
    if not _world_ready:
        lib.shim_create_world(gx, gy, gz)
        _world_ready = True
        _gen += 1

def destroy_world():
    global _world_ready, _accum
    if _world_ready:
        _world_ready = False
        _live.clear()
        _live_joints.clear()
        _pending_remove.clear()
        lib.shim_destroy_world()
        body_nodes.clear()
        body_info.clear()
    _accum = 0.0
    origin[0] = origin[1] = origin[2] = 0.0





_info = {}

def _first_args(a):
    try:
        return tuple(round(float(x), 1) for x in a[:3])
    except (TypeError, ValueError):
        return None

def _track(fn, store, name="?"):
    def wrapper(*a):
        h = fn(*a)
        if h >= 0:
            store.add(h)
            if store is _live:
                _info[h] = (_gen, name, _first_args(a))
        return h
    return wrapper

for _name in ("shim_add_static_box", "shim_add_dynamic_cube", "shim_add_dynamic_sphere",
              "shim_add_dynamic_capsule", "shim_add_dynamic_hull", "shim_add_static_mesh",
              "shim_add_static_mesh_at", "shim_add_static_point", "shim_add_dynamic_box"):
    setattr(lib, _name, _track(getattr(lib, _name), _live, _name))
post_step_hooks = []

def step(frame_dt):
    global _accum
    if not _world_ready:
        return
    if _pending_remove:
        for h in sorted(set(_pending_remove)):
 
            _raw_remove_body(h)
        _pending_remove.clear()
    _accum += min(frame_dt, 0.1)
    steps = 0
    while _accum >= PHYSICS_DT:
        lib.shim_step(PHYSICS_DT, SUBSTEPS)
        _accum -= PHYSICS_DT
        steps += 1
    if steps:
        for fn in post_step_hooks:
            fn(steps * PHYSICS_DT)


def add_static_mesh_indexed(pos_abs, verts, idx):
    """Static indexed triangle mesh.
    pos_abs : absolute (un-shifted) world position of the body origin (x, y, z)
    verts   : (N,3) float32, C-contiguous, LOCAL to pos_abs
    idx     : flat int32, 3 per triangle
    Returns the body handle or None. Caller must keep verts/idx alive (cheap insurance)."""
    if not _world_ready:
        return None
    px, py, pz = pos_abs[0] - origin[0], pos_abs[1] - origin[1], pos_abs[2] - origin[2]
    h = lib.shim_add_static_mesh_at(
        px, py, pz,
        verts.ctypes.data_as(PF), len(verts),
        idx.ctypes.data_as(PI), len(idx) // 3)
    return h if h >= 0 else None
def rebuild_chunk_collision(chunk, flat_verts):

    remove_chunk_collision(chunk)
    if flat_verts is None or len(flat_verts) == 0:
        return
    v = np.ascontiguousarray(np.asarray(flat_verts, dtype=np.float32).reshape(-1, 3))
    idx = np.arange(len(v), dtype=np.int32)
    p = chunk.pos
    px, py, pz = p.x - origin[0], p.y - origin[1], p.z - origin[2]
    h = lib.shim_add_static_mesh_at(
        px, py, pz,
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(v),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(v) // 3)
    chunk.phys_handle = h if h >= 0 else None
    chunk._phys_keep = (v, idx)


def remove_chunk_collision(chunk):
    h = getattr(chunk, "phys_handle", None)
    remove_body(h)
    chunk.phys_handle = None
    chunk._phys_keep = None