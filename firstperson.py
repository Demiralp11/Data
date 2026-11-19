"""
firstperson.py  --  first-person camera + movement for terrain.py

Two modes (Tab toggles, or the FLY button on mobile):
  walk : gravity, jump, slope limit, step-up, walls, falls into holes you dig
  fly  : free camera, W/S along the view direction, Q/E (or Space) up/down

Input, all merged every frame:
  keyboard      WASD, arrows, Space, Shift, Q/E, F dig, G fill, Tab fly/walk
  mouse         hold left button and drag to look  (browser: click to lock the pointer)
  touch         pass a mobile_controls.TouchControls and it takes over look / move / buttons

Usage
    fp = FirstPersonController(base, chunks, touch=touch_controls)   # touch optional
    fp.spawn(0, 0)
    ...every frame...
    fp.update(dt)
    chunks.update(base.camera.get_pos(base.render))

The controller never needs the terrain *meshes* to exist: it queries the SDF directly,
so you can walk before the first chunks have finished streaming.
"""
import math
import sys

import numpy as np
from panda3d.core import KeyboardButton, MouseButton, Vec3

import terrain

# ---- tuning ---------------------------------------------------------------
EYE_HEIGHT = 1.7
WALK_SPEED = 7.0
SPRINT_MULT = 1.7
FLY_SPEED = 20.0
FLY_SPRINT_MULT = 4.0
GRAVITY = 24.0
JUMP_SPEED = 8.5
TERMINAL_SPEED = 60.0
STEP_UP = 0.6              # ledges up to this high are walked up without jumping
SNAP_DOWN = 0.8            # stay glued to the ground when walking downhill
MAX_SLOPE = math.tan(math.radians(55.0))
EDIT_INTERVAL = 0.18       # seconds between digs/fills while the button is held
EDIT_REACH = 60.0

MOUSE_DRAG_H, MOUSE_DRAG_P = 90.0, 60.0     # degrees per full screen drag
TOUCH_SENS = 0.22                            # degrees per CSS pixel of finger travel
WEB_SENS = 0.12                              # degrees per pixel with browser pointer lock
KEY_TURN_H, KEY_TURN_P = 90.0, 60.0          # arrow keys, degrees per second

JS_POINTER_LOCK = """
window.plx = 0; window.ply = 0; window.plk = false;
document.addEventListener('pointerlockchange', function () { window.plk = !!document.pointerLockElement; });
document.addEventListener('mousemove', function (e) {
  if (document.pointerLockElement) { window.plx += e.movementX; window.ply += e.movementY; }
});
document.addEventListener('mousedown', function (e) {
  var c = document.getElementById('canvas') || document.querySelector('canvas');
  if (c && e.target === c && !document.pointerLockElement) { c.requestPointerLock(); }
});
"""


def _key(ch):
    try:
        return KeyboardButton.ascii_key(ch)
    except TypeError:                       # older Panda builds want bytes
        return KeyboardButton.ascii_key(ch.encode())


class WebPointerLock:
    """Browser only: lock the pointer to the canvas so the mouse can't leave the page."""

    def __init__(self):
        self.win = None
        self.locked = False
        if sys.platform != "emscripten":
            return
        try:
            import platform
            self.win = platform.window
            self.win.eval(JS_POINTER_LOCK)
        except Exception as e:                           # pragma: no cover
            print("pointer lock unavailable:", e)
            self.win = None

    def poll(self):
        """(dx, dy) pixels since last call; (0, 0) unless the pointer is locked."""
        if self.win is None:
            return 0.0, 0.0
        try:
            w = self.win
            self.locked = bool(w.plk)
            dx, dy = float(w.plx), float(w.ply)
            if dx or dy:
                w.eval("window.plx = 0; window.ply = 0;")
            return (dx, dy) if self.locked else (0.0, 0.0)
        except Exception as e:                           # pragma: no cover
            print("pointer lock polling failed:", e)
            self.win = None
            self.locked = False
            return 0.0, 0.0


class FirstPersonController:
    def __init__(self, base, chunks, touch=None):
        self.base = base
        self.chunks = chunks
        self.touch = touch
        self.heading, self.pitch = 0.0, -10.0
        self.pos = np.array([0.0, 0.0, 50.0], np.float64)      # feet position
        self.vz = 0.0
        self.grounded = False
        self.flying = False
        self.spawn_point = (0.0, 0.0)
        self._edit_cd = 0.0
        self._last_mouse = None
        self._fly_taps_seen = 0

        self.k = {n: _key(c) for n, c in (("w", "w"), ("a", "a"), ("s", "s"), ("d", "d"),
                                          ("q", "q"), ("e", "e"), ("f", "f"), ("g", "g"))}
        self.k["space"] = KeyboardButton.space()
        self.k["shift"] = KeyboardButton.shift()
        self.k["left"] = KeyboardButton.left()
        self.k["right"] = KeyboardButton.right()
        self.k["up"] = KeyboardButton.up()
        self.k["down"] = KeyboardButton.down()

        base.accept("tab", self.toggle_fly)
        self.lock = None if (touch is not None and touch.active) else WebPointerLock()
        self.apply_camera()

    # ------------------------------------------------------------------ api
    @property
    def eye(self):
        return Vec3(self.pos[0], self.pos[1], self.pos[2] + EYE_HEIGHT)

    def spawn(self, x=0.0, y=0.0, heading=0.0):
        self.spawn_point = (float(x), float(y))
        self.pos[:] = (x, y, self.chunks.height_at(x, y) + 2.0)
        self.vz, self.grounded, self.heading = 0.0, False, heading
        self.apply_camera()

    def teleport(self, x, y, z):
        self.pos[:] = (x, y, z)
        self.vz, self.grounded = 0.0, False
        self.apply_camera()

    def toggle_fly(self):
        self.flying = not self.flying
        self.vz, self.grounded = 0.0, False
        if self.touch is not None and self.touch.active:
            self.touch.set_fly(self.flying)

    def edit(self, kind, radius=None):
        """Dig or fill the terrain where the crosshair points."""
        b = self.base
        origin = b.camera.get_pos(b.render)
        fwd = b.render.get_relative_vector(b.camera, Vec3(0, 1, 0))
        fwd.normalize()
        hit = self.chunks.ray_hit(origin, fwd, max_dist=EDIT_REACH)
        if hit is not None:
            self.chunks.edit(kind, hit, radius or terrain.EDIT_RADIUS)

    def apply_camera(self):
        cam = self.base.camera
        cam.set_pos(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]) + EYE_HEIGHT)
        cam.set_hpr(self.heading, self.pitch, 0)

    # --------------------------------------------------------------- update
    def _down(self, name):
        return self.base.mouseWatcherNode.is_button_down(self.k[name])

    def update(self, dt):
        dt = min(dt, 0.05)
        touch = self.touch if (self.touch is not None and self.touch.active) else None
        ts = touch.poll() if touch is not None else None

        # ---- look
        if ts is not None:
            self.heading -= ts.look_dx * TOUCH_SENS
            self.pitch -= ts.look_dy * TOUCH_SENS
            if ts.fly_taps & 1:
                self.toggle_fly()
        else:
            self._mouse_look()
        self.heading += (self._down("left") - self._down("right")) * KEY_TURN_H * dt
        self.pitch += (self._down("up") - self._down("down")) * KEY_TURN_P * dt
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self.heading %= 360.0

        # ---- movement input (keyboard + stick)
        f = float(self._down("w")) - float(self._down("s"))
        r = float(self._down("d")) - float(self._down("a"))
        jump = self._down("space")
        sprint = self._down("shift")
        up = float(self._down("e"))
        dn = float(self._down("q"))
        dig, fill = self._down("f"), self._down("g")
        if ts is not None:
            f += ts.move_y
            r += ts.move_x
            jump = jump or ts.jump
            sprint = sprint or ts.sprint
            dn = max(dn, float(ts.down))
            dig, fill = dig or ts.dig, fill or ts.fill
        n = math.hypot(f, r)
        if n > 1.0:
            f, r = f / n, r / n

        if self.flying:
            self._fly(dt, f, r, up + (1.0 if jump else 0.0) - dn, sprint)
        else:
            self._walk(dt, f, r, jump, sprint)

        # ---- dig / fill
        self._edit_cd -= dt
        if (dig or fill) and self._edit_cd <= 0.0:
            self.edit("dig" if dig else "fill")
            self._edit_cd = EDIT_INTERVAL

        if self.pos[2] < terrain.Z_MIN - 20.0:                       # fell out of the world
            self.spawn(*self.spawn_point, heading=self.heading)
        self.apply_camera()

    def _mouse_look(self):
        if self.lock is not None:
            dx, dy = self.lock.poll()
            if self.lock.locked:
                self.heading -= dx * WEB_SENS
                self.pitch -= dy * WEB_SENS
                self._last_mouse = None
                return
        mw = self.base.mouseWatcherNode
        if mw.has_mouse() and mw.is_button_down(MouseButton.one()):
            m = mw.get_mouse()
            if self._last_mouse is not None:
                self.heading -= (m.x - self._last_mouse[0]) * MOUSE_DRAG_H
                self.pitch += (m.y - self._last_mouse[1]) * MOUSE_DRAG_P
            self._last_mouse = (m.x, m.y)
        else:
            self._last_mouse = None

    # ------------------------------------------------------------------ fly
    def _fly(self, dt, f, r, vert, sprint):
        h, p = math.radians(self.heading), math.radians(self.pitch)
        fwd = np.array([-math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
        right = np.array([math.cos(h), math.sin(h), 0.0])
        move = fwd * f + right * r + np.array([0.0, 0.0, vert])
        self.pos += move * FLY_SPEED * (FLY_SPRINT_MULT if sprint else 1.0) * dt

    # ----------------------------------------------------------------- walk
    def _walk(self, dt, f, r, jump, sprint):
        h = math.radians(self.heading)
        wx = -math.sin(h) * f + math.cos(h) * r
        wy = math.cos(h) * f + math.sin(h) * r
        speed = WALK_SPEED * (SPRINT_MULT if sprint else 1.0)
        dx, dy = wx * speed * dt, wy * speed * dt
        x, y, z = (float(v) for v in self.pos)

        # candidates: full move, slide along x, slide along y, stay put (always last)
        cands = [(x + dx, y + dy), (x + dx, y), (x, y + dy)] if (dx or dy) else []
        cands.append((x, y))
        xs = np.array([c[0] for c in cands], np.float32)
        ys = np.array([c[1] for c in cands], np.float32)
        window_down = SNAP_DOWN if self.grounded else 0.6 + max(0.0, -self.vz) * dt
        ok, g, top = self.chunks.ground_batch(xs, ys, np.full(len(cands), z, np.float32),
                                              up=STEP_UP, down=window_down, with_top=True)

        pick = len(cands) - 1
        for i, (cx, cy) in enumerate(cands[:-1]):
            if top[i]:
                continue                                             # wall taller than a step
            if ok[i] and self.grounded:
                rise = float(g[i]) - z
                if rise > math.hypot(cx - x, cy - y) * MAX_SLOPE + 0.02:
                    continue                                         # too steep to climb
            pick = i
            break
        nx, ny = cands[pick]
        c_ok, c_g = bool(ok[pick]), float(g[pick])

        if pick == len(cands) - 1 and top[pick]:                     # buried (terrain was filled around us)
            ok2, g2 = self.chunks.ground_batch([x], [y], [z], up=6.0, down=0.5)
            if ok2[0]:
                z, self.grounded, self.vz = float(g2[0]), True, 0.0
            self.pos[:] = (nx, ny, z)
            return

        if self.grounded:
            if jump:
                self.vz = JUMP_SPEED
                self.grounded = False
            elif c_ok:
                z = c_g                                              # follow the ground
            else:
                self.grounded, self.vz = False, 0.0                  # walked off a ledge
        if not self.grounded:
            self.vz = max(self.vz - GRAVITY * dt, -TERMINAL_SPEED)
            z_new = z + self.vz * dt
            if c_ok and c_g >= z_new:
                z = c_g
                if self.vz <= 0.0:
                    self.vz, self.grounded = 0.0, True
            else:
                z = z_new
        self.pos[:] = (nx, ny, z)
