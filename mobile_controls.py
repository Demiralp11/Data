"""
mobile_controls.py  --  on-screen touch controls for firstperson.py

Layout
    left  42% of the screen : floating move stick (appears where the thumb lands)
    right 58% of the screen : drag anywhere to look
    buttons (right side)    : JUMP, RUN (hold), DIG, FILL (hold to repeat), DOWN (fly only)
    top right               : FLY toggle, fullscreen

Two backends, picked automatically:
  * browser (pygbag / pyodide): real multi-touch HTML overlay. Stick and look work at the same
    time. The page state is exposed as plain window.tc* numbers and polled each frame.
  * native (desktop / Android): a Panda3D overlay. It is driven by the mouse pointer, so only
    one finger at a time, but it lets you test the layout on a desktop with  --touch.

    touch = TouchControls(base)            # auto: on for touch devices only
    touch = TouchControls(base, mode=True) # force on (desktop testing)
    touch.active                           # bool
    touch.poll() -> TouchState             # call once per frame (FirstPersonController does)
    touch.set_fly(bool)                    # keep the FLY / DOWN buttons in sync

Browser override: append ?touch=1 or ?touch=0 to the page URL.
"""
import math
import sys

from panda3d.core import (CardMaker, MouseButton, PNMImage, TextNode, Texture,
                          TransparencyAttrib)


class TouchState:
    """One frame of input. Sticks are -1..1 (move_y +1 = forward), look is in pixels."""
    __slots__ = ("move_x", "move_y", "look_dx", "look_dy",
                 "jump", "sprint", "down", "dig", "fill", "fly_taps")

    def __init__(self):
        self.move_x = self.move_y = 0.0
        self.look_dx = self.look_dy = 0.0
        self.jump = self.sprint = self.down = self.dig = self.fill = False
        self.fly_taps = 0


# =============================================================================
# Browser backend
# =============================================================================
JS_DETECT = r"""
(function () {
  var q = /[?&]touch=([01])/.exec(location.search);
  var t = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
  window.tcIsTouch = q ? parseInt(q[1], 10) : (t ? 1 : 0);
})();
"""

JS_TOUCH_UI = r"""
(function () {
  if (window.tcInit) return;
  window.tcInit = true;
  var S = window;
  S.tcMX = 0; S.tcMY = 0; S.tcLX = 0; S.tcLY = 0;
  S.tcJump = 0; S.tcSprint = 0; S.tcDown = 0; S.tcDig = 0; S.tcFill = 0; S.tcFlyN = 0;

  var css = document.createElement('style');
  css.textContent =
    '#tc-root{position:fixed;left:0;top:0;width:100%;height:100%;pointer-events:none;z-index:99999;' +
      'touch-action:none;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none;' +
      '-webkit-tap-highlight-color:transparent;font-family:system-ui,-apple-system,sans-serif}' +
    '.tc-zone{position:absolute;top:0;bottom:0;pointer-events:auto;touch-action:none}' +
    '#tc-left{left:0;width:42%}#tc-right{right:0;width:58%}' +
    '.tc-ring{position:absolute;width:128px;height:128px;border-radius:50%;' +
      'background:rgba(255,255,255,.10);border:2px solid rgba(255,255,255,.35)}' +
    '#tc-base{display:none}' +
    '#tc-hint{left:calc(36px + env(safe-area-inset-left));bottom:calc(44px + env(safe-area-inset-bottom));opacity:.55}' +
    '.tc-knob{position:absolute;left:36px;top:36px;width:56px;height:56px;border-radius:50%;' +
      'background:rgba(255,255,255,.5)}' +
    '.tc-btn{position:absolute;pointer-events:auto;touch-action:none;display:flex;align-items:center;' +
      'justify-content:center;border-radius:50%;background:rgba(0,0,0,.38);' +
      'border:2px solid rgba(255,255,255,.55);color:#fff;font-weight:700;font-size:13px;letter-spacing:.5px}' +
    '.tc-btn.on{background:rgba(255,255,255,.5);color:#000}' +
    '.tc-btn.active{border-color:#7cf;color:#7cf}';
  document.head.appendChild(css);

  var root = document.createElement('div');
  root.id = 'tc-root';
  document.body.appendChild(root);
  root.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  document.addEventListener('gesturestart', function (e) { e.preventDefault(); });
  root.addEventListener('touchmove', function (e) { e.preventDefault(); }, { passive: false });

  function el(cls, id, parent) {
    var d = document.createElement('div');
    d.className = cls; if (id) d.id = id;
    (parent || root).appendChild(d);
    return d;
  }
  var left = el('tc-zone', 'tc-left');
  var right = el('tc-zone', 'tc-right');
  var hint = el('tc-ring', 'tc-hint');
  var base = el('tc-ring', 'tc-base');
  var knob = el('tc-knob', null, base);
  el('tc-knob', null, hint);

  // ---- move stick (floating) -------------------------------------------------
  var R = 64, jid = null, jx = 0, jy = 0;
  left.addEventListener('pointerdown', function (e) {
    if (jid !== null) return;
    jid = e.pointerId; left.setPointerCapture(jid);
    jx = e.clientX; jy = e.clientY;
    base.style.left = (jx - R) + 'px'; base.style.top = (jy - R) + 'px';
    base.style.display = 'block'; hint.style.visibility = 'hidden';
    knob.style.transform = 'translate(0px,0px)';
    e.preventDefault();
  });
  left.addEventListener('pointermove', function (e) {
    if (e.pointerId !== jid) return;
    var dx = e.clientX - jx, dy = e.clientY - jy, l = Math.hypot(dx, dy);
    if (l > R) { dx *= R / l; dy *= R / l; l = R; }
    knob.style.transform = 'translate(' + dx + 'px,' + dy + 'px)';
    if (l / R < 0.12) { S.tcMX = 0; S.tcMY = 0; return; }
    S.tcMX = dx / R; S.tcMY = -dy / R;
  });
  function jend(e) {
    if (e.pointerId !== jid) return;
    jid = null; S.tcMX = 0; S.tcMY = 0;
    base.style.display = 'none'; hint.style.visibility = 'visible';
  }
  left.addEventListener('pointerup', jend);
  left.addEventListener('pointercancel', jend);
  left.addEventListener('lostpointercapture', jend);

  // ---- look pad --------------------------------------------------------------
  var lid = null, lx = 0, ly = 0;
  right.addEventListener('pointerdown', function (e) {
    if (lid !== null) return;
    lid = e.pointerId; right.setPointerCapture(lid);
    lx = e.clientX; ly = e.clientY; e.preventDefault();
  });
  right.addEventListener('pointermove', function (e) {
    if (e.pointerId !== lid) return;
    S.tcLX += e.clientX - lx; S.tcLY += e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
  });
  function lend(e) { if (e.pointerId === lid) lid = null; }
  right.addEventListener('pointerup', lend);
  right.addEventListener('pointercancel', lend);
  right.addEventListener('lostpointercapture', lend);

  // ---- buttons ---------------------------------------------------------------
  function btn(label, css, key, tap) {
    var b = el('tc-btn'); b.textContent = label; b.style.cssText += css;
    var id = null;
    b.addEventListener('pointerdown', function (e) {
      if (id !== null) return;
      id = e.pointerId; b.setPointerCapture(id); b.classList.add('on');
      if (tap) { if (key) S[key]++; } else { S[key] = 1; }
      e.preventDefault();
    });
    function up(e) {
      if (e.pointerId !== id) return;
      id = null; b.classList.remove('on');
      if (!tap && key) S[key] = 0;
    }
    b.addEventListener('pointerup', up);
    b.addEventListener('pointercancel', up);
    b.addEventListener('lostpointercapture', up);
    return b;
  }
  var SR = 'env(safe-area-inset-right)', SB = 'env(safe-area-inset-bottom)', ST = 'env(safe-area-inset-top)';
  var jump = btn('JUMP', 'right:calc(28px + ' + SR + ');bottom:calc(40px + ' + SB + ');width:88px;height:88px;font-size:15px', 'tcJump', false);
  btn('RUN',  'right:calc(124px + ' + SR + ');bottom:calc(30px + ' + SB + ');width:62px;height:62px', 'tcSprint', false);
  btn('DIG',  'right:calc(28px + ' + SR + ');bottom:calc(150px + ' + SB + ');width:62px;height:62px', 'tcDig', false);
  btn('FILL', 'right:calc(104px + ' + SR + ');bottom:calc(130px + ' + SB + ');width:62px;height:62px', 'tcFill', false);
  var down = btn('DOWN', 'right:calc(196px + ' + SR + ');bottom:calc(56px + ' + SB + ');width:62px;height:62px;display:none', 'tcDown', false);
  var fly = btn('FLY', 'right:calc(14px + ' + SR + ');top:calc(14px + ' + ST + ');width:50px;height:50px;font-size:12px', 'tcFlyN', true);

  S.tcSetFly = function (on) {
    fly.classList.toggle('active', !!on);
    down.style.display = on ? 'flex' : 'none';
    jump.textContent = on ? 'UP' : 'JUMP';
    if (!on) S.tcDown = 0;
  };

  // ---- fullscreen (not available on iPhone Safari) ---------------------------
  var de = document.documentElement;
  var rfs = de.requestFullscreen || de.webkitRequestFullscreen;
  if (rfs) {
    var fs = btn('\u26F6', 'right:calc(74px + ' + SR + ');top:calc(14px + ' + ST + ');width:50px;height:50px;font-size:20px', null, true);
    fs.addEventListener('click', function () {
      try { if (document.fullscreenElement || document.webkitFullscreenElement) {
              (document.exitFullscreen || document.webkitExitFullscreen).call(document);
            } else { rfs.call(de); } } catch (err) {}
    });
  }
})();
"""


def _js_window():
    try:                                                    # pygbag
        import platform
        return platform.window
    except Exception:
        pass
    try:                                                    # pyodide
        import js
        return js.window
    except Exception:
        return None


class WebTouchBackend:
    def __init__(self, win):
        self.w = win
        self._flyn = 0
        self._look = (0.0, 0.0)
        win.eval(JS_TOUCH_UI)

    def poll(self):
        w, s = self.w, TouchState()
        try:
            s.move_x, s.move_y = float(w.tcMX), float(w.tcMY)
            s.look_dx, s.look_dy = float(w.tcLX), float(w.tcLY)
            s.jump, s.sprint, s.down = bool(w.tcJump), bool(w.tcSprint), bool(w.tcDown)
            s.dig, s.fill = bool(w.tcDig), bool(w.tcFill)
            n = int(w.tcFlyN)
            s.fly_taps, self._flyn = n - self._flyn, n
            if s.look_dx or s.look_dy:
                w.eval("window.tcLX = 0; window.tcLY = 0;")
        except Exception as e:                                          # pragma: no cover
            print("touch polling failed:", e)
        return s

    def set_fly(self, on):
        try:
            self.w.eval("window.tcSetFly && window.tcSetFly(%d);" % (1 if on else 0))
        except Exception:                                               # pragma: no cover
            pass


# =============================================================================
# Native backend (Panda3D overlay, single pointer)
# =============================================================================
def _disk_texture(size=64):
    img = PNMImage(size, size, 4)
    c = (size - 1) / 2.0
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - c, y - c) / c
            a = 0.0 if d > 1.0 else (1.0 if d < 0.93 else 0.55)
            img.set_xel_a(x, y, 1.0, 1.0, 1.0, a)
    tex = Texture("touch-disk")
    tex.load(img)
    return tex


class PandaTouchBackend:
    STICK_R = 0.22
    # name: (offset from right edge, y, radius, label, tap?)  -- aspect2d units
    BUTTONS = {
        "jump":   (0.30, -0.62, 0.17, "JUMP", False),
        "sprint": (0.58, -0.80, 0.11, "RUN", False),
        "dig":    (0.30, -0.30, 0.11, "DIG", False),
        "fill":   (0.55, -0.40, 0.11, "FILL", False),
        "down":   (0.80, -0.70, 0.11, "DOWN", False),
        "fly":    (0.14, 0.86, 0.09, "FLY", True),
    }

    def __init__(self, base):
        self.base = base
        self.tex = _disk_texture()
        self.root = base.aspect2d.attach_new_node("touch-ui")
        self.nodes, self.labels = {}, {}
        self.held = {n: False for n in self.BUTTONS}
        self.fly_taps = 0
        self.fly_on = False
        self.mode = None                  # None | "stick" | "look" | ("btn", name)
        self.origin = (0.0, 0.0)
        self.stick = (0.0, 0.0)
        self.last = (0.0, 0.0)
        self.look = [0.0, 0.0]
        self._aspect = None

        self.hint = self._disk(self.STICK_R, (1, 1, 1, 0.12))
        self.base_disk = self._disk(self.STICK_R, (1, 1, 1, 0.18))
        self.knob = self._disk(0.09, (1, 1, 1, 0.5))
        self.base_disk.hide()
        self.knob.hide()
        for name, (_, _, r, label, _) in self.BUTTONS.items():
            n = self._disk(r, (0, 0, 0, 0.4))
            t = TextNode("lbl")
            t.set_text(label)
            t.set_align(TextNode.ACenter)
            t.set_text_color(1, 1, 1, 1)
            tn = n.attach_new_node(t)
            tn.set_scale(r * 0.85)
            tn.set_pos(0, 0, -r * 0.18)
            self.nodes[name] = n
            self.labels[name] = t
        self.nodes["down"].hide()
        self._layout()

    def _disk(self, radius, rgba):
        cm = CardMaker("disk")
        cm.set_frame(-radius, radius, -radius, radius)
        n = self.root.attach_new_node(cm.generate())
        n.set_texture(self.tex)
        n.set_transparency(TransparencyAttrib.M_alpha)
        n.set_color(*rgba)
        return n

    def _layout(self):
        a = self.base.get_aspect_ratio()
        self._aspect = a
        self.hint.set_pos(-a + 0.42, 0, -0.62)
        for name, (ox, y, _, _, _) in self.BUTTONS.items():
            self.nodes[name].set_pos(a - ox, 0, y)

    def _pointer(self):
        mw = self.base.mouseWatcherNode
        if not mw.has_mouse():
            return None
        m = mw.get_mouse()
        return m.x * self._aspect, m.y

    def _hit_button(self, ax, ay):
        for name, (ox, y, r, _, _) in self.BUTTONS.items():
            if name == "down" and not self.fly_on:
                continue
            if math.hypot(ax - (self._aspect - ox), ay - y) <= r * 1.15:
                return name
        return None

    def _press(self, p):
        ax, ay = p
        name = self._hit_button(ax, ay)
        if name is not None:
            self.mode = ("btn", name)
            if self.BUTTONS[name][4]:
                self.fly_taps += 1
            else:
                self.held[name] = True
            self.nodes[name].set_color(1, 1, 1, 0.6)
        elif ax < -0.16 * self._aspect:
            self.mode, self.origin, self.stick = "stick", (ax, ay), (0.0, 0.0)
            self.base_disk.set_pos(ax, 0, ay)
            self.knob.set_pos(ax, 0, ay)
            self.base_disk.show()
            self.knob.show()
            self.hint.hide()
        else:
            self.mode = "look"
        self.last = p

    def _release(self):
        if isinstance(self.mode, tuple):
            name = self.mode[1]
            self.held[name] = False
            self.nodes[name].set_color(0, 0, 0, 0.4)
        elif self.mode == "stick":
            self.stick = (0.0, 0.0)
            self.base_disk.hide()
            self.knob.hide()
            self.hint.show()
        self.mode = None

    def poll(self):
        if abs(self.base.get_aspect_ratio() - self._aspect) > 1e-3:
            self._layout()
        mw = self.base.mouseWatcherNode
        down = mw.has_mouse() and mw.is_button_down(MouseButton.one())
        p = self._pointer() if down else None
        if p is not None and self.mode is None:
            self._press(p)
        elif p is None and self.mode is not None:
            self._release()
        elif p is not None:
            if self.mode == "stick":
                dx, dy = p[0] - self.origin[0], p[1] - self.origin[1]
                l = math.hypot(dx, dy)
                if l > self.STICK_R:
                    dx, dy, l = dx * self.STICK_R / l, dy * self.STICK_R / l, self.STICK_R
                self.knob.set_pos(self.origin[0] + dx, 0, self.origin[1] + dy)
                self.stick = (0.0, 0.0) if l / self.STICK_R < 0.12 else (dx / self.STICK_R, dy / self.STICK_R)
            elif self.mode == "look":
                half_w = self.base.win.get_x_size() / 2.0
                self.look[0] += (p[0] - self.last[0]) / self._aspect * half_w
                self.look[1] -= (p[1] - self.last[1]) * self.base.win.get_y_size() / 2.0
            self.last = p

        s = TouchState()
        s.move_x, s.move_y = self.stick
        s.look_dx, s.look_dy = self.look
        self.look = [0.0, 0.0]
        h = self.held
        s.jump, s.sprint, s.down, s.dig, s.fill = h["jump"], h["sprint"], h["down"], h["dig"], h["fill"]
        s.fly_taps, self.fly_taps = self.fly_taps, 0
        return s

    def set_fly(self, on):
        self.fly_on = bool(on)
        self.labels["jump"].set_text("UP" if on else "JUMP")
        self.labels["fly"].set_text_color(*((0.5, 0.8, 1, 1) if on else (1, 1, 1, 1)))
        if on:
            self.nodes["down"].show()
        else:
            self.nodes["down"].hide()
            self.held["down"] = False


# =============================================================================
# Facade
# =============================================================================
class TouchControls:
    def __init__(self, base, mode=None):
        """mode: None = auto (touch devices only), True = force on, False = off."""
        self.backend = None
        self.active = False
        if mode is False:
            return
        if sys.platform == "emscripten":
            win = _js_window()
            if win is None:
                return
            try:
                win.eval(JS_DETECT)
                on = bool(int(win.tcIsTouch))
            except Exception as e:                                      # pragma: no cover
                print("touch detection failed:", e)
                on = False
            if mode is True or on:
                self.backend = WebTouchBackend(win)
        else:
            on = hasattr(sys, "getandroidapilevel")
            if mode is True or on:
                self.backend = PandaTouchBackend(base)
        self.active = self.backend is not None

    def poll(self):
        return self.backend.poll() if self.active else TouchState()

    def set_fly(self, on):
        if self.active:
            self.backend.set_fly(on)
