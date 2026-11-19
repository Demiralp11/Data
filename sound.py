"""
Sound for the terrain toy.

BROWSER (pygbag): every sound is SYNTHESIZED by the browser's WebAudio API from small recipes below,
so the game has sound with no audio files at all.

REPLACING A SOUND: drop a file named after the sound into an `sfx` folder next to main.py:

        sfx/shoot.ogg       sfx/explosion.ogg       sfx/rotor.ogg  ...

The file is used instead of the generated sound (.ogg is best for the web; .wav and .mp3 also work).
Anything you do not replace keeps its generated sound.  Run `python sound.py` to print every sound name.
Loops (rotor, rotor2, ufo) are looped automatically, so give them a seamless clip.

DESKTOP: the generated sounds only exist in the browser build; on desktop a sound plays only if you provide
a file for it (played through Panda3D), everything else is silent.

Game code calls:   snd.play("explosion", pos=Vec3(...), vol=1.0)      snd.loop("rotor", True, 0.5, pos)
`pos` makes the sound quieter with distance and pans it left/right relative to the camera.
"""
import base64
import json
import os
import sys
import time

WEB = sys.platform == "emscripten"
try:
    SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfx")
except NameError:
    SFX_DIR = "sfx"
EXTS = (".ogg", ".wav", ".mp3")
MAX_DIST = 130.0                       # farther than this a positional sound is silent

# one-shot sounds: name -> minimum seconds between plays (stops machine guns from becoming a buzz)
SOUNDS = {
    "shoot": 0.05, "turret": 0.07, "mg": 0.08, "throw": 0.15,
    "explosion": 0.10, "blast": 0.08, "slam": 0.30, "missile": 0.20, "plasma": 0.30,
    "spit": 0.15, "acid": 0.10, "hit": 0.08, "hurt": 0.15, "zombie_die": 0.05, "growl": 0.50,
    "beep": 0.10, "deny": 0.20, "gunship": 1.00, "wave_start": 1.00, "wave_clear": 1.00,
    "boss": 2.00, "beam": 0.40, "lock": 0.40,
}
LOOPS = ("rotor", "rotor2", "ufo")      # continuous sounds (rotor2 = evacuation helicopter)

# ---------------------------------------------------------------------------------------------------------
# The synthesizer.  Each recipe schedules oscillators / filtered noise on a gain bus `o`.
#   tone(o, type, startHz, endHz, seconds, volume, delay)       noise(o, seconds, volume, filter, f0, f1, delay, Q)
# Edit a recipe to change its sound, or add a new name here AND to SOUNDS above.
# ---------------------------------------------------------------------------------------------------------
JS_SFX = r"""
(function () {
  if (window.sfx) { return; }
  var S = window.sfx = { ctx: null, master: null, noiseBuf: null, buffers: {}, last: {}, loops: {},
                         voices: 0, muted: false, vol: 0.6, gap: {} };

  S.ensure = function () {
    if (!S.ctx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { return false; }
      S.ctx = new AC();
      S.master = S.ctx.createGain();
      S.master.gain.value = S.muted ? 0 : S.vol;
      S.master.connect(S.ctx.destination);
      var n = S.ctx.sampleRate, b = S.ctx.createBuffer(1, n, S.ctx.sampleRate), d = b.getChannelData(0);
      for (var i = 0; i < n; i++) { d[i] = Math.random() * 2 - 1; }
      S.noiseBuf = b;
    }
    if (S.ctx.state === 'suspended') { S.ctx.resume(); }
    return true;
  };
  ['mousedown', 'keydown', 'touchstart'].forEach(function (ev) {
    document.addEventListener(ev, function () { S.ensure(); });
  });

  function bus(vol, pan) {
    var g = S.ctx.createGain(); g.gain.value = vol;
    if (S.ctx.createStereoPanner) {
      var p = S.ctx.createStereoPanner(); p.pan.value = pan; g.connect(p); p.connect(S.master);
    } else { g.connect(S.master); }
    return g;
  }
  function tone(o, type, f0, f1, dur, vol, delay) {
    var t = S.ctx.currentTime + (delay || 0), osc = S.ctx.createOscillator(), g = S.ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(f0, t);
    osc.frequency.exponentialRampToValueAtTime(Math.max(f1, 1), t + dur);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + Math.min(0.012, dur * 0.25));
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.connect(g); g.connect(o); osc.start(t); osc.stop(t + dur + 0.05);
  }
  function noise(o, dur, vol, ftype, f0, f1, delay, q) {
    var t = S.ctx.currentTime + (delay || 0), src = S.ctx.createBufferSource(),
        f = S.ctx.createBiquadFilter(), g = S.ctx.createGain();
    src.buffer = S.noiseBuf; src.loop = true;
    f.type = ftype; f.Q.value = q || 0.7;
    f.frequency.setValueAtTime(f0, t);
    f.frequency.exponentialRampToValueAtTime(Math.max(f1, 20), t + dur);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f); f.connect(g); g.connect(o); src.start(t); src.stop(t + dur + 0.05);
  }

  var R = {
    shoot:      function (o) { noise(o, 0.09, 0.9, 'highpass', 2500, 1200); tone(o, 'square', 260, 50, 0.12, 0.35); },
    turret:     function (o) { noise(o, 0.05, 0.5, 'highpass', 2000, 1500); tone(o, 'square', 520, 160, 0.07, 0.22); },
    mg:         function (o) { noise(o, 0.04, 0.35, 'highpass', 3000, 2000); tone(o, 'sawtooth', 420, 200, 0.05, 0.15); },
    throw:      function (o) { noise(o, 0.22, 0.35, 'bandpass', 400, 1400, 0, 2); },
    explosion:  function (o) { noise(o, 0.9, 1.0, 'lowpass', 1800, 60); tone(o, 'sine', 110, 28, 0.7, 0.9); },
    blast:      function (o) { noise(o, 0.5, 0.7, 'lowpass', 1500, 100); tone(o, 'sine', 90, 35, 0.4, 0.6); },
    slam:       function (o) { tone(o, 'sine', 80, 22, 0.6, 1.0); noise(o, 0.45, 0.7, 'lowpass', 500, 50); },
    missile:    function (o) { noise(o, 0.5, 0.4, 'bandpass', 600, 1800, 0, 1.5); tone(o, 'sawtooth', 120, 300, 0.5, 0.12); },
    plasma:     function (o) { tone(o, 'sine', 300, 900, 0.35, 0.3); tone(o, 'triangle', 600, 200, 0.35, 0.2, 0.05); },
    spit:       function (o) { tone(o, 'sine', 300, 800, 0.18, 0.3); noise(o, 0.1, 0.25, 'bandpass', 900, 500, 0, 3); },
    acid:       function (o) { noise(o, 0.25, 0.6, 'bandpass', 900, 300, 0, 4); tone(o, 'sine', 600, 150, 0.18, 0.25); },
    hit:        function (o) { tone(o, 'square', 180, 70, 0.1, 0.35); noise(o, 0.05, 0.4, 'lowpass', 2000, 500); },
    hurt:       function (o) { tone(o, 'sawtooth', 220, 60, 0.25, 0.5); },
    zombie_die: function (o) { tone(o, 'sawtooth', 140, 45, 0.3, 0.3); noise(o, 0.18, 0.35, 'lowpass', 900, 200); },
    growl:      function (o) { tone(o, 'sawtooth', 70, 45, 0.8, 0.5); noise(o, 0.7, 0.3, 'lowpass', 400, 120); },
    beep:       function (o) { tone(o, 'sine', 880, 880, 0.1, 0.3); tone(o, 'sine', 1320, 1320, 0.12, 0.3, 0.09); },
    deny:       function (o) { tone(o, 'square', 140, 110, 0.18, 0.3); },
    gunship:    function (o) { noise(o, 0.9, 0.4, 'bandpass', 300, 900, 0, 1.5); tone(o, 'sawtooth', 90, 140, 0.9, 0.15); },
    wave_start: function (o) { tone(o, 'sawtooth', 98, 98, 0.7, 0.3); tone(o, 'sawtooth', 147, 147, 0.7, 0.25); },
    wave_clear: function (o) { [523, 659, 784, 1047].forEach(function (f, i) { tone(o, 'sine', f, f, 0.22, 0.28, i * 0.11); }); },
    boss:       function (o) { tone(o, 'sawtooth', 60, 40, 1.4, 0.6); noise(o, 1.2, 0.4, 'lowpass', 300, 80); },
    beam:       function (o) { tone(o, 'sawtooth', 900, 300, 0.4, 0.25); tone(o, 'square', 1800, 600, 0.3, 0.1); },
    lock:       function (o) { tone(o, 'sine', 1200, 1200, 0.15, 0.2); tone(o, 'sine', 1200, 1200, 0.15, 0.2, 0.25); tone(o, 'sine', 1200, 1200, 0.15, 0.2, 0.5); }
  };

  var LOOPS = {
    rotor: function (o) {
      var src = S.ctx.createBufferSource(), f = S.ctx.createBiquadFilter(), g = S.ctx.createGain(),
          lfo = S.ctx.createOscillator(), lg = S.ctx.createGain();
      src.buffer = S.noiseBuf; src.loop = true; f.type = 'bandpass'; f.frequency.value = 260; f.Q.value = 1.2;
      g.gain.value = 0.5; lfo.frequency.value = 17; lg.gain.value = 0.45; lfo.connect(lg); lg.connect(g.gain);
      src.connect(f); f.connect(g); g.connect(o); src.start(); lfo.start();
      return [src, lfo];
    },
    ufo: function (o) {
      var a = S.ctx.createOscillator(), b = S.ctx.createOscillator(), g = S.ctx.createGain(),
          lfo = S.ctx.createOscillator(), lg = S.ctx.createGain();
      a.type = 'sine'; a.frequency.value = 82; b.type = 'sine'; b.frequency.value = 86;
      g.gain.value = 0.5; lfo.frequency.value = 5; lg.gain.value = 0.2; lfo.connect(lg); lg.connect(g.gain);
      a.connect(g); b.connect(g); g.connect(o); a.start(); b.start(); lfo.start();
      return [a, b, lfo];
    }
  };
  LOOPS.rotor2 = LOOPS.rotor;

  S.play = function (name, vol, pan) {
    if (S.muted || !S.ensure() || S.ctx.state !== 'running') { return; }
    var now = S.ctx.currentTime, gap = S.gap[name] || 0.03;
    if (S.last[name] !== undefined && now - S.last[name] < gap) { return; }
    S.last[name] = now;
    if (S.voices > 30) { return; }
    var o = bus(vol, pan || 0);
    if (S.buffers[name]) {
      var src = S.ctx.createBufferSource(); src.buffer = S.buffers[name]; src.connect(o);
      S.voices++; src.onended = function () { S.voices--; }; src.start();
    } else if (R[name]) {
      R[name](o); S.voices++; setTimeout(function () { S.voices--; }, 1500);
    }
  };
  S.loop = function (name, on, vol) {
    if (!S.ensure() || S.ctx.state !== 'running') { return; }
    var L = S.loops[name];
    if (on) {
      if (!L) {
        var o = bus(0.0001, 0), nodes;
        if (S.buffers[name]) {
          var s = S.ctx.createBufferSource(); s.buffer = S.buffers[name]; s.loop = true; s.connect(o); s.start(); nodes = [s];
        } else if (LOOPS[name]) { nodes = LOOPS[name](o); } else { return; }
        L = S.loops[name] = { o: o, nodes: nodes };
      }
      L.o.gain.setTargetAtTime(S.muted ? 0 : vol, S.ctx.currentTime, 0.08);
    } else if (L) {
      L.o.gain.setTargetAtTime(0.0001, S.ctx.currentTime, 0.1);
      var old = L.nodes; delete S.loops[name];
      setTimeout(function () { old.forEach(function (n) { try { n.stop(); } catch (e) {} }); }, 600);
    }
  };
  S.stopAll = function () { for (var k in S.loops) { S.loop(k, false, 0); } };
  S.setVol = function (v) { S.vol = v; if (S.master) { S.master.gain.value = S.muted ? 0 : v; } };
  S.setMute = function (m) {
    S.muted = !!m;
    if (S.master) { S.master.gain.value = S.muted ? 0 : S.vol; }
    if (S.muted) { S.stopAll(); }
  };
  S.load = function (name, b64) {              // a replacement file sent over from Python
    if (!S.ensure()) { return; }
    var bin = atob(b64), n = bin.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) { u[i] = bin.charCodeAt(i); }
    S.ctx.decodeAudioData(u.buffer, function (buf) { S.buffers[name] = buf; },
                          function () { console.log('sfx: could not decode ' + name); });
  };
})();
"""


class Sound:
    def __init__(self, app, volume=0.6):
        self.app = app
        self.volume = volume
        self.muted = False
        self._win = None            # browser: the JS window
        self._desk = None           # desktop: name -> list of Panda3D sounds
        self._rr = {}
        self._last = {}
        self._loop_state = {}
        self._loop_t = {}
        self.files = self._scan()   # name -> path of replacement files that exist
        if WEB:
            self._init_web()
        else:
            self._init_desktop()

    # ------------------------------------------------------------------ setup
    def _scan(self):
        files = {}
        for name in list(SOUNDS) + list(LOOPS):
            for ext in EXTS:
                p = os.path.join(SFX_DIR, name + ext)
                if os.path.isfile(p):
                    files[name] = p
                    break
        return files

    def _init_web(self):
        try:
            import platform
            self._win = platform.window
            self._win.eval(JS_SFX)
            self._win.eval("window.sfx.gap = %s; window.sfx.setVol(%.2f);" % (json.dumps(SOUNDS), self.volume))
            for name, path in self.files.items():
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                self._win.eval("window.sfx.load('%s', '%s');" % (name, b64))
            if self.files:
                print("sound: using replacement files for", ", ".join(sorted(self.files)))
        except Exception as e:
            print("sound unavailable:", e)
            self._win = None

    def _init_desktop(self):
        self._desk = {}
        loader = getattr(self.app, "loader", None)
        if loader is None or not self.files:
            return
        try:
            from panda3d.core import Filename
            for name, path in self.files.items():
                fn = Filename.from_os_specific(path)
                self._desk[name] = [loader.load_sfx(fn) for _ in range(4 if name in SOUNDS else 1)]
        except Exception as e:
            print("desktop sound unavailable:", e)
            self._desk = {}

    # ------------------------------------------------------------------ helpers
    def _spatial(self, pos):
        """(attenuation 0..1, pan -1..1) for a world position, relative to the camera."""
        from panda3d.core import Vec3
        app = self.app
        cam = app.camera.get_pos(app.render)
        d = Vec3(pos.x - cam.x, pos.y - cam.y, pos.z - cam.z)
        dist = d.length()
        att = max(0.0, 1.0 - dist / MAX_DIST) ** 1.5
        pan = 0.0
        if dist > 1e-3:
            d.normalize()
            right = app.render.get_relative_vector(app.camera, Vec3(1, 0, 0))
            pan = max(-1.0, min(1.0, right.dot(d) * 0.8))
        return att, pan

    # ------------------------------------------------------------------ API
    def play(self, name, pos=None, vol=1.0):
        if self.muted or (self._win is None and not self._desk):
            return
        now = time.perf_counter()
        if now - self._last.get(name, -1.0) < SOUNDS.get(name, 0.03):
            return
        pan = 0.0
        if pos is not None:
            att, pan = self._spatial(pos)
            vol *= att
        if vol < 0.03:
            return
        self._last[name] = now
        if self._win is not None:
            try:
                self._win.eval("window.sfx.play('%s', %.2f, %.2f);" % (name, vol, pan))
            except Exception as e:
                print("sound failed:", e)
                self._win = None
        else:
            lst = self._desk.get(name)
            if lst:
                i = self._rr.get(name, 0)
                self._rr[name] = (i + 1) % len(lst)
                s = lst[i]
                s.set_volume(min(1.0, vol * self.volume))
                s.set_balance(pan)
                s.play()

    def loop(self, name, on, vol=1.0, pos=None):
        if self._win is None and not self._desk:
            return
        if on and pos is not None:
            att, _pan = self._spatial(pos)
            vol *= att
        on = bool(on) and not self.muted
        key = (on, round(vol * 10) / 10.0)          # only talk to the browser when something changed...
        now = time.perf_counter()
        if self._loop_state.get(name) == key and now - self._loop_t.get(name, 0.0) < 1.0:
            return                                  # ...or once a second (the first call may land before the first click)
        self._loop_state[name] = key
        self._loop_t[name] = now
        if self._win is not None:
            try:
                self._win.eval("window.sfx.loop('%s', %s, %.2f);" % (name, "true" if on else "false", key[1]))
            except Exception as e:
                print("sound failed:", e)
                self._win = None
        else:
            lst = self._desk.get(name)
            if lst:
                s = lst[0]
                if on:
                    s.set_loop(True)
                    s.set_volume(min(1.0, key[1] * self.volume))
                    if s.status() != s.PLAYING:
                        s.play()
                else:
                    s.stop()

    def stop_all(self):
        for name in LOOPS:
            self.loop(name, False)
        self._loop_state.clear()

    def toggle_mute(self):
        self.muted = not self.muted
        if self._win is not None:
            try:
                self._win.eval("window.sfx.setMute(%s);" % ("true" if self.muted else "false"))
            except Exception:
                pass
        if self.muted:
            self._loop_state.clear()
            for lst in (self._desk or {}).values():
                for s in lst:
                    s.stop()
        return self.muted


if __name__ == "__main__":
    print("one-shot sounds :", ", ".join(SOUNDS))
    print("looping sounds  :", ", ".join(LOOPS))
    print("replace any of them by putting <name>.ogg in:", SFX_DIR)
