# /// script
# dependencies = [
#    "panda3d",
#    "numpy",
# ]
# ///
"""
Demo: streamed SDF terrain + first-person controller + mobile controls.

    python main.py              desktop; touch UI only if the device is a touch device
    python main.py --touch      force the on-screen touch UI (test the layout with the mouse)
    python main.py --view 200   shorter view distance (default 310 desktop / 200 touch)
    browser (pygbag)            touch UI appears automatically on phones/tablets;
                                add ?touch=1 to the URL to force it

Desktop keys: WASD move, Shift sprint, Space jump, mouse drag look, F dig, G fill, Tab fly/walk,
Q/E down/up while flying, Esc quit.
"""
import sys

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode, loadPrcFileData

import terrain
from firstperson import FirstPersonController
from mobile_controls import TouchControls

SKY = (0.55, 0.70, 0.85)


class App(ShowBase):
    def __init__(self, touch=None, view=None):
        super().__init__()
        self.disableMouse()
        self.set_background_color(*SKY)

        self.touch = TouchControls(self, mode=touch)
        if view is None:
            view = 200 if self.touch.active else 310
        terrain.set_view_distance(view)                  # before ChunkManager
        self.camLens.set_near_far(0.2, terrain.FOG_END)
        self.camLens.set_fov(90)

        self.chunks = terrain.ChunkManager(self)
        self.fp = FirstPersonController(self, self.chunks, touch=self.touch)
        self.fp.spawn(0, 0)
        self.chunks.prewarm_height(0, 0)

        if sys.platform != "emscripten":
            self.accept("escape", sys.exit)

        self.cross = OnscreenText(text="+", pos=(0, -0.02), scale=0.07, fg=(1, 1, 1, 0.8),
                                  shadow=(0, 0, 0, 0.6), mayChange=False)
        self.hud = OnscreenText(text="", pos=(-1.3, 0.92), scale=0.045, fg=(1, 1, 1, 1),
                                shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True)
        self.taskMgr.add(self.update, "update")

    def update(self, task):
        dt = globalClock.get_dt()
        self.fp.update(dt)
        self.chunks.update(self.camera.get_pos(self.render))

        total, meshed, _, tris, avg = self.chunks.stats()
        mode = "FLY" if self.fp.flying else "WALK"
        text = f"{mode}  chunks {total} ({meshed} meshed)  tris {tris:,}  edits {len(self.chunks.edits)}"
        if not self.touch.active:
            text += "\nWASD move  Shift sprint  Space jump  drag look  F dig  G fill  Tab fly"
        self.hud.setText(text)
        return task.cont


def main():
    args = sys.argv[1:]
    touch = True if "--touch" in args else (False if "--no-touch" in args else None)
    view = int(args[args.index("--view") + 1]) if "--view" in args else None
    loadPrcFileData("", "sync-video 0\nshow-frame-rate-meter true")
    App(touch=touch, view=view).run()


if __name__ == "__main__":
    main()
