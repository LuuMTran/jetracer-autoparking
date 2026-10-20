"""Local gamepad, terminal keyboard, and synthetic controls; normalized outputs."""
import os
import select
import sys
import time


KEYBOARD_ACTIONS = {
    "w": (0., .7),     # forward straight
    "s": (0., -.7),    # reverse straight
    "u": (-1., .7),    # forward left
    "i": (1., .7),     # forward right
    "j": (-1., -.7),   # reverse left
    "k": (1., -.7),    # reverse right
}
KEYBOARD_HELP = "W/S straight; U/I forward left/right; J/K reverse left/right; Space neutral; Escape emergency stop; F finish"


class Controls:
    def __init__(self, cfg, kind="gamepad"):
        self.cfg, self.kind = cfg, kind
        self.finish = False
        self.stop = False
        self.active = False
        self.last_key = 0.
        self.action = (0., 0.)
        self.terminal_state = None
        self.pg = None
        if kind == "gamepad":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
            self.pg = pygame
            pygame.display.init()
            pygame.joystick.init()
            index = cfg["gamepad"]["index"]
            if pygame.joystick.get_count() <= index:
                raise RuntimeError("Connect a gamepad to the Jetson (or use --controller keyboard)")
            self.joy = pygame.joystick.Joystick(index)
            self.joy.init()
            g = cfg["gamepad"]
            if max(g["steering_axis"], g["throttle_axis"]) >= self.joy.get_numaxes() or max(g["deadman_button"], g["stop_button"], g["finish_button"]) >= self.joy.get_numbuttons():
                self.close()
                raise ValueError("Gamepad mapping exceeds available axes/buttons")
        elif kind == "keyboard":
            import termios
            import tty
            if not sys.stdin.isatty():
                raise ValueError("Keyboard control requires a terminal/SSH, not a notebook cell")
            self.terminal_state = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        elif kind != "synthetic":
            raise ValueError("Unknown controller")

    def poll(self):
        if self.kind == "synthetic":
            self.active = True
            return (0.25, -0.5)
        if self.kind == "keyboard":
            # Commands expire unless keys repeat. Space neutralizes; Escape ends the run.
            while select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "\x1b":
                    self.stop = True
                elif key == "f":
                    self.finish = True
                elif key == " ":
                    self.action = (0., 0.)
                    self.last_key = 0.
                elif key in KEYBOARD_ACTIONS:
                    self.action = KEYBOARD_ACTIONS[key]
                    self.last_key = time.monotonic()
            self.active = time.monotonic() - self.last_key < 0.25
            return (self.action[0] * self.cfg["hardware"]["right_steering_sign"], self.action[1]) if self.active else (0., 0.)
        pg, g = self.pg, self.cfg["gamepad"]
        for event in pg.event.get():
            if event.type == getattr(pg, "JOYDEVICEREMOVED", -1):
                raise RuntimeError("Gamepad disconnected")
        if pg.joystick.get_count() <= g["index"]:
            raise RuntimeError("Gamepad disconnected")
        self.stop = bool(self.joy.get_button(g["stop_button"]))
        self.finish = bool(self.joy.get_button(g["finish_button"]))
        self.active = bool(self.joy.get_button(g["deadman_button"]))
        def axis(index, sign):
            value = self.joy.get_axis(index) * sign
            d = g["deadzone"]
            return 0. if abs(value) <= d else (abs(value) - d) / (1 - d) * (1 if value > 0 else -1)
        return (axis(g["steering_axis"], g["steering_sign"]) * self.cfg["hardware"]["right_steering_sign"],
                axis(g["throttle_axis"], g["throttle_sign"])) if self.active else (0., 0.)

    def close(self):
        if self.terminal_state is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.terminal_state)
        if self.pg:
            self.pg.joystick.quit()
            self.pg.display.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class NotebookControls:
    """Optional notebook buttons. Each click leases a command for only 0.20 s."""
    def __init__(self, cfg):
        import ipywidgets as widgets
        from IPython.display import display
        self.finish, self.stop, self.active = False, False, False
        self.action, self.expires = (0., 0.), 0.
        self.cfg = cfg
        directions = [("Forward left", (-1., .7)), ("Forward", (0., .7)), ("Forward right", (1., .7)),
                      ("Reverse left", (-1., -.7)), ("Reverse", (0., -.7)), ("Reverse right", (1., -.7))]
        self.buttons = []
        for label, action in directions:
            button = widgets.Button(description=label)
            def click(_, value=action):
                self.action = value
                self.expires = time.monotonic() + .2
            button.on_click(click)
            self.buttons.append(button)
        finish = widgets.Button(description="Finish episode", button_style="success")
        stop = widgets.Button(description="STOP", button_style="danger")
        finish.on_click(lambda _: setattr(self, "finish", True))
        stop.on_click(lambda _: setattr(self, "stop", True))
        self.buttons += [finish, stop]
        display(widgets.VBox([widgets.HTML("Each direction click lasts 0.20 s. Use SSH keyboard for smooth demonstrations."),
                              widgets.HBox(self.buttons[:3]), widgets.HBox(self.buttons[3:6]),
                              widgets.HBox(self.buttons[6:])]))

    def poll(self):
        self.active = time.monotonic() < self.expires
        return (self.action[0] * self.cfg["hardware"]["right_steering_sign"], self.action[1]) if self.active else (0., 0.)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop = True
        for button in self.buttons:
            button.disabled = True
