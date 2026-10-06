import numpy as np
import matplotlib.pyplot as plt

from arm import L1, L2, forward_kinematics, inverse_kinematics

STEPS = 50          # frames per move (more = slower, smoother)
FRAME_MS = 20       # milliseconds between frames


class ArmSimulator:
    def __init__(self):
        self.theta = np.array([np.radians(90), np.radians(-90)])  # start pose
        self.path = []                     # joint angles still to animate
        self.trail_x, self.trail_y = [], []

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        ax = self.ax
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title("Click anywhere to move the arm")

        # Workspace: the hand can only reach the ring between these circles
        ax.add_patch(plt.Circle((0, 0), L1 + L2, fill=False, ls="--", color="grey"))
        ax.add_patch(plt.Circle((0, 0), abs(L1 - L2), fill=False, ls=":", color="grey"))

        (self.arm_line,) = ax.plot([], [], "-o", linewidth=4, markersize=10)
        (self.trail_line,) = ax.plot([], [], "-", linewidth=1, alpha=0.5)
        (self.target_marker,) = ax.plot([], [], "x", markersize=12, markeredgewidth=3)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self.step)
        self.timer.start()

        self.redraw()

    def redraw(self):
        elbow, hand = forward_kinematics(*self.theta)
        self.arm_line.set_data([0, elbow[0], hand[0]], [0, elbow[1], hand[1]])
        self.trail_x.append(hand[0])
        self.trail_y.append(hand[1])
        self.trail_line.set_data(self.trail_x, self.trail_y)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:        # ignore clicks outside the plot
            return
        x, y = event.xdata, event.ydata
        self.target_marker.set_data([x], [y])

        result = inverse_kinematics(x, y)
        if result is None:
            self.target_marker.set_color("red")
            self.ax.set_title(f"({x:.2f}, {y:.2f}) is out of reach")
            self.fig.canvas.draw_idle()
            return

        self.target_marker.set_color("green")
        self.ax.set_title(f"Moving to ({x:.2f}, {y:.2f})")

        # Turn the shortest way round (e.g. 350° -> 10° goes +20°, not -340°)
        target = np.array(result)
        delta = (target - self.theta + np.pi) % (2 * np.pi) - np.pi

        # Ease-in-out: start slow, speed up, slow down (like a real motor)
        u = np.linspace(0, 1, STEPS)[1:]
        s = (1 - np.cos(np.pi * u)) / 2
        start = self.theta.copy()
        self.path = [start + delta * k for k in s]

    def step(self):
        if self.path:
            self.theta = self.path.pop(0)
            self.redraw()


if __name__ == "__main__":
    sim = ArmSimulator()
    plt.show()