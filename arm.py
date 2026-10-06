import numpy as np
import matplotlib.pyplot as plt

L1 = 1.0  # length of link 1 (metres)
L2 = 0.7  # length of link 2


def forward_kinematics(theta1, theta2):
    """Return (elbow, hand) positions for joint angles in radians."""
    elbow = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    hand = elbow + np.array([L2 * np.cos(theta1 + theta2),
                             L2 * np.sin(theta1 + theta2)])
    return elbow, hand


def draw_arm(theta1, theta2):
    elbow, hand = forward_kinematics(theta1, theta2)
    xs = [0, elbow[0], hand[0]]
    ys = [0, elbow[1], hand[1]]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, "-o", linewidth=4, markersize=10)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(f"θ1={np.degrees(theta1):.0f}°, θ2={np.degrees(theta2):.0f}°  "
                 f"→ hand at ({hand[0]:.2f}, {hand[1]:.2f})")
    plt.show()


if __name__ == "__main__":
    draw_arm(np.radians(45), np.radians(30))