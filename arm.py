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


def inverse_kinematics(x, y):
    """Return (theta1, theta2) in radians to put the hand at (x, y),
    or None if the point is out of reach."""
    # Law of cosines: the cosine of the elbow angle
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # cos can only be between -1 and 1; outside that, the point can't be reached
    if c2 > 1 or c2 < -1:
        return None

    theta2 = np.arccos(c2)  # elbow angle
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2),
                                           L1 + L2 * np.cos(theta2))
    return theta1, theta2


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
    for target in [(1.2, 0.8), (0.0, 1.5), (2.0, 0.0), (0.2, 0.0)]:
        result = inverse_kinematics(*target)
        if result is None:
            print(f"{target}: unreachable")
            continue
        t1, t2 = result
        _, hand = forward_kinematics(t1, t2)
        print(f"{target}: θ1={np.degrees(t1):.1f}°, θ2={np.degrees(t2):.1f}° "
              f"-> FK gives {hand.round(3)}")

    draw_arm(*inverse_kinematics(1.2, 0.8))