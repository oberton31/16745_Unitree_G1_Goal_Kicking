import numpy as np
from qpsolvers import solve_qp
from matplotlib import pyplot as plt

def optimize_impulse(q_des, q_i, N=100, J_max=np.inf, dt=0.01, m=0.43) -> np.array:
    g = 9.81
    N = N - 1 # go from 0th to Nth timestep -> N total 

    P = np.array([[(N/m * dt)**2, 0],
                  [0, (N/m * dt)**2]])
    q_T = (N * dt) / m * np.array([q_i[0] - q_des[0], q_i[1] - q_des[1] - (N - 1) * N * dt**2 * g / 2])
    q = q_T.T

    G = np.array([[1, 0],
                  [0, 1],
                  [-1, 0],
                  [0, -1]])
    
    h = np.array([J_max, J_max, 0, 0])

    J = solve_qp(P, q, G, h, solver="clarabel")

    return J

def get_contact_point(J, r=0.22, ball_center=np.array([0, 0.11])) -> np.array:
    norm_J = J / np.linalg.norm(J)
    p_c = ball_center  - norm_J * r
    return p_c


def visualize_contact(J, p_c, r=0.22, ball_center=np.array([0, 0.11])) -> None:
    norm_J = J / np.linalg.norm(J) * r
    fig, ax = plt.subplots(figsize=(6, 6))
    ball = plt.Circle(ball_center, r, color='blue', fill=False, linewidth=2)    
    ax.add_patch(ball)
    ax.plot(ball_center[0], ball_center[1], 'o', color='red', label='Ball Center')
    ax.quiver(
        p_c[0], p_c[1], norm_J[0], norm_J[1],
        angles='xy', scale_units='xy', scale=1, color='green', label='Impulse Vector Direction'
    )
    ax.plot(p_c[0], p_c[1], 'x', color='purple', label='Contact Point')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    
    # Add legend and title
    ax.legend()
    ax.set_title('Ball Contact Point Visualization')
    ax.set_xlim(ball_center[0] - 2 * r, ball_center[0] +  2 * r)
    ax.set_ylim(ball_center[1] - 2 * r, ball_center[1] + 2 * r)
    ax.set_aspect('equal')

    plt.grid()
    plt.show()

def dynamics_rollout(J, q_des, q_i, N = 100, dt=0.01, m = 0.43) -> None:
    v_i = J / m
    x = np.zeros((N, 4))
    g = np.array([0, 9.81])
    x[0, :] = np.concatenate((q_i, v_i))
    for k in range(1, N):
        v_k = v_i - k * dt * g
        q_k = q_i + k * J/m * dt - (k-1) * k * dt**2 * g / 2
        x[k, :] = np.concatenate((q_k, v_k))
    plt.plot(q_des[0], q_des[1], 'ro', label='Target')
    plt.plot(x[:, 0], x[:, 1], label='Position (q)')
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    J = optimize_impulse([20, 3], [0, 0])
    dynamics_rollout(J, [20, 3], [0, 0])
    p_c = get_contact_point(J)
    visualize_contact(J, p_c)