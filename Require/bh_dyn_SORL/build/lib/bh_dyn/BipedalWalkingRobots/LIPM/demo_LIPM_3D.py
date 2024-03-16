import numpy as np
import matplotlib.pyplot as plt

from LIPM.LIPM_3D import LIPM3D

def myChaBu(q0, q1, qd0=0, qd1=0, qdd0=0, qdd1=0, tv=np.arange(0, 1, 0.002)):
    T = max(tv)
    t = np.array(tv)

    F = q0
    E = qd0
    D = qdd0 / 2

    MAT_A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
    MAT_b = np.array([q1 - F - E * T - D * T ** 2,
                      qd1 - E - 2 * D * T,
                      qdd1 - 2 * D])
    x = np.linalg.solve(MAT_A, MAT_b)
    C, B, A = x

    tt = np.column_stack((t ** 5, t ** 4, t ** 3, t ** 2, t, np.ones_like(t)))
    c = np.array([A, B, C, D, E, F])

    qt = np.dot(tt, c)

    c = np.array([np.zeros_like(A), 5 * A, 4 * B, 3 * C, 2 * D, E])
    qdt = np.dot(tt, c.T)


    c = np.array([np.zeros_like(A), np.zeros_like(B), 20 * A, 12 * B, 6 * C, 2 * D])
    qddt = np.dot(tt, c.T)

    return qt, qdt, qddt


# %% ---------------------------------------------------------------- LIPM control
print('\n--------- Program start from here ...')

COM_pos_x = list()
COM_pos_y = list()
COM_pos_vx = list()
COM_pos_vy = list()
COM_pos_ax = list()
COM_pos_ay = list()
left_foot_pos_x = list()
left_foot_pos_y = list()
left_foot_pos_z = list()
right_foot_pos_x = list()
right_foot_pos_y = list()
right_foot_pos_z = list()

COM_pos_z_0 = 1.0

# Initialize the COM position and velocity
COM_pos_0 = [0, 0.1327, COM_pos_z_0 - 0.227]  # [0, 0, 0] [-0.4, 0.2, 1.0]
COM_v0 = [-0.723, 0.001]  # [1.0, 0.01] [1.0, -0.01] [0.1, 0.56]

# Initialize the foot positions
left_foot_pos = [-0.2275, 0.1327, 0]  # [0.2, 0.1, -1] [-0.2, 0.3, 0]
right_foot_pos = [0.2275, 0.1327, 0]  # [0.6, -0.5, -1] [0.2, -0.3, 0]

delta_t = 0.002

s_x = 0.2775 * 2  # default: 0.5
s_y = 0.4
a = 4.0
b = 6.0
theta = 0.0

LIPM_model = LIPM3D(dt=delta_t, T_sup=0.8, T_dbl=0.2)
LIPM_model.initializeModel(COM_pos_0, left_foot_pos, right_foot_pos)

LIPM_model.support_leg = 'left_leg'  # set the support leg to right leg in next step
if LIPM_model.support_leg is 'left_leg':
    support_foot_pos = LIPM_model.left_foot_pos
    LIPM_model.p_x = LIPM_model.left_foot_pos[0]
    LIPM_model.p_y = LIPM_model.left_foot_pos[1]
else:
    support_foot_pos = LIPM_model.right_foot_pos
    LIPM_model.p_x = LIPM_model.right_foot_pos[0]
    LIPM_model.p_y = LIPM_model.right_foot_pos[1]

LIPM_model.x_0 = LIPM_model.COM_pos[0] - support_foot_pos[0]
LIPM_model.y_0 = LIPM_model.COM_pos[1] - support_foot_pos[1]
LIPM_model.vx_0 = COM_v0[0]
LIPM_model.vy_0 = COM_v0[1]

step_num = 0
total_time = 6  # seconds    default: 30
global_time = 0

swing_data_len = int(LIPM_model.T_sup / delta_t)
swing_foot_pos = np.zeros((swing_data_len, 3))
double_foot_len = int(LIPM_model.T_dbl / delta_t)
j = 0

switch_index = swing_data_len

for i in range(int(total_time / delta_t)):
    global_time += delta_t

    LIPM_model.step()

    if step_num >= 1:
        if LIPM_model.support_leg is 'left_leg':
            LIPM_model.right_foot_pos = [swing_foot_pos[j, 0], swing_foot_pos[j, 1], swing_foot_pos[j, 2]]
        else:
            LIPM_model.left_foot_pos = [swing_foot_pos[j, 0], swing_foot_pos[j, 1], swing_foot_pos[j, 2]]
        j += 1

    # record data
    COM_pos_x.append(LIPM_model.x_t + support_foot_pos[0])
    COM_pos_y.append(LIPM_model.y_t + support_foot_pos[1])
    COM_pos_vx.append(LIPM_model.vx_t)
    COM_pos_vy.append(LIPM_model.vy_t)
    COM_pos_ax.append(LIPM_model.x_t / LIPM_model.T_c ** 2)
    COM_pos_ay.append(LIPM_model.y_t / LIPM_model.T_c ** 2)
    left_foot_pos_x.append(LIPM_model.left_foot_pos[0])
    left_foot_pos_y.append(LIPM_model.left_foot_pos[1])
    left_foot_pos_z.append(LIPM_model.left_foot_pos[2])
    right_foot_pos_x.append(LIPM_model.right_foot_pos[0])
    right_foot_pos_y.append(LIPM_model.right_foot_pos[1])
    right_foot_pos_z.append(LIPM_model.right_foot_pos[2])

    # switch the support leg
    if (i > 0) and (i % switch_index == 0):
        j = 0

        LIPM_model.switchSupportLeg()  # switch the support leg
        step_num += 1

        # theta -= 0.04 # set zero for walking forward, set non-zero for turn left and right

        # if step_num >= 5: # stop forward after 5 steps
        #     s_x = 0.0
        #
        # if step_num >= 10:
        #     s_y = 0.0

        if LIPM_model.support_leg is 'left_leg':
            support_foot_pos = LIPM_model.left_foot_pos
            LIPM_model.p_x = LIPM_model.left_foot_pos[0]
            LIPM_model.p_y = LIPM_model.left_foot_pos[1]
        else:
            support_foot_pos = LIPM_model.right_foot_pos
            LIPM_model.p_x = LIPM_model.right_foot_pos[0]
            LIPM_model.p_y = LIPM_model.right_foot_pos[1]

        # calculate the next foot locations, with modification, stable
        x_0, vx_0, y_0, vy_0 = LIPM_model.calculateXtVt(
            LIPM_model.T_sup)  # calculate the xt and yt as the initial state for next step

        if LIPM_model.support_leg is 'left_leg':
            x_0 = x_0 + LIPM_model.left_foot_pos[0]  # need the absolute position for next step
            y_0 = y_0 + LIPM_model.left_foot_pos[1]  # need the absolute position for next step
        else:
            x_0 = x_0 + LIPM_model.right_foot_pos[0]  # need the absolute position for next step
            y_0 = y_0 + LIPM_model.right_foot_pos[1]  # need the absolute position for next step

        LIPM_model.calculateFootLocationForNextStep(s_x, s_y, a, b, theta, x_0, vx_0, y_0, vy_0)
        # print('p_star=', LIPM_model.p_x_star, LIPM_model.p_y_star)

        # calculate the foot positions for swing phase
        if LIPM_model.support_leg is 'left_leg':
            right_foot_target_pos = [LIPM_model.p_x_star, LIPM_model.p_y_star, 0]
            swing_foot_pos = np.zeros((swing_data_len, 3))

            # swing_foot_pos[:, 0] = np.linspace(LIPM_model.right_foot_pos[0], right_foot_target_pos[0], swing_data_len)
            # swing_foot_pos[:, 1] = np.linspace(LIPM_model.right_foot_pos[1], right_foot_target_pos[1], swing_data_len)
            # swing_foot_pos[0, 2] = -1
            # swing_foot_pos[1:swing_data_len - 1, 2] = -0.9  # 0.1
            # swing_foot_pos[swing_data_len - 1, 2] = -1

            swing_foot_pos[:double_foot_len // 2, 0] = np.ones((double_foot_len // 2)) * LIPM_model.right_foot_pos[0]
            swing_foot_pos[:double_foot_len // 2, 1] = np.ones((double_foot_len // 2)) * LIPM_model.right_foot_pos[1]
            swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 0], _, _ = myChaBu(
                LIPM_model.right_foot_pos[0], right_foot_target_pos[0], 0, 0, 0, 0,
                np.arange(0, delta_t * (swing_data_len - double_foot_len), delta_t))
            swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 1], _, _ = myChaBu(
                LIPM_model.right_foot_pos[1], right_foot_target_pos[1], 0, 0, 0, 0,
                np.arange(0, delta_t * (swing_data_len - double_foot_len), delta_t))
            swing_foot_pos[swing_data_len - double_foot_len // 2:, 0] = np.ones((double_foot_len // 2)) * \
                                                                        right_foot_target_pos[0]
            swing_foot_pos[swing_data_len - double_foot_len // 2:, 1] = np.ones((double_foot_len // 2)) * \
                                                                        right_foot_target_pos[1]

            # 平滑摆动脚轨迹
            swing_foot_pos[:double_foot_len // 2, 2] = 0
            swing_foot_pos[swing_data_len - double_foot_len // 2:, 2] = 0
            _, swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 2], _ = myChaBu(0,
                                                                                                          0.1 * 8 / 15,
                                                                                                          tv=np.arange(0,
                                                                                                              1,
                                                                                                              1 / (swing_data_len - double_foot_len)))
            # swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 2] = COM_pos_z_0


        else:
            # left_foot_target_pos = [LIPM_model.p_x_star, LIPM_model.p_y_star, -1]
            # swing_foot_pos[:, 0] = np.linspace(LIPM_model.left_foot_pos[0], left_foot_target_pos[0], swing_data_len)
            # swing_foot_pos[:, 1] = np.linspace(LIPM_model.left_foot_pos[1], left_foot_target_pos[1], swing_data_len)
            # swing_foot_pos[0, 2] = -1
            # swing_foot_pos[1:swing_data_len - 1, 2] = -0.9  # 0.1
            # swing_foot_pos[swing_data_len - 1, 2] = -1

            left_foot_target_pos = [LIPM_model.p_x_star, LIPM_model.p_y_star, 0]
            swing_foot_pos = np.zeros((swing_data_len, 3))

            swing_foot_pos[:double_foot_len // 2, 0] = np.ones((double_foot_len // 2)) * LIPM_model.left_foot_pos[0]
            swing_foot_pos[:double_foot_len // 2, 1] = np.ones((double_foot_len // 2)) * LIPM_model.left_foot_pos[1]
            swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 0], _, _ = myChaBu(
                LIPM_model.left_foot_pos[0], left_foot_target_pos[0], 0, 0, 0, 0,
                np.arange(0, delta_t * (swing_data_len - double_foot_len), delta_t))
            swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 1], _, _ = myChaBu(
                LIPM_model.left_foot_pos[1], left_foot_target_pos[1], 0, 0, 0, 0,
                np.arange(0, delta_t * (swing_data_len - double_foot_len), delta_t))
            swing_foot_pos[swing_data_len - double_foot_len // 2:, 0] = np.ones((double_foot_len // 2)) * \
                                                                        left_foot_target_pos[0]
            swing_foot_pos[swing_data_len - double_foot_len // 2:, 1] = np.ones((double_foot_len // 2)) * \
                                                                        left_foot_target_pos[1]

            # 平滑摆动脚轨迹
            swing_foot_pos[:double_foot_len // 2, 2] = 0
            swing_foot_pos[swing_data_len - double_foot_len // 2:, 2] = 0
            _, swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 2], _ = myChaBu(0,
                                                                                                          0.1 * 8 / 15,
                                                                                                          tv=np.arange(0,
                                                                                                              1,
                                                                                                              1 / (swing_data_len - double_foot_len)))
            # swing_foot_pos[double_foot_len // 2:swing_data_len - double_foot_len // 2, 2] -= COM_pos_z_0

data_len = len(left_foot_pos_x)

for i in range(1, data_len - double_foot_len // 2):
    if i > 1 and (i - 1) % switch_index == switch_index - double_foot_len // 2:
        COM_smooth_x, _, _ = myChaBu(
            COM_pos_x[i], COM_pos_x[i + double_foot_len - 1],
            COM_pos_vx[i], COM_pos_vx[i + double_foot_len - 1],
            COM_pos_ax[i], COM_pos_ax[i + double_foot_len - 1],
            np.arange(0, delta_t * double_foot_len, delta_t)
        )

        COM_pos_x[i:i + double_foot_len] = COM_smooth_x

        COM_smooth_y, _, _ = myChaBu(
            COM_pos_y[i], COM_pos_y[i + double_foot_len - 1],
            COM_pos_vy[i], COM_pos_vy[i + double_foot_len - 1],
            COM_pos_ay[i], COM_pos_ay[i + double_foot_len - 1],
            np.arange(0, delta_t * double_foot_len, delta_t)
        )

        COM_pos_y[i:i + double_foot_len] = COM_smooth_y

        i += double_foot_len // 2 - 1
        if i > data_len - double_foot_len // 2:
            break

# ------------------------------------------------- animation plot

print('--------- plot')
# from matplotlib import gridspec
#
# fig = plt.figure(figsize=(10, 8))
# spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.5, 1])
# ax = fig.add_subplot(spec[0], projection='3d')
# # ax.set_aspect('equal') # bugs
# ax.set_xlim(-2.0, 2.0)
# ax.set_ylim(-1.0, 8.0)
# ax.set_zlim(-1.0, 2.0)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')
#
# # view angles
# ax.view_init(20, -150)
# LIPM_3D_ani = LIPM_3D_Animate()
# ani_3D = FuncAnimation(fig, ani_3D_update, frames=range(1, data_len), init_func=ani_3D_init, interval=1.0 / delta_t,
#                        blit=True, repeat=True)
# # writer = PillowWriter(fps=30)
# # ani_3D.save('../pic/LIPM_3D.gif', writer='imagemagick', fps=30)
#
# bx = fig.add_subplot(spec[1], autoscale_on=False)
# bx.set_xlim(-0.8, 0.8)
# bx.set_ylim(-0.5, 5.0)
# bx.set_aspect('equal')
# bx.set_xlabel('x (m)')
# bx.set_ylabel('y (m)')
# bx.grid(ls='--')
#
# COM_pos_str = 'COM = (%.2f, %.2f)'
# ani_text_COM_pos = bx.text(0.05, 0.9, '', transform=bx.transAxes)
#
# original_ani, = bx.plot(0, 0, marker='o', markersize=2, color='k')
# left_foot_pos_ani, = bx.plot([], [], 'o', lw=2, color='b')
# COM_traj_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], markersize=2, color='g')
# COM_pos_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], marker='o', markersize=6, color='r')
# left_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='b')
# right_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='m')
#
# ani_2D = FuncAnimation(fig=fig, init_func=ani_2D_init, func=ani_2D_update, frames=range(1, data_len),
#                        interval=1.0 / delta_t, blit=True, repeat=True)
# # ani_2D.save('./pic/COM_trajectory.gif', fps=30)
#
# plt.show()


# 模拟球的坐标时间序列
t = np.linspace(0, total_time, data_len)  # 时间序列
x = COM_pos_x  # 球的 x 坐标
y = COM_pos_y  # 球的 y 坐标

# 绘制球的坐标随时间变化的图形
plt.figure(1)
plt.plot(t, x, label='X coordinate')
plt.plot(t, y, label='Y coordinate')
plt.xlabel('Time')
plt.ylabel('Coordinate')
plt.title('Ball Coordinate over Time')
plt.legend()
plt.grid(True)

plt.figure(2)
# 创建三个子图
lfig, laxes = plt.subplots(3, 1, figsize=(8, 10))
lx = left_foot_pos_x  # 球的 x 坐标
ly = left_foot_pos_y  # 球的 y 坐标
lz = left_foot_pos_z  # 球的 y 坐标

# 第一个子图：X 坐标随时间变化
laxes[0].plot(t, lx, label='X coordinate', color='blue')
laxes[0].set_xlabel('Time')
laxes[0].set_ylabel('X Coordinate')
laxes[0].set_title('L-X Coordinate over Time')
laxes[0].legend()
laxes[0].grid(True)

# 第二个子图：Y 坐标随时间变化
laxes[1].plot(t, ly, label='Y coordinate', color='green')
laxes[1].set_xlabel('Time')
laxes[1].set_ylabel('Y Coordinate')
laxes[1].set_title('L-Y Coordinate over Time')
laxes[1].legend()
laxes[1].grid(True)

# 第三个子图：Z 坐标随时间变化
laxes[2].plot(t, lz, label='Z coordinate', color='red')
laxes[2].set_xlabel('Time')
laxes[2].set_ylabel('Z Coordinate')
laxes[2].set_title('L-Z Coordinate over Time')
laxes[2].legend()
laxes[2].grid(True)

plt.tight_layout()

plt.figure(3)
# 创建三个子图
rfig, raxes = plt.subplots(3, 1, figsize=(8, 10))
rx = right_foot_pos_x  # 球的 x 坐标
ry = right_foot_pos_y  # 球的 y 坐标
rz = right_foot_pos_z  # 球的 y 坐标

# 第一个子图：X 坐标随时间变化
raxes[0].plot(t, rx, label='X coordinate', color='blue')
raxes[0].set_xlabel('Time')
raxes[0].set_ylabel('X Coordinate')
raxes[0].set_title('R-X Coordinate over Time')
raxes[0].legend()
raxes[0].grid(True)

# 第二个子图：Y 坐标随时间变化
raxes[1].plot(t, ry, label='Y coordinate', color='green')
raxes[1].set_xlabel('Time')
raxes[1].set_ylabel('Y Coordinate')
raxes[1].set_title('R-Y Coordinate over Time')
raxes[1].legend()
raxes[1].grid(True)

# 第三个子图：Z 坐标随时间变化
raxes[2].plot(t, rz, label='Z coordinate', color='red')
raxes[2].set_xlabel('Time')
raxes[2].set_ylabel('Z Coordinate')
raxes[2].set_title('R-Z Coordinate over Time')
raxes[2].legend()
raxes[2].grid(True)

# 调整子图布局
plt.tight_layout()
plt.show()

print('---------  Program terminated')
