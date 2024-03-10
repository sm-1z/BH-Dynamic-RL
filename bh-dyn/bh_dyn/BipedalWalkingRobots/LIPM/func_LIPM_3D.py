import numpy as np
# from LIPM import LIPM_3D
from bh_dyn.BipedalWalkingRobots.LIPM import LIPM_3D


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
# print('\n--------- Program start from here ...')


def func(foot_step, total_time, theta=0.0, T_sup=0.8, T_dbl=0.2, COM_pos_0=[0, 0.1327, 1 - 0.227],
                 delta_t=0.01,
                 COM_v0=[-0.723, 0.001]):
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

    # Initialize the COM position and velocity

    # Initialize the foot positions
    left_foot_pos = [-0.2275, 0.1327, 0]  # [0.2, 0.1, -1] [-0.2, 0.3, 0]
    right_foot_pos = [0.2275, 0.1327, 0]  # [0.6, -0.5, -1] [0.2, -0.3, 0]

    s_x = 0.2775 * 2  # default: 0.5
    s_y = foot_step
    a = 4.0
    b = 6.0

    LIPM_model = LIPM_3D.LIPM3D(dt=delta_t, T_sup=T_sup, T_dbl=T_dbl)
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

            # calculate the foot positions for swing phase
            if LIPM_model.support_leg is 'left_leg':
                right_foot_target_pos = [LIPM_model.p_x_star, LIPM_model.p_y_star, 0]
                swing_foot_pos = np.zeros((swing_data_len, 3))

                swing_foot_pos[:double_foot_len // 2, 0] = np.ones((double_foot_len // 2)) * LIPM_model.right_foot_pos[
                    0]
                swing_foot_pos[:double_foot_len // 2, 1] = np.ones((double_foot_len // 2)) * LIPM_model.right_foot_pos[
                    1]
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
                                                                                                              tv=np.arange(
                                                                                                                  0,
                                                                                                                  1,
                                                                                                                  1 / (
                                                                                                                          swing_data_len - double_foot_len)))


            else:

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
                                                                                                              tv=np.arange(
                                                                                                                  0,
                                                                                                                  1,
                                                                                                                  1 / (
                                                                                                                          swing_data_len - double_foot_len)))

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

    print('---------  Program terminated')

    return COM_pos_x, COM_pos_y, left_foot_pos_x, left_foot_pos_y, left_foot_pos_z, right_foot_pos_x, right_foot_pos_y, right_foot_pos_z
