from LIPM import func_LIPM_3D as func
import numpy as np
import matplotlib.pyplot as plt

from ikine import ikineLeft
from ikine import ikineRight

total_time = 6

(
    COM_pos_x,
    COM_pos_y,
    left_foot_pos_x,
    left_foot_pos_y,
    left_foot_pos_z,
    right_foot_pos_x,
    right_foot_pos_y,
    right_foot_pos_z,
) = func.func(0.4, total_time)
data_len = len(COM_pos_x)
print(data_len)
t = np.linspace(0, total_time, data_len)  # 时间序列
print(t)

"""
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
"""

left_joint_pos = np.zeros((data_len, 7))
right_joint_pos = np.zeros((data_len, 7))

left_inangle = np.array([0, np.pi / 2, -np.pi / 2])
right_inangle = np.array([0, np.pi / 2, np.pi / 2])

q_left_tmp = np.zeros((7,))
q_right_tmp = np.zeros((7,))

COM_pos_z_0 = 1

for i in range(data_len):
    q_left = ikineLeft.left_leg_inv(
        left_inangle,
        np.array(
            [
                left_foot_pos_x[i] - COM_pos_x[i],
                left_foot_pos_y[i] - COM_pos_y[i],
                left_foot_pos_z[i] - COM_pos_z_0,
            ]
        ),
    )

    q_right = ikineLeft.left_leg_inv(
        left_inangle,
        np.array(
            [
                left_foot_pos_x[i] - COM_pos_x[i],
                left_foot_pos_y[i] - COM_pos_y[i],
                left_foot_pos_z[i] - COM_pos_z_0,
            ]
        ),
    )

    q_left_tmp[:4] = q_left[:4]
    q_left_tmp[5:] = q_left[4:]
    q_left_tmp[4] = -q_left[3]

    q_right_tmp[:4] = q_right[:4]
    q_right_tmp[5:] = q_right[4:]
    q_right_tmp[4] = -q_right[3]

    left_joint_pos[i, :] = q_left_tmp
    right_joint_pos[i, :] = q_right_tmp


plt.figure(figsize=(10, 6))
for i in range(7):
    plt.plot(t, left_joint_pos[:, i], label=f"Curve {i+1}")

# 添加标题和图例
plt.title("Seven Curves")
plt.legend()

# 显示图形
plt.show()
