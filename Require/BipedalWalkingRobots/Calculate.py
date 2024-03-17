# from LIPM import func_LIPM_3D as func
import LIPM.func_LIPM_3D as func
import numpy as np
from ikine import ikineLeft, ikineRight

# from ikine import ikineLeft
# from ikine import ikineRight

def Calculate(time=6):
    total_time = time
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

        q_right = ikineRight.right_leg_inv(
            right_inangle,
            np.array(
                [
                    right_foot_pos_x[i] - COM_pos_x[i],
                    right_foot_pos_y[i] - COM_pos_y[i],
                    right_foot_pos_z[i] - COM_pos_z_0,
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
        print(f'------Success:{i}------')
        
    print('--------Success----------')

    data_to_save = {
    'COM_pos_x': COM_pos_x,
    'COM_pos_y': COM_pos_y,
    'left_foot_pos_x': left_foot_pos_x,
    'left_foot_pos_y': left_foot_pos_y,
    'left_foot_pos_z': left_foot_pos_z,
    'right_foot_pos_x': right_foot_pos_x,
    'right_foot_pos_y': right_foot_pos_y,
    'right_foot_pos_z': right_foot_pos_z,
    'left_joint_pos': left_joint_pos,
    'right_joint_pos': right_joint_pos
}
    np.savez('bipedal_robot_data.npz', **data_to_save)

    return COM_pos_x,COM_pos_y,left_foot_pos_x,left_foot_pos_y,left_foot_pos_z,right_foot_pos_x,right_foot_pos_y,right_foot_pos_z,left_joint_pos,right_joint_pos

if __name__=='__main__':
    Calculate(time=6)