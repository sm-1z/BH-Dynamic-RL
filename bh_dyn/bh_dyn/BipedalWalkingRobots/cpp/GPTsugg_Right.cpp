#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

//extern "C" {
using namespace Eigen;
using namespace std;

// 正运动学函数，根据关节变量计算末端执行器的位置和姿态
Matrix4d forward_kinematics(VectorXd q, VectorXd L, VectorXd D, VectorXd Phi) {
    // 实现正运动学的具体计算
    // ...
   // Matrix4d T_current = MatrixXd::Zero(4, 4); // 计算结果
    Matrix<double, 4, 4>T_current;
    T_current(0, 0) = cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(5)) * sin(q(1)) * sin(q(2)) - cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(2)) * cos(q(5)) * sin(q(0)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(5)) * sin(q(0)) * sin(q(2)) - cos(q(0)) * cos(q(1)) * sin(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(2)) * cos(q(5)) * sin(q(1));
    T_current(0, 1) = cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(2)) * sin(q(0)) * sin(q(5)) - cos(q(0)) * cos(q(1)) * cos(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * sin(q(0)) * sin(q(2)) * sin(q(5)) - cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * sin(q(1)) * sin(q(2)) * sin(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(2)) * sin(q(1)) * sin(q(5));
    T_current(0, 2) = cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * sin(q(0)) * sin(q(2)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(2)) * sin(q(0)) + cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(2)) * sin(q(1)) + sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * sin(q(1)) * sin(q(2));
    T_current(1, 0) = sin(q(1)) * sin(q(5)) + cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(1)) * cos(q(5)) * sin(q(2)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(1)) * cos(q(2)) * cos(q(5));
    T_current(1, 1) = cos(q(5)) * sin(q(1)) - cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(1)) * sin(q(2)) * sin(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(1)) * cos(q(2)) * sin(q(5));
    T_current(1, 2) = cos(Phi(0) + Phi(1) + Phi(2) - q(2) + q(4)) * cos(q(1));
    T_current(2, 0) = cos(q(1)) * sin(q(0)) * sin(q(5)) - cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(2)) * cos(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(5)) * sin(q(2)) - cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(5)) * sin(q(0)) * sin(q(1)) * sin(q(2)) + sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(2)) * cos(q(5)) * sin(q(0)) * sin(q(1));
    T_current(2, 1) = cos(q(1)) * cos(q(5)) * sin(q(0)) + cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(2)) * sin(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * sin(q(2)) * sin(q(5)) + cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * sin(q(0)) * sin(q(1)) * sin(q(2)) * sin(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(2)) * sin(q(0)) * sin(q(1)) * sin(q(5));
    T_current(2, 2) = cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * sin(q(2)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(0)) * cos(q(2)) - cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(2)) * sin(q(0)) * sin(q(1)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * sin(q(0)) * sin(q(1)) * sin(q(2));
    
    T_current(0, 3) = L(0) - (cos(q(3)) * (sin(q(0)) * sin(q(2)) + cos(q(0)) * cos(q(2)) * sin(q(1))) - sin(q(3)) * (cos(q(2)) * sin(q(0)) - cos(q(0)) * sin(q(1)) * sin(q(2)))) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0))) - (cos(q(3)) * (cos(q(2)) * sin(q(0)) - cos(q(0)) * sin(q(1)) * sin(q(2))) + sin(q(3)) * (sin(q(0)) * sin(q(2)) + cos(q(0)) * cos(q(2)) * sin(q(1)))) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0))) - L(1) * (sin(q(0)) * sin(q(2)) + cos(q(0)) * cos(q(2)) * sin(q(1))) - D(0) * sin(q(0)) - L(5) * (cos(q(5)) * (sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * (cos(q(3)) * (sin(q(0)) * sin(q(2)) + cos(q(0)) * cos(q(2)) * sin(q(1))) - sin(q(3)) * (cos(q(2)) * sin(q(0)) - cos(q(0)) * sin(q(1)) * sin(q(2)))) + cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * (cos(q(3)) * (cos(q(2)) * sin(q(0)) - cos(q(0)) * sin(q(1)) * sin(q(2))) + sin(q(3)) * (sin(q(0)) * sin(q(2)) + cos(q(0)) * cos(q(2)) * sin(q(1))))) + cos(q(0)) * cos(q(1)) * sin(q(5))) + D(1) * cos(q(0)) * cos(q(1));
    T_current(1, 3) = D(0) + L(5) * (sin(q(1)) * sin(q(5)) + cos(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(1)) * cos(q(5)) * sin(q(2)) - sin(Phi(0) + Phi(1) + Phi(2) + q(4)) * cos(q(1)) * cos(q(2)) * cos(q(5))) - D(1) * sin(q(1)) + sin(q(2) - q(3)) * cos(q(1)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0))) - L(1) * cos(q(1)) * cos(q(2)) - cos(q(2) - q(3)) * cos(q(1)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)));
    T_current(2, 3) = -(cos(q(3)) * (cos(q(0)) * sin(q(2)) - cos(q(2)) * sin(q(0)) * sin(q(1))) - sin(q(3)) * (cos(q(0)) * cos(q(2)) + sin(q(0)) * sin(q(1)) * sin(q(2)))) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0))) - (cos(q(3)) * (cos(q(0)) * cos(q(2)) + sin(q(0)) * sin(q(1)) * sin(q(2))) + sin(q(3)) * (cos(q(0)) * sin(q(2)) - cos(q(2)) * sin(q(0)) * sin(q(1)))) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0))) - L(1) * (cos(q(0)) * sin(q(2)) - cos(q(2)) * sin(q(0)) * sin(q(1))) - D(0) * cos(q(0)) - L(5) * (cos(q(5)) * (sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * (cos(q(3)) * (cos(q(0)) * sin(q(2)) - cos(q(2)) * sin(q(0)) * sin(q(1))) - sin(q(3)) * (cos(q(0)) * cos(q(2)) + sin(q(0)) * sin(q(1)) * sin(q(2)))) + cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * (cos(q(3)) * (cos(q(0)) * cos(q(2)) + sin(q(0)) * sin(q(1)) * sin(q(2))) + sin(q(3)) * (cos(q(0)) * sin(q(2)) - cos(q(2)) * sin(q(0)) * sin(q(1))))) - cos(q(1)) * sin(q(0)) * sin(q(5))) - D(1) * cos(q(1)) * sin(q(0));

    T_current(3, 0) = 0;
    T_current(3, 1) = 0;
    T_current(3, 2) = 0;
    T_current(3, 3) = 1;

    return T_current;
}

// 计算雅可比矩阵函数
MatrixXd compute_jacobian(VectorXd q, VectorXd L, VectorXd D, VectorXd Phi) {
    // 实现雅可比矩阵的具体计算
    // MatrixXd J = MatrixXd::Zero(6, 6); // 计算结果
    Matrix<double, 6, 6>J;
    J(0, 0) = (cos(q(1)) * sin(q(5)) + sin(q(1)) * (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5))) - sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3))))) * (D(0) - cos(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + sin(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))))) - (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3))) + sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5)))) * (sin(q(1)) * (sin(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + cos(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))))) - cos(q(1)) * (D(1) - L(5) * sin(q(5))));
    J(0, 1) = (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5))) - sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3)))) * (D(1) - L(5) * sin(q(5))) + sin(q(5)) * (sin(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + cos(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))));
    J(0, 2) = (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3))) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * cos(q(5))) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))));
    J(0, 3) = sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)));
    J(0, 4) = 0;
    J(0, 5) = 0;

    J(1, 0) = (cos(q(1)) * cos(q(5)) - sin(q(1)) * (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5))) - sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5))))) * (D(0) - cos(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + sin(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))))) + (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5))) + sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5)))) * (sin(q(1)) * (sin(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + cos(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))))) - cos(q(1)) * (D(1) - L(5) * sin(q(5))));
    J(1, 1) = cos(q(5)) * (sin(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + cos(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))))) - (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5))) - sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5)))) * (D(1) - L(5) * sin(q(5)));
    J(1, 2) = -(cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5))) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) - (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) * sin(q(5)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) * sin(q(5))) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))));
    J(1, 3) = cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(5)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(5)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)));
    J(1, 4) = 0;
    J(1, 5) = L(5);

    J(2, 0) = -(sin(q(1)) * (sin(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + cos(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))))) - cos(q(1)) * (D(1) - L(5) * sin(q(5)))) * (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3))) - sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)))) - sin(q(1)) * (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3))) + sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)))) * (D(0) - cos(q(2)) * (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) + sin(q(2)) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))));
    J(2, 1) = -(D(1) - L(5) * sin(q(5))) * (cos(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3))) + sin(q(2)) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3))));
    J(2, 2) = (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)) + sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3))) * (L(1) + cos(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) + sin(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) - (sin(q(3)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(q(3)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)))) * (cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(3)) - sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * sin(q(3)));
    J(2, 3) = -sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * (L(4) * sin(Phi(0) + Phi(1) - q(3)) + L(3) * sin(Phi(0)) + L(5) * sin(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5))) - cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * (L(2) + L(4) * cos(Phi(0) + Phi(1) - q(3)) + L(3) * cos(Phi(0)) + L(5) * cos(Phi(0) + Phi(1) + Phi(2) - q(3) + q(4)) * cos(q(5)));
    J(2, 4) = -L(5) * cos(q(5));
    J(2, 5) = 0;

    J(3, 0) = sin((q(1))) * sin((q(5))) - cos((q(1))) * (cos((q(2))) * (cos((q(3))) * cos((q(5))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) + cos((q(5))) * sin((q(3))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))) - sin((q(2))) * (cos((q(3))) * cos((q(5))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) - cos((q(5))) * sin((q(3))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))));
    J(3, 1) = cos((q(2))) * (cos((q(3))) * cos((q(5))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) - cos((q(5))) * sin((q(3))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))) + sin((q(2))) * (cos((q(3))) * cos((q(5))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) + cos((q(5))) * sin((q(3))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))));
    J(3, 2) = -sin((q(5)));
    J(3, 3) = sin((q(5)));
    J(3, 4) = sin((q(5)));
    J(3, 5) = 0;

    J(4, 0) = cos((q(1))) * (cos((q(2))) * (cos((q(3))) * sin((q(5))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) + sin((q(3))) * sin((q(5))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))) - sin((q(2))) * (cos((q(3))) * sin((q(5))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) - sin((q(3))) * sin((q(5))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))))) + cos((q(5))) * sin((q(1)));
    J(4, 1) = -cos((q(2))) * (cos((q(3))) * sin((q(5))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) - sin((q(3))) * sin((q(5))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))) - sin((q(2))) * (cos((q(3))) * sin((q(5))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) + sin((q(3))) * sin((q(5))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))));
    J(4, 2) = -cos((q(5)));
    J(4, 3) = cos((q(5)));
    J(4, 4) = cos((q(5)));
    J(4, 5) = 0;

    J(5, 0) = cos((q(1))) * (cos((q(2))) * (cos((q(3))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) - sin((q(3))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))) + sin((q(2))) * (cos((q(3))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) + sin((q(3))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))));
    J(5, 1) = cos((q(2))) * (cos((q(3))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) + sin((q(3))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4)))) - sin((q(2))) * (cos((q(3))) * cos((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))) - sin((q(3))) * sin((Phi(0)) + (Phi(1)) + (Phi(2)) - (q(3)) + (q(4))));
    J(5, 2) = 0;
    J(5, 3) = 0;
    J(5, 4) = 0;
    J(5, 5) = 1;


    //MatrixXd J; // 计算结果
    return J;
}

//int main() 
double RightLeg_Inv(double R0_inagle, double R1_inagle, double R2_inagle, double R0_inp, double R1_inp, double R2_inp)
{
    // 输入参数
    double pi = M_PI;
    //VectorXd inangle(3), inp(3);
    //inangle << 0, pi / 2, pi / 2;
    //inp << 0.25, 0.15, -0.9;//0.2275, 0.1327, -1.1;

    // 期望的末端执行器姿态和位置矩阵
    Matrix4d T_desired(4,4);
    T_desired << cos(R1_inagle) * cos(R2_inagle), sin(R0_inagle)* sin(R1_inagle)* cos(R2_inagle) - cos(R0_inagle) * sin(R2_inagle), cos(R0_inagle)* sin(R1_inagle)* cos(R2_inagle) + sin(R0_inagle) * sin(R2_inagle), R0_inp,
                cos(R1_inagle)* sin(R2_inagle), sin(R0_inagle)* sin(R1_inagle)* sin(R2_inagle) + cos(R0_inagle) * cos(R2_inagle), cos(R0_inagle)* sin(R1_inagle)* sin(R2_inagle) - sin(R0_inagle) * cos(R2_inagle), R1_inp,
                -sin(R1_inagle), sin(R0_inagle)* cos(R1_inagle), cos(R0_inagle)* cos(R1_inagle), R2_inp,
                 0, 0, 0, 1;

    // 初始化参数
    //VectorXd L(6), D(2), Phi(3), Theta0(6);
    Matrix<double, 6, 1>L;
    L << 0.25, 0.125, 0.087, 0.42125, 0.45675, 0.03;
    Matrix<double, 2, 1>D;
    D << 0.141, -0.0225;
    Matrix<double, 3, 1>Phi;
    Phi << -35.96 / 180 * pi, 121.83 / 180 * pi, 5.02 / 180 * pi;// /180*pi
    Matrix<double, 6, 1>Theta0;
    Theta0 << 0 / 180 * pi, 0 / 180 * pi, 50.3 / 180 * pi, 37.3 / 180 * pi, -40.57 / 180 * pi, 0 / 180 * pi;// /180*pi
     
   // RowVectorXd L_tran = L.transpose();
   // RowVectorXd D_tran = D.transpose();
   // RowVectorXd Phi_tran = Phi.transpose();


    // 初始化关节变量
    Matrix<double, 6, 1> q;
    q << 0, 0, 0, 0, 0, 0;
    q += Theta0;
   // RowVectorXd q_tran = q.transpose();

    // 定义迭代次数和迭代步长
    int max_iterations = 1000;
    double step_size = 0.01;
    double lambda = 0.1;
    int limit = 100;
    int rejcount = 0;
    Matrix<double, 6, 6> W = Matrix<double, 6, 6>::Identity();//MatrixXd W = MatrixXd::Identity(6,6);
    Matrix<double, 3, 3> W3 = Matrix<double, 3, 3>::Identity();

    // 定义迭代终止条件
    double tolerance = 1e-6;

    // 开始迭代求解
    for (int i = 1; i <= max_iterations; ++i) {
        // 计算当前末端执行器的位置和姿态
        Matrix4d T_current = forward_kinematics(q, L, D, Phi);//_tran

        // 计算末端执行器的误差
        Matrix4d TD = T_current.inverse() * T_desired;
        Matrix3d S_block = TD.block(0, 0, 3, 3);
        Matrix3d S = S_block - W3;
        VectorXd error(6);
        Vector3d Vex;
        Vex << 0.5 * (S(2, 1) - S(1, 2)),
            0.5 * (S(0, 2) - S(2, 0)),
            0.5 * (S(1, 0) - S(0, 1));
        error << TD.block(0, 3, 3, 1), Vex;
        Matrix<double, 6, 1> errorW = W * error ;

        // 判断是否达到收敛条件
        if (errorW.norm() < tolerance) { //error.norm() < tolerance
            std::cout << "Reaching the convergence condition" << std::endl;
            break;
        }

        // 计算雅可比矩阵
        Matrix<double, 6, 6> J = compute_jacobian(q, L, D, Phi);//_tran
        Matrix<double, 6, 6> JtJ  = J.transpose() * W * J;

        // 计算关节变量的变化量
        Matrix<double, 6, 1> delta_q = (JtJ + (lambda)*MatrixXd::Identity(JtJ.rows(), JtJ.cols())).inverse() * J.transpose() * W * error;
        Matrix<double, 6, 1> q_new = q + delta_q;
        Matrix<double, 1, 6> q_new_tran = q_new.transpose();

        // 计算新的误差
        Matrix4d TD_new = forward_kinematics(q_new, L, D, Phi).inverse() * T_desired; //_tran
        Matrix3d S2 = TD_new.block( 0, 0, 3, 3) - MatrixXd::Identity(3,3);
        Vector3d Vex2;
            Vex2 << 0.5 * (S2(2, 1) - S2(1, 2)),
                0.5 * (S2(0, 2) - S2(2, 0)),
                0.5 * (S2(1, 0) - S2(0, 1));
            Matrix<double, 6, 1> error_new;
        error_new << TD_new.block( 0, 3 ,3, 1), Vex2;

        Matrix<double, 6, 1> errorNewW = W * error_new ;

        if (errorNewW.norm() < errorW.norm()) {
            q = q_new;
            error = error_new;
            lambda = lambda / 2;
            rejcount = 0;
        }
        else {
            lambda = lambda * 2;
            rejcount = rejcount+1;
            if (rejcount > limit) {
                std::cout << "Unable to solve" << std::endl;
                break;
            }
            continue;
        }

        // 更新关节变量
        q = q + step_size * delta_q;

        // 可选：对关节变量进行边界处理，确保在合理范围内
        // q = bound_joint_values(q); // 进行关节变量的边界处理
    }

    q = q - Theta0;

    // 输出最终的关节变量
//std::cout << "Final results:" << std::endl;
//std::cout << q << std::endl;

//return 0;
    double q_output_R[] = { q(0),q(1),q(2),q(3),q(4),q(5) };
    return q_output_R;
}

//}