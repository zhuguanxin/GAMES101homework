#define _USE_MATH_DEFINES
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include<cmath>

int main() {
    // 定义点P(2,1)
    Eigen::Vector3f p(2.0f, 1.0f, 1.0f);

    // 创建旋转矩阵(逆时针旋转45度)
    float angle = 45.0f * M_PI / 180.0f;
    Eigen::Matrix3f rotation = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).matrix();

    // 创建平移向量(1,2)
    Eigen::Vector3f translation(1.0f, 2.0f, 0.0f);

    // 创建仿射变换矩阵
    Eigen::Affine3f transform = Eigen::Translation3f(translation) * Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ());

    // 应用变换
    Eigen::Vector3f transformed_p = transform * p;

    // 输出结果
    std::cout << "变换后的坐标: (" << transformed_p.x() << ", " << transformed_p.y() << ")" << std::endl;

    return 0;
}