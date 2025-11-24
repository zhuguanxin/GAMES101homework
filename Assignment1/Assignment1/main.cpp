#define _USE_MATH_DEFINES
#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    //Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    // 逐个元素地构建模型变换矩阵并返回该矩阵。在此函数中，你只需要实现三维中绕z 轴旋转的变换矩阵，而不用处理平移与缩放。
    rotation_angle = fmod(rotation_angle, 360);
    if (rotation_angle < 0) {
        rotation_angle += 360;
    }

    // 角度转换成弧度
    float angle = rotation_angle * M_PI / 180.0f;

    // 计算sin和cos
    float cos_theta = cos(angle);
    float sin_theta = sin(angle);

    // 创建4*4单位矩阵
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // 填充旋转矩阵:绕z轴旋转
    model(0, 0) = cos_theta;
    model(0, 1) = -sin_theta;
    model(1, 0) = sin_theta;
    model(1, 1) = cos_theta;
    std::cout<<"angle = "<<angle<<std::endl;
    std::cout << "model = "<<std::endl << model << std::endl;
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    // 使用给定的参数逐个元素地构建透视投影矩阵并返回该矩阵。
    // 将 fov 从度转换为弧度
    float fov_rad = eye_fov * M_PI / 180.0;

    // 计算 t (top)
    float t = tan(fov_rad / 2) * abs(zNear);
    // 计算 r (right)
    float r = t * aspect_ratio;

    // 填充投影矩阵
    projection(0, 0) = zNear / r;
    projection(1, 1) = zNear / t;
    projection(2, 2) = -(zFar + zNear) / (zFar - zNear);
    projection(2, 3) = -2 * zFar * zNear / (zFar - zNear);
    projection(3, 2) = -1;
    projection(3, 3) = 0;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth); // 清空缓冲区

        r.set_model(get_model_matrix(angle)); // MVP变换：设置model矩阵
        r.set_view(get_view_matrix(eye_pos)); // MVP变换：设置view矩阵
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50)); // MVP变换：设置projection矩阵

        std::cout << "Before draw, angle: " << angle << std::endl;
        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        std::cout << "After draw" << std::endl;

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
