#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

/**
 * @brief 使用 de Casteljau 算法递归计算贝塞尔曲线上的点
 * 
 * de Casteljau 算法是一种数值稳定的方法，用于计算贝塞尔曲线上的点。
 * 该算法通过递归地在控制点之间进行线性插值来工作：
 * 1. 在相邻控制点之间进行线性插值，得到新的点集
 * 2. 对新点集重复此过程，直到只剩下一个点
 * 3. 这个最终点就是贝塞尔曲线在参数 t 处的点
 * 
 * @param control_points 贝塞尔曲线的控制点数组
 * @param t 参数值，范围 [0, 1]，表示在曲线上的位置
 *          t = 0 对应起始点，t = 1 对应终点
 * @return cv::Point2f 贝塞尔曲线在参数 t 处的坐标点
 */
cv::Point2f recursive_bezier(const std::vector<cv::Point2f>& control_points, float t)
{
    // 递归终止条件：当只剩一个控制点时，直接返回该点
    if (control_points.size() == 1)
    {
        return cv::Point2f(control_points[0]);
    }
    else
    {
        // 创建两个子序列进行递归计算
        // left_control_points: 包含前 n-1 个控制点 [P0, P1, ..., P(n-2)]
        std::vector<cv::Point2f> left_control_points(control_points.begin(), control_points.end() - 1);
        // right_control_points: 包含后 n-1 个控制点 [P1, P2, ..., P(n-1)]
        std::vector<cv::Point2f> right_control_points(control_points.begin() + 1, control_points.end());
        
        // 递归计算左右两个子序列的贝塞尔点
        cv::Point2f left_bezier = recursive_bezier(left_control_points, t);
        cv::Point2f right_bezier = recursive_bezier(right_control_points, t);

        // 在两个递归结果之间进行线性插值
        // 公式：B(t) = (1-t) * B_left(t) + t * B_right(t)
        return cv::Point2f(left_bezier * (1 - t) + right_bezier * t);
    }
}

/**
 * @brief 使用 de Casteljau 算法绘制完整的贝塞尔曲线
 * 
 * 该函数通过在参数范围 [0, 1] 内进行密集采样，调用递归贝塞尔算法
 * 计算曲线上的每个点，并将这些点绘制到图像窗口中，形成连续的贝塞尔曲线。
 * 
 * 算法流程：
 * 1. 从 t=0 开始，以小步长 delta 递增
 * 2. 对每个 t 值调用 recursive_bezier 函数计算曲线上的点
 * 3. 将计算得到的点在图像中标记为绿色像素
 * 4. 重复直到 t 接近 1，完成整条曲线的绘制
 * 
 * @param control_points 贝塞尔曲线的控制点数组，支持任意阶数的贝塞尔曲线
 * @param window 用于绘制的图像窗口（OpenCV Mat 对象）
 *               函数会直接在此图像上绘制绿色的贝塞尔曲线
 */
void bezier(const std::vector<cv::Point2f>& control_points, cv::Mat& window)
{
    // 参数 t 的初始值，从曲线起点开始
    double t = 0.0;
    
    // 参数步长，决定曲线的平滑度
    // 步长越小，曲线越平滑，但计算量越大
    double delta = 0.001;
    
    // 遍历参数范围 [0, 1)，生成贝塞尔曲线上的所有点
    while (t < 1.0)
    {
        // 调用递归算法计算当前参数 t 对应的曲线点
        cv::Point2f point = recursive_bezier(control_points, t);
        
        // 在图像窗口中绘制该点
        // at<cv::Vec3b>(y, x)[1] 表示访问 BGR 图像的绿色通道
        // 将绿色通道值设为 255（最大亮度），形成绿色像素点
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
        
        // 增加参数 t，移向曲线上的下一个点
        t += delta;
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            //naive_bezier(control_points, window);
               bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
