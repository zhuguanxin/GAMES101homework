//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdexcept>

// 加载顶点位置数据
rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();  // 为顶点位置数据分配唯一ID
    pos_buf.emplace(id, positions); // 将数据存储在 pos_buf 映射表中

    return {id}; // 返回位置缓冲区ID，用于后续引用
}

// 加载索引数据
rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id(); // 管理三角形的顶点索引
    ind_buf.emplace(id, indices); // 每个 Vector3i 包含构成一个三角形的三个顶点索引

    return {id};
}

// Bresenham's line drawing algorithm
// Code taken from a stack overflow answer: https://stackoverflow.com/a/16405254
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;

    dx=x2-x1;
    dy=y2-y1;
    dx1=fabs(dx); // 绝对值
    dy1=fabs(dy);
    px=2*dy1-dx1; // 决策参数
    py=2*dx1-dy1;

    if(dy1<=dx1)
    {
        if(dx>=0)
        {
            x=x1;
            y=y1;
            xe=x2;
        }
        else
        {
            x=x2;
            y=y2;
            xe=x1;
        }
        Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
        set_pixel(point,line_color);
        for(i=0;x<xe;i++)
        {
            x=x+1;
            if(px<0)
            {
                px=px+2*dy1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    y=y+1;
                }
                else
                {
                    y=y-1;
                }
                px=px+2*(dy1-dx1);
            }
//            delay(0);
            Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
            set_pixel(point,line_color);
        }
    }
    else
    {
        if(dy>=0)
        {
            x=x1;
            y=y1;
            ye=y2;
        }
        else
        {
            x=x2;
            y=y2;
            ye=y1;
        }
        Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
        set_pixel(point,line_color);
        for(i=0;y<ye;i++)
        {
            y=y+1;
            if(py<=0)
            {
                py=py+2*dx1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    x=x+1;
                }
                else
                {
                    x=x-1;
                }
                py=py+2*(dx1-dy1);
            }
//            delay(0);
            Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
            set_pixel(point,line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w); // 坐标转换工具：将3D向量转换为4D齐次坐标，默认w分量为1.0
}

void rst::rasterizer::draw(rst::pos_buf_id pos_buffer, rst::ind_buf_id ind_buffer, rst::Primitive type)
{
    if (type != rst::Primitive::Triangle)
    {
        throw std::runtime_error("Drawing primitives other than triangle is not implemented yet!");
    }
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];

    float f1 = (100 - 0.1) / 2.0;
    float f2 = (100 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model; // MVP变换
    for (auto& i : ind)
    {
        Triangle t;

        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f), // MVP变换:变换第一个顶点
                mvp * to_vec4(buf[i[1]], 1.0f), // MVP变换:变换第二个顶点
                mvp * to_vec4(buf[i[2]], 1.0f) // MVP变换:变换第三个顶点
        };

        for (auto& vec : v) {
            vec /= vec.w(); // 透视除法：齐次坐标标准化
        }

        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0); // 视口变换：[-1,1] → [0,width]
            vert.y() = 0.5*height*(vert.y()+1.0); // 视口变换：[-1,1] → [0,height]
            vert.z() = vert.z() * f1 + f2; // 视口变换：深度值映射
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        t.setColor(0, 255.0,  0.0,  0.0); // 设置顶点属性：红色顶点
        t.setColor(1, 0.0  ,255.0,  0.0); // 设置顶点属性：绿色顶点
        t.setColor(2, 0.0  ,  0.0,255.0); // 设置顶点属性：蓝色顶点

        rasterize_wireframe(t);
    }
}

void rst::rasterizer::rasterize_wireframe(const Triangle& t)
{
    draw_line(t.c(), t.a()); // 绘制边 CA
    draw_line(t.c(), t.b()); // 绘制边 CB
    draw_line(t.b(), t.a()); // 绘制边 BA
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m) // 变换矩阵设置
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v) // 变换矩阵设置
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p) // 变换矩阵设置
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) // 清除缓冲区
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0}); // 颜色缓冲区：清除为黑色 (0,0,0)
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity()); // 深度缓冲区：清除为无穷大（最远距离）
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h); // 构造函数:分配颜色缓冲区
    depth_buf.resize(w * h); // 构造函数:分配深度缓冲区
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-y)*width + x; // 索引计算
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int x = point.x();
    int y = point.y();
    if (x < 0 || x >= width ||
        y < 0 || y >= height) return; // 像素设置：边界检查
    // 这里作业原版的代码好像错了，y坐标是从0（屏幕顶部）开始的，因此需要对高度-1，否则会报数组溢出的错
    auto ind = (height- 1 -y)*width + x; // 像素设置：计算像素索引。是因为OpenCV的坐标系Y轴向下，而图形学通常Y轴向上，需要进行Y坐标翻转
    frame_buf[ind] = color;  // 像素设置：设置颜色
}

