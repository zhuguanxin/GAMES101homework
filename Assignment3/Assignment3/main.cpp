#define _USE_MATH_DEFINES  // 必须在include之前
#include <cmath>
#include <iostream>
#include<opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
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

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

/**
 * 纹理片段着色器 - 基于Blinn-Phong光照模型的纹理渲染
 * 
 * 功能：将纹理颜色作为漫反射系数(kd)，结合Blinn-Phong光照模型计算最终像素颜色
 * 
 * @param payload 片段着色器输入数据，包含：
 *   - texture: 纹理对象指针
 *   - tex_coords: 当前像素的纹理坐标(u,v)
 *   - view_pos: 像素在视空间中的位置
 *   - normal: 像素的法向量
 * 
 * @return 计算得到的最终像素颜色(RGB，范围0-255)
 * 
 * 光照模型：Blinn-Phong = 环境光 + 漫反射 + 镜面反射
 * - 环境光(Ambient): ka * 环境光强度
 * - 漫反射(Diffuse): kd * 光源强度 * max(0, N·L) (Lambert定律)  
 * - 镜面反射(Specular): ks * 光源强度 * max(0, N·H)^p (Blinn-Phong模型)
 */
Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    // === 步骤1：纹理采样 ===
    Eigen::Vector3f return_color = { 0, 0, 0 };
    if (payload.texture)
    {
        // 从纹理中采样颜色值
        // 将纹理坐标限制在[0,1]范围内，防止越界访问
        float u = std::clamp(static_cast<double>(payload.tex_coords(0)), 0.0, 1.0);
        float v = std::clamp(static_cast<double>(payload.tex_coords(1)), 0.0, 1.0);
        return_color = payload.texture->getColor(u, v);
    }
    
    // 将采样得到的纹理颜色转换为向量格式
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    // === 步骤2：材质参数设置 ===
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);  // 环境光反射系数(很小的值，产生微弱环境光)
    Eigen::Vector3f kd = texture_color / 255.f;                  // 漫反射系数(使用纹理颜色，归一化到[0,1])
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937); // 镜面反射系数(较高值产生明显高光)

    // === 步骤3：光源和场景参数设置 ===
    auto l1 = light{ {20, 20, 20}, {500, 500, 500} };   // 光源1：位置(20,20,20)，强度(500,500,500)
    auto l2 = light{ {-20, 20, 0}, {500, 500, 500} };   // 光源2：位置(-20,20,0)，强度(500,500,500)

    std::vector<light> lights = { l1, l2 };              // 光源列表（支持多光源）
    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };   // 环境光强度
    Eigen::Vector3f eye_pos{ 0, 0, 10 };                 // 观察者位置（摄像机位置）

    float p = 150;  // 高光指数（控制高光的锐利程度，值越大高光越集中）

    // === 步骤4：从输入数据中提取必要信息 ===
    Eigen::Vector3f color = texture_color;  // 当前像素的基础颜色（来自纹理）
    Eigen::Vector3f point = payload.view_pos; // 当前像素在视空间中的位置
    Eigen::Vector3f normal = payload.normal;  // 当前像素的法向量

    Eigen::Vector3f result_color = { 0, 0, 0 }; // 最终计算结果

    // === 步骤5：多光源光照计算循环 ===
    for (auto& light : lights)
    {
        // 计算从像素指向光源的单位向量（光照方向）
        Eigen::Vector3f light_dir = (light.position - point).normalized();
        
        // 计算从像素指向观察者的单位向量（视线方向）
        Eigen::Vector3f view_dir = (eye_pos - point).normalized();
        
        // 计算半角向量（Blinn-Phong模型的核心，用于计算镜面反射）
        // 半角向量是光线方向和视线方向的角平分线
        Eigen::Vector3f h = (light_dir + view_dir).normalized();
        
        // 计算光强衰减（距离平方反比定律）
        // 光强 = 原始光强 / 距离?
        Eigen::Vector3f attenuated_light = light.intensity / ((light.position - point).norm() * (light.position - point).norm());
        
        // === Blinn-Phong光照模型的三个分量 ===
        
        // 1. 环境光分量（Ambient）
        // 模拟来自四面八方的间接光照，与光源位置和方向无关
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        
        // 2. 漫反射分量（Diffuse）- Lambert余弦定律
        // 强度与表面法向量和光线方向的夹角余弦成正比
        // max(0, N·L) 确保光线从背面照射时不产生负值
        Eigen::Vector3f diffuse = kd.cwiseProduct(attenuated_light) * std::max(0.0f, normal.dot(light_dir));
        
        // 3. 镜面反射分量（Specular）- Blinn-Phong模型
        // 使用半角向量计算，比传统Phong模型更高效
        // 强度与法向量和半角向量夹角的p次幂成正比
        Eigen::Vector3f specular = ks.cwiseProduct(attenuated_light) * std::pow(std::max(0.0f, normal.dot(h)), p);

        // 累加当前光源对该像素的总贡献
        result_color += (diffuse + ambient + specular);
    }

    // === 步骤6：颜色空间转换和输出 ===
    // 将颜色从[0,1]范围转换到[0,255]范围（适合显示设备）
    return result_color * 255.f;
}

/**
 * Phong片段着色器 - 基于Blinn-Phong光照模型的标准光照渲染
 * 
 * 功能：使用顶点颜色作为漫反射系数，实现经典的Blinn-Phong光照效果
 * 
 * @param payload 片段着色器输入数据，包含：
 *   - color: 顶点颜色（作为漫反射系数使用）
 *   - view_pos: 像素在视空间中的位置
 *   - normal: 像素的法向量
 *   - texture: 纹理对象（在此着色器中未使用）
 * 
 * @return 计算得到的最终像素颜色(RGB，范围0-255)
 * 
 * 光照模型：Blinn-Phong = 环境光 + 漫反射 + 镜面反射
 * - 环境光(Ambient): ka * 环境光强度（模拟间接光照）
 * - 漫反射(Diffuse): kd * 光源强度 * max(0, N·L) (Lambert余弦定律)  
 * - 镜面反射(Specular): ks * 光源强度 * max(0, N·H)^p (Blinn-Phong高光模型)
 * 
 * 与texture_fragment_shader的区别：
 * - 使用顶点颜色而非纹理颜色作为漫反射系数
 * - 适用于纯色或顶点着色的模型
 */
Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    // === 步骤1：材质参数设置 ===
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);  // 环境光反射系数（低值产生柔和环境光）
    Eigen::Vector3f kd = payload.color;                          // 漫反射系数（使用顶点颜色，通常已归一化）
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937); // 镜面反射系数（高值产生明亮高光）

    // === 步骤2：光源配置 ===
    auto l1 = light{{20, 20, 20}, {500, 500, 500}};   // 主光源：右上前方，高强度白光
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};   // 辅助光源：左上方，补充光照

    std::vector<light> lights = {l1, l2};              // 双光源配置（模拟摄影棚照明）
    Eigen::Vector3f amb_light_intensity{10, 10, 10};   // 环境光强度（提供基础照明）
    Eigen::Vector3f eye_pos{0, 0, 10};                 // 观察者/摄像机位置

    float p = 150;  // 高光指数（控制高光集中度，值越大高光越尖锐）

    // === 步骤3：从输入载荷中提取渲染所需数据 ===
    Eigen::Vector3f color = payload.color;    // 片段的基础颜色（来自顶点插值）
    Eigen::Vector3f point = payload.view_pos; // 片段在视空间中的3D位置
    Eigen::Vector3f normal = payload.normal;  // 片段的表面法向量（用于光照计算）

    Eigen::Vector3f result_color = {0, 0, 0}; // 累积最终颜色结果

    // === 步骤4：多光源光照计算主循环 ===
    for (auto& light : lights)
    {
        // --- 4.1：计算基础光照向量 ---
        
        // 计算从片段指向光源的单位方向向量（入射光方向）
        Eigen::Vector3f light_dir = (light.position - point).normalized();
        
        // 计算从片段指向观察者的单位方向向量（视线方向）
        Eigen::Vector3f view_dir = (eye_pos - point).normalized();
        
        // 计算半角向量（Blinn-Phong模型核心）
        // 半角向量是光线方向与视线方向的角平分线，用于高效计算镜面反射
        Eigen::Vector3f h = (light_dir + view_dir).normalized();
        
        // --- 4.2：计算光照强度衰减 ---
        // 公式都是GB2312编码的
        // 基于物理的距离平方衰减：I = I? / r?
        // 距离越远，光照强度衰减越快，模拟真实光照传播
        Eigen::Vector3f attenuated_light = light.intensity / ((light.position - point).norm() * (light.position - point).norm());
        
        // --- 4.3：计算Blinn-Phong光照模型的三个分量 ---
        
        // 环境光分量（Ambient Reflection）
        // 模拟来自环境的间接光照，不依赖光源方向，提供物体的基础可见性（逐元素乘法）
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        
        // 漫反射分量（Diffuse Reflection - Lambert余弦定律）
        // 光照强度与表面法向量和光线方向夹角的余弦成正比
        // max(0, N·L) 确保背向光源的表面不会产生负光照
        Eigen::Vector3f diffuse = kd.cwiseProduct(attenuated_light) * std::max(0.0f, normal.dot(light_dir));
        
        // 镜面反射分量（Specular Reflection - Blinn-Phong高光模型）
        // 使用半角向量计算，比传统Phong模型计算更高效
        // 高光强度与法向量和半角向量夹角的p次幂成正比
        // 指数p控制高光的锐利程度：p越大，高光越集中越亮
        Eigen::Vector3f specular = ks.cwiseProduct(attenuated_light) * std::pow(std::max(0.0f, normal.dot(h)), p);

        // --- 4.4：累加当前光源的总贡献 ---
        // 将环境光、漫反射和镜面反射三个分量相加
        result_color += (diffuse + ambient + specular);
    }

    // === 步骤5：颜色空间转换和最终输出 ===
    // 将计算结果从[0,1]浮点范围转换为[0,255]整数范围
    // 适配标准RGB显示设备的颜色表示
    return result_color * 255.f;
}



/**
 * 位移映射片段着色器 - 通过高度图实现真实几何位移和法向量扰动
 * 
 * 功能：位移映射是凹凸映射的高级版本，不仅扰动表面法向量，还实际改变顶点位置，
 *      创造真实的几何变形效果，结合完整的Blinn-Phong光照模型渲染
 * 
 * 核心特点：
 * 1. 真实几何位移：根据高度图实际移动顶点位置
 * 2. 法向量重计算：基于梯度信息重新计算表面法向量
 * 3. 完整光照计算：使用位移后的几何进行Blinn-Phong光照
 * 4. 高质量渲染：比凹凸映射更真实，但计算成本更高
 * 
 * 与凹凸映射的区别：
 * - 凹凸映射：仅修改法向量，不改变几何形状
 * - 位移映射：既修改法向量，又实际位移几何顶点
 * 
 * @param payload 包含顶点位置、法向量、纹理坐标、纹理等完整片段信息
 * @return 经过位移映射和完整光照计算后的最终像素颜色(RGB，范围0-255)
 */
Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    // === 材质参数设置 ===
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);  // 环境光反射系数
    Eigen::Vector3f kd = payload.color;                          // 漫反射系数（使用顶点颜色）
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937); // 镜面反射系数

    // === 光源配置 ===
    auto l1 = light{{20, 20, 20}, {500, 500, 500}};   // 主光源：右上前方高强度照明
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};   // 辅助光源：左上方补充照明

    std::vector<light> lights = {l1, l2};              // 双光源系统
    Eigen::Vector3f amb_light_intensity{10, 10, 10};   // 环境光强度
    Eigen::Vector3f eye_pos{0, 0, 10};                 // 观察者/摄像机位置

    float p = 150;  // 高光指数（控制镜面反射的锐利程度）

    // === 从输入载荷提取基础几何信息 ===
    Eigen::Vector3f color = payload.color;    // 片段基础颜色
    Eigen::Vector3f point = payload.view_pos; // 片段在视空间中的原始位置
    Eigen::Vector3f normal = payload.normal;  // 片段的原始表面法向量

    // === 位移映射参数 ===
    float kh = 0.2, kn = 0.1;  // kh: 高度缩放系数，kn: 位移和法向量扰动强度系数

    // === 位移映射核心算法实现 ===
    
    // 步骤1: 提取原始法向量分量，构建切线空间基础
    // 设 n = normal = (x, y, z)
    float x = normal.x(), y = normal.y(), z = normal.z();
    
    // 步骤2: 构建切线向量 t
    // 公式：t = (x*y/√(x?+z?), √(x?+z?), z*y/√(x?+z?))
    // 确保切线向量与法向量正交，形成切线空间坐标系的第一个基向量
    Eigen::Vector3f t(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    
    // 步骤3: 构建副切线向量 b = n × t
    // 通过叉积确保副切线向量同时垂直于法向量和切线向量
    Eigen::Vector3f b(normal.cross(t));
    
    // 步骤4: 构建TBN变换矩阵 (Tangent-Bitangent-Normal)
    // 用于在切线空间和世界空间之间进行坐标变换
    Eigen::Matrix3f TBN;
    TBN << t[0], b[0], x,    // 第一行：各基向量的x分量
           t[1], b[1], y,    // 第二行：各基向量的y分量
           t[2], b[2], z;    // 第三行：各基向量的z分量

    // 步骤5: 获取纹理信息和安全的纹理坐标
    float w = payload.texture->width, h = payload.texture->height;  // 纹理分辨率
    float u = std::clamp(static_cast<double>(payload.tex_coords(0)), 0.0, 1.0);  // U坐标约束
    float v = std::clamp(static_cast<double>(payload.tex_coords(1)), 0.0, 1.0);  // V坐标约束

    // 步骤6: 计算高度图的梯度信息
    // dU：U方向（水平）的高度变化率，用于计算切线方向的法向量变化
    float dU = kh * kn * (payload.texture->getColor(u + 1 / w, v).norm() - payload.texture->getColor(u, v).norm());
    
    // dV：V方向（垂直）的高度变化率，用于计算副切线方向的法向量变化
    float dV = kh * kn * (payload.texture->getColor(u, (v + 1) / h).norm() - payload.texture->getColor(u, v).norm());
    
    // 步骤7: 构建切线空间中的法向量扰动
    // ln = (-dU, -dV, 1)：切线空间中的扰动法向量
    // 负号确保高度增加时法向量向外凸出
    Eigen::Vector3f ln(-dU, -dV, 1);
    
    // === 关键步骤：几何位移 ===
    // 步骤8: 根据高度图实际位移顶点位置
    // 新位置 = 原位置 + 位移强度 * 法向量 * 当前位置的高度值
    // 这是位移映射与凹凸映射的核心区别：实际改变几何形状
    point = point + kn * normal * payload.texture->getColor(u, v).norm();
    
    // 步骤9: 基于梯度重新计算表面法向量
    // 将切线空间的扰动法向量转换到世界空间
    normal = (TBN * ln).normalized();

    // === Blinn-Phong光照计算（使用位移后的几何） ===
    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // 注意：以下所有光照计算都基于位移后的顶点位置和重新计算的法向量
        
        // 计算从位移后的顶点指向光源的方向向量
        Eigen::Vector3f light_dir = (light.position - point).normalized();
        
        // 计算从位移后的顶点指向观察者的方向向量
        Eigen::Vector3f view_dir = (eye_pos - point).normalized();
        
        // 计算半角向量（Blinn-Phong模型的核心）
        Eigen::Vector3f h = (light_dir + view_dir).normalized();
        
        // 基于位移后的位置计算光照衰减（距离平方反比定律）
        Eigen::Vector3f attenuated_light = light.intensity / ((light.position - point).norm() * (light.position - point).norm());
        
        // === Blinn-Phong光照模型三分量计算 ===
        
        // 环境光分量：提供基础照明，与几何细节无关
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        
        // 漫反射分量：使用重新计算的法向量，体现位移后的表面朝向
        Eigen::Vector3f diffuse = kd.cwiseProduct(attenuated_light) * std::max(0.0f, normal.dot(light_dir));
        
        // 镜面反射分量：使用重新计算的法向量，产生符合新几何的高光效果
        Eigen::Vector3f specular = ks.cwiseProduct(attenuated_light) * std::pow(std::max(0.0f, normal.dot(h)), p);

        // 累加当前光源的总贡献
        result_color += (diffuse + ambient + specular);
    }

    // === 最终输出 ===
    // 返回经过位移映射处理和完整光照计算后的最终颜色
    // 颜色范围从[0,1]转换为[0,255]以适配显示设备
    return result_color * 255.f;
}


/**
 * 凹凸映射片段着色器 - 通过高度图扰动法向量实现表面细节
 * 
 * 功能：使用纹理作为高度图，通过计算高度变化的梯度来扰动表面法向量，
 *      从而在不改变几何形状的情况下模拟表面的凹凸细节
 * 
 * 核心原理：
 * 1. 从高度图采样获取当前位置及邻近位置的高度值
 * 2. 计算高度变化的梯度(dU, dV)
 * 3. 构建切线空间坐标系统(TBN矩阵)
 * 4. 将梯度信息转换为法向量扰动
 * 5. 输出扰动后的法向量用于可视化
 * 
 * @param payload 包含纹理坐标、法向量、纹理等片段信息
 * @return 扰动后的法向量颜色(RGB格式，范围0-255)
 */
Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    // === 材质和光照参数设置（虽然此处未用于光照计算，但保持完整性） ===
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);  // 环境光反射系数
    Eigen::Vector3f kd = payload.color;                          // 漫反射系数（顶点颜色）
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937); // 镜面反射系数

    // 光源配置（此实现中仅用于演示，实际未进行光照计算）
    auto l1 = light{{20, 20, 20}, {500, 500, 500}};   // 主光源
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};   // 辅助光源

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};   // 环境光强度
    Eigen::Vector3f eye_pos{0, 0, 10};                 // 观察者位置

    float p = 150;  // 高光指数

    // 从输入载荷中提取基础数据
    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    // === 凹凸映射核心参数 ===
    float kh = 0.2, kn = 0.1;  // kh: 高度缩放系数，kn: 法向量扰动强度系数

    // === 凹凸映射算法实现 ===
    
    // 步骤1: 提取原始法向量的分量
    // 设 n = normal = (x, y, z)
    float x = normal.x(), y = normal.y(), z = normal.z();
    
    // 步骤2: 构建切线向量 t
    // 公式：t = (x*y/sqrt(x?+z?), sqrt(x?+z?), z*y/sqrt(x?+z?))
    // 这个公式确保切线向量与法向量垂直，形成切线空间的一个基向量
    Eigen::Vector3f t(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    
    // 步骤3: 构建副切线向量 b
    // b = n × t (法向量与切线向量的叉积)
    // 叉积确保副切线向量同时垂直于法向量和切线向量
    Eigen::Vector3f b(normal.cross(t));
    
    // 步骤4: 构建TBN变换矩阵 (Tangent-Bitangent-Normal)
    // TBN矩阵用于将切线空间的向量转换到世界空间
    // 矩阵形式：[t b n] = [切线 副切线 法向量]
    Eigen::Matrix3f TBN;
    TBN << t[0], b[0], x,    // 第一行：T、B、N的x分量
           t[1], b[1], y,    // 第二行：T、B、N的y分量  
           t[2], b[2], z;    // 第三行：T、B、N的z分量

    // 步骤5: 获取纹理信息和坐标
    float w = payload.texture->width, h = payload.texture->height;  // 纹理尺寸
    float u = std::clamp(static_cast<double>(payload.tex_coords(0)), 0.0, 1.0);  // U坐标限制在[0,1]
    float v = std::clamp(static_cast<double>(payload.tex_coords(1)), 0.0, 1.0);  // V坐标限制在[0,1]

    // 步骤6: 计算高度图梯度
    // dU = kh * kn * (h(u+1/w,v) - h(u,v)) - U方向的高度变化率
    // 比较当前位置与右侧相邻像素的高度差，计算U方向梯度
    float dU = kh * kn * (payload.texture->getColor(u + 1 / w, v).norm() - payload.texture->getColor(u, v).norm());
    
    // dV = kh * kn * (h(u,v+1/h) - h(u,v)) - V方向的高度变化率  
    // 比较当前位置与上方相邻像素的高度差，计算V方向梯度
    float dV = kh * kn * (payload.texture->getColor(u, (v + 1) / h).norm() - payload.texture->getColor(u, v).norm());
    
    // 步骤7: 构建切线空间中的法向量扰动
    // ln = (-dU, -dV, 1) - 切线空间中的扰动法向量
    // 负号确保高度增加时法向量向外偏移，符合物理直觉
    // z分量为1表示保持主要的向上方向
    Eigen::Vector3f ln(-dU, -dV, 1);
    
    // 步骤8: 将扰动法向量从切线空间转换到世界空间
    // 最终法向量 = normalize(TBN * ln)
    // TBN变换将切线空间的扰动应用到世界空间的法向量上
    normal = (TBN * ln).normalized();

    // === 输出处理 ===
    // 此实现直接输出扰动后的法向量作为颜色，用于可视化凹凸效果
    // 在实际应用中，这个扰动法向量应该用于后续的光照计算
    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = normal;  // 将法向量作为颜色输出

    // 将法向量从[-1,1]范围转换到[0,255]显示范围
    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
