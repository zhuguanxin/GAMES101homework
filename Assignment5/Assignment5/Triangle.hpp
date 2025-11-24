#pragma once

#include "Object.hpp"

#include <cstring>

/**
 * @brief 光线与三角形相交测试函数 - 实现Möller-Trumbore算法
 * 
 * 该函数使用高效的Möller-Trumbore算法来测试光线是否与三角形相交。
 * 该算法直接计算重心坐标和交点距离，避免了先计算平面方程的步骤。
 * 
 * 算法原理：
 * 1. 光线方程：P = orig + t * dir (t ≥ 0)
 * 2. 三角形重心坐标：P = (1-u-v)*v0 + u*v1 + v*v2 (u≥0, v≥0, u+v≤1)
 * 3. 联立求解得到 t, u, v 的值
 * 
 * @param v0    三角形顶点0的世界坐标
 * @param v1    三角形顶点1的世界坐标  
 * @param v2    三角形顶点2的世界坐标
 * @param orig  光线起点（原点）
 * @param dir   光线方向向量（应为单位向量）
 * @param tnear 输出参数：交点到光线起点的距离（t值）
 * @param u     输出参数：重心坐标u分量（对应顶点v1的权重）
 * @param v     输出参数：重心坐标v分量（对应顶点v2的权重）
 * @return      如果光线与三角形相交返回true，否则返回false
 */
bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, const Vector3f& orig,
                          const Vector3f& dir, float& tnear, float& u, float& v)
{
    // ==================== 构建三角形的边向量和光线起点向量 ====================
    
    // // E1, E2: 三角形的两条边向量，用于构建三角形的局部坐标系
    // // S: 从三角形顶点v0到光线起点的向量
    Vector3f E1 = v1 - v0, E2 = v2 - v0, S = orig - v0;
    
    // // ==================== 计算中间向量（Möller-Trumbore算法的核心步骤） ====================
    
    // 这个向量用于计算重心坐标u和检测算法的数值稳定性

    Vector3f S1 = crossProduct(dir,E2);
    
    // S2 = S × E1：起点向量与边E1的叉积  
    // 这个向量用于计算距离t和重心坐标v
    Vector3f S2 = crossProduct(S,E1);

    // ==================== 计算交点参数 ====================
    
    // 计算交点距离 t：光线参数方程中的参数
    // t = (S2 · E2) / (S1 · E1)
    tnear = 1.0 / dotProduct(S1,E1) * dotProduct(S2,E2);
    
    // 计算重心坐标 u：对应顶点v1的权重
    // u = (S1 · S) / (S1 · E1)
    u = 1.0 / dotProduct(S1,E1) * dotProduct(S1,S);
    
    // 计算重心坐标 v：对应顶点v2的权重  
    // v = (S2 · dir) / (S1 · E1)
    v = 1.0 / dotProduct(S1,E1) * dotProduct(S2,dir);

    // ==================== 相交性检测 ====================
    
    // 检测条件：
    // 1. tnear > 0：交点在光线的正方向上（不是背后）
    // 2. u > 0：重心坐标u为正（在三角形内部）
    // 3. v > 0：重心坐标v为正（在三角形内部）  
    // 4. (1-u-v) > 0：重心坐标的第三个分量为正，即u+v < 1（在三角形内部）
    //
    // 重心坐标解释：三角形内任意点P可表示为：
    // P = (1-u-v)*v0 + u*v1 + v*v2
    // 其中 u≥0, v≥0, u+v≤1 确保点在三角形内部

    return (tnear > 0 && u > 0 && v > 0 && (1 - u - v) > 0);
}

class MeshTriangle : public Object
{
public:
    MeshTriangle(const Vector3f* verts, const uint32_t* vertsIndex, const uint32_t& numTris, const Vector2f* st)
    {
        uint32_t maxIndex = 0;
        for (uint32_t i = 0; i < numTris * 3; ++i)
            if (vertsIndex[i] > maxIndex)
                maxIndex = vertsIndex[i];
        maxIndex += 1;
        vertices = std::unique_ptr<Vector3f[]>(new Vector3f[maxIndex]);
        memcpy(vertices.get(), verts, sizeof(Vector3f) * maxIndex);
        vertexIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numTris * 3]);
        memcpy(vertexIndex.get(), vertsIndex, sizeof(uint32_t) * numTris * 3);
        numTriangles = numTris;
        stCoordinates = std::unique_ptr<Vector2f[]>(new Vector2f[maxIndex]);
        memcpy(stCoordinates.get(), st, sizeof(Vector2f) * maxIndex);
    }

    bool intersect(const Vector3f& orig, const Vector3f& dir, float& tnear, uint32_t& index,
                   Vector2f& uv) const override
    {
        bool intersect = false;
        for (uint32_t k = 0; k < numTriangles; ++k)
        {
            const Vector3f& v0 = vertices[vertexIndex[k * 3]];
            const Vector3f& v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vector3f& v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, orig, dir, t, u, v) && t < tnear)
            {
                tnear = t;
                uv.x = u;
                uv.y = v;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    void getSurfaceProperties(const Vector3f&, const Vector3f&, const uint32_t& index, const Vector2f& uv, Vector3f& N,
                              Vector2f& st) const override
    {
        const Vector3f& v0 = vertices[vertexIndex[index * 3]];
        const Vector3f& v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vector3f& v2 = vertices[vertexIndex[index * 3 + 2]];
        Vector3f e0 = normalize(v1 - v0);
        Vector3f e1 = normalize(v2 - v1);
        N = normalize(crossProduct(e0, e1));
        const Vector2f& st0 = stCoordinates[vertexIndex[index * 3]];
        const Vector2f& st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vector2f& st2 = stCoordinates[vertexIndex[index * 3 + 2]];
        st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
    }

    Vector3f evalDiffuseColor(const Vector2f& st) const override
    {
        float scale = 5;
        float pattern = (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
        return lerp(Vector3f(0.815, 0.235, 0.031), Vector3f(0.937, 0.937, 0.231), pattern);
    }

    std::unique_ptr<Vector3f[]> vertices;
    uint32_t numTriangles;
    std::unique_ptr<uint32_t[]> vertexIndex;
    std::unique_ptr<Vector2f[]> stCoordinates;
};
