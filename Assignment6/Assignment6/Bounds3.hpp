//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BOUNDS3_H
#define RAYTRACING_BOUNDS3_H
#include "Ray.hpp"
#include "Vector.hpp"
#include <limits>
#include <array>

class Bounds3
{
  public:
    Vector3f pMin, pMax; // two points to specify the bounding box
    Bounds3()
    {
        double minNum = std::numeric_limits<double>::lowest();
        double maxNum = std::numeric_limits<double>::max();
        pMax = Vector3f(minNum, minNum, minNum);
        pMin = Vector3f(maxNum, maxNum, maxNum);
    }
    Bounds3(const Vector3f p) : pMin(p), pMax(p) {}
    Bounds3(const Vector3f p1, const Vector3f p2)
    {
        pMin = Vector3f(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = Vector3f(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
    }

    Vector3f Diagonal() const { return pMax - pMin; }
    int maxExtent() const
    {
        Vector3f d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    double SurfaceArea() const
    {
        Vector3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    Vector3f Centroid() { return 0.5 * pMin + 0.5 * pMax; }
    Bounds3 Intersect(const Bounds3& b)
    {
        return Bounds3(Vector3f(fmax(pMin.x, b.pMin.x), fmax(pMin.y, b.pMin.y),
                                fmax(pMin.z, b.pMin.z)),
                       Vector3f(fmin(pMax.x, b.pMax.x), fmin(pMax.y, b.pMax.y),
                                fmin(pMax.z, b.pMax.z)));
    }

    Vector3f Offset(const Vector3f& p) const
    {
        Vector3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    bool Overlaps(const Bounds3& b1, const Bounds3& b2)
    {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
        return (x && y && z);
    }

    bool Inside(const Vector3f& p, const Bounds3& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
                p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
    }
    inline const Vector3f& operator[](int i) const
    {
        return (i == 0) ? pMin : pMax;
    }

    inline bool IntersectP(const Ray& ray, const Vector3f& invDir,
                           const std::array<int, 3>& dirisNeg) const;
};



/**
 * 光线与轴对齐包围盒（AABB）求交测试
 * 使用Slab方法（平板方法）进行快速求交计算
 *
 * @param ray 待测试的光线
 * @param invDir 光线方向的倒数 (1/dx, 1/dy, 1/dz)，用于避免除法运算提高性能
 * @param dirIsNeg 光线方向的符号标记 [x<0, y<0, z<0]，用于简化逻辑判断
 * @return 如果光线与包围盒相交返回true，否则返回false
 */
inline bool Bounds3::IntersectP(const Ray& ray, const Vector3f& invDir,
    const std::array<int, 3>& dirIsNeg) const
{
    // 参数说明：
    // invDir: 光线方向的倒数(1/x, 1/y, 1/z)，使用倒数是因为乘法比除法快
    // dirIsNeg: 光线方向符号[int(x<0), int(y<0), int(z<0)]，用于简化逻辑判断

    // TODO: 测试光线是否与包围盒相交

    // === Slab方法原理 ===
    // 光线方程：P(t) = origin + t * direction
    // 平板方程：在某轴上的两个平行平面，如X轴上的x=pMin.x和x=pMax.x
    // 求交点参数t：t = (plane_coord - origin_coord) / direction_coord

    double tmin(0.0), tmax(0.0);  // 临时变量存储t值

    // === X轴方向的平板测试 ===
    // 计算光线与X轴两个垂直平面的交点参数t
    tmin = (pMin.x - ray.origin.x) * invDir.x;  // 与x=pMin.x平面的交点
    tmax = (pMax.x - ray.origin.x) * invDir.x;  // 与x=pMax.x平面的交点

    // 根据光线方向确定进入和退出的t值
    // 如果方向为正(dirIsNeg[0]==0)：tmin对应进入，tmax对应退出
    // 如果方向为负(dirIsNeg[0]==1)：tmax对应进入，tmin对应退出
    double tmin_x = dirIsNeg[0] > 0 ? tmax : tmin;  // X方向进入包围盒的t值
    double tmax_x = dirIsNeg[0] > 0 ? tmin : tmax;  // X方向退出包围盒的t值

    // === Y轴方向的平板测试 ===
    // 计算光线与Y轴两个垂直平面的交点参数t
    tmin = (pMin.y - ray.origin.y) * invDir.y;  // 与y=pMin.y平面的交点
    tmax = (pMax.y - ray.origin.y) * invDir.y;  // 与y=pMax.y平面的交点

    // 根据光线方向确定Y轴进入和退出的t值
    double tmin_y = dirIsNeg[1] > 0 ? tmax : tmin;  // Y方向进入包围盒的t值
    double tmax_y = dirIsNeg[1] > 0 ? tmin : tmax;  // Y方向退出包围盒的t值

    // === Z轴方向的平板测试 ===
    // 计算光线与Z轴两个垂直平面的交点参数t
    tmin = (pMin.z - ray.origin.z) * invDir.z;  // 与z=pMin.z平面的交点
    tmax = (pMax.z - ray.origin.z) * invDir.z;  // 与z=pMax.z平面的交点

    // 根据光线方向确定Z轴进入和退出的t值
    double tmin_z = dirIsNeg[2] > 0 ? tmax : tmin;  // Z方向进入包围盒的t值
    double tmax_z = dirIsNeg[2] > 0 ? tmin : tmax;  // Z方向退出包围盒的t值

    // === 最终相交判断 ===
    // 光线与包围盒相交的充分必要条件：
    // 所有轴向上的"最晚进入时间" < 所有轴向上的"最早退出时间"
    // 即：max(tmin_x, tmin_y, tmin_z) < min(tmax_x, tmax_y, tmax_z)

    double t_enter = std::max(tmin_x, std::max(tmin_y, tmin_z));  // 最后进入包围盒的时间
    double t_exit = std::min(tmax_x, std::min(tmax_y, tmax_z));   // 最先退出包围盒的时间

    if (t_enter < t_exit)
        return true;   // 相交：光线在某个时间段内完全位于包围盒内
    else
        return false;  // 不相交：光线错过了包围盒
}

inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b1.pMin, b2.pMin);
    ret.pMax = Vector3f::Max(b1.pMax, b2.pMax);
    return ret;
}

inline Bounds3 Union(const Bounds3& b, const Vector3f& p)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b.pMin, p);
    ret.pMax = Vector3f::Max(b.pMax, p);
    return ret;
}

#endif // RAYTRACING_BOUNDS3_H
