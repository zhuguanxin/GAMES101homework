#include <algorithm>
#include <cassert>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node;
    }
    else {
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
        int dim = centroidBounds.maxExtent();
        switch (dim) {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                       f2->getBounds().Centroid().x;
            });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                       f2->getBounds().Centroid().y;
            });
            break;
        case 2:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                       f2->getBounds().Centroid().z;
            });
            break;
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + (objects.size() / 2);
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

/**
 * 在BVH树中递归查找光线与场景物体的最近交点
 * 使用深度优先搜索遍历BVH树，利用包围盒快速剔除不相交的子树
 *
 * @param node 当前遍历的BVH树节点
 * @param ray 待测试的光线
 * @return 返回最近的交点信息，如果没有交点则返回空的Intersection对象
 */
Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO: 遍历BVH树查找交点

    // === 预计算光线参数以优化包围盒求交性能 ===

    // 计算光线方向的符号数组：如果方向分量>0则为1，否则为0
    // 用于在包围盒求交时快速确定进入/退出面
    std::array<int, 3> dirIsNeg = {
        int(ray.direction.x > 0),  // X方向是否为正
        int(ray.direction.y > 0),  // Y方向是否为正
        int(ray.direction.z > 0)   // Z方向是否为正
    };

    // 计算光线方向的倒数，避免在包围盒求交时进行除法运算
    // 注意：这里假设direction的各分量都不为0
    Vector3f invDir = Vector3f(
        1.0f / ray.direction.x,    // X方向倒数
        1.0f / ray.direction.y,    // Y方向倒数
        1.0f / ray.direction.z     // Z方向倒数
    );

    // === 早期剔除：包围盒求交测试 ===

    // 如果光线与当前节点的包围盒不相交，直接返回空交点
    // 这是BVH加速的核心：快速剔除大量不可能相交的几何体
    if (node->bounds.IntersectP(ray, invDir, dirIsNeg) == false)
        return Intersection();  // 返回空交点（表示无交点）

    // === 叶子节点处理：直接几何求交 ===

    // 判断是否为叶子节点（没有子节点）
    if (node->left == nullptr && node->right == nullptr)
        // 叶子节点包含实际的几何对象，直接进行精确的几何求交测试
        return node->object->getIntersection(ray);

    // === 内部节点处理：递归遍历子树 ===

    // 递归测试左子树，获取左子树中的最近交点
    Intersection hit1 = getIntersection(node->left, ray);

    // 递归测试右子树，获取右子树中的最近交点
    Intersection hit2 = getIntersection(node->right, ray);

    // === 选择最近交点 ===

    // 比较两个交点的距离，返回距离光线起点更近的交点
    // 如果某个交点无效（distance为无穷大），另一个交点会被自动选中
    return (hit1.distance < hit2.distance) ? hit1 : hit2;
}