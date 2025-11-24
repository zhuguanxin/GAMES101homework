//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

/**
 * Implementation of Monte Carlo Path Tracing Algorithm
 * @param ray The incident ray to cast into the scene
 * @param depth Current recursion depth (for limiting infinite recursion)
 * @return The color contribution from this ray path
 */
 Vector3f Scene::castRay(const Ray& ray, int depth) const
 {
     // Find the intersection point between the ray and scene objects
     Intersection intersection = intersect(ray);
     Vector3f hitcolor = Vector3f(1);  // Initialize color to black

     // Case 1: Ray directly hits a light source
     if (intersection.emit.norm() > 0)
         hitcolor = Vector3f(1);  // Return light emission (simplified as white)

     // Case 2: Ray hits an object surface
     else if (intersection.happened)
     {
         // Calculate fundamental vectors for shading calculations
         Vector3f wo = normalize(-ray.direction);    // Outgoing direction (view direction)
         Vector3f p = intersection.coords;           // Intersection point coordinates
         Vector3f N = normalize(intersection.normal); // Surface normal at intersection point

         // === DIRECT LIGHTING CALCULATION ===

         // Sample a random point on all light sources in the scene
         float pdf_light = 0.0f;
         Intersection inter;
         sampleLight(inter, pdf_light);              // Get random light sample and its PDF
         Vector3f x = inter.coords;                  // Position of sampled light point
         Vector3f ws = normalize(x - p);             // Direction from hit point to light sample
         Vector3f NN = normalize(inter.normal);      // Normal of the light surface

         Vector3f L_dir = Vector3f(0);

         // Shadow test: Check if the light sample is visible from the hit point
         if ((intersect(Ray(p, ws)).coords - x).norm() < 0.01)
         {
             // Calculate direct lighting contribution using the rendering equation:
             // L_dir = Le * BRDF * cos(θ_out) * cos(θ_in) / (distance² * pdf_light)
             L_dir = inter.emit *                                    // Light source emission
                 intersection.m->eval(wo, ws, N) *               // BRDF evaluation at hit point
                 dotProduct(ws, N) *                             // Cosine factor: angle between light direction and surface normal
                 dotProduct(-ws, NN) /                           // Cosine factor: angle between light direction and light surface normal
                 (((x - p).norm() * (x - p).norm()) * pdf_light); // Distance squared attenuation divided by sampling PDF
         }

         // === INDIRECT LIGHTING CALCULATION (Russian Roulette Sampling) ===

         Vector3f L_indir = Vector3f(0);
         float P_RR = get_random_float();            // Generate random number for Russian Roulette termination

         // Russian Roulette: Probabilistically continue path tracing to avoid infinite recursion
         if (P_RR < Scene::RussianRoulette)
         {
             Vector3f wi = intersection.m->sample(wo, N);    // Sample new incident direction using BRDF importance sampling

             // Recursively calculate indirect lighting contribution:
             // L_indir = L_incoming * BRDF * cos(θ) / (BRDF_pdf * RussianRoulette_probability)
             L_indir = castRay(Ray(p, wi), depth) *          // Recursively trace ray in sampled direction
                 intersection.m->eval(wi, wo, N) *     // BRDF evaluation for the sampled direction
                 dotProduct(wi, N) /                   // Cosine term between incident direction and surface normal
                 (intersection.m->pdf(wi, wo, N) * Scene::RussianRoulette); // BRDF sampling PDF * Russian Roulette probability
         }

         // Combine direct and indirect lighting contributions
         hitcolor = L_indir + L_dir;
     }

     return hitcolor;
 }

//  /**
//  * Monte Carlo 路径追踪算法实现
//  * @param ray 投射到场景中的入射光线
//  * @param depth 当前递归深度（用于限制无限递归）
//  * @return 此光线路径的颜色贡献
//  */
// Vector3f Scene::castRay(const Ray& ray, int depth) const
// {
//     // 寻找光线与场景物体的交点
//     Intersection intersection = intersect(ray);
//     Vector3f hitcolor = Vector3f(1);  // 初始化颜色为黑色

//     // 情况1：光线直接击中光源
//     if (intersection.emit.norm() > 0)
//         hitcolor = Vector3f(0);  // 返回光源发射（简化为白色）

//     // 情况2：光线击中物体表面
//     else if (intersection.happened)
//     {
//         // 计算着色计算所需的基本向量
//         Vector3f wo = normalize(-ray.direction);    // 出射方向（观察方向）
//         Vector3f p = intersection.coords;           // 交点坐标
//         Vector3f N = normalize(intersection.normal); // 交点处的表面法向量

//         // === 直接光照计算 ===

//         // 在场景中的所有光源上采样一个随机点
//         float pdf_light = 0.0f;
//         Intersection inter;
//         sampleLight(inter, pdf_light);              // 获取随机光源采样及其概率密度函数值
//         Vector3f x = inter.coords;                  // 采样光源点的位置
//         Vector3f ws = normalize(x - p);             // 从击中点到光源采样点的方向
//         Vector3f NN = normalize(inter.normal);      // 光源表面的法向量

//         Vector3f L_dir = Vector3f(0);

//         // 阴影测试：检查光源采样点是否从击中点可见
//         if ((intersect(Ray(p, ws)).coords - x).norm() < 0.01)
//         {
//             // 使用渲染方程计算直接光照贡献：
//             // L_dir = Le * BRDF * cos(θ_out) * cos(θ_in) / (距离² * pdf_light)
//             L_dir = inter.emit *                                    // 光源发射
//                     intersection.m->eval(wo, ws, N) *               // 在击中点的BRDF评估
//                     dotProduct(ws, N) *                             // 余弦因子：光线方向与表面法向量的夹角
//                     dotProduct(-ws, NN) /                           // 余弦因子：光线方向与光源表面法向量的夹角
//                     (((x - p).norm() * (x - p).norm()) * pdf_light); // 距离平方衰减除以采样概率密度函数
//         }

//         // === 间接光照计算（俄罗斯轮盘赌采样）===

//         Vector3f L_indir = Vector3f(0);
//         float P_RR = get_random_float();            // 为俄罗斯轮盘赌终止生成随机数

//         // 俄罗斯轮盘赌：概率性地继续路径追踪以避免无限递归
//         if (P_RR < Scene::RussianRoulette)
//         {
//             Vector3f wi = intersection.m->sample(wo, N);    // 使用BRDF重要性采样获得新的入射方向

//             // 递归计算间接光照贡献：
//             // L_indir = L_incoming * BRDF * cos(θ) / (BRDF_pdf * RussianRoulette_probability)
//             L_indir = castRay(Ray(p, wi), depth) *          // 在采样方向上递归追踪光线
//                       intersection.m->eval(wi, wo, N) *     // 对采样方向的BRDF评估
//                       dotProduct(wi, N) /                   // 入射方向与表面法向量之间的余弦项
//                       (intersection.m->pdf(wi, wo, N) * Scene::RussianRoulette); // BRDF采样概率密度函数 * 俄罗斯轮盘赌概率
//         }

//         // 组合直接和间接光照贡献
//         hitcolor = L_indir + L_dir;
//     }

//     return hitcolor;
// }