#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include "Vector.hpp"
#include "Renderer.hpp"
#include "Scene.hpp"
#include <optional>

inline float deg2rad(const float &deg)
{ return deg * M_PI/180.0; }

// Compute reflection direction
Vector3f reflect(const Vector3f &I, const Vector3f &N)
{
    return I - 2 * dotProduct(I, N) * N;
}

// [comment]
// Compute refraction direction using Snell's law
//
// We need to handle with care the two possible situations:
//
//    - When the ray is inside the object
//
//    - When the ray is outside.
//
// If the ray is outside, you need to make cosi positive cosi = -N.I
//
// If the ray is inside, you need to invert the refractive indices and negate the normal N
// [/comment]
Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    Vector3f n = N;
    if (cosi < 0) { cosi = -cosi; } else { std::swap(etai, etat); n= -N; }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

// [comment]
// Compute Fresnel equation
//
// \param I is the incident view direction
//
// \param N is the normal at the intersection point
//
// \param ior is the material refractive index
// [/comment]
float fresnel(const Vector3f &I, const Vector3f &N, const float &ior)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    if (cosi > 0) {  std::swap(etai, etat); }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        return 1;
    }
    else {
        float cost = sqrtf(std::max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        return (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}

// [comment]
// Returns true if the ray intersects an object, false otherwise.
//
// \param orig is the ray origin
// \param dir is the ray direction
// \param objects is the list of objects the scene contains
// \param[out] tNear contains the distance to the cloesest intersected object.
// \param[out] index stores the index of the intersect triangle if the interesected object is a mesh.
// \param[out] uv stores the u and v barycentric coordinates of the intersected point
// \param[out] *hitObject stores the pointer to the intersected object (used to retrieve material information, etc.)
// \param isShadowRay is it a shadow ray. We can return from the function sooner as soon as we have found a hit.
// [/comment]
std::optional<hit_payload> trace(
        const Vector3f &orig, const Vector3f &dir,
        const std::vector<std::unique_ptr<Object> > &objects)
{
    float tNear = kInfinity;
    std::optional<hit_payload> payload;
    for (const auto & object : objects)
    {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (object->intersect(orig, dir, tNearK, indexK, uvK) && tNearK < tNear)
        {
            payload.emplace();
            payload->hit_obj = object.get();
            payload->tNear = tNearK;
            payload->index = indexK;
            payload->uv = uvK;
            tNear = tNearK;
        }
    }

    return payload;
}

// [comment]
// Implementation of the Whitted-style light transport algorithm (E [S*] (D|G) L)
//
// This function is the function that compute the color at the intersection point
// of a ray defined by a position and a direction. Note that thus function is recursive (it calls itself).
//
// If the material of the intersected object is either reflective or reflective and refractive,
// then we compute the reflection/refraction direction and cast two new rays into the scene
// by calling the castRay() function recursively. When the surface is transparent, we mix
// the reflection and refraction color using the result of the fresnel equations (it computes
// the amount of reflection and refraction depending on the surface normal, incident view direction
// and surface refractive index).
//
// If the surface is diffuse/glossy we use the Phong illumation model to compute the color
// at the intersection point.
// [/comment]
Vector3f castRay(
        const Vector3f &orig, const Vector3f &dir, const Scene& scene,
        int depth)
{
    if (depth > scene.maxDepth) {
        return Vector3f(0.0,0.0,0.0);
    }

    Vector3f hitColor = scene.backgroundColor;
    if (auto payload = trace(orig, dir, scene.get_objects()); payload)
    {
        Vector3f hitPoint = orig + dir * payload->tNear;
        Vector3f N; // normal
        Vector2f st; // st coordinates
        payload->hit_obj->getSurfaceProperties(hitPoint, dir, payload->index, payload->uv, N, st);
        switch (payload->hit_obj->materialType) {
            case REFLECTION_AND_REFRACTION:
            {
                Vector3f reflectionDirection = normalize(reflect(dir, N));
                Vector3f refractionDirection = normalize(refract(dir, N, payload->hit_obj->ior));
                Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                                             hitPoint - N * scene.epsilon :
                                             hitPoint + N * scene.epsilon;
                Vector3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0) ?
                                             hitPoint - N * scene.epsilon :
                                             hitPoint + N * scene.epsilon;
                Vector3f reflectionColor = castRay(reflectionRayOrig, reflectionDirection, scene, depth + 1);
                Vector3f refractionColor = castRay(refractionRayOrig, refractionDirection, scene, depth + 1);
                float kr = fresnel(dir, N, payload->hit_obj->ior);
                hitColor = reflectionColor * kr + refractionColor * (1 - kr);
                break;
            }
            case REFLECTION:
            {
                float kr = fresnel(dir, N, payload->hit_obj->ior);
                Vector3f reflectionDirection = reflect(dir, N);
                Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                                             hitPoint + N * scene.epsilon :
                                             hitPoint - N * scene.epsilon;
                hitColor = castRay(reflectionRayOrig, reflectionDirection, scene, depth + 1) * kr;
                break;
            }
            default:
            {
                // [comment]
                // We use the Phong illumation model int the default case. The phong model
                // is composed of a diffuse and a specular reflection component.
                // [/comment]
                Vector3f lightAmt = 0, specularColor = 0;
                Vector3f shadowPointOrig = (dotProduct(dir, N) < 0) ?
                                           hitPoint + N * scene.epsilon :
                                           hitPoint - N * scene.epsilon;
                // [comment]
                // Loop over all lights in the scene and sum their contribution up
                // We also apply the lambert cosine law
                // [/comment]
                for (auto& light : scene.get_lights()) {
                    Vector3f lightDir = light->position - hitPoint;
                    // square of the distance between hitPoint and the light
                    float lightDistance2 = dotProduct(lightDir, lightDir);
                    lightDir = normalize(lightDir);
                    float LdotN = std::max(0.f, dotProduct(lightDir, N));
                    // is the point in shadow, and is the nearest occluding object closer to the object than the light itself?
                    auto shadow_res = trace(shadowPointOrig, lightDir, scene.get_objects());
                    bool inShadow = shadow_res && (shadow_res->tNear * shadow_res->tNear < lightDistance2);

                    lightAmt += inShadow ? 0 : light->intensity * LdotN;
                    Vector3f reflectionDirection = reflect(-lightDir, N);

                    specularColor += powf(std::max(0.f, -dotProduct(reflectionDirection, dir)),
                        payload->hit_obj->specularExponent) * light->intensity;
                }

                hitColor = lightAmt * payload->hit_obj->evalDiffuseColor(st) * payload->hit_obj->Kd + specularColor * payload->hit_obj->Ks;
                break;
            }
        }
    }

    return hitColor;
}

/**
 * @brief 主渲染函数 - 光线追踪渲染器的核心入口
 *
 * 该函数实现了基本的光线追踪渲染流程：
 * 1. 为图像中的每个像素生成主光线
 * 2. 将光线投射到场景中进行相交测试和着色计算
 * 3. 将渲染结果保存为PPM图像文件
 *
 * @param scene 包含场景几何体、光源、材质等信息的场景对象
 */
void Renderer::Render(const Scene& scene)
{
    // 创建帧缓冲区，用于存储每个像素的RGB颜色值
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    // 根据视场角计算投影缩放因子
    // fov是全视场角，这里取一半然后计算正切值来获得屏幕边界
    float scale = std::tan(deg2rad(scene.fov * 0.5f));

    // 计算图像宽高比，用于校正水平方向的像素坐标
    float imageAspectRatio = scene.width / (float)scene.height;

    // 摄像机位置设置为原点，作为所有光线的起始点
    Vector3f eye_pos(0);

    // 帧缓冲区的线性索引，用于按行优先顺序填充像素数据
    int m = 0;

    // 外层循环：遍历图像的每一行（从上到下）
    for (int j = 0; j < scene.height; ++j)
    {
        // 内层循环：遍历当前行的每一列（从左到右）
        for (int i = 0; i < scene.width; ++i)
        {
            // === 生成通过当前像素的主光线方向 ===
            float x;
            float y;

            // 第一步：将屏幕像素坐标转换为标准化坐标[-0.5, 0.5]
            // i/width 得到[0,1]范围，减去0.5得到[-0.5, 0.5]
            x = (float)i / scene.width - 0.5;

            // 对于y坐标需要垂直翻转，因为：
            // - 屏幕坐标系：原点在左上角，y轴向下
            // - 世界坐标系：y轴向上
            y = (float)(scene.height - j) / scene.height - 0.5;

            // 第二步：将标准化坐标转换到世界空间
            // 应用视场角缩放和宽高比校正
            x *= scale * imageAspectRatio;  // 水平方向需要宽高比校正
            y *= scale;                     // 垂直方向只需视场角缩放

            // 第三步：构建光线方向向量
            // z=-1表示光线指向屏幕内部（右手坐标系中的负z方向）
            Vector3f dir = Vector3f(x, y, -1);

            // 归一化方向向量，确保它是单位向量
            // 这对于后续的光线-几何体相交计算很重要
            dir = normalize(dir);

            // 投射光线到场景中，执行光线追踪算法
            // 返回该像素的最终颜色值并存储到帧缓冲区
            framebuffer[m++] = castRay(eye_pos, dir, scene, 0);
        }

        // 更新渲染进度，用于显示渲染百分比
        UpdateProgress(j / (float)scene.height);
    }

    // === 将渲染结果保存为PPM图像文件 ===

    // 以二进制写模式打开输出文件
    FILE* fp = fopen("binary.ppm", "wb");

    // 写入PPM文件头信息：
    // P6: 表示二进制RGB格式
    // width height: 图像尺寸
    // 255: 最大颜色值（8位）
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);

    // 遍历帧缓冲区，将浮点颜色值转换为8位整数格式
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];  // RGB三个通道

        // 将[0,1]范围的浮点颜色值映射到[0,255]的整数范围
        // clamp函数确保颜色值不会超出有效范围
        color[0] = (char)(255 * clamp(0, 1, framebuffer[i].x));  // Red通道
        color[1] = (char)(255 * clamp(0, 1, framebuffer[i].y));  // Green通道
        color[2] = (char)(255 * clamp(0, 1, framebuffer[i].z));  // Blue通道

        // 将RGB数据写入文件
        fwrite(color, 1, 3, fp);
    }

    // 关闭文件，完成图像保存
    fclose(fp);
}
