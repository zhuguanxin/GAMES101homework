# GAMES101 现代计算机图形学入门 - 课程作业

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-lightgrey.svg)]()
[![Visual Studio](https://img.shields.io/badge/IDE-Visual%20Studio%202022-purple.svg)]()
[![Language](https://img.shields.io/badge/language-C%2B%2B-blue.svg)]()

本仓库包含了 GAMES101《现代计算机图形学入门》课程的所有编程作业实现。

## 📚 课程简介

GAMES101 是由闫令琪老师主讲的计算机图形学入门课程，涵盖了现代计算机图形学的基础理论和实践应用。

**课程官网**: [GAMES101](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)

## 🎯 作业列表

### Assignment 0 - 矩阵变换基础
- **内容**: 实现基本的 2D 变换（旋转、平移）
- **技术栈**: Eigen 线性代数库
- **主要功能**: 
  - 点的旋转变换（逆时针旋转45度）
  - 平移变换
  - 复合变换的应用

## 🛠️ 开发环境

### 系统要求
- **操作系统**: Windows 11
- **开发环境**: Visual Studio 2022
- **编译器**: MSVC (Microsoft Visual C++)
- **C++ 标准**: C++17 或更高版本

### 依赖库
- **Eigen**: 用于线性代数运算的 C++ 模板库
- **OpenCV** (部分作业): 用于图像处理和显示

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone git@github.com:zhuguanxin/GAMES101homework.git
```

### 2. 环境配置

**参考我的笔记**: [GAMES101作业笔记](https://zhuanlan.zhihu.com/p/1976419613685334201)

### 3. 编译运行

1. 在 Visual Studio 中设置为 **Release** 或 **Debug** 模式
2. 选择 **x64** 平台
3. 按 `Ctrl + F5` 编译并运行

## 📁 项目结构

```
GAMES101homework/
├── Assignment0/           # 作业0 
│   ├── Assignment0.sln   # Visual Studio 解决方案
│   ├── Assignment0.vcxproj  # 项目文件
│   ├── main.cpp          # 主程序
│   └── test.jpg          # 测试图片
├── Assignment1/           # 作业1 
├── Assignment2/           # 作业2 
├── ...
├── .gitignore            # Git 忽略文件
└── README.md             # 项目说明文档
```

