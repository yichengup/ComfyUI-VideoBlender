# ComfyUI-VideoBlender
## Video clip mixing

一个用于视频混合和合成的ComfyUI插件。

## 功能特点

- 提供多种混合模式
- 支持图层变换（位置、缩放、旋转）
- GPU加速支持

## 安装说明

1. 确保已安装ComfyUI
2. 克隆本仓库到ComfyUI的custom_nodes目录：
```bash
cd custom_nodes
git clone https://github.com/yourusername/ComfyUI-VideoBlender.git
```
3. 安装依赖：
```bash
cd ComfyUI-VideoBlender
pip install -r requirements.txt
```
![image](https://github.com/user-attachments/assets/a0a328de-a7d0-4c28-a1e1-813479ab99ae)


https://github.com/user-attachments/assets/b21f4026-8304-4380-8a72-0d4a1c8b193b

![3e8344fe06a019b730a41197beef93a](https://github.com/user-attachments/assets/9011da9c-3823-43a7-8c85-f2cbabcb4ecf)


https://github.com/user-attachments/assets/28ae478f-4f2c-4da6-bed4-a52df2ac42bb



## 使用方法

### 基本用法

1. 加载视频：使用LoadVideo节点加载视频文件
2. 创建图层：使用VideoBlendLayer节点设置混合模式和变换参数
3. 合成图层：使用VideoBlendStack节点将多个图层合成
4. 导出视频：使用SaveVideo节点保存结果

### 节点说明

#### VideoBlendLayer
- 输入：视频帧序列
- 参数：
  - blend_mode：混合模式
  - opacity：透明度
  - position：位置
  - scale：缩放
  - rotation：旋转
- 输出：处理后的帧序列

#### VideoBlendStack
- 输入：多个视频层
- 参数：
  - canvas_size：画布尺寸
  - background_color：背景颜色
- 输出：合成后的帧序列



## 示例工作流

在`example_workflow`目录中提供了示例工作流文件，展示了基本的视频混合操作。



## 注意事项

1. 视频文件大小：
   - 建议处理前压缩大型视频
   - 注意内存使用情况

2. 性能考虑：
   - 使用GPU可显著提升性能

3. 兼容性：
   - 支持常见视频格式
   - 建议使用MP4、MOV格式

## 更新日志

### v1.0.0
- 初始版本发布
- 实现基本的视频混合功能
- 支持多种混合模式
- 添加图层变换功能


## 许可证

MIT License 
