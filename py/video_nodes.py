import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Union
import torch.nn.functional as F

from .blend_modes import BlendModes
from .utils import VideoUtils

class VideoBlendLayer:
    """视频图层节点"""
    
    # 定义为类属性而不是实例属性
    BLEND_MODES = list(BlendModes.MODES.keys())
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "blend_mode": (s.BLEND_MODES,),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "position_x": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1
                }),
                "position_y": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 0.1
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LAYER_INFO")
    FUNCTION = "process_layer"
    CATEGORY = "VideoBlender"
    
    def process_layer(self, frames: List[np.ndarray], blend_mode: str, opacity: float,
                     position_x: int, position_y: int, scale: float, rotation: float) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """处理视频图层
        
        Args:
            frames: 输入帧序列
            blend_mode: 混合模式
            opacity: 不透明度
            position_x: X轴位置
            position_y: Y轴位置
            scale: 缩放比例
            rotation: 旋转角度
            
        Returns:
            Tuple[List[np.ndarray], Dict[str, Any]]: 处理后的帧序列和图层信息
        """
        processed_frames = []
        for frame in frames:
            # 应用变换
            transformed = VideoUtils.transform_frame(
                frame,
                position=(position_x, position_y),
                scale=scale,
                rotation=rotation
            )
            processed_frames.append(transformed)
            
        layer_info = {
            'blend_mode': blend_mode,
            'opacity': opacity,
            'position': (position_x, position_y),
            'scale': scale,
            'rotation': rotation
        }
        
        return (processed_frames, layer_info)

class VideoBlendStack:
    """视频图层堆栈节点（基础版）"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_frames": ("IMAGE",),
                "blend_frames": ("IMAGE",),
                "layer_info": ("LAYER_INFO",),
                "canvas_width": ("INT", {
                    "default": 1920,
                    "min": 1,
                    "max": 7680,
                    "step": 1
                }),
                "canvas_height": ("INT", {
                    "default": 1080,
                    "min": 1,
                    "max": 4320,
                    "step": 1
                }),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_layers"
    CATEGORY = "VideoBlender"
    
    def blend_layers(self, base_frames: Union[torch.Tensor, List], blend_frames: Union[torch.Tensor, List],
                    layer_info: Dict[str, Any], canvas_width: int, canvas_height: int,
                    background_color: str) -> Tuple[torch.Tensor]:
        """合成视频图层
        
        Args:
            base_frames: 基础帧序列(tensor或列表)
            blend_frames: 要混合的帧序列(tensor或列表)
            layer_info: 图层信息
            canvas_width: 画布宽度
            canvas_height: 画布高度
            background_color: 背景颜色 (十六进制格式，如 "#FFFFFF")
            
        Returns:
            Tuple[torch.Tensor]: 合成后的帧序列，格式为(B,H,W,C)，值范围[0,255]，类型uint8
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 解析背景颜色
        try:
            bg_color = background_color.lstrip('#')
            bg_r = int(bg_color[0:2], 16) / 255.0
            bg_g = int(bg_color[2:4], 16) / 255.0
            bg_b = int(bg_color[4:6], 16) / 255.0
        except Exception as e:
            bg_r = bg_g = bg_b = 1.0
        
        # 标准化输入
        if isinstance(base_frames, list):
            if len(base_frames) == 0:
                raise ValueError("Empty base_frames list")
            if isinstance(base_frames[0], np.ndarray):
                base_frames = torch.from_numpy(np.stack(base_frames))
            elif isinstance(base_frames[0], torch.Tensor):
                base_frames = torch.stack(base_frames)
                
        if isinstance(blend_frames, list):
            if len(blend_frames) == 0:
                raise ValueError("Empty blend_frames list")
            if isinstance(blend_frames[0], np.ndarray):
                blend_frames = torch.from_numpy(np.stack(blend_frames))
            elif isinstance(blend_frames[0], torch.Tensor):
                blend_frames = torch.stack(blend_frames)
        
        # 处理单帧输入
        if base_frames.dim() == 3:
            base_frames = base_frames.unsqueeze(0)
        if blend_frames.dim() == 3:
            blend_frames = blend_frames.unsqueeze(0)
        
        # 转换为BCHW格式用于处理
        if base_frames.size(-1) in [1, 3]:  # 如果是BHWC格式
            base_frames = base_frames.permute(0, 3, 1, 2)
        if blend_frames.size(-1) in [1, 3]:  # 如果是BHWC格式
            blend_frames = blend_frames.permute(0, 3, 1, 2)
        
        # 转换为float32并规范化到[0,1]
        base_frames = base_frames.float()
        blend_frames = blend_frames.float()
        
        # 使用min-max标准化确保数值范围一致性
        if base_frames.max() > 1.0 or blend_frames.max() > 1.0:
            base_frames = base_frames / 255.0 if base_frames.max() > 1.0 else base_frames
            blend_frames = blend_frames / 255.0 if blend_frames.max() > 1.0 else blend_frames
        
        # 确保通道数为3
        if base_frames.size(1) == 1:
            base_frames = base_frames.repeat(1, 3, 1, 1)
        if blend_frames.size(1) == 1:
            blend_frames = blend_frames.repeat(1, 3, 1, 1)
        
        # 调整尺寸
        if base_frames.size(2) != canvas_height or base_frames.size(3) != canvas_width:
            base_frames = F.interpolate(
                base_frames,
                size=(canvas_height, canvas_width),
                mode='bilinear',
                align_corners=False
            )
        if blend_frames.size(2) != canvas_height or blend_frames.size(3) != canvas_width:
            blend_frames = F.interpolate(
                blend_frames,
                size=(canvas_height, canvas_width),
                mode='bilinear',
                align_corners=False
            )
            
        # 创建输出tensor (BCHW格式)
        num_frames = base_frames.size(0)
        output = torch.zeros((num_frames, 3, canvas_height, canvas_width), 
                           dtype=torch.float32, device='cpu')
        
        # 设置背景颜色
        background = torch.tensor([bg_r, bg_g, bg_b], device=device).view(1, 3, 1, 1)
        output = output.to(device)
        output += background

        try:
            # 移动到GPU
            base_frames = base_frames.to(device)
            blend_frames = blend_frames.to(device)
            
            # 批量应用混合
            for i in range(num_frames):
                result = BlendModes.apply_blend(
                    base_frames[i],
                    blend_frames[i],
                    mode=layer_info['blend_mode'],
                    opacity=layer_info['opacity']
                )
                output[i] = result.cpu()
                
            # 清理GPU内存
            del base_frames
            del blend_frames
            torch.cuda.empty_cache()
            
        except Exception as e:
            raise e
            
        # 确保值范围在[0,1]之间
        output = output.clamp(0.0, 1.0)
        
        # 转换为uint8格式并改变为BHWC
        output = (output * 255.0).to(torch.uint8)
        output = output.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        return (output,)

class VideoBlendStackAdvanced:
    """视频多图层堆栈节点（高级版）"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames_1": ("IMAGE",),
                "frames_2": ("IMAGE",),
                "layer_info_1": ("LAYER_INFO",),
                "layer_info_2": ("LAYER_INFO",),
                "canvas_width": ("INT", {
                    "default": 1024,  # 默认值改为8的倍数
                    "min": 8,         # 最小值改为8
                    "max": 7680,
                    "step": 8         # 步进值改为8
                }),
                "canvas_height": ("INT", {
                    "default": 576,   # 默认值改为8的倍数
                    "min": 8,         # 最小值改为8
                    "max": 4320,
                    "step": 8         # 步进值改为8
                }),
                "background_mode": (["first_frame", "color"], {
                    "default": "first_frame"
                }),
                "background_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False
                })
            },
            "optional": {
                "frames_3": ("IMAGE",),
                "frames_4": ("IMAGE",),
                "frames_5": ("IMAGE",),
                "frames_6": ("IMAGE",),
                "frames_7": ("IMAGE",),
                "frames_8": ("IMAGE",),
                "layer_info_3": ("LAYER_INFO",),
                "layer_info_4": ("LAYER_INFO",),
                "layer_info_5": ("LAYER_INFO",),
                "layer_info_6": ("LAYER_INFO",),
                "layer_info_7": ("LAYER_INFO",),
                "layer_info_8": ("LAYER_INFO",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_layers"
    CATEGORY = "VideoBlender"
    
    def blend_layers(self, **kwargs) -> Tuple[torch.Tensor]:
        """合成多个视频图层
        
        Args:
            kwargs: 包含所有输入参数的字典
                frames_1, frames_2, ...: 图层帧序列
                layer_info_1, layer_info_2, ...: 图层信息
                canvas_width: 画布宽度
                canvas_height: 画布高度
                background_mode: 背景模式
                background_color: 背景颜色
                
        Returns:
            Tuple[torch.Tensor]: 合成后的帧序列
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 提取基本参数并调整为8的倍数
        canvas_width = (kwargs['canvas_width'] + 7) // 8 * 8
        canvas_height = (kwargs['canvas_height'] + 7) // 8 * 8
        background_mode = kwargs['background_mode']
        background_color = kwargs['background_color']
        
        # 收集有效的图层
        layers = []
        for i in range(1, 9):
            frames_key = f"frames_{i}"
            info_key = f"layer_info_{i}"
            
            if frames_key in kwargs and info_key in kwargs:
                frames = kwargs[frames_key]
                layer_info = kwargs[info_key]
                
                if frames is not None and layer_info is not None:
                    layers.append((frames, layer_info))
        
        if not layers:
            raise ValueError("至少需要一个有效的图层")
            
        # 处理第一个图层
        base_frames, base_info = layers[0]
        
        # 标准化输入
        if isinstance(base_frames, list):
            if len(base_frames) == 0:
                raise ValueError("Empty base_frames list")
            if isinstance(base_frames[0], np.ndarray):
                base_frames = torch.from_numpy(np.stack(base_frames))
            elif isinstance(base_frames[0], torch.Tensor):
                base_frames = torch.stack(base_frames)
                
        # 处理单帧输入
        if base_frames.dim() == 3:
            base_frames = base_frames.unsqueeze(0)
            
        # 转换为BCHW格式
        if base_frames.size(-1) in [1, 3]:  # 如果是BHWC格式
            base_frames = base_frames.permute(0, 3, 1, 2)
            
        # 转换为float32并规范化到[0,1]
        base_frames = base_frames.float()
        if base_frames.max() > 1.0:
            base_frames = base_frames / 255.0
            
        # 确保通道数为3
        if base_frames.size(1) == 1:
            base_frames = base_frames.repeat(1, 3, 1, 1)
            
        # 调整尺寸
        if base_frames.size(2) != canvas_height or base_frames.size(3) != canvas_width:
            base_frames = F.interpolate(
                base_frames,
                size=(canvas_height, canvas_width),
                mode='bilinear',
                align_corners=False
            )
            
        # 创建输出tensor (BCHW格式)
        num_frames = base_frames.size(0)
        output = torch.zeros((num_frames, 3, canvas_height, canvas_width), 
                           dtype=torch.float32, device=device)
        
        # 根据背景模式设置初始背景
        if background_mode == "first_frame":
            # 使用第一帧作为背景
            first_frame = base_frames[0].to(device)
            output = first_frame.expand(num_frames, -1, -1, -1)
        else:
            # 使用指定的背景颜色
            try:
                bg_color = background_color.lstrip('#')
                bg_r = int(bg_color[0:2], 16) / 255.0
                bg_g = int(bg_color[2:4], 16) / 255.0
                bg_b = int(bg_color[4:6], 16) / 255.0
            except Exception as e:
                bg_r = bg_g = bg_b = 0.0  # 默认黑色背景
                
            background = torch.tensor([bg_r, bg_g, bg_b], device=device).view(1, 3, 1, 1)
            output = output.to(device)
            output += background

        try:
            # 移动基础图层到GPU
            base_frames = base_frames.to(device)
            
            # 首先混合第一个图层
            for i in range(num_frames):
                result = BlendModes.apply_blend(
                    output[i],
                    base_frames[i],
                    mode=base_info['blend_mode'],
                    opacity=base_info['opacity']
                )
                output[i] = result
                
            # 清理基础图层内存
            del base_frames
            torch.cuda.empty_cache()
            
            # 处理其余图层
            for blend_frames, layer_info in layers[1:]:
                # 标准化当前图层
                if isinstance(blend_frames, list):
                    if isinstance(blend_frames[0], np.ndarray):
                        blend_frames = torch.from_numpy(np.stack(blend_frames))
                    elif isinstance(blend_frames[0], torch.Tensor):
                        blend_frames = torch.stack(blend_frames)
                        
                if blend_frames.dim() == 3:
                    blend_frames = blend_frames.unsqueeze(0)
                    
                if blend_frames.size(-1) in [1, 3]:
                    blend_frames = blend_frames.permute(0, 3, 1, 2)
                    
                blend_frames = blend_frames.float()
                if blend_frames.max() > 1.0:
                    blend_frames = blend_frames / 255.0
                    
                if blend_frames.size(1) == 1:
                    blend_frames = blend_frames.repeat(1, 3, 1, 1)
                    
                if blend_frames.size(2) != canvas_height or blend_frames.size(3) != canvas_width:
                    blend_frames = F.interpolate(
                        blend_frames,
                        size=(canvas_height, canvas_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                # 移动到GPU并混合
                blend_frames = blend_frames.to(device)
                
                for i in range(num_frames):
                    result = BlendModes.apply_blend(
                        output[i],
                        blend_frames[i],
                        mode=layer_info['blend_mode'],
                        opacity=layer_info['opacity']
                    )
                    output[i] = result
                    
                # 清理当前图层内存
                del blend_frames
                torch.cuda.empty_cache()
                
        except Exception as e:
            raise e
            
        # 确保值范围在[0,1]之间
        output = output.clamp(0.0, 1.0)
        
        # 转换为uint8格式并改变为BHWC
        output = (output * 255.0).to(torch.uint8)
        output = output.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        return (output,)

class VideoPreprocess:
    """视频预处理节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "interpolation": (["bilinear", "bicubic", "lanczos"], {
                    "default": "lanczos"
                }),
                "edge_feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "sharpness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "anti_aliasing": ("BOOLEAN", {
                    "default": True
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_frames"
    CATEGORY = "VideoBlender"
    
    def preprocess_frames(self, frames: Union[torch.Tensor, List], 
                         interpolation: str = "lanczos",
                         edge_feather: float = 0.0,
                         smoothing: float = 0.0,
                         sharpness: float = 0.5,
                         anti_aliasing: bool = True) -> Tuple[torch.Tensor]:
        """预处理视频帧"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 标准化输入
        if isinstance(frames, list):
            if len(frames) == 0:
                raise ValueError("Empty frames list")
            if isinstance(frames[0], np.ndarray):
                frames = torch.from_numpy(np.stack(frames))
            elif isinstance(frames[0], torch.Tensor):
                frames = torch.stack(frames)
        
        # 处理单帧输入
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
            
        # 转换为BCHW格式
        if frames.size(-1) in [1, 3]:  # 如果是BHWC格式
            frames = frames.permute(0, 3, 1, 2)
            
        # 转换为float32并规范化到[0,1]
        frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0
            
        # 确保通道数为3
        if frames.size(1) == 1:
            frames = frames.repeat(1, 3, 1, 1)
            
        # 移动到GPU
        frames = frames.to(device)
        
        try:
            # 1. 应用平滑处理
            if smoothing > 0:
                kernel_size = max(3, int(smoothing * 10) * 2 + 1)
                sigma = smoothing * 2.0
                kernel = self._gaussian_kernel(kernel_size, sigma).to(device)
                padding = kernel_size // 2
                
                # 分离卷积以提高效率
                frames = F.pad(frames, (padding, padding, padding, padding), mode='reflect')
                frames = F.conv2d(frames, kernel.view(1, 1, -1, 1).repeat(3, 1, 1, 1), 
                                groups=3, padding=0)
                frames = F.conv2d(frames, kernel.view(1, 1, 1, -1).repeat(3, 1, 1, 1), 
                                groups=3, padding=0)
                
            # 2. 边缘羽化
            if edge_feather > 0:
                mask = self._create_feather_mask(frames.shape[2:], edge_feather).to(device)
                frames = frames * mask
                
            # 3. 锐化处理
            if sharpness > 0.5:  # 只在锐化值大于默认值时应用
                sharp_strength = (sharpness - 0.5) * 2  # 映射到0-1范围
                kernel = self._sharpen_kernel(sharp_strength).to(device)
                frames = F.pad(frames, (1, 1, 1, 1), mode='reflect')
                frames = F.conv2d(frames, kernel.repeat(3, 1, 1, 1), groups=3, padding=0)
                
            # 4. 抗锯齿处理
            if anti_aliasing:
                # 计算上采样尺寸（确保是整数）
                up_h = int(frames.size(2) * 1.5)
                up_w = int(frames.size(3) * 1.5)
                
                # 计算下采样尺寸（确保是整数）
                down_h = frames.size(2)
                down_w = frames.size(3)
                
                # 先上采样
                frames = F.interpolate(frames, 
                                    size=(up_h, up_w),
                                    mode=interpolation,
                                    align_corners=False if interpolation != 'nearest' else None)
                                    
                # 再下采样回原始尺寸
                frames = F.interpolate(frames, 
                                    size=(down_h, down_w),
                                    mode=interpolation,
                                    align_corners=False if interpolation != 'nearest' else None)
                
            # 确保值范围在[0,1]之间
            frames = frames.clamp(0.0, 1.0)
            
            # 转换回BHWC格式
            frames = frames.permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            # 转换为uint8
            frames = (frames * 255.0).to(torch.uint8)
            
            return (frames,)
            
        except Exception as e:
            raise RuntimeError(f"预处理过程中出错: {str(e)}")
            
    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """创建高斯核"""
        x = torch.linspace(-sigma, sigma, kernel_size)
        kernel = torch.exp(-x**2 / (2*sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
        
    def _create_feather_mask(self, size: Tuple[int, int], strength: float) -> torch.Tensor:
        """创建羽化遮罩"""
        h, w = size
        y = torch.linspace(0, 1, h).view(-1, 1)
        x = torch.linspace(0, 1, w).view(1, -1)
        
        # 创建边缘渐变
        mask_y = torch.min(y / strength, (1 - y) / strength).clamp(0, 1)
        mask_x = torch.min(x / strength, (1 - x) / strength).clamp(0, 1)
        mask = torch.min(mask_y, mask_x)
        
        return mask.view(1, 1, h, w)
        
    def _sharpen_kernel(self, strength: float) -> torch.Tensor:
        """创建锐化核"""
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32)
        kernel = kernel * strength + torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=torch.float32)
        return kernel.view(1, 1, 3, 3)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "VideoBlendLayer": VideoBlendLayer,
    "VideoBlendStack": VideoBlendStack,
    "VideoBlendStackAdvanced": VideoBlendStackAdvanced,
    "VideoPreprocess": VideoPreprocess
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoBlendLayer": "Video Blend Layer (YC)",
    "VideoBlendStack": "Video Blend Stack (YC)",
    "VideoBlendStackAdvanced": "Video Blend Stack Advanced (YC)",
    "VideoPreprocess": "Video Preprocess (YC)"
} 
