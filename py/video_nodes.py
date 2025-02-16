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
    """视频图层堆栈节点"""
    
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

# 节点注册
NODE_CLASS_MAPPINGS = {
    "VideoBlendLayer": VideoBlendLayer,
    "VideoBlendStack": VideoBlendStack
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoBlendLayer": "Video Blend Layer (YC)",
    "VideoBlendStack": "Video Blend Stack (YC)"
} 