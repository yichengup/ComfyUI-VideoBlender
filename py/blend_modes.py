import torch
import torch.nn.functional as F

class BlendModes:
    @staticmethod
    def normal(base, blend, opacity=1.0):
        """正常模式 (GPU)"""
        return (1.0 - opacity) * base + opacity * blend

    @staticmethod
    def multiply(base, blend, opacity=1.0):
        """正片叠底 (GPU)
        结果 = 基础层 × 混合层
        """
        blended = base * blend
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def screen(base, blend, opacity=1.0):
        """滤色 (GPU)
        结果 = 1 - (1 - 基础层) × (1 - 混合层)
        """
        blended = 1.0 - (1.0 - base) * (1.0 - blend)
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def darken(base, blend, opacity=1.0):
        """变暗 (GPU)
        结果 = min(基础层, 混合层)
        """
        blended = torch.minimum(base, blend)
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def lighten(base, blend, opacity=1.0):
        """变亮 (GPU)
        结果 = max(基础层, 混合层)
        """
        blended = torch.maximum(base, blend)
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def overlay(base, blend, opacity=1.0):
        """叠加 (GPU)
        if 基础层 <= 0.5:
            结果 = 2 × 基础层 × 混合层
        else:
            结果 = 1 - 2 × (1 - 基础层) × (1 - 混合层)
        """
        mask = base <= 0.5
        blended = torch.zeros_like(base)
        blended[mask] = 2 * base[mask] * blend[mask]
        blended[~mask] = 1.0 - 2 * (1.0 - base[~mask]) * (1.0 - blend[~mask])
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def color_dodge(base, blend, opacity=1.0):
        """颜色减淡 (GPU)
        结果 = 基础层 / (1 - 混合层)
        """
        blend = torch.clamp(blend, 1e-7, 1.0)  # 防止除零
        blended = base / (1.0 - blend)
        blended = torch.clamp(blended, 0.0, 1.0)
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def color_burn(base, blend, opacity=1.0):
        """颜色加深 (GPU)
        结果 = 1 - (1 - 基础层) / 混合层
        """
        blend = torch.clamp(blend, 1e-7, 1.0)  # 防止除零
        blended = 1.0 - (1.0 - base) / blend
        blended = torch.clamp(blended, 0.0, 1.0)
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def soft_light(base, blend, opacity=1.0):
        """柔光 (GPU)"""
        mask = blend <= 0.5
        blended = torch.zeros_like(base)
        blended[mask] = base[mask] - (1.0 - 2 * blend[mask]) * base[mask] * (1.0 - base[mask])
        blended[~mask] = base[~mask] + (2 * blend[~mask] - 1.0) * (torch.sqrt(base[~mask]) - base[~mask])
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def hard_light(base, blend, opacity=1.0):
        """强光 (GPU)"""
        mask = blend <= 0.5
        blended = torch.zeros_like(base)
        blended[mask] = 2 * base[mask] * blend[mask]
        blended[~mask] = 1.0 - 2 * (1.0 - base[~mask]) * (1.0 - blend[~mask])
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def difference(base, blend, opacity=1.0):
        """差值 (GPU)"""
        blended = torch.abs(base - blend)
        return (1.0 - opacity) * base + opacity * blended

    @staticmethod
    def exclusion(base, blend, opacity=1.0):
        """排除 (GPU)"""
        blended = base + blend - 2.0 * base * blend
        return (1.0 - opacity) * base + opacity * blended

    # 混合模式映射
    MODES = {
        'normal': normal,            # 正常
        'multiply': multiply,        # 正片叠底
        'screen': screen,           # 滤色
        'darken': darken,          # 变暗
        'lighten': lighten,        # 变亮
        'overlay': overlay,         # 叠加
        'color_dodge': color_dodge, # 颜色减淡
        'color_burn': color_burn,   # 颜色加深
        'soft_light': soft_light,   # 柔光
        'hard_light': hard_light,    # 强光
        'difference': difference,
        'exclusion': exclusion
    }

    @classmethod
    def apply_blend(cls, base, blend, mode='normal', opacity=1.0):
        """应用混合模式 (GPU版本)
        
        Args:
            base: 基础图层 (torch.Tensor, 范围[0,1])
            blend: 混合图层 (torch.Tensor, 范围[0,1])
            mode: 混合模式
            opacity: 不透明度
            
        Returns:
            torch.Tensor: 混合结果
        """
        if mode not in cls.MODES:
            raise ValueError(f"不支持的混合模式: {mode}")
            
        # 确保输入是tensor
        if not isinstance(base, torch.Tensor):
            base = torch.from_numpy(base).float()
        if not isinstance(blend, torch.Tensor):
            blend = torch.from_numpy(blend).float()
            
        # 移动到GPU
        device = base.device
        if device.type == 'cpu' and torch.cuda.is_available():
            device = torch.device('cuda')
            base = base.to(device)
            blend = blend.to(device)
            
        # 确保值范围在[0,1]之间，使用min-max标准化
        if base.max() > 1.0 or blend.max() > 1.0:
            base = base / 255.0 if base.max() > 1.0 else base
            blend = blend / 255.0 if blend.max() > 1.0 else blend
        
        # 确保形状相同
        if base.shape != blend.shape:
            blend = F.interpolate(
                blend.unsqueeze(0) if blend.dim() == 3 else blend,
                size=base.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if blend.dim() == 4:
                blend = blend.squeeze(0)
                
        # 应用混合模式
        blend_func = cls.MODES[mode]
        result = blend_func(base, blend, opacity)
        
        # 确保输出范围正确，使用clamp而不是简单截断
        result = torch.clamp(result, 0.0, 1.0)
        
        return result 