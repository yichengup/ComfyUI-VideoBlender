import os
import cv2
import numpy as np
from typing import Tuple, List, Union
import torch

class VideoUtils:
    @staticmethod
    def load_video(video_path: str) -> Tuple[List[np.ndarray], dict]:
        """加载视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Tuple[List[np.ndarray], dict]: 帧列表和视频信息
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': len(frames),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        return frames, info

    @staticmethod
    def save_video(frames: List[np.ndarray], output_path: str, fps: float, fourcc: str = 'mp4v') -> str:
        """保存视频文件
        
        Args:
            frames: 帧列表
            output_path: 输出路径
            fps: 帧率
            fourcc: 编码格式
            
        Returns:
            str: 输出文件路径
        """
        if not frames:
            raise ValueError("没有帧可以保存")
            
        height, width = frames[0].shape[:2]
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        
        out = cv2.VideoWriter(output_path, fourcc_code, fps, (width, height))
        for frame in frames:
            out.write(frame)
            
        out.release()
        return output_path

    @staticmethod
    def resize_frame(frame: np.ndarray, size: Tuple[int, int], keep_aspect: bool = True) -> np.ndarray:
        """调整帧大小
        
        Args:
            frame: 输入帧
            size: 目标尺寸 (width, height)
            keep_aspect: 是否保持宽高比
            
        Returns:
            np.ndarray: 调整后的帧
        """
        if keep_aspect:
            h, w = frame.shape[:2]
            target_w, target_h = size
            
            # 计算缩放比例
            scale = min(target_w/w, target_h/h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 调整大小
            resized = cv2.resize(frame, (new_w, new_h))
            
            # 创建目标尺寸的画布
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # 计算偏移量使图像居中
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            # 将调整后的图像放置在画布中心
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return result
        else:
            return cv2.resize(frame, size)

    @staticmethod
    def transform_frame(frame: Union[np.ndarray, torch.Tensor], 
                       position: Tuple[int, int] = (0, 0),
                       scale: Union[float, Tuple[float, float]] = 1.0,
                       rotation: float = 0.0) -> np.ndarray:
        """变换帧
        
        Args:
            frame: 输入帧（numpy数组或torch张量）
            position: 位置偏移 (x, y)
            scale: 缩放比例，可以是单个数值或(x_scale, y_scale)
            rotation: 旋转角度（度）
            
        Returns:
            np.ndarray: 变换后的帧
        """
        # 转换torch.Tensor为numpy数组
        if isinstance(frame, torch.Tensor):
            if frame.ndim == 4:  # batch of images
                frame = frame[0]
            if frame.ndim == 3:
                if frame.shape[0] in [1, 3, 4]:  # 如果是CHW格式
                    frame = frame.permute(1, 2, 0)
            frame = frame.cpu().numpy()
            # 确保值范围在0-255之间
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # 验证输入图像
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"输入frame类型错误: {type(frame)}, 需要numpy.ndarray")
            
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"输入图像尺寸无效: {frame.shape}")
            
        if len(frame.shape) != 3:
            raise ValueError(f"输入图像维度错误: {frame.shape}, 需要(H,W,C)格式")
            
        h, w = frame.shape[:2]
        
        # 处理缩放参数
        if isinstance(scale, (int, float)):
            scale_x = scale_y = float(scale)
        else:
            scale_x, scale_y = scale
            
        # 验证变换参数
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError(f"缩放比例必须大于0: scale_x={scale_x}, scale_y={scale_y}")
            
        # 创建变换矩阵
        center = (w/2, h/2)
        
        try:
            # 缩放矩阵
            scale_matrix = cv2.getRotationMatrix2D(center, 0, scale_x)
            scale_matrix[0,2] += position[0]
            scale_matrix[1,2] += position[1]
            
            # 旋转矩阵
            if rotation != 0:
                rot_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                scale_matrix = np.dot(rot_matrix, scale_matrix)
            
            # 应用变换
            result = cv2.warpAffine(frame, scale_matrix, (w, h))
            
            # 验证输出
            if result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0:
                raise ValueError(f"变换后图像尺寸无效: {result.shape}")
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"图像变换失败: {str(e)}, 输入尺寸: {frame.shape}")

    @staticmethod
    def sync_frame_counts(frame_lists: List[List[np.ndarray]], mode: str = 'min') -> List[List[np.ndarray]]:
        """同步多个帧序列的长度
        
        Args:
            frame_lists: 帧序列列表
            mode: 同步模式 ('min'/'max')
            
        Returns:
            List[List[np.ndarray]]: 同步后的帧序列列表
        """
        if not frame_lists:
            return []
            
        # 获取所有序列的长度
        lengths = [len(frames) for frames in frame_lists]
        
        if mode == 'min':
            target_len = min(lengths)
        else:  # mode == 'max'
            target_len = max(lengths)
            
        result = []
        for frames in frame_lists:
            if len(frames) == target_len:
                result.append(frames)
            elif len(frames) < target_len:
                # 通过重复最后一帧来扩展
                extended = frames + [frames[-1]] * (target_len - len(frames))
                result.append(extended)
            else:
                # 截断到目标长度
                result.append(frames[:target_len])
                
        return result

    @staticmethod
    def create_preview(frames: List[np.ndarray], 
                      max_frames: int = 10, 
                      preview_size: Tuple[int, int] = (320, 240)) -> List[np.ndarray]:
        """创建预览帧序列
        
        Args:
            frames: 输入帧序列
            max_frames: 最大预览帧数
            preview_size: 预览尺寸
            
        Returns:
            List[np.ndarray]: 预览帧序列
        """
        if not frames:
            return []
            
        # 选择预览帧
        step = max(1, len(frames) // max_frames)
        preview_frames = frames[::step][:max_frames]
        
        # 调整尺寸
        return [VideoUtils.resize_frame(frame, preview_size) for frame in preview_frames]

    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """将PyTorch张量转换为NumPy数组
        
        Args:
            tensor: 输入张量
            
        Returns:
            np.ndarray: 转换后的NumPy数组
        """
        if tensor.ndim == 4:  # batch of images
            tensor = tensor[0]
        if tensor.ndim == 3:
            # 假设输入格式为 CxHxW
            tensor = tensor.permute(1, 2, 0)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    @staticmethod
    def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
        """将NumPy数组转换为PyTorch张量
        
        Args:
            array: 输入数组
            
        Returns:
            torch.Tensor: 转换后的张量
        """
        if array.ndim == 3:
            # 假设输入格式为 HxWxC
            array = array.transpose(2, 0, 1)
        tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor 