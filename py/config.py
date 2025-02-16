import os

# 插件版本
VERSION = "1.0.0"

# 默认配置
DEFAULT_CONFIG = {
    # 视频处理
    "max_video_size": (7680, 4320),  # 8K分辨率
    "default_video_size": (1920, 1080),  # 1080p
    "default_fps": 30.0,
    "max_fps": 240.0,
    
    # 缓存设置
    "cache_dir": "cache",
    "max_cache_size": 1024 * 1024 * 1024,  # 1GB
    "cache_cleanup_threshold": 0.9,  # 90%
    
    # 性能设置
    "use_gpu": True,
    "gpu_id": 0,
    "batch_size": 32,
    "num_workers": 4,
    
    # 预览设置
    "preview_size": (320, 240),
    "max_preview_frames": 10,
    
    # 输出设置
    "default_output_format": "mp4",
    "default_video_codec": "h264",
    "default_video_bitrate": "5000k",
    
    # 混合模式设置
    "default_blend_mode": "normal",
    "default_opacity": 1.0,
    
    # 错误处理
    "max_retries": 3,
    "retry_delay": 1.0,  # 秒
}

def get_cache_dir():
    """获取缓存目录路径"""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), DEFAULT_CONFIG["cache_dir"])
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def get_temp_dir():
    """获取临时文件目录路径"""
    temp_dir = os.path.join(get_cache_dir(), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def cleanup_cache():
    """清理缓存目录"""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return
        
    # 获取所有缓存文件
    total_size = 0
    files = []
    for root, _, filenames in os.walk(cache_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            size = os.path.getsize(file_path)
            files.append((file_path, size))
            total_size += size
            
    # 检查是否需要清理
    if total_size > DEFAULT_CONFIG["max_cache_size"]:
        # 按修改时间排序
        files.sort(key=lambda x: os.path.getmtime(x[0]))
        
        # 删除旧文件直到缓存大小低于阈值
        target_size = DEFAULT_CONFIG["max_cache_size"] * DEFAULT_CONFIG["cache_cleanup_threshold"]
        while total_size > target_size and files:
            file_path, size = files.pop(0)
            try:
                os.remove(file_path)
                total_size -= size
            except OSError:
                pass 