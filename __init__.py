import os
import sys
import importlib.util
from .py.config import VERSION, DEFAULT_CONFIG

# 确保py目录在Python路径中
py_dir = os.path.join(os.path.dirname(__file__), "py")
if py_dir not in sys.path:
    sys.path.append(py_dir)

# 导入节点定义
from .py.video_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 获取web目录路径
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# 创建必要的目录
def setup_directories():
    """创建插件所需的目录结构"""
    dirs = [
        os.path.join(os.path.dirname(__file__), "cache"),
        os.path.join(os.path.dirname(__file__), "cache", "temp"),
        os.path.join(os.path.dirname(__file__), "web", "js")
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# 初始化插件
def init_plugin():
    """初始化插件"""
    try:
        # 创建目录
        setup_directories()
        
        # 检查依赖
        required_packages = [
            "opencv-python",
            "numpy",
            "torch",
            "pillow",
            "moviepy"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package.replace("-", ""))
            except ImportError:
                missing_packages.append(package)
                
        if missing_packages:
            print(f"警告: 缺少以下依赖包: {', '.join(missing_packages)}")
            print("请运行: pip install -r requirements.txt")
            
        print(f"VideoBlender插件 v{VERSION} 初始化完成")
        
    except Exception as e:
        print(f"初始化插件时出错: {str(e)}")

# 运行初始化
init_plugin()

# 导出必要的变量
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 