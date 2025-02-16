// 视频预览组件
class VideoPreview {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        
        // 配置选项
        this.options = {
            width: options.width || 200,
            height: options.height || 150,
            backgroundColor: options.backgroundColor || "#000",
            controlsHeight: options.controlsHeight || 30,
            fps: options.fps || 30
        };
        
        // 设置画布尺寸
        this.canvas.width = this.options.width;
        this.canvas.height = this.options.height;
        
        // 初始化状态
        this.frames = [];
        this.currentFrame = 0;
        this.isPlaying = false;
        this.lastFrameTime = 0;
        
        // 创建控制界面
        this.createControls();
    }
    
    createControls() {
        // 创建控制容器
        this.controls = document.createElement("div");
        this.controls.style.position = "absolute";
        this.controls.style.bottom = "0";
        this.controls.style.left = "0";
        this.controls.style.width = "100%";
        this.controls.style.height = `${this.options.controlsHeight}px`;
        this.controls.style.backgroundColor = "rgba(0,0,0,0.5)";
        this.controls.style.display = "flex";
        this.controls.style.alignItems = "center";
        this.controls.style.padding = "0 5px";
        
        // 播放/暂停按钮
        this.playButton = document.createElement("button");
        this.playButton.innerHTML = "▶";
        this.playButton.onclick = () => this.togglePlay();
        this.controls.appendChild(this.playButton);
        
        // 进度条
        this.progress = document.createElement("input");
        this.progress.type = "range";
        this.progress.min = 0;
        this.progress.max = 100;
        this.progress.value = 0;
        this.progress.style.flex = "1";
        this.progress.style.margin = "0 10px";
        this.progress.oninput = (e) => this.seekTo(e.target.value);
        this.controls.appendChild(this.progress);
        
        // 帧计数
        this.frameCounter = document.createElement("span");
        this.frameCounter.style.color = "#fff";
        this.frameCounter.style.marginRight = "5px";
        this.frameCounter.innerHTML = "0/0";
        this.controls.appendChild(this.frameCounter);
        
        // 添加到画布容器
        this.canvas.parentElement.appendChild(this.controls);
    }
    
    setFrames(frames) {
        this.frames = frames;
        this.currentFrame = 0;
        this.progress.max = Math.max(0, frames.length - 1);
        this.updateDisplay();
    }
    
    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playButton.innerHTML = this.isPlaying ? "⏸" : "▶";
        
        if (this.isPlaying) {
            this.play();
        }
    }
    
    play() {
        if (!this.isPlaying || !this.frames.length) return;
        
        const now = performance.now();
        const elapsed = now - this.lastFrameTime;
        
        if (elapsed >= (1000 / this.options.fps)) {
            this.currentFrame = (this.currentFrame + 1) % this.frames.length;
            this.updateDisplay();
            this.lastFrameTime = now;
        }
        
        requestAnimationFrame(() => this.play());
    }
    
    seekTo(frame) {
        this.currentFrame = Math.min(Math.max(0, parseInt(frame)), this.frames.length - 1);
        this.updateDisplay();
    }
    
    updateDisplay() {
        // 清除画布
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 绘制当前帧
        if (this.frames.length && this.frames[this.currentFrame]) {
            this.ctx.drawImage(this.frames[this.currentFrame], 0, 0, this.canvas.width, this.canvas.height);
        }
        
        // 更新进度条
        this.progress.value = this.currentFrame;
        
        // 更新帧计数
        this.frameCounter.innerHTML = `${this.currentFrame + 1}/${this.frames.length}`;
    }
    
    destroy() {
        this.isPlaying = false;
        if (this.controls && this.controls.parentElement) {
            this.controls.parentElement.removeChild(this.controls);
        }
    }
}

export { VideoPreview }; 