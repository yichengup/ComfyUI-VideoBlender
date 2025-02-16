import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// 注册插件
app.registerExtension({
    name: "YC.VideoBlender",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 处理视频混合节点
        if (nodeData.name === "VideoBlendLayer" || 
            nodeData.name === "VideoBlendStack" || 
            nodeData.name === "VideoFrameSync") {
            
            // 保存原始方法
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            const origOnExecuted = nodeType.prototype.onExecuted;
            
            // 添加预览支持
            nodeType.prototype.onNodeCreated = function() {
                if (origOnNodeCreated) {
                    origOnNodeCreated.apply(this, arguments);
                }
                
                // 确保节点已经创建完成
                requestAnimationFrame(() => {
                    if (!this.widgets_values) {
                        this.widgets_values = {};
                    }
                    
                    // 获取或创建容器
                    let container = this.widgets?.find(w => w.name === "preview_container")?.element;
                    if (!container) {
                        container = document.createElement("div");
                        container.style.width = "100%";
                        container.style.height = "200px";
                        container.style.position = "relative";
                        container.style.marginBottom = "10px";
                        
                        // 添加到节点的widget区域
                        if (this.widgets) {
                            const widget = {
                                name: "preview_container",
                                type: "preview",
                                element: container
                            };
                            this.widgets.push(widget);
                        }
                    }
                    
                    // 添加预览画布
                    this.preview = document.createElement("canvas");
                    this.preview.style.width = "100%";
                    this.preview.style.height = "calc(100% - 30px)";
                    this.preview.style.backgroundColor = "#000";
                    container.appendChild(this.preview);
                    
                    // 初始化预览状态
                    this.previewData = null;
                    this.isPreviewPlaying = false;
                    this.currentPreviewFrame = 0;
                    
                    // 添加预览控制
                    this.addPreviewControls(container);
                });
            };
            
            // 添加预览控制
            nodeType.prototype.addPreviewControls = function(container) {
                const controls = document.createElement("div");
                controls.style.position = "absolute";
                controls.style.bottom = "0";
                controls.style.left = "0";
                controls.style.width = "100%";
                controls.style.height = "30px";
                controls.style.backgroundColor = "rgba(0,0,0,0.5)";
                controls.style.display = "flex";
                controls.style.alignItems = "center";
                controls.style.padding = "0 5px";
                
                // 播放/暂停按钮
                const playButton = document.createElement("button");
                playButton.innerHTML = "▶";
                playButton.style.marginRight = "5px";
                playButton.onclick = () => this.togglePreview();
                controls.appendChild(playButton);
                
                // 进度条
                const progress = document.createElement("input");
                progress.type = "range";
                progress.min = 0;
                progress.max = 100;
                progress.value = 0;
                progress.style.flex = "1";
                progress.style.margin = "0 10px";
                progress.oninput = (e) => this.seekTo(parseInt(e.target.value));
                controls.appendChild(progress);
                
                // 帧计数
                const counter = document.createElement("span");
                counter.style.color = "#fff";
                counter.style.marginLeft = "5px";
                counter.innerHTML = "0/0";
                controls.appendChild(counter);
                
                // 保存引用
                this.controls = {
                    playButton,
                    progress,
                    counter
                };
                
                container.appendChild(controls);
            };
            
            // 切换预览播放状态
            nodeType.prototype.togglePreview = function() {
                if (!this.previewData) return;
                
                this.isPreviewPlaying = !this.isPreviewPlaying;
                this.controls.playButton.innerHTML = this.isPreviewPlaying ? "⏸" : "▶";
                
                if (this.isPreviewPlaying) {
                    this.playPreview();
                }
            };
            
            // 播放预览
            nodeType.prototype.playPreview = function() {
                if (!this.isPreviewPlaying || !this.previewData) return;
                
                const ctx = this.preview.getContext("2d");
                const frame = this.previewData[this.currentPreviewFrame];
                
                if (frame) {
                    // 绘制当前帧
                    ctx.drawImage(frame, 0, 0, this.preview.width, this.preview.height);
                    
                    // 更新帧索引
                    this.currentPreviewFrame = (this.currentPreviewFrame + 1) % this.previewData.length;
                    
                    // 更新进度条和计数器
                    this.updateControls();
                    
                    // 继续播放
                    requestAnimationFrame(() => this.playPreview());
                }
            };
            
            // 跳转到指定帧
            nodeType.prototype.seekTo = function(frame) {
                if (!this.previewData) return;
                
                this.isPreviewPlaying = false;
                this.controls.playButton.innerHTML = "▶";
                
                this.currentPreviewFrame = Math.min(Math.max(0, frame), this.previewData.length - 1);
                
                const ctx = this.preview.getContext("2d");
                const frameData = this.previewData[this.currentPreviewFrame];
                if (frameData) {
                    ctx.drawImage(frameData, 0, 0, this.preview.width, this.preview.height);
                }
                
                this.updateControls();
            };
            
            // 更新控制器状态
            nodeType.prototype.updateControls = function() {
                if (!this.previewData || !this.controls) return;
                
                this.controls.progress.value = this.currentPreviewFrame;
                this.controls.counter.innerHTML = `${this.currentPreviewFrame + 1}/${this.previewData.length}`;
            };
            
            // 处理节点执行完成
            nodeType.prototype.onExecuted = function(message) {
                if (origOnExecuted) {
                    origOnExecuted.apply(this, arguments);
                }
                
                // 更新预览
                if (message.preview) {
                    this.updatePreview(message.preview);
                }
            };
            
            // 更新预览数据
            nodeType.prototype.updatePreview = function(previewData) {
                this.previewData = previewData;
                this.currentPreviewFrame = 0;
                
                if (this.controls) {
                    this.controls.progress.max = Math.max(0, previewData.length - 1);
                }
                
                // 显示第一帧
                if (previewData && previewData.length > 0) {
                    const ctx = this.preview.getContext("2d");
                    ctx.drawImage(previewData[0], 0, 0, this.preview.width, this.preview.height);
                    this.updateControls();
                }
            };
        }
    }
}); 