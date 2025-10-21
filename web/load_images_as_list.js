import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 注册自定义节点扩展
app.registerExtension({
    name: "hhy.LoadImagesAsList",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LoadImagesAsList") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 初始化属性
                this.selectedFiles = [];  // 存储选择的文件对象和 base64 数据
                this.isProcessing = false;
                
                // 获取 images_data widget
                this.imagesDataWidget = this.widgets.find(w => w.name === "images_data");
                
                // 添加选择图片按钮
                this.addWidget("button", "📁 Select Images", null, () => {
                    if (!this.isProcessing) {
                        this.showMultiFileDialog();
                    }
                });
                
                // 添加清除按钮
                this.addWidget("button", "🗑️ Clear All", null, () => {
                    if (!this.isProcessing) {
                        this.selectedFiles = [];
                        this.updateImagesData();
                        this.updateTitle();
                    }
                });
                
                // 设置初始标题
                this.updateTitle();
                
                return result;
            };
            
            // 显示多文件选择对话框
            nodeType.prototype.showMultiFileDialog = function() {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = "image/*";
                input.multiple = true;
                
                input.onchange = async (e) => {
                    const files = Array.from(e.target.files);
                    if (files.length === 0) return;
                    
                    this.isProcessing = true;
                    console.log(`[LoadImagesAsList] Selected ${files.length} files`);
                    
                    // 显示处理中状态
                    this.title = `⏳ Processing 0/${files.length}...`;
                    if (this.graph && this.graph.canvas) {
                        this.graph.canvas.setDirty(true);
                    }
                    
                    const startTime = Date.now();
                    this.selectedFiles = [];
                    
                    for (let i = 0; i < files.length; i++) {
                        const file = files[i];
                        const percentage = Math.round(((i + 1) / files.length) * 100);
                        
                        // 更新标题显示进度
                        this.title = `⏳ Processing ${i + 1}/${files.length} (${percentage}%)`;
                        if (this.graph && this.graph.canvas) {
                            this.graph.canvas.setDirty(true);
                        }
                        
                        try {
                            // 将文件转为 base64
                            const base64Data = await this.fileToBase64(file);
                            
                            this.selectedFiles.push({
                                name: file.name,
                                data: base64Data
                            });
                            
                            console.log(`[LoadImagesAsList] ✓ ${i + 1}/${files.length}: ${file.name}`);
                        } catch (error) {
                            console.error(`[LoadImagesAsList] ✗ Error processing ${file.name}:`, error);
                        }
                    }
                    
                    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                    
                    // 更新 images_data
                    this.updateImagesData();
                    
                    this.isProcessing = false;
                    this.updateTitle();
                    
                    console.log(`[LoadImagesAsList] ✅ Processing complete: ${this.selectedFiles.length}/${files.length} files, Time ${elapsed}s`);
                };
                
                input.click();
            };
            
            // 将文件转为 base64
            nodeType.prototype.fileToBase64 = function(file) {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = reject;
                    reader.readAsDataURL(file);
                });
            };
            
            // 更新 images_data widget
            nodeType.prototype.updateImagesData = function() {
                if (this.imagesDataWidget) {
                    if (this.selectedFiles.length === 0) {
                        this.imagesDataWidget.value = "";
                    } else {
                        // 将文件列表转为 JSON
                        this.imagesDataWidget.value = JSON.stringify(this.selectedFiles);
                    }
                }
            };
            
            // 更新节点标题
            nodeType.prototype.updateTitle = function() {
                if (this.selectedFiles.length === 0) {
                    this.title = "Load Images as List";
                } else {
                    this.title = `📊 ${this.selectedFiles.length} images selected`;
                }
                
                // 强制刷新画布
                if (this.graph && this.graph.canvas) {
                    this.graph.canvas.setDirty(true);
                }
            };
            
            // 序列化 - 保存文件数据到工作流
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                const result = onSerialize?.apply(this, arguments) || o;
                
                // 存储选择的图片数据到节点
                result.selected_files = this.selectedFiles;
                
                return result;
            };
            
            // 反序列化 - 从工作流恢复文件数据
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                const result = onConfigure?.apply(this, arguments);
                
                if (o.selected_files && Array.isArray(o.selected_files)) {
                    this.selectedFiles = o.selected_files;
                    this.updateImagesData();
                    this.updateTitle();
                }
                
                return result;
            };
            
            // 右键菜单
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                getExtraMenuOptions?.apply(this, arguments);
                
                options.unshift(
                    {
                        content: "📁 Select Images",
                        callback: () => {
                            if (!this.isProcessing) {
                                this.showMultiFileDialog();
                            }
                        },
                    },
                    {
                        content: "🗑️ Clear All",
                        callback: () => {
                            if (!this.isProcessing) {
                                this.selectedFiles = [];
                                this.updateImagesData();
                                this.updateTitle();
                            }
                        },
                    },
                    {
                        content: "📋 Show Files List",
                        callback: () => {
                            if (this.selectedFiles.length === 0) {
                                alert("No images selected yet");
                            } else {
                                const fileList = this.selectedFiles
                                    .map((file, i) => `${i + 1}. ${file.name}`)
                                    .join('\n');
                                alert(`Selected Files (${this.selectedFiles.length}):\n\n${fileList}`);
                            }
                        },
                    }
                );
            };
        }
    },
});
