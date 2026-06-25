
#!/bin/bash

# 配置参数
WORK_DIR="work_dirs/"
HF_REPO="zhouyik/mask_tokenizer_train_weights"  # 替换为你的Hugging Face仓库
UPLOAD_INTERVAL=7200  # 2小时 = 7200秒
LOG_FILE="upload.log"
export HF_ENDPOINT=https://hf-mirror.com

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查依赖
check_dependencies() {
    if ! command -v huggingface-cli &> /dev/null; then
        log "错误: huggingface-cli 未安装"
        log "请运行: pip install huggingface_hub"
        exit 1
    fi

    if [ ! -d "$WORK_DIR" ]; then
        log "错误: 工作目录 $WORK_DIR 不存在"
        exit 1
    fi
}

# 上传函数
upload_pth_files() {
    log "开始扫描 $WORK_DIR 目录下的 .pth 文件..."

    # 使用find查找所有.pth文件
    pth_files=$(find "$WORK_DIR" -name "*.pth" -type f)

    if [ -z "$pth_files" ]; then
        log "未找到任何 .pth 文件"
        return 0
    fi

    file_count=$(echo "$pth_files" | wc -l)
    log "找到 $file_count 个 .pth 文件"

    # 上传每个文件
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            log "正在上传: $file"

            # 获取相对路径用于在仓库中的路径
            relative_path=${file#$WORK_DIR/}

            # 使用huggingface-cli上传
            if huggingface-cli upload "$HF_REPO" "$file" "$relative_path" --token "${HF_TOKEN:?set HF_TOKEN env var}" --quiet; then
                log "✓ 上传成功: $file"
            else
                log "✗ 上传失败: $file"
            fi
        fi
    done <<< "$pth_files"

    log "本次上传完成"
}

# 主循环
main() {
    log "脚本启动 - 目标仓库: $HF_REPO"
    log "工作目录: $WORK_DIR"
    log "上传间隔: $UPLOAD_INTERVAL 秒 ($(($UPLOAD_INTERVAL / 3600)) 小时)"

    # 检查依赖
    check_dependencies

    # 检查Hugging Face登录状态
#    if ! huggingface-cli whoami &> /dev/null; then
#        log "错误: 未登录Hugging Face"
#        log "请运行: huggingface-cli login"
#        exit 1
#    fi

    # 主循环
    while true; do
        upload_pth_files
        log "等待 $UPLOAD_INTERVAL 秒后进行下次扫描..."
        sleep "$UPLOAD_INTERVAL"
    done
}

# 信号处理
cleanup() {
    log "收到退出信号，正在清理..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# 启动主程序
main "$@"