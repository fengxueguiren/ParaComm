#!/bin/bash

# ================== 配置区 ==================
FREE_MEMORY_THRESHOLD=24576          # 空余显存阈值 (MB)，24GB = 24576
USED_MEMORY_THRESHOLD=3000           # 已用显存阈值 (MB)，示例 8GB 8192. 默认是用used模式时的阈值
CHECK_INTERVAL=600                  # 检查间隔（秒）

# ================== 命令行参数处理 ==================
if [ $# -eq 0 ]; then
    echo "错误：请指定要监控的 GPU ID"
    echo "用法: $0 <GPU_ID> [MODE]"
    echo "MODE 可选值:"
    echo "   used     → 当已用显存 < USED_MEMORY_THRESHOLD 时触发（当前默认模式）"
    echo "   free     → 当空余显存 > FREE_MEMORY_THRESHOLD 时触发"
    echo "示例:"
    echo "   $0 1              # 默认模式：已用显存 < 8GB"
    echo "   $0 1 used         # 已用显存 < 8GB"
    echo "   $0 1 free         # 空余显存 > 24GB"
    exit 1
fi

GPU_ID=$1
MODE=${2:-used}                      # 默认模式为 used

# ================== 根据模式设置触发条件 ==================
case "$MODE" in
    free)
        THRESHOLD=$FREE_MEMORY_THRESHOLD
        CONDITION_DESC="空余显存 > ${FREE_MEMORY_THRESHOLD} MB (24GB)"
        QUERY_FIELD="memory.free"
        COMPARE_OP="-gt"
        ;;
    used)
        THRESHOLD=$USED_MEMORY_THRESHOLD
        CONDITION_DESC="已用显存 < ${USED_MEMORY_THRESHOLD} MB"
        QUERY_FIELD="memory.used"
        COMPARE_OP="-lt"
        ;;
    *)
        echo "错误：未知的 MODE: $MODE"
        echo "支持的模式: free 或 used"
        exit 1
        ;;
esac

# ================== 动态生成 COMMAND ==================

YAML_FILE="opencood/hypes_yaml/OPV2V/pointpillar_sparse_dict_paracom_gpuid${GPU_ID}.yaml"
COMMAND="CUDA_VISIBLE_DEVICES=${GPU_ID} python opencood/tools/train_comm.py -y ${YAML_FILE}"

# ================== 检查 YAML 文件是否存在 ==================
echo "=================================================="
echo "正在检查配置文件..."
if [ ! -f "${YAML_FILE}" ]; then
    echo "❌ 错误：配置文件不存在！"
    echo "   期望路径: ${YAML_FILE}"
    exit 1
fi
echo "✅ 配置文件检查通过: ${YAML_FILE}"
echo "=================================================="

# ===========================================
echo "GPU 显存监控脚本启动"
echo "监控目标     : GPU ${GPU_ID}"
echo "触发模式     : ${MODE} （默认）"
echo "触发条件     : ${CONDITION_DESC}"
echo "检查间隔     : ${CHECK_INTERVAL} 秒"
echo "训练命令     : ${COMMAND}"
echo "=================================================="

while true; do
    # 获取指定 GPU 的显存信息
    MEM_VALUE=$(nvidia-smi --query-gpu=${QUERY_FIELD} --format=csv,noheader,nounits -i ${GPU_ID} | awk '{print $1}')

    if [ -z "$MEM_VALUE" ]; then
        echo "错误：无法获取 GPU ${GPU_ID} 的显存信息，请确认 GPU 存在。"
        exit 1
    fi

    MEM_GB=$(awk "BEGIN {printf \"%.2f\", $MEM_VALUE/1024}")

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID} 当前 ${QUERY_FIELD}: ${MEM_VALUE} MB (${MEM_GB} GB)"

    # 根据不同模式判断是否满足触发条件
    if [ "${MEM_VALUE}" ${COMPARE_OP} "${THRESHOLD}" ]; then
        echo "=================================================="
        echo "✅ 触发条件满足！"
        echo "   当前 ${QUERY_FIELD}: ${MEM_GB} GB"
        echo "   条件: ${CONDITION_DESC}"
        echo "🚀 开始执行训练任务..."
        echo "=================================================="

        # 执行训练命令
        eval ${COMMAND}

        echo "=================================================="
        echo "🎉 训练任务已结束！"
        echo "=================================================="
        break
    fi

    sleep ${CHECK_INTERVAL}
done

echo "脚本执行完毕。"