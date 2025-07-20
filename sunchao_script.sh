#!/bin/bash
SESSION_NAME="eval_session"
# SCRIPT_PATH="eval.sh"  # 替换为eval.sh的实际路径
SCRIPT_PATH="chair_eval.sh"  # 替换为eval.sh的实际路径

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo "错误：tmux未安装，请先安装tmux"
    echo "安装命令：sudo apt-get install tmux 或 sudo yum install tmux"
    exit 1
fi

# 检查会话是否已存在
tmux has-session -t $SESSION_NAME 2>/dev/null

# 如果会话不存在则创建
if [ $? != 0 ]; then
    # 创建新会话并执行评估脚本
    tmux new-session -d -s $SESSION_NAME "bash $SCRIPT_PATH"
    echo "✅ 已启动新的tmux会话 [$SESSION_NAME] 执行任务"
else
    echo "⚠️ 会话 [$SESSION_NAME] 已存在，请先处理现有会话"
    echo "查看会话命令：tmux attach -t $SESSION_NAME"
    exit 1
fi

# 使用说明
echo ""
echo "------------------------------------------------------"
echo "任务已在后台运行，可通过以下命令管理："
echo "1. 查看任务输出： tmux attach -t $SESSION_NAME"
echo "2. 退出查看模式： 按 Ctrl+B 松开后按 D"
echo "3. 终止会话：     tmux kill-session -t $SESSION_NAME"
echo "------------------------------------------------------"