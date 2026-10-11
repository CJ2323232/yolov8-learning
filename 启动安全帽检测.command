#!/bin/zsh
# Finder 双击此文件，进入文件所在项目并启动本地网页。
cd -- "$(dirname -- "$0")" || exit 1
if [[ ! -x ".venv/bin/python" ]]; then
    echo "请先在项目目录创建 .venv，并安装 requirements-gradio.txt 中的依赖。"
    read "?按回车退出"
    exit 1
fi
exec ".venv/bin/python" app.py
