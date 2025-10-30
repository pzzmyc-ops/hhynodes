#!/bin/bash
echo "编译SECRET_KEY库文件..."
python3 build_key.py
if [ $? -eq 0 ]; then
    echo "编译完成！"
else
    echo "编译失败！"
fi
