#!/usr/bin/env python3
"""
简单的密钥库编译脚本
"""

import os
import platform
import subprocess
import sys

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(current_dir, "secret_key.c")
    
    if not os.path.exists(source_file):
        print(f"错误: 源文件不存在 {source_file}")
        return False
    
    system = platform.system()
    
    if system == "Windows":
        output_file = os.path.join(current_dir, "secret_key.dll")
        # 使用nvcc编译DLL
        cmd = ["nvcc", "--shared", "-O2", "-o", output_file, source_file]
    else:
        output_file = os.path.join(current_dir, "secret_key.so")
        # Linux/macOS使用nvcc编译SO
        cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC", "-O2", "-o", output_file, source_file]
    
    print(f"编译命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            if os.path.exists(output_file):
                print(f"编译成功: {output_file}")
                return True
            else:
                print("编译完成但未找到输出文件")
                return False
        else:
            print(f"编译失败: {result.stderr}")
            return False
    
    except FileNotFoundError:
        print("错误: 未找到nvcc编译器，请确保已安装CUDA工具包")
        return False
    except Exception as e:
        print(f"编译过程出错: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
