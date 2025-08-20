#!/usr/bin/env python3
"""
合并 Qwen2.5-VL 模型分片为单个 safetensors 文件
"""

import os
import json
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
from tqdm import tqdm


def merge_model_shards(model_dir: str, output_file: str = None):
    """
    合并分片的 safetensors 文件为单个文件
    
    Args:
        model_dir: 包含分片模型文件的目录
        output_file: 输出文件路径 (可选，默认为 model_dir/model_merged.safetensors)
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise ValueError(f"模型目录不存在: {model_dir}")
    
    # 读取索引文件以了解分片结构
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        raise ValueError(f"索引文件未找到: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    print(f"📋 找到索引文件，包含 {len(index_data.get('weight_map', {}))} 个权重")
    
    # 获取模型总大小信息
    total_size = index_data.get('metadata', {}).get('total_size', 0)
    if total_size > 0:
        print(f"📊 模型总大小: {total_size / (1024**3):.2f} GB")
    
    # 收集所有唯一的分片文件
    shard_files = set(index_data.get('weight_map', {}).values())
    print(f"📁 需要合并的分片文件数: {len(shard_files)}")
    for shard in sorted(shard_files):
        shard_path = model_path / shard
        if shard_path.exists():
            size = os.path.getsize(shard_path) / (1024**3)
            print(f"   - {shard} ({size:.2f} GB)")
        else:
            print(f"   - {shard} (文件不存在!)")
    
    # 从所有分片加载张量
    merged_state_dict = {}
    total_params = 0
    
    print(f"\n🔄 开始合并模型分片...")
    
    for shard_file in tqdm(sorted(shard_files), desc="合并分片", unit="file"):
        shard_path = model_path / shard_file
        if not shard_path.exists():
            print(f"⚠️  警告: 分片文件未找到: {shard_path}")
            continue
            
        # 获取文件大小
        shard_size = os.path.getsize(shard_path)
        tqdm.write(f"📥 加载分片: {shard_file} ({shard_size / (1024**3):.2f} GB)")
        
        try:
            shard_data = load_file(str(shard_path))
        except Exception as e:
            tqdm.write(f"❌ 加载分片失败 {shard_file}: {e}")
            continue
        
        shard_params = 0
        for key, tensor in shard_data.items():
            if key in merged_state_dict:
                tqdm.write(f"⚠️  警告: 发现重复键: {key}")
            merged_state_dict[key] = tensor
            shard_params += tensor.numel()
            
        total_params += shard_params
        tqdm.write(f"   ✅ 已加载 {len(shard_data)} 个张量，{shard_params:,} 个参数")
    
    print(f"\n📈 合并统计:")
    print(f"   - 总张量数: {len(merged_state_dict)}")
    print(f"   - 总参数数: {total_params:,}")
    print(f"   - 预估大小: {total_params * 2 / (1024**3):.2f} GB (假设FP16)")
    
    # 确定输出文件路径
    if output_file is None:
        output_file = str(model_path / "model_merged.safetensors")
    
    # 保存合并后的模型
    print(f"\n💾 保存合并后的模型到: {output_file}")
    try:
        # 显示保存进度
        print("   正在写入文件，请稍候...")
        save_file(merged_state_dict, output_file)
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None
    
    # 获取文件大小
    file_size = os.path.getsize(output_file)
    file_size_gb = file_size / (1024 ** 3)
    print(f"✅ 合并模型保存成功!")
    print(f"📊 文件大小: {file_size_gb:.2f} GB")
    
    # 为合并后的模型创建新的索引文件
    print(f"\n📝 创建新的索引文件...")
    merged_index = {
        "metadata": {
            "total_size": file_size,
            "merged_from_shards": len(shard_files),
            "original_total_size": total_size,
            "merged_at": str(Path(output_file).name)
        },
        "weight_map": {key: "model_merged.safetensors" for key in merged_state_dict.keys()}
    }
    
    merged_index_file = str(model_path / "model_merged.safetensors.index.json")
    with open(merged_index_file, 'w', encoding='utf-8') as f:
        json.dump(merged_index, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 创建索引文件: {merged_index_file}")
    
    # 验证合并结果
    print(f"\n🔍 验证合并结果...")
    try:
        # 快速加载验证（只读取元数据）
        import safetensors
        with safetensors.safe_open(output_file, framework="pt") as f:
            keys = list(f.keys())
            print(f"✅ 验证通过: 可以正常读取合并后的文件")
            print(f"   - 实际张量数: {len(keys)}")
            print(f"   - 样本键名: {keys[:3]}..." if len(keys) > 3 else f"   - 所有键名: {keys}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return None
    
    return output_file


def get_user_input():
    """获取用户输入的模型路径和输出路径"""
    print("=" * 60)
    print("🤖 Qwen2.5-VL 模型分片合并工具")
    print("=" * 60)
    
    # 获取模型目录路径
    while True:
        model_dir = input("\n📁 请输入模型目录路径 (包含 model.safetensors.index.json 的目录): ").strip()
        
        if not model_dir:
            print("❌ 路径不能为空，请重新输入")
            continue
            
        # 去除引号
        model_dir = model_dir.strip('"').strip("'")
        
        if not os.path.exists(model_dir):
            print(f"❌ 目录不存在: {model_dir}")
            continue
            
        index_file = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            print(f"❌ 未找到索引文件: {index_file}")
            print("   请确保输入的是包含模型文件的正确目录")
            continue
            
        print(f"✅ 模型目录: {model_dir}")
        break
    
    # 获取输出文件路径
    print(f"\n📝 输出文件设置:")
    default_output = os.path.join(model_dir, "model_merged.safetensors")
    print(f"   默认输出路径: {default_output}")
    
    while True:
        output_choice = input("\n是否使用默认输出路径? (y/n) [默认: y]: ").strip().lower()
        
        if output_choice == '' or output_choice == 'y' or output_choice == 'yes':
            output_file = default_output
            break
        elif output_choice == 'n' or output_choice == 'no':
            output_file = input("请输入自定义输出文件路径: ").strip()
            output_file = output_file.strip('"').strip("'")
            
            if not output_file:
                print("❌ 输出路径不能为空")
                continue
                
            # 检查输出目录是否存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"✅ 创建输出目录: {output_dir}")
                except Exception as e:
                    print(f"❌ 无法创建输出目录: {e}")
                    continue
            break
        else:
            print("❌ 请输入 y 或 n")
            continue
    
    print(f"✅ 输出文件: {output_file}")
    
    # 确认信息
    print(f"\n📋 合并设置确认:")
    print(f"   源目录: {model_dir}")
    print(f"   输出文件: {output_file}")
    
    while True:
        confirm = input("\n确认开始合并? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return model_dir, output_file
        elif confirm in ['n', 'no']:
            print("❌ 用户取消操作")
            return None, None
        else:
            print("❌ 请输入 y 或 n")


def main():
    try:
        # 获取用户输入
        model_dir, output_file = get_user_input()
        
        if not model_dir:
            return 0
        
        print("\n🚀 开始合并 Qwen2.5-VL 模型...")
        merged_file = merge_model_shards(model_dir, output_file)
        
        if merged_file:
            print(f"\n🎉 成功! 合并后的模型已保存到: {merged_file}")
            
            # 使用说明
            print("\n📖 使用说明:")
            print("1. 现在可以通过指向模型目录来使用合并后的模型")
            print("2. 加载代码会自动检测并使用合并后的文件")
            print(f"3. 模型目录: {model_dir}")
            print("4. 合并后的文件: model_merged.safetensors")
            print("5. 新的索引文件: model_merged.safetensors.index.json")
            
            # 备份建议
            print("\n💡 建议:")
            print("1. 原始分片文件仍然保留，可以作为备份")
            print("2. 如果需要节省空间，确认合并文件正常工作后可以删除原始分片")
            print("3. 建议保留原始的 model.safetensors.index.json 作为备份")
            
            # 等待用户确认
            input("\n按 Enter 键退出...")
        else:
            print(f"\n❌ 合并失败!")
            input("\n按 Enter 键退出...")
            return 1
        
    except KeyboardInterrupt:
        print(f"\n\n❌ 用户中断操作")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 键退出...")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
