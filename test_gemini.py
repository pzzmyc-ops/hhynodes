#!/usr/bin/env python3
"""
Gemini API测试脚本
用于验证Gemini API是否能正常调用和返回物体检测结果
"""

import json
import os
from PIL import Image
import requests
from io import BytesIO

# 测试Gemini API是否可用
def test_gemini_import():
    """测试Gemini包导入"""
    try:
        from google import genai
        from google.genai import types
        print("✅ Google Genai 包导入成功")
        return True, genai, types
    except ImportError as e:
        print(f"❌ Google Genai 包导入失败: {e}")
        print("请安装: pip install google-genai")
        return False, None, None

def test_gemini_client(api_key=None):
    """测试Gemini客户端初始化"""
    success, genai, types = test_gemini_import()
    if not success:
        return False, None, None
    
    try:
        # 使用提供的API密钥或默认密钥
        if not api_key:
            api_key = "AIzaSyBz9Wjf8yAm5zknsEOsaB72NAzWxC8q81k"
        client = genai.Client(api_key=api_key)
        print("✅ Gemini 客户端初始化成功")
        print(f"🔑 使用API密钥: {api_key[:10]}...")
        return True, client, types
    except Exception as e:
        print(f"❌ Gemini 客户端初始化失败: {e}")
        print("请检查API密钥是否有效")
        return False, None, None

def create_test_image():
    """创建一个简单的测试图像"""
    # 创建一个简单的测试图像 - 红色矩形
    img = Image.new('RGB', (400, 300), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # 绘制一些简单的形状用于测试
    draw.rectangle([50, 50, 150, 100], fill='red', outline='black', width=2)
    draw.rectangle([200, 150, 300, 200], fill='blue', outline='black', width=2)
    draw.ellipse([250, 50, 350, 100], fill='green', outline='black', width=2)
    
    return img

def test_gemini_detection():
    """测试Gemini物体检测"""
    print("\n🔍 开始测试Gemini物体检测...")
    
    # 1. 测试客户端
    success, client, types = test_gemini_client()
    if not success:
        return False, None
    
    # 2. 加载本地测试图像
    image_path = r"C:\Users\Administrator\Desktop\原图\new\poster_Me&TwoBadBoys@2x_ipad.png"
    print(f"📷 加载本地图像: {image_path}")
    
    try:
        test_image = Image.open(image_path)
        print(f"✅ 图像加载成功，尺寸: {test_image.size}")
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        # 如果本地图像加载失败，使用创建的测试图像
        print("📷 使用默认测试图像...")
        test_image = create_test_image()
    
    # 3. 准备检测参数
    prompt = "Detect the woman face in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    
    try:
        print(f"🚀 调用Gemini API (模型: gemini-2.5-flash)...")
        print(f"🎯 检测目标: woman face")
        
        # 配置响应格式
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        # 调用API (固定使用gemini-2.5-flash)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[test_image, prompt],
            config=config
        )
        
        print("✅ API调用成功!")
        print(f"📝 原始响应: {response.text}")
        
        # 4. 解析响应
        try:
            response_data = json.loads(response.text)
            print(f"📦 原始响应数据类型: {type(response_data)}")
            print(f"📦 响应内容: {response_data}")
            
            # 处理不同的响应格式
            bounding_boxes = []
            
            if isinstance(response_data, dict):
                # 如果是单个对象，转换为列表
                if "box_2d" in response_data:
                    bounding_boxes = [response_data]
                else:
                    print("⚠️  响应中没有找到box_2d字段")
                    return True, []
            elif isinstance(response_data, list):
                # 如果已经是列表，直接使用
                bounding_boxes = response_data
            else:
                print(f"⚠️  未知的响应格式: {type(response_data)}")
                return True, []
            
            print(f"📦 解析到 {len(bounding_boxes)} 个检测结果:")
            
            for i, bbox in enumerate(bounding_boxes):
                print(f"  {i+1}. {bbox}")
            
            # 5. 测试坐标转换
            if bounding_boxes:
                print(f"\n🔄 转换坐标 (图像尺寸: {test_image.size}):")
                converted_boxes = []
                
                for bbox in bounding_boxes:
                    if isinstance(bbox, dict):
                        box_2d = bbox.get("box_2d", [])
                        label = bbox.get("label", "detected_object")
                        
                        if len(box_2d) >= 4:
                            # Convert normalized coordinates (0-1000) to absolute coordinates
                            # Gemini format: [ymin, xmin, ymax, xmax]
                            abs_y1 = int(box_2d[0] / 1000 * test_image.height)  # ymin
                            abs_x1 = int(box_2d[1] / 1000 * test_image.width)   # xmin
                            abs_y2 = int(box_2d[2] / 1000 * test_image.height)  # ymax
                            abs_x2 = int(box_2d[3] / 1000 * test_image.width)   # xmax
                            
                            # Ensure coordinates are in correct order
                            if abs_x1 > abs_x2:
                                abs_x1, abs_x2 = abs_x2, abs_x1
                            if abs_y1 > abs_y2:
                                abs_y1, abs_y2 = abs_y2, abs_y1
                            
                            converted_box = [abs_x1, abs_y1, abs_x2, abs_y2]
                            converted_boxes.append({"bbox": converted_box, "label": label})
                            
                            print(f"  {label}:")
                            print(f"    原始 (归一化): {box_2d}")
                            print(f"    转换后 (绝对): {converted_box}")
                        else:
                            print(f"  警告: box_2d格式不正确: {box_2d}")
                    else:
                        print(f"  警告: 检测结果不是字典格式: {bbox}")
                
                return True, converted_boxes
            else:
                print("⚠️  没有检测到任何对象")
                return True, []
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始响应可能不是有效的JSON格式")
            # 尝试提取可能的JSON部分
            response_text = response.text.strip()
            if response_text.startswith('[') or response_text.startswith('{'):
                print("尝试直接解析响应...")
                return True, response_text
            return False, None
            
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return False, None

def test_coordinate_conversion():
    """测试坐标转换"""
    print("\n🔄 测试坐标转换...")
    
    # 模拟Gemini响应
    mock_response = [
        {"box_2d": [100, 200, 300, 400], "label": "rectangle"},
        {"box_2d": [500, 600, 700, 800], "label": "circle"}
    ]
    
    img_width, img_height = 400, 300
    
    print(f"图像尺寸: {img_width} x {img_height}")
    print("原始坐标 (归一化到0-1000):")
    
    converted_boxes = []
    for i, bbox in enumerate(mock_response):
        box_2d = bbox["box_2d"]
        label = bbox["label"]
        
        print(f"  {label}: {box_2d}")
        
        # 转换坐标
        abs_y1 = int(box_2d[0] / 1000 * img_height)
        abs_x1 = int(box_2d[1] / 1000 * img_width)
        abs_y2 = int(box_2d[2] / 1000 * img_height)
        abs_x2 = int(box_2d[3] / 1000 * img_width)
        
        converted_box = [abs_x1, abs_y1, abs_x2, abs_y2]
        converted_boxes.append({"bbox": converted_box, "label": label})
        
        print(f"    -> 绝对坐标: {converted_box}")
    
    print("✅ 坐标转换测试完成")
    return converted_boxes

def main():
    """主测试函数"""
    print("🧪 Gemini API 测试开始")
    print("=" * 50)
    
    # 检查环境变量
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"🔑 检测到API密钥: {api_key[:10]}...")
    else:
        print("⚠️  未检测到API密钥环境变量 (GOOGLE_API_KEY 或 GEMINI_API_KEY)")
    
    # 1. 测试坐标转换
    test_coordinate_conversion()
    
    # 2. 测试实际API调用
    success, results = test_gemini_detection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过!")
        return True
    else:
        print("❌ 测试失败，请检查配置")
        return False

if __name__ == "__main__":
    main() 