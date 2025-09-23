#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础功能测试脚本

测试项目的核心模块是否能正常导入和基本功能。
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    
    try:
        # 测试配置管理器
        from config_manager import ConfigManager
        print("✅ ConfigManager 导入成功")
        
        # 测试文档处理器
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor 导入成功")
        
        # 测试分块策略
        from chunk_strategies import BaseChunkingStrategy, ChunkingConfig
        print("✅ ChunkStrategy 导入成功")
        
        # 测试质量评估器
        from quality_evaluator import QualityEvaluator
        print("✅ QualityEvaluator 导入成功")
        
        # 测试工具模块
        from utils import Logger, ErrorHandler, PerformanceMonitor, FileUtils, TextUtils
        print("✅ Utils 模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    try:
        # 测试配置管理器
        from config_manager import ConfigManager
        config = ConfigManager()
        test_config = config.get_config()
        print("✅ ConfigManager 初始化成功")
        
        # 测试文本工具
        from utils import TextUtils
        text_utils = TextUtils()
        test_text = "这是一个测试文本。This is a test text."
        cleaned = text_utils.clean_text(test_text)
        print(f"✅ TextUtils 文本清理成功: {len(cleaned)} 字符")
        
        # 测试文件工具
        from utils import FileUtils
        file_utils = FileUtils()
        safe_name = file_utils.safe_filename("test file.txt")
        print(f"✅ FileUtils 安全文件名生成: {safe_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False

def test_configuration():
    """测试配置功能"""
    print("\n=== 测试配置功能 ===")
    
    try:
        from config_manager import ConfigManager
        
        # 创建配置管理器
        config = ConfigManager()
        
        # 测试默认配置
        app_config = config.get_config()
        print(f"✅ 获取配置成功: {type(app_config)}")
        
        # 测试配置节获取
        llm_config = config.get_llm_config()
        print(f"✅ 获取LLM配置成功: {llm_config.model}")
        
        # 测试配置更新
        config.update_config({'test_key': 'test_value'})
        print("✅ 配置更新成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("Embedding增强项目 - 基础功能测试")
    print("=" * 50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    # 运行测试
    tests = [
        ("模块导入", test_imports),
        ("基本功能", test_basic_functionality),
        ("配置功能", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
        print()
    
    # 测试总结
    print("=" * 50)
    print(f"测试总结: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有基础测试通过！项目代码结构正确。")
        print("\n下一步建议:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行完整测试: python main_app.py --help")
        print("3. 启动Web界面: python web_interface.py")
    else:
        print("⚠️ 部分测试失败，请检查相关模块。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)