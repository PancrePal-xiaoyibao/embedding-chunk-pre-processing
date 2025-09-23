#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown文档Embedding增强项目 - 主入口文件

本项目通过智能预处理，对指定目录的Markdown文档进行优化，
便于后续的Embedding效果增强，显著提升RAG系统的检索效果。

主要功能：
- 智能分块：基于语义和结构的多策略chunk分割
- 关键词增强：医学专业术语提取和同义词扩展
- 质量保证：全面的chunk质量评估和优化建议
- 用户友好：直观的Web界面和命令行工具
- 高性能：本地化优先，LLM备用的混合处理模式

使用方法：
    python main.py --help                    # 查看帮助
    python main.py --web                     # 启动Web界面
    python main.py --cli                     # 启动命令行界面
    python main.py --file input.md           # 处理单个文件
    python main.py --dir input_dir           # 处理目录

作者: Embedding增强项目团队
版本: 1.0
更新日期: 2025
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """
    主入口函数
    
    功能：
        - 解析命令行参数
        - 初始化应用程序
        - 根据参数启动相应的界面或处理模式
    
    异常处理：
        - ImportError: 模块导入失败时的处理
        - SystemExit: 程序正常退出
        - Exception: 其他未预期的异常
    """
    try:
        # 导入主应用程序
        from src.interfaces.main_app import main as app_main
        
        # 调用主应用程序的main函数
        app_main()
        
    except ImportError as e:
        print(f"错误: 无法导入必要的模块 - {e}")
        print("请确保已安装所有依赖项：pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()