#!/usr/bin/env python3
"""
配置生成脚本 - Configuration Generator Script

快速生成AI Services配置文件的工具。
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config.config import create_config_template, get_default_config


def create_minimal_config() -> str:
    """创建最小化配置内容
    
    Returns:
        str: 最小化配置的YAML内容
    """
    return """# AI Services 最小化配置
version: "1.0"

services:
  chat:
    default_provider: "ollama"
    providers:
      ollama:
        base_url: "http://localhost:11434"
        model_name: "qwen3:1.7b"
        
  embedding:
    default_provider: "ollama"
    providers:
      ollama:
        base_url: "http://localhost:11434"
        model_name: "nomic-embed-text:latest"
        
  rerank:
    default_provider: "ollama"
    providers:
      ollama:
        base_url: "http://localhost:11434"
        model_name: "qwen3:1.7b"

logging:
  level: "INFO"
"""


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成AI Services配置文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置类型:
  minimal  - 最小化配置（仅包含必要设置）
  example  - 示例配置（包含常用设置和注释）
  full     - 完整配置（包含所有可用选项）

示例:
  python generate_config.py config.yaml minimal
  python generate_config.py my_config.yaml example
  python generate_config.py full_config.yaml full
        """
    )
    
    parser.add_argument(
        'output_file',
        help='输出配置文件路径'
    )
    
    parser.add_argument(
        'config_type',
        choices=['minimal', 'example', 'full'],
        default='example',
        nargs='?',
        help='配置类型 (默认: example)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制覆盖已存在的文件'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_file)
    
    # 检查文件是否已存在
    if output_path.exists() and not args.force:
        print(f"❌ 文件已存在: {output_path}")
        print("使用 --force 参数强制覆盖")
        sys.exit(1)
    
    try:
        if args.config_type == 'minimal':
            print(f"🔧 生成最小化配置: {output_path}")
            content = create_minimal_config()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif args.config_type == 'example':
            print(f"🔧 生成示例配置: {output_path}")
            # 复制示例配置文件
            example_path = Path(__file__).parent / "config.example.yaml"
            if example_path.exists():
                with open(example_path, 'r', encoding='utf-8') as src:
                    content = src.read()
                with open(output_path, 'w', encoding='utf-8') as dst:
                    dst.write(content)
            else:
                print("❌ 示例配置文件不存在，使用完整配置模板")
                create_config_template(str(output_path), 'yaml')
                
        elif args.config_type == 'full':
            print(f"🔧 生成完整配置: {output_path}")
            create_config_template(str(output_path), 'yaml')
        
        print(f"✅ 配置文件已生成: {output_path}")
        
        # 提供后续步骤建议
        print("\n📝 后续步骤:")
        print(f"  1. 编辑配置文件: {output_path}")
        print(f"  2. 验证配置: python validate_config.py {output_path}")
        print("  3. 运行测试: python quick_start.py")
        
        print("\n💡 提示:")
        print("  - 确保Ollama服务正在运行: ollama serve")
        print("  - 检查所需模型是否已安装: ollama list")
        print("  - 根据需要调整模型名称和服务地址")
        
    except Exception as e:
        print(f"❌ 生成配置文件失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()