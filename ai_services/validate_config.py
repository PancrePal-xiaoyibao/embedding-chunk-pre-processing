#!/usr/bin/env python3
"""
配置验证脚本 - Configuration Validation Script

用于验证AI Services配置文件的正确性。
"""

import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config.config import validate_config, get_default_config


def load_config_file(file_path: str) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式错误时抛出
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML格式错误: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {e}")


def print_config_summary(config: Dict[str, Any]) -> None:
    """打印配置摘要
    
    Args:
        config: 配置字典
    """
    print("\n📋 配置摘要:")
    print(f"  版本: {config.get('version', 'N/A')}")
    
    services = config.get('services', {})
    
    # Chat服务
    chat_config = services.get('chat', {})
    chat_provider = chat_config.get('default_provider', 'N/A')
    print(f"  Chat服务: {chat_provider}")
    if chat_provider in chat_config.get('providers', {}):
        chat_model = chat_config['providers'][chat_provider].get('model_name', 'N/A')
        print(f"    模型: {chat_model}")
    
    # Embedding服务
    embedding_config = services.get('embedding', {})
    embedding_provider = embedding_config.get('default_provider', 'N/A')
    print(f"  Embedding服务: {embedding_provider}")
    if embedding_provider in embedding_config.get('providers', {}):
        embedding_model = embedding_config['providers'][embedding_provider].get('model_name', 'N/A')
        print(f"    模型: {embedding_model}")
    
    # Rerank服务
    rerank_config = services.get('rerank', {})
    rerank_provider = rerank_config.get('default_provider', 'N/A')
    print(f"  Rerank服务: {rerank_provider}")
    if rerank_provider in rerank_config.get('providers', {}):
        rerank_model = rerank_config['providers'][rerank_provider].get('model_name', 'N/A')
        print(f"    模型: {rerank_model}")
    
    # 日志配置
    logging_config = config.get('logging', {})
    log_level = logging_config.get('level', 'N/A')
    log_file = logging_config.get('file', 'console')
    print(f"  日志级别: {log_level}")
    print(f"  日志输出: {'文件' if log_file else '控制台'}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="验证AI Services配置文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python validate_config.py config.yaml
  python validate_config.py --default
  python validate_config.py config.json --verbose
        """
    )
    
    parser.add_argument(
        'config_file',
        nargs='?',
        help='配置文件路径 (YAML或JSON格式)'
    )
    
    parser.add_argument(
        '--default',
        action='store_true',
        help='验证默认配置'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    args = parser.parse_args()
    
    try:
        if args.default:
            print("🔍 验证默认配置...")
            config = get_default_config()
            config_source = "默认配置"
        elif args.config_file:
            print(f"🔍 验证配置文件: {args.config_file}")
            config = load_config_file(args.config_file)
            config_source = args.config_file
        else:
            # 尝试查找常见的配置文件
            possible_files = ['config.yaml', 'config.yml', 'config.json']
            config_file = None
            
            for file in possible_files:
                if Path(file).exists():
                    config_file = file
                    break
            
            if config_file:
                print(f"🔍 找到配置文件: {config_file}")
                config = load_config_file(config_file)
                config_source = config_file
            else:
                print("❌ 未找到配置文件，请指定配置文件路径或使用 --default 验证默认配置")
                print("可用的配置文件名: config.yaml, config.yml, config.json")
                sys.exit(1)
        
        # 验证配置
        errors = validate_config(config)
        
        if errors:
            print(f"\n❌ 配置验证失败 ({config_source}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            sys.exit(1)
        else:
            print(f"\n✅ 配置验证通过! ({config_source})")
        
        # 显示配置摘要
        if args.verbose:
            print_config_summary(config)
        
        print("\n💡 提示:")
        print("  - 确保Ollama服务正在运行: ollama serve")
        print("  - 检查模型是否已安装: ollama list")
        print("  - 运行快速测试: python quick_start.py")
        
    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ 格式错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()