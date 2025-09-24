#!/usr/bin/env python3
"""
AI Services 配置管理示例

演示如何使用不同的配置方式来创建和管理AI服务。
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_services import AIServiceFactory
from ai_services.config import (
    get_default_config, 
    create_config_template, 
    validate_config,
    load_config_from_env,
    merge_configs
)


def default_config_example():
    """默认配置示例"""
    print("=== 默认配置示例 ===")
    
    # 获取默认配置
    config = get_default_config()
    print("默认配置结构:")
    print(f"  版本: {config['version']}")
    print(f"  服务数量: {len(config['services'])}")
    
    for service_name, service_config in config['services'].items():
        providers = list(service_config['providers'].keys())
        default_provider = service_config['default_provider']
        print(f"  {service_name}: 默认提供商={default_provider}, 可用提供商={providers}")
    
    # 使用默认配置创建工厂
    factory = AIServiceFactory.create_default()
    print("\n✅ 使用默认配置创建工厂成功")
    
    return config


def config_template_example():
    """配置模板示例"""
    print("\n=== 配置模板示例 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建YAML配置模板
        yaml_path = os.path.join(temp_dir, "config.yaml")
        create_config_template(yaml_path, format="yaml")
        print(f"✅ YAML配置模板已创建: {yaml_path}")
        
        # 创建JSON配置模板
        json_path = os.path.join(temp_dir, "config.json")
        create_config_template(json_path, format="json")
        print(f"✅ JSON配置模板已创建: {json_path}")
        
        # 显示YAML配置内容的前几行
        print("\nYAML配置模板预览:")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:15]  # 显示前15行
            for line in lines:
                print(f"  {line.rstrip()}")
            if len(lines) >= 15:
                print("  ...")
        
        # 从配置文件创建工厂
        try:
            factory = AIServiceFactory.from_config_file(yaml_path)
            print("\n✅ 从YAML配置文件创建工厂成功")
        except Exception as e:
            print(f"\n❌ 从配置文件创建工厂失败: {e}")


def custom_config_example():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 创建自定义配置
    custom_config = {
        "version": "1.0",
        "services": {
            "chat": {
                "default_provider": "ollama",
                "providers": {
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "llama2:7b",  # 指定特定版本
                        "timeout": 60.0,  # 增加超时时间
                        "max_retries": 5,  # 增加重试次数
                        "options": {
                            "temperature": 0.3,  # 降低温度
                            "top_p": 0.8,
                            "top_k": 30
                        }
                    }
                }
            },
            "embedding": {
                "default_provider": "local",  # 使用本地模型
                "providers": {
                    "local": {
                        "model_name": "all-MiniLM-L6-v2",
                        "device": "cpu"
                    },
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "nomic-embed-text",
                        "timeout": 30.0,
                        "max_retries": 3
                    }
                }
            },
            "rerank": {
                "default_provider": "cross_encoder",  # 使用交叉编码器
                "providers": {
                    "cross_encoder": {
                        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        "device": "cpu",
                        "batch_size": 16
                    },
                    "embedding_based": {
                        "similarity_method": "cosine",
                        "normalize_scores": True
                    }
                }
            }
        },
        "logging": {
            "level": "DEBUG",  # 启用调试日志
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    print("自定义配置特点:")
    print("  - Chat: 使用llama2:7b模型，降低温度参数")
    print("  - Embedding: 默认使用本地模型")
    print("  - Rerank: 默认使用交叉编码器")
    print("  - 启用DEBUG日志级别")
    
    # 验证配置
    errors = validate_config(custom_config)
    if errors:
        print(f"\n❌ 配置验证失败:")
        for error in errors:
            print(f"    {error}")
        return
    
    print("\n✅ 配置验证通过")
    
    # 使用自定义配置创建工厂
    try:
        factory = AIServiceFactory.from_config(custom_config)
        print("✅ 使用自定义配置创建工厂成功")
        
        # 测试服务创建
        chat_service = factory.create_service("chat")
        print("✅ Chat服务创建成功")
        
        # 显示实际使用的提供商
        chat_providers = factory.get_available_providers("chat")
        embedding_providers = factory.get_available_providers("embedding")
        rerank_providers = factory.get_available_providers("rerank")
        
        print(f"\n可用提供商:")
        print(f"  Chat: {chat_providers}")
        print(f"  Embedding: {embedding_providers}")
        print(f"  Rerank: {rerank_providers}")
        
    except Exception as e:
        print(f"\n❌ 使用自定义配置失败: {e}")


def env_config_example():
    """环境变量配置示例"""
    print("\n=== 环境变量配置示例 ===")
    
    # 设置环境变量
    original_env = {}
    env_vars = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_CHAT_MODEL": "llama2:13b",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text:latest",
        "LOG_LEVEL": "WARNING"
    }
    
    print("设置环境变量:")
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)  # 保存原值
        os.environ[key] = value
        print(f"  {key}={value}")
    
    try:
        # 从环境变量加载配置
        env_config = load_config_from_env()
        
        print("\n从环境变量加载的配置:")
        print(f"  Ollama Base URL: {env_config['services']['chat']['providers']['ollama']['base_url']}")
        print(f"  Chat Model: {env_config['services']['chat']['providers']['ollama']['model_name']}")
        print(f"  Embedding Model: {env_config['services']['embedding']['providers']['ollama']['model_name']}")
        print(f"  Log Level: {env_config['logging']['level']}")
        
        # 使用环境变量配置创建工厂
        factory = AIServiceFactory.from_config(env_config)
        print("\n✅ 使用环境变量配置创建工厂成功")
        
    except Exception as e:
        print(f"\n❌ 环境变量配置失败: {e}")
    
    finally:
        # 恢复原始环境变量
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def config_merge_example():
    """配置合并示例"""
    print("\n=== 配置合并示例 ===")
    
    # 基础配置
    base_config = get_default_config()
    
    # 覆盖配置
    override_config = {
        "services": {
            "chat": {
                "providers": {
                    "ollama": {
                        "model_name": "codellama",  # 更改模型
                        "options": {
                            "temperature": 0.1  # 更改温度
                        }
                    }
                }
            },
            "embedding": {
                "default_provider": "local"  # 更改默认提供商
            }
        },
        "logging": {
            "level": "ERROR"  # 更改日志级别
        }
    }
    
    print("基础配置:")
    print(f"  Chat模型: {base_config['services']['chat']['providers']['ollama']['model_name']}")
    print(f"  Chat温度: {base_config['services']['chat']['providers']['ollama']['options']['temperature']}")
    print(f"  Embedding默认提供商: {base_config['services']['embedding']['default_provider']}")
    print(f"  日志级别: {base_config['logging']['level']}")
    
    print("\n覆盖配置:")
    print(f"  Chat模型: {override_config['services']['chat']['providers']['ollama']['model_name']}")
    print(f"  Chat温度: {override_config['services']['chat']['providers']['ollama']['options']['temperature']}")
    print(f"  Embedding默认提供商: {override_config['services']['embedding']['default_provider']}")
    print(f"  日志级别: {override_config['logging']['level']}")
    
    # 合并配置
    merged_config = merge_configs(base_config, override_config)
    
    print("\n合并后配置:")
    print(f"  Chat模型: {merged_config['services']['chat']['providers']['ollama']['model_name']}")
    print(f"  Chat温度: {merged_config['services']['chat']['providers']['ollama']['options']['temperature']}")
    print(f"  Chat Top-P: {merged_config['services']['chat']['providers']['ollama']['options']['top_p']}")  # 保持原值
    print(f"  Embedding默认提供商: {merged_config['services']['embedding']['default_provider']}")
    print(f"  日志级别: {merged_config['logging']['level']}")
    
    # 验证合并后的配置
    errors = validate_config(merged_config)
    if errors:
        print(f"\n❌ 合并配置验证失败:")
        for error in errors:
            print(f"    {error}")
    else:
        print("\n✅ 合并配置验证通过")


def config_validation_example():
    """配置验证示例"""
    print("\n=== 配置验证示例 ===")
    
    # 有效配置
    valid_config = get_default_config()
    errors = validate_config(valid_config)
    print(f"有效配置验证结果: {len(errors)} 个错误")
    
    # 无效配置示例
    invalid_configs = [
        {
            "name": "缺少version字段",
            "config": {
                "services": {
                    "chat": {
                        "default_provider": "ollama",
                        "providers": {"ollama": {"base_url": "http://localhost:11434"}}
                    }
                }
            }
        },
        {
            "name": "缺少services字段",
            "config": {
                "version": "1.0"
            }
        },
        {
            "name": "默认提供商不存在",
            "config": {
                "version": "1.0",
                "services": {
                    "chat": {
                        "default_provider": "nonexistent",
                        "providers": {
                            "ollama": {
                                "base_url": "http://localhost:11434",
                                "model_name": "llama2"
                            }
                        }
                    }
                }
            }
        },
        {
            "name": "Ollama提供商缺少必需字段",
            "config": {
                "version": "1.0",
                "services": {
                    "chat": {
                        "default_provider": "ollama",
                        "providers": {
                            "ollama": {
                                "base_url": "http://localhost:11434"
                                # 缺少 model_name
                            }
                        }
                    }
                }
            }
        }
    ]
    
    print("\n无效配置验证:")
    for example in invalid_configs:
        print(f"\n  {example['name']}:")
        errors = validate_config(example['config'])
        if errors:
            for error in errors:
                print(f"    ❌ {error}")
        else:
            print("    ✅ 意外通过验证")


def main():
    """主函数"""
    print("🔧 AI Services 配置管理示例")
    print("=" * 50)
    
    # 各种配置示例
    default_config_example()
    config_template_example()
    custom_config_example()
    env_config_example()
    config_merge_example()
    config_validation_example()
    
    print("\n🎉 所有配置示例运行完成！")
    print("\n💡 提示:")
    print("  - 使用 create_config_template() 生成配置模板")
    print("  - 使用 validate_config() 验证配置有效性")
    print("  - 支持YAML和JSON格式的配置文件")
    print("  - 可以通过环境变量覆盖配置")
    print("  - 使用 merge_configs() 合并多个配置")


if __name__ == "__main__":
    main()