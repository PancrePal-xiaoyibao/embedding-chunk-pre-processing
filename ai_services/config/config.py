"""
Configuration Management - 配置管理

提供默认配置、配置模板生成和配置验证功能。
"""

import os
import json
import yaml
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import ServiceProvider, ServiceType


def get_default_config() -> Dict[str, Any]:
    """获取默认配置
    
    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        "version": "1.0",
        "services": {
            "chat": {
                "default_provider": "ollama",
                "providers": {
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "llama2",
                        "timeout": 30.0,
                        "max_retries": 3,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40
                        }
                    }
                }
            },
            "embedding": {
                "default_provider": "ollama",
                "providers": {
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "nomic-embed-text",
                        "timeout": 30.0,
                        "max_retries": 3
                    },
                    "local": {
                        "model_name": "all-MiniLM-L6-v2",
                        "device": "cpu"
                    }
                }
            },
            "rerank": {
                "default_provider": "embedding_based",
                "providers": {
                    "embedding_based": {
                        "similarity_method": "cosine",
                        "normalize_scores": True
                    },
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "llama2",
                        "timeout": 30.0,
                        "max_retries": 3
                    },
                    "cross_encoder": {
                        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        "device": "cpu",
                        "batch_size": 32
                    }
                }
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None
        }
    }


def create_config_template(output_path: str, format: str = "yaml") -> None:
    """创建配置模板文件
    
    Args:
        output_path: 输出文件路径
        format: 配置文件格式 ("yaml" 或 "json")
        
    Raises:
        ValueError: 不支持的格式时抛出
        IOError: 文件写入失败时抛出
    """
    if format not in ["yaml", "json"]:
        raise ValueError(f"不支持的配置格式: {format}")
    
    config = get_default_config()
    
    # 添加注释（仅YAML格式）
    if format == "yaml":
        config_with_comments = _add_yaml_comments(config)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(config_with_comments)
        except Exception as e:
            raise IOError(f"写入配置文件失败: {e}")
    
    elif format == "json":
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"写入配置文件失败: {e}")
    
    print(f"配置模板已创建: {output_path}")


def _add_yaml_comments(config: Dict[str, Any]) -> str:
    """为YAML配置添加注释
    
    Args:
        config: 配置字典
        
    Returns:
        str: 带注释的YAML字符串
    """
    yaml_content = f"""# AI Services Configuration
# AI服务配置文件

version: "{config['version']}"

# 服务配置
services:
  
  # 聊天服务配置
  chat:
    default_provider: "{config['services']['chat']['default_provider']}"
    providers:
      
      # Ollama聊天服务
      ollama:
        base_url: "{config['services']['chat']['providers']['ollama']['base_url']}"  # Ollama服务地址
        model_name: "{config['services']['chat']['providers']['ollama']['model_name']}"  # 默认模型
        timeout: {config['services']['chat']['providers']['ollama']['timeout']}  # 请求超时时间（秒）
        max_retries: {config['services']['chat']['providers']['ollama']['max_retries']}  # 最大重试次数
        stream: {str(config['services']['chat']['providers']['ollama']['stream']).lower()}  # 是否使用流式响应
        options:
          temperature: {config['services']['chat']['providers']['ollama']['options']['temperature']}  # 生成温度
          top_p: {config['services']['chat']['providers']['ollama']['options']['top_p']}  # Top-p采样
          top_k: {config['services']['chat']['providers']['ollama']['options']['top_k']}  # Top-k采样
  
  # 嵌入服务配置
  embedding:
    default_provider: "{config['services']['embedding']['default_provider']}"
    providers:
      
      # Ollama嵌入服务
      ollama:
        base_url: "{config['services']['embedding']['providers']['ollama']['base_url']}"  # Ollama服务地址
        model_name: "{config['services']['embedding']['providers']['ollama']['model_name']}"  # 嵌入模型
        timeout: {config['services']['embedding']['providers']['ollama']['timeout']}  # 请求超时时间（秒）
        max_retries: {config['services']['embedding']['providers']['ollama']['max_retries']}  # 最大重试次数
      
      # 本地嵌入服务
      local:
        model_name: "{config['services']['embedding']['providers']['local']['model_name']}"  # 本地模型名称
        device: "{config['services']['embedding']['providers']['local']['device']}"  # 计算设备 (cpu/cuda)
  
  # 重排序服务配置
  rerank:
    default_provider: "{config['services']['rerank']['default_provider']}"
    providers:
      
      # 基于嵌入的重排序
      embedding_based:
        similarity_method: "{config['services']['rerank']['providers']['embedding_based']['similarity_method']}"  # 相似度计算方法
        normalize_scores: {str(config['services']['rerank']['providers']['embedding_based']['normalize_scores']).lower()}  # 是否标准化分数
      
      # Ollama重排序服务
      ollama:
        base_url: "{config['services']['rerank']['providers']['ollama']['base_url']}"  # Ollama服务地址
        model_name: "{config['services']['rerank']['providers']['ollama']['model_name']}"  # 重排序模型
        timeout: {config['services']['rerank']['providers']['ollama']['timeout']}  # 请求超时时间（秒）
        max_retries: {config['services']['rerank']['providers']['ollama']['max_retries']}  # 最大重试次数
      
      # 交叉编码器重排序
      cross_encoder:
        model_name: "{config['services']['rerank']['providers']['cross_encoder']['model_name']}"  # 交叉编码器模型
        device: "{config['services']['rerank']['providers']['cross_encoder']['device']}"  # 计算设备
        batch_size: {config['services']['rerank']['providers']['cross_encoder']['batch_size']}  # 批处理大小

# 日志配置
logging:
  level: "{config['logging']['level']}"  # 日志级别 (DEBUG/INFO/WARNING/ERROR)
  format: "{config['logging']['format']}"  # 日志格式
  file: {config['logging']['file']}  # 日志文件路径 (null表示输出到控制台)
"""
    
    return yaml_content


def validate_config(config: Dict[str, Any]) -> List[str]:
    """验证配置
    
    Args:
        config: 配置字典
        
    Returns:
        List[str]: 验证错误列表，空列表表示验证通过
    """
    errors = []
    
    # 检查必需的顶级字段
    required_fields = ["version", "services"]
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必需字段: {field}")
    
    if "services" not in config:
        return errors
    
    services = config["services"]
    
    # 检查服务配置
    required_services = ["chat", "embedding", "rerank"]
    for service in required_services:
        if service not in services:
            errors.append(f"缺少服务配置: {service}")
            continue
        
        service_config = services[service]
        
        # 检查默认提供商
        if "default_provider" not in service_config:
            errors.append(f"服务 {service} 缺少 default_provider")
        
        # 检查提供商配置
        if "providers" not in service_config:
            errors.append(f"服务 {service} 缺少 providers 配置")
            continue
        
        providers = service_config["providers"]
        default_provider = service_config.get("default_provider")
        
        if default_provider and default_provider not in providers:
            errors.append(f"服务 {service} 的默认提供商 {default_provider} 不存在于 providers 中")
        
        # 验证具体提供商配置
        for provider_name, provider_config in providers.items():
            provider_errors = _validate_provider_config(service, provider_name, provider_config)
            errors.extend(provider_errors)
    
    return errors


def _validate_provider_config(service: str, provider: str, config: Dict[str, Any]) -> List[str]:
    """验证提供商配置
    
    Args:
        service: 服务名称
        provider: 提供商名称
        config: 提供商配置
        
    Returns:
        List[str]: 验证错误列表
    """
    errors = []
    
    # Ollama提供商的通用验证
    if provider == "ollama":
        required_fields = ["base_url", "model_name"]
        for field in required_fields:
            if field not in config:
                errors.append(f"服务 {service} 的 {provider} 提供商缺少字段: {field}")
        
        # 验证URL格式
        if "base_url" in config:
            base_url = config["base_url"]
            if not isinstance(base_url, str) or not base_url.strip():
                errors.append(f"服务 {service} 的 {provider} 提供商的 base_url 无效")
    
    # 本地提供商验证
    elif provider == "local":
        if service == "embedding":
            required_fields = ["model_name"]
            for field in required_fields:
                if field not in config:
                    errors.append(f"服务 {service} 的 {provider} 提供商缺少字段: {field}")
    
    # 交叉编码器验证
    elif provider == "cross_encoder":
        if service == "rerank":
            required_fields = ["model_name"]
            for field in required_fields:
                if field not in config:
                    errors.append(f"服务 {service} 的 {provider} 提供商缺少字段: {field}")
    
    # 基于嵌入的重排序验证
    elif provider == "embedding_based":
        if service == "rerank":
            if "similarity_method" in config:
                method = config["similarity_method"]
                if method not in ["cosine", "dot", "euclidean"]:
                    errors.append(f"服务 {service} 的 {provider} 提供商的 similarity_method 无效: {method}")
    
    return errors


def load_config_from_env() -> Dict[str, Any]:
    """从环境变量加载配置
    
    Returns:
        Dict[str, Any]: 从环境变量构建的配置字典
    """
    config = get_default_config()
    
    # 从环境变量覆盖配置
    env_mappings = {
        "OLLAMA_BASE_URL": ["services", "chat", "providers", "ollama", "base_url"],
        "OLLAMA_CHAT_MODEL": ["services", "chat", "providers", "ollama", "model_name"],
        "OLLAMA_EMBEDDING_MODEL": ["services", "embedding", "providers", "ollama", "model_name"],
        "OLLAMA_RERANK_MODEL": ["services", "rerank", "providers", "ollama", "model_name"],
        "LOG_LEVEL": ["logging", "level"],
        "LOG_FILE": ["logging", "file"],
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            _set_nested_value(config, config_path, value)
    
    return config


def _set_nested_value(config: Dict[str, Any], path: List[str], value: Any) -> None:
    """设置嵌套字典的值
    
    Args:
        config: 配置字典
        path: 配置路径
        value: 要设置的值
    """
    current = config
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[path[-1]] = value


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        Dict[str, Any]: 合并后的配置
    """
    import copy
    
    result = copy.deepcopy(base_config)
    
    def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_dict(base[key], value)
            else:
                base[key] = value
    
    _merge_dict(result, override_config)
    return result


def get_provider_config(
    config: Dict[str, Any], 
    service_type: ServiceType, 
    provider: ServiceProvider
) -> Dict[str, Any]:
    """获取特定提供商的配置
    
    Args:
        config: 完整配置
        service_type: 服务类型
        provider: 提供商
        
    Returns:
        Dict[str, Any]: 提供商配置
        
    Raises:
        KeyError: 配置不存在时抛出
    """
    service_name = service_type.value
    provider_name = provider.value
    
    try:
        return config["services"][service_name]["providers"][provider_name]
    except KeyError as e:
        raise KeyError(f"配置不存在: services.{service_name}.providers.{provider_name}")


def create_minimal_config() -> Dict[str, Any]:
    """创建最小配置
    
    Returns:
        Dict[str, Any]: 最小配置字典
    """
    return {
        "version": "1.0",
        "services": {
            "chat": {
                "default_provider": "ollama",
                "providers": {
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "llama2"
                    }
                }
            },
            "embedding": {
                "default_provider": "ollama",
                "providers": {
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "nomic-embed-text"
                    }
                }
            },
            "rerank": {
                "default_provider": "embedding_based",
                "providers": {
                    "embedding_based": {
                        "similarity_method": "cosine"
                    }
                }
            }
        }
    }