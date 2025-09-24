"""
AI Services - 配置模块

这个包包含了AI服务模块的配置管理功能：
- 默认配置：预定义的服务配置
- 配置加载：从文件、字典、环境变量加载配置
- 配置验证：确保配置的完整性和有效性
- 配置模板：生成配置文件模板
- 配置合并：合并多个配置源

支持YAML和JSON格式的配置文件。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    # 配置获取和创建
    get_default_config,
    create_config_template,
    create_minimal_config,
    
    # 配置验证和加载
    validate_config,
    load_config_from_env,
    merge_configs,
    get_provider_config
)

__all__ = [
    # 配置获取和创建
    "get_default_config",
    "create_config_template", 
    "create_minimal_config",
    
    # 配置验证和加载
    "validate_config",
    "load_config_from_env",
    "merge_configs",
    "get_provider_config"
]