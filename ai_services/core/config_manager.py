"""
AI Services Config Manager - 配置管理器

提供统一的配置管理功能，支持多种配置格式和验证。
"""

import os
import yaml
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

from .interfaces import ConfigManager, ServiceType, ServiceProvider
from .exceptions import ConfigurationError, ValidationError


@dataclass
class ServiceConfig:
    """服务配置数据类
    
    Args:
        provider: 服务提供商
        base_url: 服务基础URL
        api_key: API密钥
        model_name: 默认模型名称
        timeout: 请求超时时间
        max_retries: 最大重试次数
        extra_params: 额外参数
    """
    provider: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        result = {
            "provider": self.provider,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        
        if self.base_url:
            result["base_url"] = self.base_url
        if self.api_key:
            result["api_key"] = self.api_key
        if self.model_name:
            result["model_name"] = self.model_name
        
        result.update(self.extra_params)
        return result


@dataclass
class AIServiceConfig:
    """AI服务完整配置
    
    Args:
        chat_services: 聊天服务配置
        embedding_services: 嵌入服务配置
        rerank_services: 重排序服务配置
        default_providers: 默认服务提供商
        logging_config: 日志配置
    """
    chat_services: Dict[str, ServiceConfig] = field(default_factory=dict)
    embedding_services: Dict[str, ServiceConfig] = field(default_factory=dict)
    rerank_services: Dict[str, ServiceConfig] = field(default_factory=dict)
    default_providers: Dict[str, str] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_service_config(self, service_type: ServiceType, provider: ServiceProvider) -> ServiceConfig:
        """获取特定服务配置
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            
        Returns:
            ServiceConfig: 服务配置
            
        Raises:
            ConfigurationError: 配置不存在时抛出
        """
        service_map = {
            ServiceType.CHAT: self.chat_services,
            ServiceType.EMBEDDING: self.embedding_services,
            ServiceType.RERANK: self.rerank_services,
        }
        
        services = service_map.get(service_type)
        if not services:
            raise ConfigurationError(f"未找到服务类型 {service_type.value} 的配置")
        
        config = services.get(provider.value)
        if not config:
            raise ConfigurationError(
                f"未找到服务提供商 {provider.value} 在 {service_type.value} 中的配置"
            )
        
        return config
    
    def get_default_provider(self, service_type: ServiceType) -> Optional[ServiceProvider]:
        """获取默认服务提供商
        
        Args:
            service_type: 服务类型
            
        Returns:
            Optional[ServiceProvider]: 默认服务提供商
        """
        provider_name = self.default_providers.get(service_type.value)
        if provider_name:
            try:
                return ServiceProvider(provider_name)
            except ValueError:
                return None
        return None


class AIServiceConfigManager(ConfigManager):
    """AI服务配置管理器
    
    负责加载、验证和管理AI服务的配置。
    
    Args:
        config_path: 配置文件路径
        logger: 日志记录器
    """
    
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.config_path = config_path
        self.logger = logger or logging.getLogger(__name__)
        self._config: Optional[AIServiceConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用初始化时的路径
            
        Returns:
            Dict[str, Any]: 原始配置数据
            
        Raises:
            ConfigurationError: 配置加载失败时抛出
        """
        path = config_path or self.config_path
        if not path:
            raise ConfigurationError("未指定配置文件路径")
        
        config_file = Path(path)
        if not config_file.exists():
            raise ConfigurationError(f"配置文件不存在: {path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    raw_config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    raw_config = json.load(f)
                else:
                    raise ConfigurationError(f"不支持的配置文件格式: {config_file.suffix}")
            
            self._raw_config = raw_config
            self.logger.info(f"成功加载配置文件: {path}")
            return raw_config
            
        except Exception as e:
            raise ConfigurationError(
                f"加载配置文件失败: {path}",
                config_path=str(path),
                original_error=e
            )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置
        
        Args:
            config: 配置数据
            
        Returns:
            bool: 配置是否有效
            
        Raises:
            ValidationError: 配置验证失败时抛出
        """
        errors = []
        
        # 验证必需的顶级键
        required_keys = ['services']
        for key in required_keys:
            if key not in config:
                errors.append(f"缺少必需的配置键: {key}")
        
        if errors:
            raise ValidationError(
                "配置验证失败",
                details={"validation_errors": errors}
            )
        
        # 验证服务配置
        services = config.get('services', {})
        for service_type in ['chat', 'embedding', 'rerank']:
            if service_type in services:
                self._validate_service_configs(services[service_type], service_type, errors)
        
        if errors:
            raise ValidationError(
                "配置验证失败",
                details={"validation_errors": errors}
            )
        
        return True
    
    def _validate_service_configs(self, service_configs: Dict[str, Any], service_type: str, errors: List[str]):
        """验证特定服务类型的配置
        
        Args:
            service_configs: 服务配置字典
            service_type: 服务类型
            errors: 错误列表
        """
        for provider, config in service_configs.items():
            if not isinstance(config, dict):
                errors.append(f"{service_type}.{provider}: 配置必须是字典格式")
                continue
            
            # 验证必需字段
            if 'provider' not in config:
                config['provider'] = provider  # 自动设置provider
            
            # 验证数值字段
            if 'timeout' in config:
                try:
                    float(config['timeout'])
                except (ValueError, TypeError):
                    errors.append(f"{service_type}.{provider}.timeout: 必须是数值")
            
            if 'max_retries' in config:
                try:
                    int(config['max_retries'])
                except (ValueError, TypeError):
                    errors.append(f"{service_type}.{provider}.max_retries: 必须是整数")
    
    def get_service_config(self, service_type: ServiceType, provider: ServiceProvider) -> Dict[str, Any]:
        """获取特定服务的配置
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            
        Returns:
            Dict[str, Any]: 服务配置
            
        Raises:
            ConfigurationError: 配置不存在时抛出
        """
        if not self._config:
            self._parse_config()
        
        try:
            service_config = self._config.get_service_config(service_type, provider)
            return service_config.to_dict()
        except Exception as e:
            raise ConfigurationError(
                f"获取服务配置失败: {service_type.value}.{provider.value}",
                original_error=e
            )
    
    def get_parsed_config(self) -> AIServiceConfig:
        """获取解析后的配置对象
        
        Returns:
            AIServiceConfig: 解析后的配置
        """
        if not self._config:
            self._parse_config()
        return self._config
    
    def _parse_config(self):
        """解析原始配置为结构化对象"""
        if not self._raw_config:
            raise ConfigurationError("未加载配置文件，请先调用 load_config()")
        
        try:
            # 解析服务配置
            services = self._raw_config.get('services', {})
            
            chat_services = {}
            embedding_services = {}
            rerank_services = {}
            
            # 解析聊天服务
            for provider, config in services.get('chat', {}).items():
                chat_services[provider] = self._parse_service_config(config, provider)
            
            # 解析嵌入服务
            for provider, config in services.get('embedding', {}).items():
                embedding_services[provider] = self._parse_service_config(config, provider)
            
            # 解析重排序服务
            for provider, config in services.get('rerank', {}).items():
                rerank_services[provider] = self._parse_service_config(config, provider)
            
            # 解析默认提供商
            default_providers = self._raw_config.get('default_providers', {})
            
            # 解析日志配置
            logging_config = self._raw_config.get('logging', {})
            
            self._config = AIServiceConfig(
                chat_services=chat_services,
                embedding_services=embedding_services,
                rerank_services=rerank_services,
                default_providers=default_providers,
                logging_config=logging_config
            )
            
        except Exception as e:
            raise ConfigurationError("解析配置失败", original_error=e)
    
    def _parse_service_config(self, config: Dict[str, Any], provider: str) -> ServiceConfig:
        """解析单个服务配置
        
        Args:
            config: 原始配置
            provider: 服务提供商
            
        Returns:
            ServiceConfig: 解析后的服务配置
        """
        # 提取已知字段
        base_url = config.get('base_url')
        api_key = config.get('api_key')
        model_name = config.get('model_name')
        timeout = float(config.get('timeout', 30.0))
        max_retries = int(config.get('max_retries', 3))
        
        # 提取额外参数
        extra_params = {}
        known_keys = {'provider', 'base_url', 'api_key', 'model_name', 'timeout', 'max_retries'}
        for key, value in config.items():
            if key not in known_keys:
                extra_params[key] = value
        
        return ServiceConfig(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            extra_params=extra_params
        )
    
    def reload_config(self):
        """重新加载配置"""
        self._raw_config = None
        self._config = None
        if self.config_path:
            self.load_config()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AIServiceConfigManager':
        """从字典创建配置管理器
        
        Args:
            config_dict: 配置字典
            
        Returns:
            AIServiceConfigManager: 配置管理器实例
        """
        manager = cls()
        manager._raw_config = config_dict
        manager.validate_config(config_dict)
        return manager
    
    @classmethod
    def from_env(cls, prefix: str = "AI_SERVICE_") -> 'AIServiceConfigManager':
        """从环境变量创建配置管理器
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            AIServiceConfigManager: 配置管理器实例
        """
        config = {}
        
        # 从环境变量构建配置
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = key[len(prefix):].lower()
                config[config_key] = value
        
        return cls.from_dict({"services": config})