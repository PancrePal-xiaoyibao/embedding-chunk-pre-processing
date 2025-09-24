"""
AI Services Factory - 统一的AI服务工厂

提供统一的接口来创建和管理各种AI服务实例。
"""

import logging
from typing import Any, Dict, Optional, Type, Union
from pathlib import Path

from .interfaces import ServiceFactory, ServiceType, ServiceProvider, BaseService
from .config_manager import AIServiceConfigManager, AIServiceConfig
from .exceptions import (
    AIServiceError, ConfigurationError, ServiceNotAvailableError,
    ConnectionError, AuthenticationError
)


class AIServiceFactory(ServiceFactory):
    """AI服务工厂类
    
    统一管理和创建各种AI服务实例，包括Chat、Embedding和Rerank服务。
    
    Args:
        config_manager: 配置管理器
        logger: 日志记录器
    """
    
    def __init__(
        self, 
        config_manager: Optional[AIServiceConfigManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)
        self._service_registry: Dict[str, Type[BaseService]] = {}
        self._service_instances: Dict[str, BaseService] = {}
        
        # 注册默认服务
        self._register_default_services()
    
    def _register_default_services(self):
        """注册默认的服务实现"""
        try:
            # 延迟导入以避免循环依赖
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from services.chat_service import OllamaChatService
            from services.embedding_service import OllamaEmbeddingService
            from services.rerank_service import OllamaRerankService
            
            # 注册Ollama服务
            self.register_service(ServiceType.CHAT, ServiceProvider.OLLAMA, OllamaChatService)
            self.register_service(ServiceType.EMBEDDING, ServiceProvider.OLLAMA, OllamaEmbeddingService)
            self.register_service(ServiceType.RERANK, ServiceProvider.OLLAMA, OllamaRerankService)
            
            self.logger.info("默认服务注册完成")
            
        except ImportError as e:
            self.logger.warning(f"部分服务注册失败: {e}")
    
    def register_service(
        self, 
        service_type: ServiceType, 
        provider: ServiceProvider, 
        service_class: Type[BaseService]
    ):
        """注册服务实现
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            service_class: 服务实现类
        """
        key = f"{service_type.value}_{provider.value}"
        self._service_registry[key] = service_class
        self.logger.debug(f"注册服务: {key} -> {service_class.__name__}")
    
    def create_service(
        self, 
        service_type: ServiceType, 
        provider: Optional[ServiceProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseService:
        """创建服务实例
        
        Args:
            service_type: 服务类型
            provider: 服务提供商，如果为None则使用默认提供商
            config: 自定义配置，如果为None则从配置管理器获取
            
        Returns:
            BaseService: 服务实例
            
        Raises:
            ServiceNotAvailableError: 服务不可用时抛出
            ConfigurationError: 配置错误时抛出
        """
        # 确定服务提供商
        if provider is None:
            provider = self._get_default_provider(service_type)
            if provider is None:
                raise ServiceNotAvailableError(
                    f"未找到服务类型 {service_type.value} 的默认提供商"
                )
        
        # 获取服务配置
        if config is None:
            config = self._get_service_config(service_type, provider)
        
        # 创建服务实例
        service_key = f"{service_type.value}_{provider.value}"
        instance_key = f"{service_key}_{id(config)}"
        
        # 检查是否已有实例
        if instance_key in self._service_instances:
            return self._service_instances[instance_key]
        
        # 获取服务类
        service_class = self._service_registry.get(service_key)
        if service_class is None:
            raise ServiceNotAvailableError(
                f"未找到服务实现: {service_type.value}.{provider.value}"
            )
        
        try:
            # 创建服务实例
            service_instance = service_class(provider=provider, config=config, logger=self.logger)
            self._service_instances[instance_key] = service_instance
            
            self.logger.info(f"创建服务实例: {service_type.value}.{provider.value}")
            return service_instance
            
        except Exception as e:
            raise ServiceNotAvailableError(
                f"创建服务实例失败: {service_type.value}.{provider.value}",
                original_error=e
            )
    
    def get_available_providers(self, service_type: ServiceType) -> list[ServiceProvider]:
        """获取可用的服务提供商
        
        Args:
            service_type: 服务类型
            
        Returns:
            list[ServiceProvider]: 可用的服务提供商列表
        """
        providers = []
        for key in self._service_registry.keys():
            if key.startswith(f"{service_type.value}_"):
                provider_name = key.split("_", 1)[1]
                try:
                    provider = ServiceProvider(provider_name)
                    providers.append(provider)
                except ValueError:
                    continue
        
        return providers
    
    def is_provider_available(self, service_type: ServiceType, provider: ServiceProvider) -> bool:
        """检查服务提供商是否可用
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            
        Returns:
            bool: 是否可用
        """
        key = f"{service_type.value}_{provider.value}"
        return key in self._service_registry
    
    def test_service_connection(
        self, 
        service_type: ServiceType, 
        provider: Optional[ServiceProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """测试服务连接
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            config: 自定义配置
            
        Returns:
            bool: 连接是否成功
            
        Raises:
            ConnectionError: 连接失败时抛出
            AuthenticationError: 认证失败时抛出
        """
        try:
            service = self.create_service(service_type, provider, config)
            
            # 调用服务的健康检查方法
            if hasattr(service, 'health_check'):
                return service.health_check()
            else:
                # 如果没有健康检查方法，尝试创建一个简单的测试请求
                return self._perform_basic_test(service, service_type)
                
        except AuthenticationError:
            raise
        except Exception as e:
            raise ConnectionError(
                f"服务连接测试失败: {service_type.value}.{provider.value if provider else 'default'}",
                original_error=e
            )
    
    def _perform_basic_test(self, service: BaseService, service_type: ServiceType) -> bool:
        """执行基础测试
        
        Args:
            service: 服务实例
            service_type: 服务类型
            
        Returns:
            bool: 测试是否成功
        """
        try:
            if service_type == ServiceType.CHAT:
                # 对于聊天服务，发送一个简单的测试消息
                if hasattr(service, 'chat'):
                    response = service.chat("Hello")
                    return response is not None
            elif service_type == ServiceType.EMBEDDING:
                # 对于嵌入服务，嵌入一个简单的文本
                if hasattr(service, 'embed'):
                    response = service.embed("test")
                    return response is not None
            elif service_type == ServiceType.RERANK:
                # 对于重排序服务，执行一个简单的重排序
                if hasattr(service, 'rerank'):
                    response = service.rerank("query", ["doc1", "doc2"])
                    return response is not None
            
            return True
            
        except Exception:
            return False
    
    def _get_default_provider(self, service_type: ServiceType) -> Optional[ServiceProvider]:
        """获取默认服务提供商
        
        Args:
            service_type: 服务类型
            
        Returns:
            Optional[ServiceProvider]: 默认服务提供商
        """
        if self.config_manager:
            try:
                config = self.config_manager.get_parsed_config()
                return config.get_default_provider(service_type)
            except Exception as e:
                self.logger.warning(f"获取默认提供商失败: {e}")
        
        # 如果没有配置管理器或获取失败，返回第一个可用的提供商
        available_providers = self.get_available_providers(service_type)
        return available_providers[0] if available_providers else None
    
    def _get_service_config(self, service_type: ServiceType, provider: ServiceProvider) -> Dict[str, Any]:
        """获取服务配置
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            
        Returns:
            Dict[str, Any]: 服务配置
            
        Raises:
            ConfigurationError: 配置获取失败时抛出
        """
        if self.config_manager:
            try:
                return self.config_manager.get_service_config(service_type, provider)
            except Exception as e:
                self.logger.warning(f"从配置管理器获取配置失败: {e}")
        
        # 返回默认配置
        return self._get_default_config(service_type, provider)
    
    def _get_default_config(self, service_type: ServiceType, provider: ServiceProvider) -> Dict[str, Any]:
        """获取默认配置
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            
        Returns:
            Dict[str, Any]: 默认配置
        """
        default_configs = {
            ServiceProvider.OLLAMA: {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "timeout": 30.0,
                "max_retries": 3,
            }
        }
        
        base_config = default_configs.get(provider, {
            "provider": provider.value,
            "timeout": 30.0,
            "max_retries": 3,
        })
        
        # 根据服务类型添加特定配置
        if service_type == ServiceType.CHAT:
            base_config.update({
                "model_name": "qwen3:1.7b" if provider == ServiceProvider.OLLAMA else "gpt-3.5-turbo"
            })
        elif service_type == ServiceType.EMBEDDING:
            base_config.update({
                "model_name": "nomic-embed-text:latest" if provider == ServiceProvider.OLLAMA else "text-embedding-ada-002"
            })
        elif service_type == ServiceType.RERANK:
            base_config.update({
                "model_name": "qwen3:1.7b" if provider == ServiceProvider.OLLAMA else "rerank-english-v3.0"
            })
        
        return base_config
    
    async def test_connections(self) -> Dict[str, bool]:
        """异步测试所有可用服务的连接状态
        
        Returns:
            Dict[str, bool]: 服务名称到连接状态的映射
        """
        results = {}
        
        # 测试所有服务类型的连接
        for service_type in ServiceType:
            providers = self.get_available_providers(service_type)
            
            for provider in providers:
                service_name = f"{service_type.value}.{provider.value}"
                try:
                    service = self.create_service(service_type, provider)
                    
                    # 调用异步健康检查方法
                    if hasattr(service, 'health_check'):
                        is_healthy = await service.health_check()
                        results[service_name] = is_healthy
                    else:
                        # 如果没有健康检查方法，尝试基础测试
                        is_healthy = await self._perform_basic_test_async(service, service_type)
                        results[service_name] = is_healthy
                        
                except Exception as e:
                    self.logger.warning(f"测试服务连接失败 {service_name}: {e}")
                    results[service_name] = False
        
        return results
    
    async def _perform_basic_test_async(self, service: BaseService, service_type: ServiceType) -> bool:
        """执行异步基础测试
        
        Args:
            service: 服务实例
            service_type: 服务类型
            
        Returns:
            bool: 测试是否成功
        """
        try:
            if service_type == ServiceType.CHAT:
                # 对于聊天服务，发送一个简单的测试消息
                if hasattr(service, 'chat_async'):
                    response = await service.chat_async("Hello")
                    return response is not None
                elif hasattr(service, 'chat'):
                    response = service.chat("Hello")
                    return response is not None
            elif service_type == ServiceType.EMBEDDING:
                # 对于嵌入服务，嵌入一个简单的文本
                if hasattr(service, 'embed_async'):
                    response = await service.embed_async("test")
                    return response is not None
                elif hasattr(service, 'embed'):
                    response = service.embed("test")
                    return response is not None
            elif service_type == ServiceType.RERANK:
                # 对于重排序服务，执行一个简单的重排序
                if hasattr(service, 'rerank_async'):
                    response = await service.rerank_async("query", ["doc1", "doc2"])
                    return response is not None
                elif hasattr(service, 'rerank'):
                    response = service.rerank("query", ["doc1", "doc2"])
                    return response is not None
            
            return True
            
        except Exception as e:
            self.logger.warning(f"基础测试失败: {e}")
            return False

    def shutdown(self):
        """关闭工厂和所有服务实例"""
        for instance in self._service_instances.values():
            try:
                if hasattr(instance, 'close'):
                    instance.close()
            except Exception as e:
                self.logger.warning(f"关闭服务实例失败: {e}")
        
        self._service_instances.clear()
        self.logger.info("AI服务工厂已关闭")
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'AIServiceFactory':
        """从配置文件创建工厂实例
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            AIServiceFactory: 工厂实例
        """
        config_manager = AIServiceConfigManager(config_path)
        config_manager.load_config()
        return cls(config_manager)
    
    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> 'AIServiceFactory':
        """从配置字典创建工厂实例
        
        Args:
            config_dict: 配置字典
            
        Returns:
            AIServiceFactory: 工厂实例
        """
        config_manager = AIServiceConfigManager.from_dict(config_dict)
        return cls(config_manager)
    
    @classmethod
    def create_default(cls) -> 'AIServiceFactory':
        """创建默认工厂实例
        
        Returns:
            AIServiceFactory: 默认工厂实例
        """
        return cls()


# 便捷函数
def create_chat_service(
    provider: Optional[ServiceProvider] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None
) -> BaseService:
    """创建聊天服务
    
    Args:
        provider: 服务提供商
        config: 自定义配置
        config_path: 配置文件路径
        
    Returns:
        BaseService: 聊天服务实例
    """
    if config_path:
        factory = AIServiceFactory.from_config_file(config_path)
    else:
        factory = AIServiceFactory.create_default()
    
    return factory.create_service(ServiceType.CHAT, provider, config)


def create_embedding_service(
    provider: Optional[ServiceProvider] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None
) -> BaseService:
    """创建嵌入服务
    
    Args:
        provider: 服务提供商
        config: 自定义配置
        config_path: 配置文件路径
        
    Returns:
        BaseService: 嵌入服务实例
    """
    if config_path:
        factory = AIServiceFactory.from_config_file(config_path)
    else:
        factory = AIServiceFactory.create_default()
    
    return factory.create_service(ServiceType.EMBEDDING, provider, config)


def create_rerank_service(
    provider: Optional[ServiceProvider] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None
) -> BaseService:
    """创建重排序服务
    
    Args:
        provider: 服务提供商
        config: 自定义配置
        config_path: 配置文件路径
        
    Returns:
        BaseService: 重排序服务实例
    """
    if config_path:
        factory = AIServiceFactory.from_config_file(config_path)
    else:
        factory = AIServiceFactory.create_default()
    
    return factory.create_service(ServiceType.RERANK, provider, config)