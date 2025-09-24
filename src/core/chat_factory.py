"""
聊天工厂模块

提供统一的聊天客户端创建和管理功能，支持多种聊天模型提供商。
参考Go版本的工厂模式设计。

Author: Assistant
Date: 2025-01-24
"""

from typing import Dict, Any, Optional, Type, Union
from enum import Enum
import logging

from .chat_interface import ChatInterface, ChatError, ChatConfigError
from .ollama_chat import OllamaChatClient, OllamaChatConfig, create_ollama_chat_client_from_config
from ..config.config_manager import ConfigManager


class ChatProvider(Enum):
    """聊天提供商枚举"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class ChatFactory:
    """聊天工厂类
    
    负责创建和管理不同类型的聊天客户端。
    支持多种聊天模型提供商的统一接口。
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """初始化聊天工厂
        
        Args:
            config_manager: 配置管理器实例，如果为None则创建默认实例
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self._providers: Dict[ChatProvider, Type[ChatInterface]] = {
            ChatProvider.OLLAMA: OllamaChatClient,
        }
        self._clients: Dict[str, ChatInterface] = {}
    
    def register_provider(self, provider: ChatProvider, client_class: Type[ChatInterface]):
        """注册聊天提供商
        
        Args:
            provider: 提供商类型
            client_class: 客户端类
        """
        self._providers[provider] = client_class
        self.logger.info(f"注册聊天提供商: {provider.value}")
    
    def create_chat_client(
        self,
        provider: Union[ChatProvider, str],
        config: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None
    ) -> ChatInterface:
        """创建聊天客户端
        
        Args:
            provider: 聊天提供商（枚举或字符串）
            config: 配置参数
            cache_key: 缓存键，如果提供则会缓存客户端实例
            
        Returns:
            ChatInterface: 聊天客户端实例
            
        Raises:
            ChatConfigError: 配置错误
        """
        # 如果是字符串，转换为枚举
        if isinstance(provider, str):
            try:
                provider = ChatProvider(provider)
            except ValueError:
                raise ChatConfigError(f"不支持的聊天提供商: {provider}")
        
        if provider not in self._providers:
            raise ChatConfigError(f"不支持的聊天提供商: {provider.value}")
        
        # 检查缓存
        if cache_key and cache_key in self._clients:
            self.logger.debug(f"使用缓存的聊天客户端: {cache_key}")
            return self._clients[cache_key]
        
        try:
            client_class = self._providers[provider]
            
            if provider == ChatProvider.OLLAMA:
                client = self._create_ollama_client(config or {})
            else:
                # 其他提供商的实现
                client = client_class(config)
            
            # 缓存客户端
            if cache_key:
                self._clients[cache_key] = client
                self.logger.debug(f"缓存聊天客户端: {cache_key}")
            
            self.logger.info(f"创建聊天客户端成功: {provider.value}")
            return client
            
        except Exception as e:
            error_msg = f"创建聊天客户端失败: {provider.value} - {str(e)}"
            self.logger.error(error_msg)
            raise ChatConfigError(error_msg) from e
    
    def _create_ollama_client(self, config: Dict[str, Any]) -> OllamaChatClient:
        """创建Ollama聊天客户端
        
        Args:
            config: 配置参数，用于覆盖默认配置
            
        Returns:
            OllamaChatClient: Ollama聊天客户端
        """
        # 优先使用配置管理器中的配置，然后用传入的config覆盖
        try:
            # 从配置管理器创建客户端
            client = create_ollama_chat_client_from_config(self.config_manager)
            
            # 如果有额外的配置参数，需要重新创建配置
            if config:
                chat_config = self.config_manager.get_chat_config()
                ollama_config = chat_config.get("ollama", {})
                
                # 合并配置
                merged_config = {
                    "base_url": config.get("base_url", ollama_config.get("base_url", "http://localhost:11434")),
                    "model_name": config.get("model_name", ollama_config.get("model_name", "qwen3:1.7b")),
                    "timeout": config.get("timeout", ollama_config.get("timeout", 300)),
                    "max_retries": config.get("max_retries", ollama_config.get("max_retries", 3)),
                    "verify_ssl": config.get("verify_ssl", ollama_config.get("verify_ssl", True)),
                    "temperature": config.get("temperature", ollama_config.get("temperature", 0.7)),
                    "max_tokens": config.get("max_tokens", ollama_config.get("max_tokens", 2048)),
                    "top_p": config.get("top_p", ollama_config.get("top_p", 0.9)),
                    "top_k": config.get("top_k", ollama_config.get("top_k", 40)),
                    "stream": config.get("stream", ollama_config.get("stream", False)),
                    "keep_alive": config.get("keep_alive", ollama_config.get("keep_alive", "5m")),
                    "system_prompt": config.get("system_prompt", ollama_config.get("system_prompt", "")),
                    "context": config.get("context", ollama_config.get("context", []))
                }
                
                ollama_config_obj = OllamaChatConfig(**merged_config)
                return OllamaChatClient(ollama_config_obj, self.config_manager)
            
            return client
            
        except Exception as e:
            self.logger.warning(f"从配置管理器创建Ollama客户端失败，使用传入配置: {str(e)}")
            # 回退到使用传入的配置
            ollama_config = OllamaChatConfig(
                base_url=config.get("base_url", "http://localhost:11434"),
                model_name=config.get("model_name", "qwen3:1.7b"),
                timeout=config.get("timeout", 300),
                max_retries=config.get("max_retries", 3),
                verify_ssl=config.get("verify_ssl", True)
            )
            return OllamaChatClient(ollama_config)
    
    def get_available_providers(self) -> list[ChatProvider]:
        """获取可用的聊天提供商列表
        
        Returns:
            list[ChatProvider]: 可用的提供商列表
        """
        return list(self._providers.keys())
    
    def test_provider_connection(self, provider: Union[ChatProvider, str], config: Optional[Dict[str, Any]] = None) -> bool:
        """测试提供商连接
        
        Args:
            provider: 聊天提供商（枚举或字符串）
            config: 配置参数
            
        Returns:
            bool: 连接是否成功
        """
        try:
            client = self.create_chat_client(provider, config)
            return client.test_connection()
        except Exception as e:
            # 如果是字符串，直接使用；如果是枚举，使用.value
            provider_name = provider if isinstance(provider, str) else provider.value
            self.logger.error(f"测试提供商连接失败: {provider_name} - {str(e)}")
            return False
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """清除缓存
        
        Args:
            cache_key: 要清除的缓存键，如果为None则清除所有缓存
        """
        if cache_key:
            if cache_key in self._clients:
                del self._clients[cache_key]
                self.logger.debug(f"清除缓存: {cache_key}")
        else:
            self._clients.clear()
            self.logger.debug("清除所有聊天客户端缓存")


# 全局工厂实例
_chat_factory = None


def get_chat_factory(config_manager: Optional[ConfigManager] = None) -> ChatFactory:
    """获取全局聊天工厂实例（单例模式）
    
    Args:
        config_manager: 配置管理器实例，仅在首次创建时使用
    
    Returns:
        ChatFactory: 聊天工厂实例
    """
    global _chat_factory
    if _chat_factory is None:
        _chat_factory = ChatFactory(config_manager)
    return _chat_factory


def create_chat_factory_from_config(config_manager: Optional[ConfigManager] = None) -> ChatFactory:
    """从配置管理器创建新的聊天工厂实例
    
    Args:
        config_manager: 配置管理器实例
        
    Returns:
        ChatFactory: 新的聊天工厂实例
    """
    return ChatFactory(config_manager)


# 便捷函数
def create_chat_client(
    provider: str,
    config: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
    config_manager: Optional[ConfigManager] = None
) -> ChatInterface:
    """创建聊天客户端的便捷函数
    
    Args:
        provider: 提供商名称（字符串）
        config: 配置参数
        cache_key: 缓存键
        config_manager: 配置管理器实例
        
    Returns:
        ChatInterface: 聊天客户端实例
    """
    factory = get_chat_factory(config_manager)
    provider_enum = ChatProvider(provider.lower())
    return factory.create_chat_client(provider_enum, config, cache_key)


def create_ollama_chat_client(
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[int] = None,
    cache_key: Optional[str] = None,
    config_manager: Optional[ConfigManager] = None
) -> ChatInterface:
    """创建Ollama聊天客户端的便捷函数
    
    Args:
        base_url: Ollama服务器地址，如果为None则使用配置文件中的值
        model_name: 模型名称，如果为None则使用配置文件中的值
        timeout: 请求超时时间，如果为None则使用配置文件中的值
        cache_key: 缓存键
        config_manager: 配置管理器实例
        
    Returns:
        ChatInterface: Ollama聊天客户端实例
    """
    config = {}
    if base_url is not None:
        config["base_url"] = base_url
    if model_name is not None:
        config["model_name"] = model_name
    if timeout is not None:
        config["timeout"] = timeout
    
    return create_chat_client("ollama", config, cache_key, config_manager)


def test_chat_connection(
    provider: str, 
    config: Optional[Dict[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None
) -> bool:
    """测试聊天连接的便捷函数
    
    Args:
        provider: 提供商名称
        config: 配置参数
        config_manager: 配置管理器实例
        
    Returns:
        bool: 连接是否成功
    """
    factory = get_chat_factory(config_manager)
    provider_enum = ChatProvider(provider.lower())
    return factory.test_provider_connection(provider_enum, config)