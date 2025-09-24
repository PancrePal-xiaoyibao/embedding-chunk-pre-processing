"""AI Services - 统一的AI服务模块

这个模块提供了统一的AI服务接口，支持Chat、Embedding和Rerank功能。
支持多种提供商，包括Ollama、本地模型等。

主要特性:
- 统一的服务工厂模式
- 异步和同步支持
- 灵活的配置管理
- 健康检查和错误处理
- 流式处理支持

快速开始:
    >>> from ai_services import AIServiceFactory
    >>> factory = AIServiceFactory.create_default()
    >>> chat_service = factory.create_service("chat")
    >>> response = chat_service.chat([{"role": "user", "content": "Hello!"}])

或者使用便捷函数:
    >>> from ai_services import create_chat_service
    >>> chat_service = create_chat_service()
    >>> response = chat_service.chat([{"role": "user", "content": "Hello!"}])
"""

__version__ = "1.0.0"
__author__ = "AI Services Team"
__email__ = "support@aiservices.com"
__description__ = "统一的AI服务模块，支持Chat、Embedding和Rerank功能"

# 核心工厂类
from .core.factory import AIServiceFactory

# 便捷函数
from .core.factory import (
    create_chat_service,
    create_embedding_service,
    create_rerank_service
)

# 服务接口
from .services.chat_service import ChatService
from .services.embedding_service import EmbeddingService
from .services.rerank_service import RerankService

# 数据模型
from .services.models import (
    # 枚举
    MessageRole,
    
    # Chat相关
    ChatMessage,
    ChatResponse,
    ChatUsage,
    
    # Embedding相关
    EmbeddingVector,
    EmbeddingResponse,
    EmbeddingUsage,
    
    # Rerank相关
    RerankResult,
    RerankResponse,
    RerankUsage,
    
    # 便捷函数
    create_user_message,
    create_assistant_message,
    create_system_message,
    create_function_message,
    create_tool_message
)

# 配置管理
from .core.config_manager import (
    AIServiceConfigManager,
    ServiceConfig,
    AIServiceConfig
)

# 配置工具函数
from .config.config import (
    get_default_config,
    create_config_template,
    validate_config,
    load_config_from_env,
    merge_configs,
    get_provider_config,
    create_minimal_config
)

# 异常类
from .core.interfaces import (
    ServiceError,
    ConfigurationError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    InvalidRequestError
)

# 枚举类型
from .core.interfaces import (
    ServiceProvider,
    ServiceType
)

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    
    # 核心工厂
    "AIServiceFactory",
    
    # 便捷函数
    "create_chat_service",
    "create_embedding_service", 
    "create_rerank_service",
    
    # 服务接口
    "ChatService",
    "EmbeddingService",
    "RerankService",
    
    # 数据模型
    "MessageRole",
    "ChatMessage",
    "ChatResponse", 
    "ChatUsage",
    "EmbeddingVector",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "RerankResult",
    "RerankResponse",
    "RerankUsage",
    
    # 消息创建函数
    "create_user_message",
    "create_assistant_message",
    "create_system_message",
    "create_function_message",
    "create_tool_message",
    
    # 配置管理
    "AIServiceConfigManager",
    "ServiceConfig",
    "AIServiceConfig",
    
    # 配置工具函数
    "get_default_config",
    "create_config_template",
    "validate_config",
    "load_config_from_env",
    "merge_configs",
    "get_provider_config",
    "create_minimal_config",
    
    # 异常类
    "ServiceError",
    "ConfigurationError",
    "ConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "InvalidRequestError",
    
    # 枚举类型
    "ServiceProvider",
    "ServiceType"
]


def get_version():
    """获取版本信息"""
    return __version__


def get_info():
    """获取模块信息"""
    return {
        "name": "ai_services",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__,
        "supported_services": ["chat", "embedding", "rerank"],
        "supported_providers": ["ollama", "local", "cross_encoder"]
    }


# 模块级别的便捷函数
def quick_chat(message: str, model: str = None, provider: str = "ollama") -> str:
    """
    快速聊天函数
    
    Args:
        message: 用户消息
        model: 模型名称
        provider: 提供商
        
    Returns:
        AI回复内容
    """
    chat_service = create_chat_service(provider=provider, model=model)
    messages = [create_user_message(message)]
    response = chat_service.chat(messages)
    return response.message.content


async def quick_chat_async(message: str, model: str = None, provider: str = "ollama") -> str:
    """
    快速异步聊天函数
    
    Args:
        message: 用户消息
        model: 模型名称
        provider: 提供商
        
    Returns:
        AI回复内容
    """
    chat_service = create_chat_service(provider=provider, model=model)
    messages = [create_user_message(message)]
    response = await chat_service.chat_async(messages)
    return response.message.content


def quick_embed(text: str, model: str = None, provider: str = "ollama") -> list:
    """
    快速嵌入函数
    
    Args:
        text: 要嵌入的文本
        model: 模型名称
        provider: 提供商
        
    Returns:
        嵌入向量
    """
    embedding_service = create_embedding_service(provider=provider, model=model)
    response = embedding_service.embed(text)
    return response.vectors[0]


async def quick_embed_async(text: str, model: str = None, provider: str = "ollama") -> list:
    """
    快速异步嵌入函数
    
    Args:
        text: 要嵌入的文本
        model: 模型名称
        provider: 提供商
        
    Returns:
        嵌入向量
    """
    embedding_service = create_embedding_service(provider=provider, model=model)
    response = await embedding_service.embed(text)
    return response.vectors[0]


def quick_rerank(query: str, documents: list, model: str = None, provider: str = "embedding_based") -> list:
    """
    快速重排序函数
    
    Args:
        query: 查询文本
        documents: 文档列表
        model: 模型名称
        provider: 提供商
        
    Returns:
        排序后的文档索引和分数
    """
    rerank_service = create_rerank_service(provider=provider, model=model)
    response = rerank_service.rerank(query, documents)
    return [(result.index, result.score) for result in response.results]


async def quick_rerank_async(query: str, documents: list, model: str = None, provider: str = "embedding_based") -> list:
    """
    快速异步重排序函数
    
    Args:
        query: 查询文本
        documents: 文档列表
        model: 模型名称
        provider: 提供商
        
    Returns:
        排序后的文档索引和分数
    """
    rerank_service = create_rerank_service(provider=provider, model=model)
    response = await rerank_service.rerank(query, documents)
    return [(result.index, result.score) for result in response.results]


# 添加快捷函数到__all__
__all__.extend([
    "get_version",
    "get_info",
    "quick_chat",
    "quick_chat_async",
    "quick_embed",
    "quick_embed_async",
    "quick_rerank",
    "quick_rerank_async"
])