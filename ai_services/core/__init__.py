"""AI Services - 核心模块

这个包包含了AI服务模块的核心组件：
- 服务工厂：统一的服务创建和管理
- 配置管理：灵活的配置加载和验证
- 接口定义：所有服务的抽象接口
- 异常处理：统一的错误处理机制

这些核心组件为整个AI服务模块提供了基础架构。
"""

from .factory import AIServiceFactory, create_chat_service, create_embedding_service, create_rerank_service
from .config_manager import AIServiceConfigManager, ServiceConfig, AIServiceConfig
from .interfaces import (
    # 枚举类型
    ServiceProvider,
    ServiceType
)
from .exceptions import (
    # 异常类
    AIServiceError as ServiceError,
    ConfigurationError,
    ServiceNotAvailableError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    ValidationError as InvalidRequestError,
    ModelNotFoundError,
    ServiceTimeoutError
)

__all__ = [
    # 工厂类
    "AIServiceFactory",
    
    # 便捷函数
    "create_chat_service",
    "create_embedding_service",
    "create_rerank_service",
    
    # 配置管理
    "AIServiceConfigManager",
    "ServiceConfig",
    "AIServiceConfig",
    
    # 异常类
    "ServiceError",
    "ConfigurationError",
    "ServiceNotAvailableError",
    "ConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "ModelNotFoundError",
    "ServiceTimeoutError",
    
    # 枚举类型
    "ServiceProvider",
    "ServiceType"
]