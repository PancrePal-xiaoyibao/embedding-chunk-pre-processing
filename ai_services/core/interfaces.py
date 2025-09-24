"""
AI Services Interfaces - 基础接口定义

定义AI服务模块中使用的基础接口和抽象类。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import asyncio


class ServiceProvider(Enum):
    """服务提供商枚举
    
    定义支持的AI服务提供商。
    """
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


class ServiceType(Enum):
    """服务类型枚举
    
    定义支持的AI服务类型。
    """
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class BaseService(ABC):
    """基础服务抽象类
    
    所有AI服务的基类，定义了通用的接口和行为。
    
    Args:
        provider: 服务提供商
        config: 服务配置
        logger: 日志记录器
    """
    
    def __init__(
        self, 
        provider: ServiceProvider,
        config: Dict[str, Any],
        logger: Optional[Any] = None
    ):
        self.provider = provider
        self.config = config
        self.logger = logger
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化服务
        
        执行服务的初始化操作，如建立连接、验证配置等。
        
        Raises:
            AIServiceError: 初始化失败时抛出
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """测试服务连接
        
        测试与服务提供商的连接是否正常。
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        检查服务的健康状态。
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        pass
    
    @abstractmethod
    async def get_models(self) -> List[str]:
        """获取可用模型列表
        
        Returns:
            List[str]: 可用模型名称列表
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭服务
        
        清理资源，关闭连接。
        """
        pass
    
    def is_initialized(self) -> bool:
        """检查服务是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        return self._initialized
    
    def get_provider(self) -> ServiceProvider:
        """获取服务提供商
        
        Returns:
            ServiceProvider: 服务提供商
        """
        return self.provider
    
    def get_config(self) -> Dict[str, Any]:
        """获取服务配置
        
        Returns:
            Dict[str, Any]: 服务配置
        """
        return self.config.copy()
    
    async def __aenter__(self):
        """异步上下文管理器入口
        
        Returns:
            BaseService: 服务实例
        """
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口
        
        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常追踪
        """
        await self.close()


class ServiceFactory(ABC):
    """服务工厂抽象类
    
    定义创建和管理AI服务的工厂接口。
    """
    
    @abstractmethod
    def create_service(
        self, 
        service_type: ServiceType,
        provider: ServiceProvider,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseService:
        """创建服务实例
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            config: 服务配置
            
        Returns:
            BaseService: 服务实例
            
        Raises:
            ServiceNotAvailableError: 服务不可用时抛出
            ConfigurationError: 配置错误时抛出
        """
        pass
    
    @abstractmethod
    def get_available_providers(self, service_type: ServiceType) -> List[ServiceProvider]:
        """获取可用的服务提供商
        
        Args:
            service_type: 服务类型
            
        Returns:
            List[ServiceProvider]: 可用的服务提供商列表
        """
        pass
    
    @abstractmethod
    def is_provider_available(self, service_type: ServiceType, provider: ServiceProvider) -> bool:
        """检查服务提供商是否可用
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            
        Returns:
            bool: 是否可用
        """
        pass


class ConfigManager(ABC):
    """配置管理器抽象类
    
    定义配置管理的接口。
    """
    
    @abstractmethod
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置数据
            
        Raises:
            ConfigurationError: 配置加载失败时抛出
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置
        
        Args:
            config: 配置数据
            
        Returns:
            bool: 配置是否有效
            
        Raises:
            ValidationError: 配置验证失败时抛出
        """
        pass
    
    @abstractmethod
    def get_service_config(
        self, 
        service_type: ServiceType, 
        provider: ServiceProvider
    ) -> Dict[str, Any]:
        """获取特定服务的配置
        
        Args:
            service_type: 服务类型
            provider: 服务提供商
            
        Returns:
            Dict[str, Any]: 服务配置
        """
        pass


class AsyncContextManager:
    """异步上下文管理器辅助类
    
    为不支持异步上下文管理器的对象提供支持。
    """
    
    def __init__(self, obj: Any, init_method: str = "initialize", close_method: str = "close"):
        self.obj = obj
        self.init_method = init_method
        self.close_method = close_method
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if hasattr(self.obj, self.init_method):
            init_func = getattr(self.obj, self.init_method)
            if asyncio.iscoroutinefunction(init_func):
                await init_func()
            else:
                init_func()
        return self.obj
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if hasattr(self.obj, self.close_method):
            close_func = getattr(self.obj, self.close_method)
            if asyncio.iscoroutinefunction(close_func):
                await close_func()
            else:
                close_func()


def ensure_async_context(obj: Any) -> AsyncContextManager:
    """确保对象支持异步上下文管理器
    
    Args:
        obj: 要包装的对象
        
    Returns:
        AsyncContextManager: 异步上下文管理器
    """
    if hasattr(obj, "__aenter__") and hasattr(obj, "__aexit__"):
        return obj
    return AsyncContextManager(obj)