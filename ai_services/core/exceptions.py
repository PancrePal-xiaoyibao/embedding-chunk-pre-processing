"""
AI Services Exceptions - 异常定义

定义AI服务模块中使用的各种异常类型。
"""

from typing import Optional, Any, Dict


class AIServiceError(Exception):
    """AI服务基础异常类
    
    所有AI服务相关异常的基类。
    
    Args:
        message: 错误消息
        error_code: 错误代码
        details: 错误详细信息
        original_error: 原始异常对象
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        """返回异常的字符串表示
        
        Returns:
            str: 格式化的错误消息
        """
        parts = [self.message]
        if self.error_code:
            parts.append(f"错误代码: {self.error_code}")
        if self.details:
            parts.append(f"详细信息: {self.details}")
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典格式
        
        Returns:
            Dict[str, Any]: 异常信息字典
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None
        }


class ConfigurationError(AIServiceError):
    """配置错误异常
    
    当配置文件无效、缺失或格式错误时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        config_path: Optional[str] = None,
        validation_errors: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_path = config_path
        self.validation_errors = validation_errors or []
        if config_path:
            self.details["config_path"] = config_path
        if validation_errors:
            self.details["validation_errors"] = validation_errors


class ServiceNotAvailableError(AIServiceError):
    """服务不可用异常
    
    当请求的服务提供商不可用或未配置时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        service_type: Optional[str] = None,
        provider: Optional[str] = None,
        available_providers: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, error_code="SERVICE_UNAVAILABLE", **kwargs)
        self.service_type = service_type
        self.provider = provider
        self.available_providers = available_providers or []
        
        if service_type:
            self.details["service_type"] = service_type
        if provider:
            self.details["provider"] = provider
        if available_providers:
            self.details["available_providers"] = available_providers


class ConnectionError(AIServiceError):
    """连接错误异常
    
    当无法连接到服务提供商时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, error_code="CONNECTION_ERROR", **kwargs)
        self.provider = provider
        self.endpoint = endpoint
        self.timeout = timeout
        
        if provider:
            self.details["provider"] = provider
        if endpoint:
            self.details["endpoint"] = endpoint
        if timeout:
            self.details["timeout"] = timeout


class AuthenticationError(AIServiceError):
    """认证错误异常
    
    当API密钥无效或认证失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="AUTH_ERROR", **kwargs)
        self.provider = provider
        if provider:
            self.details["provider"] = provider


class RateLimitError(AIServiceError):
    """速率限制错误异常
    
    当达到API调用速率限制时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, error_code="RATE_LIMIT", **kwargs)
        self.provider = provider
        self.retry_after = retry_after
        
        if provider:
            self.details["provider"] = provider
        if retry_after:
            self.details["retry_after"] = retry_after


class ValidationError(AIServiceError):
    """验证错误异常
    
    当输入参数验证失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value
        self.expected_type = expected_type
        
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)
        if expected_type:
            self.details["expected_type"] = expected_type


class ModelNotFoundError(AIServiceError):
    """模型未找到异常
    
    当请求的模型不存在或不可用时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        available_models: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, error_code="MODEL_NOT_FOUND", **kwargs)
        self.model_name = model_name
        self.provider = provider
        self.available_models = available_models or []
        
        if model_name:
            self.details["model_name"] = model_name
        if provider:
            self.details["provider"] = provider
        if available_models:
            self.details["available_models"] = available_models


class ServiceTimeoutError(AIServiceError):
    """服务超时异常
    
    当服务调用超时时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="TIMEOUT", **kwargs)
        self.timeout = timeout
        self.operation = operation
        
        if timeout:
            self.details["timeout"] = timeout
        if operation:
            self.details["operation"] = operation