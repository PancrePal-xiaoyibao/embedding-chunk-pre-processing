"""
Chat接口模块

提供统一的聊天模型接口，支持多种聊天模型提供商。
参考Go版本的设计模式，实现标准化的聊天接口。

Author: Assistant
Date: 2025-01-24
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum
import asyncio


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """聊天消息数据类
    
    Attributes:
        role: 消息角色（system/user/assistant）
        content: 消息内容
        metadata: 可选的元数据
    """
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的消息
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata or {}
        }


@dataclass
class ChatOptions:
    """聊天选项配置
    
    Attributes:
        temperature: 温度参数，控制输出随机性 (0.0-2.0)
        top_p: Top P 参数，核采样参数 (0.0-1.0)
        max_tokens: 最大生成token数
        max_completion_tokens: 最大完成token数
        frequency_penalty: 频率惩罚 (-2.0-2.0)
        presence_penalty: 存在惩罚 (-2.0-2.0)
        seed: 随机种子，用于可重现的输出
        thinking: 是否启用思考模式
        stream: 是否使用流式输出
    """
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    thinking: Optional[bool] = None
    stream: bool = False


@dataclass
class ChatResponse:
    """聊天响应数据类
    
    Attributes:
        message: 响应消息
        model_name: 使用的模型名称
        usage: token使用情况
        finish_reason: 完成原因
        metadata: 响应元数据
    """
    message: ChatMessage
    model_name: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StreamResponse:
    """流式响应数据类
    
    Attributes:
        delta: 增量内容
        finish_reason: 完成原因
        usage: token使用情况（仅在最后一个chunk中）
        metadata: 响应元数据
    """
    delta: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatInterface(ABC):
    """聊天接口抽象基类
    
    定义了所有聊天模型必须实现的标准接口。
    参考Go版本的Chat接口设计。
    """
    
    @abstractmethod
    def chat(self, messages: List[ChatMessage], options: Optional[ChatOptions] = None) -> ChatResponse:
        """进行非流式聊天
        
        Args:
            messages: 聊天消息列表
            options: 聊天选项配置
            
        Returns:
            ChatResponse: 聊天响应
            
        Raises:
            ChatError: 聊天过程中的错误
        """
        pass
    
    @abstractmethod
    def chat_stream(self, messages: List[ChatMessage], options: Optional[ChatOptions] = None) -> AsyncGenerator[StreamResponse, None]:
        """进行流式聊天
        
        Args:
            messages: 聊天消息列表
            options: 聊天选项配置
            
        Yields:
            StreamResponse: 流式响应数据
            
        Raises:
            ChatError: 聊天过程中的错误
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称
        
        Returns:
            str: 模型名称
        """
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """获取模型ID
        
        Returns:
            str: 模型ID
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """测试连接
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """检查模型是否可用
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 模型是否可用
        """
        pass


class ChatError(Exception):
    """聊天相关错误"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """初始化聊天错误
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
        """
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ChatConnectionError(ChatError):
    """聊天连接错误"""
    pass


class ChatModelError(ChatError):
    """聊天模型错误"""
    pass


class ChatConfigError(ChatError):
    """聊天配置错误"""
    pass


# 工具函数
def create_message(role: Union[str, MessageRole], content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """创建聊天消息的便捷函数
    
    Args:
        role: 消息角色
        content: 消息内容
        metadata: 可选的元数据
        
    Returns:
        ChatMessage: 聊天消息对象
    """
    if isinstance(role, str):
        role = MessageRole(role)
    return ChatMessage(role=role, content=content, metadata=metadata)


def create_system_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """创建系统消息
    
    Args:
        content: 消息内容
        metadata: 可选的元数据
        
    Returns:
        ChatMessage: 系统消息对象
    """
    return create_message(MessageRole.SYSTEM, content, metadata)


def create_user_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """创建用户消息
    
    Args:
        content: 消息内容
        metadata: 可选的元数据
        
    Returns:
        ChatMessage: 用户消息对象
    """
    return create_message(MessageRole.USER, content, metadata)


def create_assistant_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """创建助手消息
    
    Args:
        content: 消息内容
        metadata: 可选的元数据
        
    Returns:
        ChatMessage: 助手消息对象
    """
    return create_message(MessageRole.ASSISTANT, content, metadata)