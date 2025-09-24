"""
AI Services Models - 数据模型定义

定义各种AI服务使用的数据模型和响应格式。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """聊天消息
    
    Args:
        role: 消息角色
        content: 消息内容
        name: 发送者名称（可选）
        function_call: 函数调用信息（可选）
        metadata: 额外元数据
    """
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的消息
        """
        result = {
            "role": self.role.value,
            "content": self.content
        }
        
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """从字典创建消息
        
        Args:
            data: 字典数据
            
        Returns:
            ChatMessage: 消息实例
        """
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            name=data.get("name"),
            function_call=data.get("function_call"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def system(cls, content: str, **kwargs) -> 'ChatMessage':
        """创建系统消息
        
        Args:
            content: 消息内容
            **kwargs: 额外参数
            
        Returns:
            ChatMessage: 系统消息
        """
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)
    
    @classmethod
    def user(cls, content: str, **kwargs) -> 'ChatMessage':
        """创建用户消息
        
        Args:
            content: 消息内容
            **kwargs: 额外参数
            
        Returns:
            ChatMessage: 用户消息
        """
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def assistant(cls, content: str, **kwargs) -> 'ChatMessage':
        """创建助手消息
        
        Args:
            content: 消息内容
            **kwargs: 额外参数
            
        Returns:
            ChatMessage: 助手消息
        """
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)


@dataclass
class ChatUsage:
    """聊天使用统计
    
    Args:
        prompt_tokens: 提示词token数
        completion_tokens: 完成token数
        total_tokens: 总token数
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典格式
        
        Returns:
            Dict[str, int]: 字典格式的使用统计
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class ChatResponse:
    """聊天响应
    
    Args:
        message: 响应消息
        model: 使用的模型
        usage: 使用统计
        finish_reason: 完成原因
        response_time: 响应时间（秒）
        metadata: 额外元数据
    """
    message: ChatMessage
    model: Optional[str] = None
    usage: Optional[ChatUsage] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的响应
        """
        result = {
            "message": self.message.to_dict()
        }
        
        if self.model:
            result["model"] = self.model
        if self.usage:
            result["usage"] = self.usage.to_dict()
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        if self.response_time:
            result["response_time"] = self.response_time
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


@dataclass
class EmbeddingVector:
    """嵌入向量
    
    Args:
        vector: 向量数据
        index: 向量索引
        metadata: 额外元数据
    """
    vector: List[float]
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的向量
        """
        return {
            "vector": self.vector,
            "index": self.index,
            "metadata": self.metadata
        }
    
    @property
    def dimension(self) -> int:
        """获取向量维度
        
        Returns:
            int: 向量维度
        """
        return len(self.vector)


@dataclass
class EmbeddingUsage:
    """嵌入使用统计
    
    Args:
        prompt_tokens: 提示词token数
        total_tokens: 总token数
    """
    prompt_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典格式
        
        Returns:
            Dict[str, int]: 字典格式的使用统计
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class EmbeddingResponse:
    """嵌入响应
    
    Args:
        embeddings: 嵌入向量列表
        model: 使用的模型
        usage: 使用统计
        response_time: 响应时间（秒）
        metadata: 额外元数据
    """
    embeddings: List[EmbeddingVector]
    model: Optional[str] = None
    usage: Optional[EmbeddingUsage] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的响应
        """
        result = {
            "embeddings": [emb.to_dict() for emb in self.embeddings]
        }
        
        if self.model:
            result["model"] = self.model
        if self.usage:
            result["usage"] = self.usage.to_dict()
        if self.response_time:
            result["response_time"] = self.response_time
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


@dataclass
class RerankResult:
    """重排序结果
    
    Args:
        index: 原始文档索引
        score: 相关性分数
        document: 文档内容
        metadata: 额外元数据
    """
    index: int
    score: float
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的结果
        """
        return {
            "index": self.index,
            "score": self.score,
            "document": self.document,
            "metadata": self.metadata
        }


@dataclass
class RerankUsage:
    """重排序使用统计
    
    Args:
        query_tokens: 查询token数
        document_tokens: 文档token数
        total_tokens: 总token数
    """
    query_tokens: int = 0
    document_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典格式
        
        Returns:
            Dict[str, int]: 字典格式的使用统计
        """
        return {
            "query_tokens": self.query_tokens,
            "document_tokens": self.document_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class RerankResponse:
    """重排序响应
    
    Args:
        results: 重排序结果列表
        model: 使用的模型
        usage: 使用统计
        response_time: 响应时间（秒）
        metadata: 额外元数据
    """
    results: List[RerankResult]
    model: Optional[str] = None
    usage: Optional[RerankUsage] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的响应
        """
        result = {
            "results": [res.to_dict() for res in self.results]
        }
        
        if self.model:
            result["model"] = self.model
        if self.usage:
            result["usage"] = self.usage.to_dict()
        if self.response_time:
            result["response_time"] = self.response_time
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


# 便捷函数
def create_chat_message(role: str, content: str, **kwargs) -> ChatMessage:
    """创建聊天消息
    
    Args:
        role: 消息角色
        content: 消息内容
        **kwargs: 额外参数
        
    Returns:
        ChatMessage: 聊天消息
    """
    return ChatMessage(role=MessageRole(role), content=content, **kwargs)


def create_system_message(content: str, **kwargs) -> ChatMessage:
    """创建系统消息
    
    Args:
        content: 消息内容
        **kwargs: 额外参数
        
    Returns:
        ChatMessage: 系统消息
    """
    return ChatMessage.system(content, **kwargs)


def create_user_message(content: str, **kwargs) -> ChatMessage:
    """创建用户消息
    
    Args:
        content: 消息内容
        **kwargs: 额外参数
        
    Returns:
        ChatMessage: 用户消息
    """
    return ChatMessage.user(content, **kwargs)


def create_assistant_message(content: str, **kwargs) -> ChatMessage:
    """创建助手消息
    
    Args:
        content: 消息内容
        **kwargs: 额外参数
        
    Returns:
        ChatMessage: 助手消息
    """
    return ChatMessage.assistant(content, **kwargs)