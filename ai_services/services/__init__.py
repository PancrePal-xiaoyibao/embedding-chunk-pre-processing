"""AI Services - 服务模块

这个包包含了所有AI服务的具体实现，包括：
- Chat服务：聊天对话功能
- Embedding服务：文本嵌入功能  
- Rerank服务：文档重排序功能
- 数据模型：所有服务使用的数据结构

每个服务都提供了统一的接口和多种实现方式。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import (
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
    create_system_message
)

from .chat_service import ChatService
from .embedding_service import EmbeddingService
from .rerank_service import RerankService

__all__ = [
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
    
    # 便捷函数
    "create_user_message",
    "create_system_message",
    "create_assistant_message",
    
    # 服务接口
    "ChatService",
    "EmbeddingService",
    "RerankService"
]