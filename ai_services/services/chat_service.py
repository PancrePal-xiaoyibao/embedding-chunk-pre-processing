"""
Chat Service Implementation - 聊天服务实现

提供各种聊天服务的具体实现，包括Ollama等。
"""

import json
import time
import logging
import requests
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from abc import ABC, abstractmethod

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import BaseService
from core.exceptions import (
    AIServiceError, ConnectionError, AuthenticationError, 
    RateLimitError, ModelNotFoundError, ServiceTimeoutError
)
from .models import ChatMessage, ChatResponse, ChatUsage, MessageRole


class ChatService(BaseService):
    """聊天服务抽象基类
    
    定义聊天服务的通用接口和行为。
    """
    
    @abstractmethod
    def chat(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """发送聊天消息
        
        Args:
            messages: 消息内容或消息列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            ChatResponse: 聊天响应
        """
        pass
    
    @abstractmethod
    async def chat_async(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """异步发送聊天消息
        
        Args:
            messages: 消息内容或消息列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            ChatResponse: 聊天响应
        """
        pass
    
    @abstractmethod
    def chat_stream(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """流式聊天
        
        Args:
            messages: 消息内容或消息列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Yields:
            ChatResponse: 流式聊天响应
        """
        pass
    
    def _normalize_messages(self, messages: Union[str, List[ChatMessage]]) -> List[ChatMessage]:
        """标准化消息格式
        
        Args:
            messages: 消息内容或消息列表
            
        Returns:
            List[ChatMessage]: 标准化的消息列表
        """
        if isinstance(messages, str):
            return [ChatMessage.user(messages)]
        elif isinstance(messages, list):
            result = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    result.append(msg)
                elif isinstance(msg, dict):
                    result.append(ChatMessage.from_dict(msg))
                else:
                    raise ValueError(f"不支持的消息格式: {type(msg)}")
            return result
        else:
            raise ValueError(f"不支持的消息格式: {type(messages)}")


class OllamaChatService(ChatService):
    """Ollama聊天服务实现
    
    提供基于Ollama的聊天功能。
    
    Args:
        config: 服务配置
        logger: 日志记录器
    """
    
    def __init__(self, provider: 'ServiceProvider', config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(provider, config, logger)
        
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "llama2")
        self.timeout = config.get("timeout", 30.0)
        self.max_retries = config.get("max_retries", 3)
        
        # 确保base_url格式正确
        if not self.base_url.startswith(("http://", "https://")):
            self.base_url = f"http://{self.base_url}"
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
        
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
        
        self.logger.info(f"初始化Ollama聊天服务: {self.base_url}")
    
    async def initialize(self) -> None:
        """初始化服务
        
        执行服务的初始化操作，如建立连接、验证配置等。
        
        Raises:
            AIServiceError: 初始化失败时抛出
        """
        try:
            # 测试连接
            if await self.test_connection():
                self._initialized = True
                self.logger.info("Ollama聊天服务初始化成功")
            else:
                raise AIServiceError("无法连接到Ollama服务")
        except Exception as e:
            raise AIServiceError(f"初始化Ollama聊天服务失败: {e}")
    
    async def test_connection(self) -> bool:
        """测试服务连接
        
        测试与服务提供商的连接是否正常。
        
        Returns:
            bool: 连接是否成功
        """
        try:
            import asyncio
            import concurrent.futures
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    lambda: requests.get(f"{self.base_url}/api/tags", timeout=5)
                )
                return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Ollama连接测试失败: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        检查服务的健康状态。
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            is_connected = await self.test_connection()
            models = await self.get_models() if is_connected else []
            
            return {
                "status": "healthy" if is_connected else "unhealthy",
                "connected": is_connected,
                "base_url": self.base_url,
                "model_count": len(models),
                "available_models": models[:5],  # 只返回前5个模型
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def get_models(self) -> List[str]:
        """获取可用模型列表
        
        Returns:
            List[str]: 可用模型名称列表
        """
        try:
            import asyncio
            import concurrent.futures
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    lambda: requests.get(f"{self.base_url}/api/tags", timeout=10)
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    return models
                else:
                    self.logger.error(f"获取模型列表失败: {response.status_code}")
                    return []
        except Exception as e:
            self.logger.error(f"获取模型列表异常: {e}")
            return []
    
    async def close(self) -> None:
        """关闭服务
        
        清理资源，关闭连接。
        """
        self._initialized = False
        self.logger.info("Ollama聊天服务已关闭")
    
    def health_check_sync(self) -> bool:
        """健康检查
        
        Returns:
            bool: 服务是否健康
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Ollama健康检查失败: {e}")
            return False
    
    def chat(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """发送聊天消息
        
        Args:
            messages: 消息内容或消息列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            ChatResponse: 聊天响应
            
        Raises:
            ConnectionError: 连接失败时抛出
            ModelNotFoundError: 模型不存在时抛出
            ServiceTimeoutError: 服务超时时抛出
        """
        start_time = time.time()
        
        # 标准化消息
        normalized_messages = self._normalize_messages(messages)
        
        # 准备请求数据
        model_name = model or self.model_name
        request_data = {
            "model": model_name,
            "messages": [msg.to_dict() for msg in normalized_messages],
            "stream": False
        }
        
        # 添加额外参数
        for key, value in kwargs.items():
            if key not in request_data:
                request_data[key] = value
        
        # 发送请求
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.chat_url,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_time = time.time() - start_time
                    
                    return self._parse_chat_response(response_data, response_time)
                
                elif response.status_code == 404:
                    raise ModelNotFoundError(f"模型不存在: {model_name}")
                
                elif response.status_code == 429:
                    raise RateLimitError("请求频率限制")
                
                else:
                    error_msg = f"Ollama API错误: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', '')}"
                    except:
                        pass
                    
                    if attempt == self.max_retries - 1:
                        raise ConnectionError(error_msg)
                    
                    self.logger.warning(f"请求失败，重试 {attempt + 1}/{self.max_retries}: {error_msg}")
                    time.sleep(2 ** attempt)  # 指数退避
            
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise ServiceTimeoutError(f"请求超时: {self.timeout}秒")
                self.logger.warning(f"请求超时，重试 {attempt + 1}/{self.max_retries}")
                time.sleep(2 ** attempt)
            
            except requests.exceptions.ConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"连接Ollama失败: {e}")
                self.logger.warning(f"连接失败，重试 {attempt + 1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt)
        
        raise ConnectionError("达到最大重试次数")
    
    async def chat_async(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """异步发送聊天消息
        
        Args:
            messages: 消息内容或消息列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            ChatResponse: 聊天响应
        """
        # 简单实现：在线程池中运行同步版本
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                lambda: self.chat(messages, model, **kwargs)
            )
    
    def chat_stream(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """流式聊天
        
        Args:
            messages: 消息内容或消息列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Yields:
            ChatResponse: 流式聊天响应
        """
        # 标准化消息
        normalized_messages = self._normalize_messages(messages)
        
        # 准备请求数据
        model_name = model or self.model_name
        request_data = {
            "model": model_name,
            "messages": [msg.to_dict() for msg in normalized_messages],
            "stream": True
        }
        
        # 添加额外参数
        for key, value in kwargs.items():
            if key not in request_data:
                request_data[key] = value
        
        try:
            response = requests.post(
                self.chat_url,
                json=request_data,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code != 200:
                raise ConnectionError(f"Ollama API错误: {response.status_code}")
            
            accumulated_content = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        
                        if 'message' in chunk_data:
                            content = chunk_data['message'].get('content', '')
                            accumulated_content += content
                            
                            # 创建增量响应
                            message = ChatMessage.assistant(accumulated_content)
                            chat_response = ChatResponse(
                                message=message,
                                model=model_name,
                                finish_reason=chunk_data.get('done_reason'),
                                metadata={"is_stream": True, "chunk": chunk_data}
                            )
                            
                            yield chat_response
                            
                            # 如果完成，跳出循环
                            if chunk_data.get('done', False):
                                break
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            raise ConnectionError(f"流式聊天失败: {e}")
    
    def _parse_chat_response(self, response_data: Dict[str, Any], response_time: float) -> ChatResponse:
        """解析聊天响应
        
        Args:
            response_data: 响应数据
            response_time: 响应时间
            
        Returns:
            ChatResponse: 解析后的响应
        """
        # 解析消息
        message_data = response_data.get('message', {})
        message = ChatMessage(
            role=MessageRole(message_data.get('role', 'assistant')),
            content=message_data.get('content', '')
        )
        
        # 解析使用统计
        usage = None
        if 'prompt_eval_count' in response_data or 'eval_count' in response_data:
            usage = ChatUsage(
                prompt_tokens=response_data.get('prompt_eval_count', 0),
                completion_tokens=response_data.get('eval_count', 0),
                total_tokens=response_data.get('prompt_eval_count', 0) + response_data.get('eval_count', 0)
            )
        
        return ChatResponse(
            message=message,
            model=response_data.get('model'),
            usage=usage,
            finish_reason=response_data.get('done_reason'),
            response_time=response_time,
            metadata={
                "eval_duration": response_data.get('eval_duration'),
                "load_duration": response_data.get('load_duration'),
                "prompt_eval_duration": response_data.get('prompt_eval_duration'),
                "total_duration": response_data.get('total_duration')
            }
        )
    
    def get_available_models_sync(self) -> List[str]:
        """获取可用模型列表（同步版本）
        
        Returns:
            List[str]: 可用模型列表
            
        Raises:
            ConnectionError: 连接失败时抛出
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            else:
                raise ConnectionError(f"获取模型列表失败: {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"连接Ollama失败: {e}")
    
    def pull_model(self, model_name: str) -> bool:
        """拉取模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否成功
            
        Raises:
            ConnectionError: 连接失败时抛出
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 拉取模型可能需要较长时间
            )
            
            return response.status_code == 200
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"拉取模型失败: {e}")