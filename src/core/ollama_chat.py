"""
Ollama聊天客户端模块

实现基于Ollama的聊天功能，支持流式和非流式聊天。
参考Go版本的OllamaChat实现。

Author: Assistant
Date: 2025-01-24
"""

import json
import asyncio
import aiohttp
import requests
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urljoin

from .chat_interface import (
    ChatInterface, ChatMessage, ChatOptions, ChatResponse, StreamResponse,
    ChatError, ChatConnectionError, ChatModelError, ChatConfigError,
    MessageRole, create_assistant_message
)
from ..config.config_manager import ConfigManager


@dataclass
class OllamaChatConfig:
    """Ollama聊天配置
    
    Attributes:
        base_url: Ollama服务器地址
        model_name: 模型名称
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        verify_ssl: 是否验证SSL证书
        temperature: 温度参数
        max_tokens: 最大token数
        top_p: top_p参数
        top_k: top_k参数
        stream: 是否启用流式输出
        keep_alive: 模型保持活跃时间
        system_prompt: 系统提示词
        context: 上下文配置
    """
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen3:1.7b"
    timeout: int = 300
    max_retries: int = 3
    verify_ssl: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    stream: bool = True
    keep_alive: str = "5m"
    system_prompt: str = "你是一个专业的AI助手，请根据用户的问题提供准确、有用的回答。"
    context: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_config_manager(cls, config_manager: Optional[ConfigManager] = None) -> 'OllamaChatConfig':
        """从配置管理器创建Ollama聊天配置
        
        Args:
            config_manager: 配置管理器实例，如果为None则创建新实例
            
        Returns:
            OllamaChatConfig: Ollama聊天配置实例
        """
        if config_manager is None:
            config_manager = ConfigManager()
        
        try:
            chat_config = config_manager.get_chat_config()
            
            # 获取Ollama特定配置
            ollama_config = chat_config.ollama or {}
            
            return cls(
                base_url=ollama_config.get("base_url", "http://localhost:11434"),
                model_name=ollama_config.get("model_name", "qwen3:1.7b"),
                timeout=ollama_config.get("timeout", 60),
                max_retries=ollama_config.get("max_retries", 3),
                verify_ssl=True,  # 默认启用SSL验证
                temperature=ollama_config.get("temperature", 0.7),
                max_tokens=ollama_config.get("max_tokens", 2048),
                top_p=ollama_config.get("top_p", 0.9),
                top_k=ollama_config.get("top_k", 40),
                stream=ollama_config.get("stream", True),
                keep_alive=ollama_config.get("keep_alive", "5m"),
                system_prompt=ollama_config.get("system_prompt", "你是一个专业的AI助手，请根据用户的问题提供准确、有用的回答。"),
                context=ollama_config.get("context", {})
            )
        except Exception as e:
            # 如果配置读取失败，使用默认配置
            logging.getLogger(__name__).warning(f"读取Chat配置失败，使用默认配置: {str(e)}")
            return cls()


class OllamaChatClient(ChatInterface):
    """Ollama聊天客户端
    
    实现基于Ollama的聊天功能，支持流式和非流式对话。
    """
    
    def __init__(self, config: Optional[OllamaChatConfig] = None, config_manager: Optional[ConfigManager] = None):
        """初始化Ollama聊天客户端
        
        Args:
            config: Ollama聊天配置，如果为None则从配置管理器读取
            config_manager: 配置管理器实例，用于读取配置
        """
        if config is None:
            self.config = OllamaChatConfig.from_config_manager(config_manager)
        else:
            self.config = config
            
        self.logger = logging.getLogger(__name__)
        
        # 确保base_url以/结尾
        if not self.config.base_url.endswith('/'):
            self.config.base_url += '/'
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """准备消息格式
        
        Args:
            messages: 聊天消息列表
            
        Returns:
            List[Dict[str, str]]: Ollama格式的消息列表
        """
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        return ollama_messages
    
    def _prepare_options(self, options: Optional[ChatOptions] = None) -> Dict[str, Any]:
        """准备选项格式
        
        Args:
            options: 聊天选项
            
        Returns:
            Dict[str, Any]: Ollama格式的选项
        """
        if not options:
            return {}
        
        ollama_options = {}
        
        # 映射选项到Ollama格式
        if options.temperature is not None:
            ollama_options["temperature"] = options.temperature
        if options.top_p is not None:
            ollama_options["top_p"] = options.top_p
        if options.seed is not None:
            ollama_options["seed"] = options.seed
        if options.max_tokens is not None:
            ollama_options["num_predict"] = options.max_tokens
        if options.frequency_penalty is not None:
            ollama_options["frequency_penalty"] = options.frequency_penalty
        if options.presence_penalty is not None:
            ollama_options["presence_penalty"] = options.presence_penalty
            
        return ollama_options
    
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
        try:
            url = urljoin(self.config.base_url, "api/chat")
            
            payload = {
                "model": self.config.model_name,
                "messages": self._prepare_messages(messages),
                "stream": False
            }
            
            # 添加选项
            ollama_options = self._prepare_options(options)
            if ollama_options:
                payload["options"] = ollama_options
            
            self.logger.debug(f"发送聊天请求到: {url}")
            self.logger.debug(f"请求负载: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama聊天请求失败: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise ChatConnectionError(error_msg, error_code=str(response.status_code))
            
            result = response.json()
            self.logger.debug(f"收到响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 解析响应
            if "message" not in result:
                raise ChatModelError("响应中缺少message字段")
            
            message_data = result["message"]
            assistant_message = create_assistant_message(
                content=message_data.get("content", ""),
                metadata={"raw_response": result}
            )
            
            # 构建响应
            chat_response = ChatResponse(
                message=assistant_message,
                model_name=self.config.model_name,
                usage=self._extract_usage(result),
                finish_reason=result.get("done_reason"),
                metadata={"raw_response": result}
            )
            
            return chat_response
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama聊天请求异常: {str(e)}"
            self.logger.error(error_msg)
            raise ChatConnectionError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"解析Ollama响应JSON失败: {str(e)}"
            self.logger.error(error_msg)
            raise ChatModelError(error_msg) from e
        except Exception as e:
            error_msg = f"聊天过程中发生未知错误: {str(e)}"
            self.logger.error(error_msg)
            raise ChatError(error_msg) from e
    
    async def chat_stream(self, messages: List[ChatMessage], options: Optional[ChatOptions] = None) -> AsyncGenerator[StreamResponse, None]:
        """进行流式聊天
        
        Args:
            messages: 聊天消息列表
            options: 聊天选项配置
            
        Yields:
            StreamResponse: 流式响应数据
            
        Raises:
            ChatError: 聊天过程中的错误
        """
        try:
            url = urljoin(self.config.base_url, "api/chat")
            
            payload = {
                "model": self.config.model_name,
                "messages": self._prepare_messages(messages),
                "stream": True
            }
            
            # 添加选项
            ollama_options = self._prepare_options(options)
            if ollama_options:
                payload["options"] = ollama_options
            
            self.logger.debug(f"发送流式聊天请求到: {url}")
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    json=payload,
                    ssl=self.config.verify_ssl
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Ollama流式聊天请求失败: {response.status} - {error_text}"
                        self.logger.error(error_msg)
                        raise ChatConnectionError(error_msg, error_code=str(response.status))
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                        
                        try:
                            chunk = json.loads(line)
                            
                            # 提取增量内容
                            delta = ""
                            if "message" in chunk and "content" in chunk["message"]:
                                delta = chunk["message"]["content"]
                            
                            # 检查是否完成
                            finish_reason = None
                            usage = None
                            if chunk.get("done", False):
                                finish_reason = chunk.get("done_reason", "stop")
                                usage = self._extract_usage(chunk)
                            
                            yield StreamResponse(
                                delta=delta,
                                finish_reason=finish_reason,
                                usage=usage,
                                metadata={"raw_chunk": chunk}
                            )
                            
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"跳过无效的JSON行: {line}")
                            continue
                            
        except aiohttp.ClientError as e:
            error_msg = f"Ollama流式聊天请求异常: {str(e)}"
            self.logger.error(error_msg)
            raise ChatConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"流式聊天过程中发生未知错误: {str(e)}"
            self.logger.error(error_msg)
            raise ChatError(error_msg) from e
    
    def _extract_usage(self, response: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """提取token使用情况
        
        Args:
            response: Ollama响应
            
        Returns:
            Optional[Dict[str, int]]: token使用情况
        """
        usage = {}
        
        if "prompt_eval_count" in response:
            usage["prompt_tokens"] = response["prompt_eval_count"]
        if "eval_count" in response:
            usage["completion_tokens"] = response["eval_count"]
        
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            return usage
        
        return None
    
    def get_model_name(self) -> str:
        """获取模型名称
        
        Returns:
            str: 模型名称
        """
        return self.config.model_name
    
    def get_model_id(self) -> str:
        """获取模型ID
        
        Returns:
            str: 模型ID
        """
        return f"ollama:{self.config.model_name}"
    
    def test_connection(self) -> bool:
        """测试连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            url = urljoin(self.config.base_url, "api/tags")
            response = requests.get(url, timeout=10, verify=self.config.verify_ssl)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"测试Ollama连接失败: {str(e)}")
            return False
    
    def is_model_available(self, model_name: str) -> bool:
        """检查模型是否可用
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 模型是否可用
        """
        try:
            url = urljoin(self.config.base_url, "api/tags")
            response = requests.get(url, timeout=10, verify=self.config.verify_ssl)
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            models = data.get("models", [])
            
            for model in models:
                if model.get("name", "").startswith(model_name):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"检查模型可用性失败: {str(e)}")
            return False
    
    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """拉取模型
        
        Args:
            model_name: 模型名称，如果为None则使用配置中的模型
            
        Returns:
            bool: 拉取是否成功
        """
        model_to_pull = model_name or self.config.model_name
        
        try:
            url = urljoin(self.config.base_url, "api/pull")
            payload = {"name": model_to_pull}
            
            self.logger.info(f"开始拉取模型: {model_to_pull}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=600,  # 拉取模型可能需要较长时间
                verify=self.config.verify_ssl
            )
            
            if response.status_code == 200:
                self.logger.info(f"模型 {model_to_pull} 拉取成功")
                return True
            else:
                self.logger.error(f"模型拉取失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"拉取模型时发生错误: {str(e)}")
            return False


# 工厂函数
def create_ollama_chat_client(
    config_manager: Optional[ConfigManager] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[int] = None,
    **kwargs
) -> OllamaChatClient:
    """创建Ollama聊天客户端的便捷函数
    
    Args:
        config_manager: 配置管理器实例，优先使用配置文件
        base_url: Ollama服务器地址，覆盖配置文件设置
        model_name: 模型名称，覆盖配置文件设置
        timeout: 请求超时时间，覆盖配置文件设置
        **kwargs: 其他配置参数
        
    Returns:
        OllamaChatClient: Ollama聊天客户端实例
    """
    # 如果提供了配置管理器或者没有提供任何参数，优先使用配置文件
    if config_manager is not None or (base_url is None and model_name is None and timeout is None):
        config = OllamaChatConfig.from_config_manager(config_manager)
        
        # 如果提供了覆盖参数，则更新配置
        if base_url is not None:
            config.base_url = base_url
        if model_name is not None:
            config.model_name = model_name
        if timeout is not None:
            config.timeout = timeout
        
        # 应用其他参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return OllamaChatClient(config)
    else:
        # 使用传统方式创建配置
        config = OllamaChatConfig(
            base_url=base_url or "http://localhost:11434",
            model_name=model_name or "qwen3:1.7b",
            timeout=timeout or 300,
            **kwargs
        )
        return OllamaChatClient(config)


def create_ollama_chat_client_from_config(config_manager: Optional[ConfigManager] = None) -> OllamaChatClient:
    """从配置文件创建Ollama聊天客户端的便捷函数
    
    Args:
        config_manager: 配置管理器实例，如果为None则创建默认实例
    
    Returns:
        OllamaChatClient: 从配置文件创建的Ollama聊天客户端实例
    """
    config = OllamaChatConfig.from_config_manager(config_manager)
    return OllamaChatClient(config, config_manager)