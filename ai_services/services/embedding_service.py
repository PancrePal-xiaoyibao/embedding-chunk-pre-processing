"""
Embedding Service Implementation - 嵌入服务实现

提供各种嵌入服务的具体实现，包括Ollama等。
"""

import time
import logging
import requests
import numpy as np
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import BaseService
from core.exceptions import (
    AIServiceError, ConnectionError, AuthenticationError, 
    RateLimitError, ModelNotFoundError, ServiceTimeoutError
)
from .models import EmbeddingVector, EmbeddingResponse, EmbeddingUsage


class EmbeddingService(BaseService):
    """嵌入服务抽象基类
    
    定义嵌入服务的通用接口和行为。
    """
    
    @abstractmethod
    def embed(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """生成文本嵌入
        
        Args:
            texts: 文本内容或文本列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            EmbeddingResponse: 嵌入响应
        """
        pass
    
    @abstractmethod
    async def embed_async(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """异步生成文本嵌入
        
        Args:
            texts: 文本内容或文本列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            EmbeddingResponse: 嵌入响应
        """
        pass
    
    def _normalize_texts(self, texts: Union[str, List[str]]) -> List[str]:
        """标准化文本格式
        
        Args:
            texts: 文本内容或文本列表
            
        Returns:
            List[str]: 标准化的文本列表
        """
        if isinstance(texts, str):
            return [texts]
        elif isinstance(texts, list):
            return [str(text) for text in texts]
        else:
            raise ValueError(f"不支持的文本格式: {type(texts)}")
    
    def compute_similarity(
        self, 
        embedding1: Union[List[float], EmbeddingVector], 
        embedding2: Union[List[float], EmbeddingVector],
        method: str = "cosine"
    ) -> float:
        """计算嵌入向量相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            method: 相似度计算方法 ("cosine", "dot", "euclidean")
            
        Returns:
            float: 相似度分数
        """
        # 提取向量数据
        vec1 = embedding1.vector if isinstance(embedding1, EmbeddingVector) else embedding1
        vec2 = embedding2.vector if isinstance(embedding2, EmbeddingVector) else embedding2
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if method == "cosine":
            # 余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return float(dot_product / (norm1 * norm2))
        
        elif method == "dot":
            # 点积
            return float(np.dot(vec1, vec2))
        
        elif method == "euclidean":
            # 欧几里得距离（转换为相似度）
            distance = np.linalg.norm(vec1 - vec2)
            return float(1 / (1 + distance))
        
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")


class OllamaEmbeddingService(EmbeddingService):
    """Ollama嵌入服务实现
    
    提供基于Ollama的嵌入功能。
    
    Args:
        config: 服务配置
        logger: 日志记录器
    """
    
    def __init__(self, provider: 'ServiceProvider', config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(provider, config, logger)
        
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "nomic-embed-text")
        self.timeout = config.get("timeout", 30.0)
        self.max_retries = config.get("max_retries", 3)
        
        # 确保base_url格式正确
        if not self.base_url.startswith(("http://", "https://")):
            self.base_url = f"http://{self.base_url}"
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
        
        self.embeddings_url = f"{self.base_url}/api/embeddings"
        
        self.logger.info(f"初始化Ollama嵌入服务: {self.base_url}")
    
    async def initialize(self) -> None:
        """异步初始化服务
        
        初始化Ollama嵌入服务，检查连接和模型可用性。
        
        Raises:
            ConnectionError: 连接失败时抛出
            ModelNotFoundError: 模型不存在时抛出
        """
        try:
            # 测试连接
            await self.test_connection()
            
            # 检查模型是否可用
            available_models = await self.get_models()
            if self.model_name not in available_models:
                self.logger.warning(f"模型 {self.model_name} 不在可用列表中，尝试拉取...")
                # 可以选择自动拉取模型或抛出异常
                # await self.pull_model_async(self.model_name)
            
            self._initialized = True
            self.logger.info(f"Ollama嵌入服务初始化成功: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Ollama嵌入服务初始化失败: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """异步测试连接
        
        测试与Ollama服务的连接是否正常。
        
        Returns:
            bool: 连接是否成功
            
        Raises:
            ConnectionError: 连接失败时抛出
        """
        try:
            import asyncio
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        self.logger.info("Ollama连接测试成功")
                        return True
                    else:
                        raise ConnectionError(f"Ollama连接测试失败: HTTP {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Ollama连接测试失败: {e}")
            raise ConnectionError(f"无法连接到Ollama服务: {e}")
    
    async def get_models(self) -> List[str]:
        """异步获取可用模型列表
        
        获取Ollama服务中可用的嵌入模型列表。
        
        Returns:
            List[str]: 可用模型名称列表
            
        Raises:
            ConnectionError: 连接失败时抛出
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        self.logger.info(f"获取到 {len(models)} 个可用模型")
                        return models
                    else:
                        raise ConnectionError(f"获取模型列表失败: HTTP {response.status}")
                        
        except Exception as e:
            self.logger.error(f"获取模型列表失败: {e}")
            raise ConnectionError(f"无法获取Ollama模型列表: {e}")
    
    async def health_check(self) -> bool:
        """异步健康检查
        
        检查Ollama嵌入服务的健康状态。
        
        Returns:
            bool: 服务是否健康
        """
        try:
            return await self.test_connection()
        except Exception as e:
            self.logger.warning(f"Ollama健康检查失败: {e}")
            return False
    
    def embed(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """生成文本嵌入
        
        Args:
            texts: 文本内容或文本列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            EmbeddingResponse: 嵌入响应
            
        Raises:
            ConnectionError: 连接失败时抛出
            ModelNotFoundError: 模型不存在时抛出
            ServiceTimeoutError: 服务超时时抛出
        """
        start_time = time.time()
        
        # 标准化文本
        normalized_texts = self._normalize_texts(texts)
        
        # 准备请求数据
        model_name = model or self.model_name
        
        embeddings = []
        total_prompt_tokens = 0
        
        # 批量处理文本（Ollama一次只能处理一个文本）
        for i, text in enumerate(normalized_texts):
            request_data = {
                "model": model_name,
                "prompt": text
            }
            
            # 添加额外参数
            for key, value in kwargs.items():
                if key not in request_data:
                    request_data[key] = value
            
            # 发送请求
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.embeddings_url,
                        json=request_data,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # 解析嵌入向量
                        embedding_vector = response_data.get('embedding', [])
                        if not embedding_vector:
                            raise AIServiceError("响应中没有嵌入向量")
                        
                        embedding = EmbeddingVector(
                            vector=embedding_vector,
                            index=i,
                            metadata={"text": text}
                        )
                        embeddings.append(embedding)
                        
                        # 估算token数（简单估算）
                        total_prompt_tokens += len(text.split())
                        break
                    
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
            else:
                raise ConnectionError("达到最大重试次数")
        
        response_time = time.time() - start_time
        
        # 创建使用统计
        usage = EmbeddingUsage(
            prompt_tokens=total_prompt_tokens,
            total_tokens=total_prompt_tokens
        )
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            usage=usage,
            response_time=response_time,
            metadata={"batch_size": len(normalized_texts)}
        )
    
    async def embed_async(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """异步生成文本嵌入
        
        Args:
            texts: 文本内容或文本列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            EmbeddingResponse: 嵌入响应
        """
        # 简单实现：在线程池中运行同步版本
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                lambda: self.embed(texts, model, **kwargs)
            )
    
    def embed_batch(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        batch_size: int = 10,
        **kwargs
    ) -> EmbeddingResponse:
        """批量生成文本嵌入
        
        Args:
            texts: 文本列表
            model: 使用的模型
            batch_size: 批次大小
            **kwargs: 额外参数
            
        Returns:
            EmbeddingResponse: 嵌入响应
        """
        all_embeddings = []
        total_usage = EmbeddingUsage()
        total_time = 0.0
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            response = self.embed(batch_texts, model, **kwargs)
            
            # 更新索引
            for j, embedding in enumerate(response.embeddings):
                embedding.index = i + j
            
            all_embeddings.extend(response.embeddings)
            
            # 累计统计
            if response.usage:
                total_usage.prompt_tokens += response.usage.prompt_tokens
                total_usage.total_tokens += response.usage.total_tokens
            
            if response.response_time:
                total_time += response.response_time
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=model or self.model_name,
            usage=total_usage,
            response_time=total_time,
            metadata={"total_batches": (len(texts) + batch_size - 1) // batch_size}
        )
    
    def get_available_models_sync(self) -> List[str]:
        """同步获取可用模型列表
        
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
    
    def close(self):
        """关闭服务"""
        self.logger.info("Ollama嵌入服务已关闭")


class LocalEmbeddingService(EmbeddingService):
    """本地嵌入服务实现
    
    使用本地模型（如sentence-transformers）生成嵌入。
    
    Args:
        config: 服务配置
        logger: 日志记录器
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        
        self.model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self.device = config.get("device", "cpu")
        
        self._model = None
        self.logger.info(f"初始化本地嵌入服务: {self.model_name}")
    
    def _load_model(self):
        """加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self.logger.info(f"成功加载模型: {self.model_name}")
            except ImportError:
                raise AIServiceError(
                    "需要安装sentence-transformers库: pip install sentence-transformers"
                )
            except Exception as e:
                raise AIServiceError(f"加载模型失败: {e}")
    
    def health_check(self) -> bool:
        """健康检查
        
        Returns:
            bool: 服务是否健康
        """
        try:
            self._load_model()
            return self._model is not None
        except Exception:
            return False
    
    def embed(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """生成文本嵌入
        
        Args:
            texts: 文本内容或文本列表
            model: 使用的模型（忽略，使用初始化时的模型）
            **kwargs: 额外参数
            
        Returns:
            EmbeddingResponse: 嵌入响应
        """
        start_time = time.time()
        
        # 加载模型
        self._load_model()
        
        # 标准化文本
        normalized_texts = self._normalize_texts(texts)
        
        # 生成嵌入
        embeddings_array = self._model.encode(normalized_texts, **kwargs)
        
        # 创建嵌入向量对象
        embeddings = []
        for i, embedding in enumerate(embeddings_array):
            embeddings.append(EmbeddingVector(
                vector=embedding.tolist(),
                index=i,
                metadata={"text": normalized_texts[i]}
            ))
        
        response_time = time.time() - start_time
        
        # 创建使用统计
        total_tokens = sum(len(text.split()) for text in normalized_texts)
        usage = EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model_name,
            usage=usage,
            response_time=response_time,
            metadata={"device": self.device}
        )
    
    async def embed_async(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """异步生成文本嵌入
        
        Args:
            texts: 文本内容或文本列表
            model: 使用的模型
            **kwargs: 额外参数
            
        Returns:
            EmbeddingResponse: 嵌入响应
        """
        # 简单实现：在线程池中运行同步版本
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                lambda: self.embed(texts, model, **kwargs)
            )
    
    def close(self):
        """关闭服务"""
        if self._model is not None:
            del self._model
            self._model = None
        self.logger.info("本地嵌入服务已关闭")