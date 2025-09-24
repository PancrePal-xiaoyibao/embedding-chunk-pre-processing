"""
Rerank Service Implementation - 重排序服务实现

提供各种重排序服务的具体实现，包括基于嵌入的重排序、基于模型的重排序等。
"""

import time
import logging
import requests
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import BaseService
from core.exceptions import (
    AIServiceError, ConnectionError, AuthenticationError, 
    RateLimitError, ModelNotFoundError, ServiceTimeoutError
)
from .models import RerankResult, RerankResponse, RerankUsage


class RerankService(BaseService):
    """重排序服务抽象基类
    
    定义重排序服务的通用接口和行为。
    """
    
    @abstractmethod
    def rerank(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
        """
        pass
    
    @abstractmethod
    async def rerank_async(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """异步对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
        """
        pass
    
    def _validate_inputs(self, query: str, documents: List[str]) -> None:
        """验证输入参数
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Raises:
            ValueError: 输入参数无效时抛出
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")
        
        if not documents:
            raise ValueError("文档列表不能为空")
        
        if not all(isinstance(doc, str) for doc in documents):
            raise ValueError("所有文档必须是字符串类型")


class EmbeddingBasedRerankService(RerankService):
    """基于嵌入的重排序服务
    
    使用嵌入向量计算相似度进行重排序。
    
    Args:
        config: 服务配置
        logger: 日志记录器
        embedding_service: 嵌入服务实例
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        logger: Optional[logging.Logger] = None,
        embedding_service=None
    ):
        super().__init__(config, logger)
        
        self.embedding_service = embedding_service
        self.similarity_method = config.get("similarity_method", "cosine")
        self.normalize_scores = config.get("normalize_scores", True)
        
        self.logger.info("初始化基于嵌入的重排序服务")
    
    def health_check(self) -> bool:
        """健康检查
        
        Returns:
            bool: 服务是否健康
        """
        if self.embedding_service is None:
            return False
        
        try:
            return self.embedding_service.health_check()
        except Exception:
            return False
    
    def rerank(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
            
        Raises:
            ValueError: 输入参数无效时抛出
            AIServiceError: 嵌入服务不可用时抛出
        """
        start_time = time.time()
        
        # 验证输入
        self._validate_inputs(query, documents)
        
        if self.embedding_service is None:
            raise AIServiceError("嵌入服务不可用")
        
        # 生成查询和文档的嵌入
        all_texts = [query] + documents
        embedding_response = self.embedding_service.embed(all_texts, model=model)
        
        if not embedding_response.embeddings or len(embedding_response.embeddings) != len(all_texts):
            raise AIServiceError("嵌入生成失败")
        
        # 提取嵌入向量
        query_embedding = embedding_response.embeddings[0].vector
        doc_embeddings = [emb.vector for emb in embedding_response.embeddings[1:]]
        
        # 计算相似度分数
        scores = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = self._compute_similarity(query_embedding, doc_embedding)
            scores.append((i, similarity))
        
        # 排序（按分数降序）
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 标准化分数
        if self.normalize_scores and scores:
            max_score = scores[0][1]
            min_score = scores[-1][1]
            score_range = max_score - min_score
            
            if score_range > 0:
                scores = [(idx, (score - min_score) / score_range) for idx, score in scores]
        
        # 应用top_k限制
        if top_k is not None:
            scores = scores[:top_k]
        
        # 创建结果
        results = []
        for rank, (original_index, score) in enumerate(scores):
            result = RerankResult(
                index=original_index,
                score=score,
                document=documents[original_index]
            )
            results.append(result)
        
        response_time = time.time() - start_time
        
        # 创建使用统计
        total_tokens = sum(len(text.split()) for text in all_texts)
        usage = RerankUsage(
            query_tokens=len(query.split()),
            document_tokens=sum(len(doc.split()) for doc in documents),
            total_tokens=total_tokens
        )
        
        return RerankResponse(
            results=results,
            model=embedding_response.model,
            usage=usage,
            response_time=response_time,
            metadata={
                "similarity_method": self.similarity_method,
                "total_documents": len(documents),
                "returned_documents": len(results)
            }
        )
    
    async def rerank_async(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """异步对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
        """
        # 简单实现：在线程池中运行同步版本
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                lambda: self.rerank(query, documents, model, top_k, **kwargs)
            )
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            float: 相似度分数
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if self.similarity_method == "cosine":
            # 余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return float(dot_product / (norm1 * norm2))
        
        elif self.similarity_method == "dot":
            # 点积
            return float(np.dot(vec1, vec2))
        
        elif self.similarity_method == "euclidean":
            # 欧几里得距离（转换为相似度）
            distance = np.linalg.norm(vec1 - vec2)
            return float(1 / (1 + distance))
        
        else:
            raise ValueError(f"不支持的相似度计算方法: {self.similarity_method}")
    
    def close(self):
        """关闭服务"""
        self.logger.info("基于嵌入的重排序服务已关闭")


class OllamaRerankService(RerankService):
    """Ollama重排序服务实现
    
    使用Ollama的生成能力进行重排序。
    
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
        
        self.generate_url = f"{self.base_url}/api/generate"
        
        self.logger.info(f"初始化Ollama重排序服务: {self.base_url}")
    
    async def initialize(self) -> None:
        """异步初始化服务
        
        初始化Ollama重排序服务，检查连接和模型可用性。
        
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
            self.logger.info(f"Ollama重排序服务初始化成功: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Ollama重排序服务初始化失败: {e}")
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
        
        获取Ollama服务中可用的模型列表。
        
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
        
        检查Ollama重排序服务的健康状态。
        
        Returns:
            bool: 服务是否健康
        """
        try:
            return await self.test_connection()
        except Exception as e:
            self.logger.warning(f"Ollama健康检查失败: {e}")
            return False
    
    def rerank(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
            
        Raises:
            ValueError: 输入参数无效时抛出
            ConnectionError: 连接失败时抛出
            ModelNotFoundError: 模型不存在时抛出
        """
        start_time = time.time()
        
        # 验证输入
        self._validate_inputs(query, documents)
        
        model_name = model or self.model_name
        
        # 构建重排序提示
        prompt = self._build_rerank_prompt(query, documents, top_k)
        
        # 发送请求
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # 低温度以获得更一致的结果
                "top_p": 0.9
            }
        }
        
        # 添加额外参数
        for key, value in kwargs.items():
            if key not in request_data:
                request_data[key] = value
        
        # 发送请求并重试
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.generate_url,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    generated_text = response_data.get('response', '')
                    
                    # 解析重排序结果
                    results = self._parse_rerank_response(generated_text, documents)
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
                    time.sleep(2 ** attempt)
            
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
        total_tokens = len(prompt.split()) + len(generated_text.split())
        usage = RerankUsage(
            query_tokens=len(query.split()),
            document_tokens=sum(len(doc.split()) for doc in documents),
            total_tokens=total_tokens
        )
        
        return RerankResponse(
            results=results,
            model=model_name,
            usage=usage,
            response_time=response_time,
            metadata={
                "method": "llm_based",
                "total_documents": len(documents),
                "returned_documents": len(results)
            }
        )
    
    async def rerank_async(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """异步对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
        """
        # 简单实现：在线程池中运行同步版本
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                lambda: self.rerank(query, documents, model, top_k, **kwargs)
            )
    
    def _build_rerank_prompt(self, query: str, documents: List[str], top_k: Optional[int]) -> str:
        """构建重排序提示
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果
            
        Returns:
            str: 重排序提示
        """
        prompt = f"""请根据查询对以下文档进行重排序，按相关性从高到低排列。

查询: {query}

文档列表:
"""
        
        for i, doc in enumerate(documents):
            prompt += f"{i}: {doc}\n"
        
        if top_k:
            prompt += f"\n请返回最相关的前{top_k}个文档的索引，格式为：索引1,索引2,索引3..."
        else:
            prompt += "\n请返回所有文档的索引，按相关性从高到低排列，格式为：索引1,索引2,索引3..."
        
        prompt += "\n只返回索引数字，用逗号分隔，不要其他解释。"
        
        return prompt
    
    def _parse_rerank_response(self, response_text: str, documents: List[str]) -> List[RerankResult]:
        """解析重排序响应
        
        Args:
            response_text: 模型响应文本
            documents: 原始文档列表
            
        Returns:
            List[RerankResult]: 重排序结果列表
        """
        results = []
        
        try:
            # 提取索引
            response_text = response_text.strip()
            
            # 尝试找到数字序列
            import re
            numbers = re.findall(r'\d+', response_text)
            
            if not numbers:
                # 如果没有找到数字，按原顺序返回
                self.logger.warning("无法解析重排序结果，按原顺序返回")
                numbers = [str(i) for i in range(len(documents))]
            
            # 验证索引并创建结果
            for rank, idx_str in enumerate(numbers):
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(documents):
                        # 计算简单的相关性分数（基于排名）
                        score = 1.0 - (rank / len(numbers))
                        
                        result = RerankResult(
                            index=idx,
                            score=score,
                            document=documents[idx]
                        )
                        results.append(result)
                except ValueError:
                    continue
            
            # 如果没有有效结果，按原顺序返回
            if not results:
                self.logger.warning("没有有效的重排序结果，按原顺序返回")
                for i, doc in enumerate(documents):
                    result = RerankResult(
                        index=i,
                        score=1.0 - (i / len(documents)),
                        document=doc
                    )
                    results.append(result)
        
        except Exception as e:
            self.logger.error(f"解析重排序结果失败: {e}")
            # 按原顺序返回
            for i, doc in enumerate(documents):
                result = RerankResult(
                    index=i,
                    score=1.0 - (i / len(documents)),
                    document=doc,
                    rank=i
                )
                results.append(result)
        
        return results
    
    def close(self):
        """关闭服务"""
        self.logger.info("Ollama重排序服务已关闭")


class CrossEncoderRerankService(RerankService):
    """交叉编码器重排序服务
    
    使用交叉编码器模型进行重排序。
    
    Args:
        config: 服务配置
        logger: 日志记录器
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        
        self.model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        
        self._model = None
        self.logger.info(f"初始化交叉编码器重排序服务: {self.model_name}")
    
    def _load_model(self):
        """加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
                self.logger.info(f"成功加载交叉编码器模型: {self.model_name}")
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
    
    def rerank(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型（忽略，使用初始化时的模型）
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
        """
        start_time = time.time()
        
        # 验证输入
        self._validate_inputs(query, documents)
        
        # 加载模型
        self._load_model()
        
        # 准备查询-文档对
        query_doc_pairs = [(query, doc) for doc in documents]
        
        # 计算相关性分数
        scores = self._model.predict(query_doc_pairs, batch_size=self.batch_size)
        
        # 创建索引-分数对并排序
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 应用top_k限制
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        # 创建结果
        results = []
        for rank, (original_index, score) in enumerate(indexed_scores):
            result = RerankResult(
                index=original_index,
                score=score,
                document=documents[original_index]
            )
            results.append(result)
        
        response_time = time.time() - start_time
        
        # 创建使用统计
        total_tokens = len(query.split()) + sum(len(doc.split()) for doc in documents)
        usage = RerankUsage(
            query_tokens=len(query.split()),
            document_tokens=sum(len(doc.split()) for doc in documents),
            total_tokens=total_tokens
        )
        
        return RerankResponse(
            results=results,
            model=self.model_name,
            usage=usage,
            response_time=response_time,
            metadata={
                "method": "cross_encoder",
                "device": self.device,
                "total_documents": len(documents),
                "returned_documents": len(results)
            }
        )
    
    async def rerank_async(
        self, 
        query: str,
        documents: List[str], 
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """异步对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            model: 使用的模型
            top_k: 返回前k个结果
            **kwargs: 额外参数
            
        Returns:
            RerankResponse: 重排序响应
        """
        # 简单实现：在线程池中运行同步版本
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                lambda: self.rerank(query, documents, model, top_k, **kwargs)
            )
    
    def close(self):
        """关闭服务"""
        if self._model is not None:
            del self._model
            self._model = None
        self.logger.info("交叉编码器重排序服务已关闭")