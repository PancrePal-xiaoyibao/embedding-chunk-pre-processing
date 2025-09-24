#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Embedding模块

提供基于Ollama本地API的embedding功能，支持多种embedding模型。
"""

import os
import json
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time


@dataclass
class OllamaEmbeddingConfig:
    """
    Ollama Embedding配置类
    
    Attributes:
        base_url: Ollama服务器基础URL
        model_name: 使用的embedding模型名称
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        batch_size: 批处理大小
        normalize_embeddings: 是否标准化embedding向量
    """
    base_url: str = "http://localhost:11434"
    model_name: str = "nomic-embed-text"
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 32
    normalize_embeddings: bool = True


class OllamaEmbeddingClient:
    """
    Ollama Embedding客户端
    
    提供与Ollama服务器交互的embedding功能
    """
    
    def __init__(self, config: OllamaEmbeddingConfig):
        """
        初始化Ollama Embedding客户端
        
        Args:
            config: Ollama embedding配置
            
        Raises:
            ConnectionError: 无法连接到Ollama服务器
            ValueError: 配置参数错误
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 从环境变量获取基础URL（如果设置了的话）
        env_url = os.getenv("OLLAMA_BASE_URL")
        if env_url:
            self.config.base_url = env_url
            self.logger.info(f"使用环境变量中的Ollama URL: {env_url}")
        
        # 验证配置
        self._validate_config()
        
        # 测试连接
        self._test_connection()
        
        # 确保模型可用
        self._ensure_model_available()
    
    def _validate_config(self):
        """
        验证配置参数
        
        Raises:
            ValueError: 配置参数错误
        """
        if not self.config.base_url:
            raise ValueError("Ollama基础URL不能为空")
        
        if not self.config.model_name:
            raise ValueError("模型名称不能为空")
        
        if self.config.timeout <= 0:
            raise ValueError("超时时间必须大于0")
        
        if self.config.max_retries < 0:
            raise ValueError("最大重试次数不能小于0")
        
        if self.config.batch_size <= 0:
            raise ValueError("批处理大小必须大于0")
    
    def _test_connection(self):
        """
        测试与Ollama服务器的连接
        
        Raises:
            ConnectionError: 无法连接到Ollama服务器
        """
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout
            )
            response.raise_for_status()
            self.logger.info(f"成功连接到Ollama服务器: {self.config.base_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"无法连接到Ollama服务器 {self.config.base_url}: {e}")
    
    def _ensure_model_available(self):
        """
        确保指定的embedding模型可用
        
        Raises:
            ValueError: 模型不可用
        """
        try:
            # 获取可用模型列表
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            # 检查模型是否可用
            if self.config.model_name not in available_models:
                self.logger.warning(f"模型 {self.config.model_name} 不在可用列表中")
                self.logger.info(f"可用模型: {available_models}")
                
                # 尝试拉取模型
                self._pull_model()
            else:
                self.logger.info(f"模型 {self.config.model_name} 已可用")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"检查模型可用性失败: {e}")
            raise ValueError(f"无法验证模型可用性: {e}")
    
    def _pull_model(self):
        """
        拉取指定的embedding模型
        
        Raises:
            RuntimeError: 模型拉取失败
        """
        try:
            self.logger.info(f"开始拉取模型: {self.config.model_name}")
            
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": self.config.model_name},
                timeout=300  # 拉取模型可能需要较长时间
            )
            response.raise_for_status()
            
            self.logger.info(f"模型 {self.config.model_name} 拉取成功")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"模型拉取失败: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的embedding向量
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: embedding向量
            
        Raises:
            RuntimeError: embedding生成失败
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的embedding向量
        
        Args:
            texts: 输入文本列表
            
        Returns:
            List[List[float]]: embedding向量列表
            
        Raises:
            RuntimeError: embedding生成失败
        """
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("没有有效的输入文本")
        
        embeddings = []
        
        # 分批处理
        for i in range(0, len(valid_texts), self.config.batch_size):
            batch_texts = valid_texts[i:i + self.config.batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取一批文本的embedding向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: embedding向量列表
            
        Raises:
            RuntimeError: embedding生成失败
        """
        embeddings = []
        
        for text in texts:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = requests.post(
                        f"{self.config.base_url}/api/embeddings",
                        json={
                            "model": self.config.model_name,
                            "prompt": text
                        },
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    embedding = result.get('embedding')
                    
                    if not embedding:
                        raise RuntimeError("服务器返回空的embedding向量")
                    
                    # 标准化embedding向量（如果需要）
                    if self.config.normalize_embeddings:
                        embedding = self._normalize_embedding(embedding)
                    
                    embeddings.append(embedding)
                    break
                    
                except requests.exceptions.RequestException as e:
                    if attempt < self.config.max_retries:
                        wait_time = 2 ** attempt  # 指数退避
                        self.logger.warning(f"请求失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.config.max_retries + 1}): {e}")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"获取embedding失败，已重试{self.config.max_retries}次: {e}")
                except Exception as e:
                    raise RuntimeError(f"处理embedding响应失败: {e}")
        
        return embeddings
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        标准化embedding向量
        
        Args:
            embedding: 原始embedding向量
            
        Returns:
            List[float]: 标准化后的embedding向量
        """
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        
        if norm == 0:
            return embedding
        
        normalized = embedding_array / norm
        return normalized.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        try:
            response = requests.post(
                f"{self.config.base_url}/api/show",
                json={"name": self.config.model_name},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """
        测试与Ollama服务器的连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self._test_connection()
            return True
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False
    
    def is_model_available(self) -> bool:
        """
        检查模型是否可用
        
        Returns:
            bool: 模型是否可用
        """
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            
            # 检查完整模型名和简化模型名
            model_available = (
                self.config.model_name in available_models or
                f"{self.config.model_name}:latest" in available_models
            )
            
            return model_available
        except Exception as e:
            self.logger.error(f"检查模型可用性失败: {e}")
            return False
    
    def pull_model(self) -> bool:
        """
        拉取模型
        
        Returns:
            bool: 拉取是否成功
        """
        try:
            self._pull_model()
            return True
        except Exception as e:
            self.logger.error(f"模型拉取失败: {e}")
            return False

    def test_embedding(self, test_text: str = "这是一个测试文本") -> Dict[str, Any]:
        """
        测试embedding功能
        
        Args:
            test_text: 测试文本
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            start_time = time.time()
            embedding = self.get_embedding(test_text)
            end_time = time.time()
            
            return {
                "success": True,
                "text": test_text,
                "embedding_dimension": len(embedding),
                "processing_time": end_time - start_time,
                "model_name": self.config.model_name,
                "base_url": self.config.base_url
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": self.config.model_name,
                "base_url": self.config.base_url
            }


class OllamaEmbeddingManager:
    """
    Ollama Embedding管理器
    
    提供高级的embedding管理功能，包括缓存、批处理优化等
    """
    
    def __init__(self, config: OllamaEmbeddingConfig, enable_cache: bool = True):
        """
        初始化Ollama Embedding管理器
        
        Args:
            config: Ollama embedding配置
            enable_cache: 是否启用缓存
        """
        self.client = OllamaEmbeddingClient(config)
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None
        self.logger = logging.getLogger(__name__)
    
    def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        获取文本的embedding向量（支持缓存）
        
        Args:
            text: 输入文本
            use_cache: 是否使用缓存
            
        Returns:
            List[float]: embedding向量
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        
        # 生成缓存键
        cache_key = self._generate_cache_key(text)
        
        # 检查缓存
        if use_cache and self.enable_cache and cache_key in self.cache:
            self.logger.debug(f"从缓存获取embedding: {text[:50]}...")
            return self.cache[cache_key]
        
        # 获取embedding
        embedding = self.client.get_embedding(text)
        
        # 存储到缓存
        if use_cache and self.enable_cache:
            self.cache[cache_key] = embedding
            self.logger.debug(f"embedding已缓存: {text[:50]}...")
        
        return embedding
    
    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        批量获取文本的embedding向量（支持缓存）
        
        Args:
            texts: 输入文本列表
            use_cache: 是否使用缓存
            
        Returns:
            List[List[float]]: embedding向量列表
        """
        if not texts:
            return []
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append([])
                continue
            
            cache_key = self._generate_cache_key(text)
            
            if use_cache and self.enable_cache and cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                embeddings.append(None)  # 占位符
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 批量获取未缓存的embedding
        if uncached_texts:
            uncached_embeddings = self.client.get_embeddings(uncached_texts)
            
            # 填充结果并更新缓存
            for idx, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings[idx] = embedding
                
                if use_cache and self.enable_cache:
                    cache_key = self._generate_cache_key(texts[idx])
                    self.cache[cache_key] = embedding
        
        return embeddings
    
    def _generate_cache_key(self, text: str) -> str:
        """
        生成缓存键
        
        Args:
            text: 输入文本
            
        Returns:
            str: 缓存键
        """
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache:
            self.cache.clear()
            self.logger.info("embedding缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        if not self.enable_cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self.cache),
            "cache_keys": list(self.cache.keys())[:10]  # 只显示前10个键
        }


def create_ollama_embedding_client(config_dict: Dict[str, Any]) -> OllamaEmbeddingManager:
    """
    创建Ollama Embedding客户端的工厂函数
    
    Args:
        config_dict: 配置字典
        
    Returns:
        OllamaEmbeddingManager: Ollama embedding管理器
    """
    config = OllamaEmbeddingConfig(
        base_url=config_dict.get("base_url", "http://localhost:11434"),
        model_name=config_dict.get("model_name", "nomic-embed-text"),
        timeout=config_dict.get("timeout", 30),
        max_retries=config_dict.get("max_retries", 3),
        batch_size=config_dict.get("batch_size", 32),
        normalize_embeddings=config_dict.get("normalize_embeddings", True)
    )
    
    enable_cache = config_dict.get("enable_cache", True)
    
    return OllamaEmbeddingManager(config, enable_cache)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = OllamaEmbeddingConfig(
        base_url="http://localhost:11434",
        model_name="nomic-embed-text"
    )
    
    try:
        # 创建客户端
        manager = OllamaEmbeddingManager(config)
        
        # 测试embedding
        test_result = manager.client.test_embedding("这是一个测试文本")
        print("测试结果:", json.dumps(test_result, indent=2, ensure_ascii=False))
        
        # 测试批量embedding
        texts = ["文本1", "文本2", "文本3"]
        embeddings = manager.get_embeddings(texts)
        print(f"批量embedding完成，获得{len(embeddings)}个向量")
        
    except Exception as e:
        print(f"测试失败: {e}")