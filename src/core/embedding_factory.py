#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding工厂模块

提供统一的embedding模型接口，支持多种embedding提供商。
"""

import os
import logging
from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod

try:
    from config.config_manager import ConfigManager, EmbeddingConfig
except ImportError:
    from src.config.config_manager import ConfigManager, EmbeddingConfig
from .ollama_embedding import create_ollama_embedding_client


class EmbeddingInterface(ABC):
    """
    Embedding接口抽象基类
    
    定义所有embedding提供商必须实现的接口方法。
    """
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        对文本进行编码，生成embedding向量
        
        Args:
            texts: 单个文本或文本列表
            **kwargs: 其他参数
            
        Returns:
            单个embedding向量或embedding向量列表
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查embedding服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        pass


class SentenceTransformersEmbedding(EmbeddingInterface):
    """
    Sentence Transformers embedding实现
    
    使用sentence-transformers库提供embedding功能。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Sentence Transformers embedding
        
        Args:
            config: Sentence Transformers配置
        """
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                config.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                device=config.get("device", "cpu")
            )
            self.logger.info(f"成功加载Sentence Transformers模型: {config.get('model_name')}")
        except ImportError:
            self.logger.error("sentence-transformers库未安装，请运行: pip install sentence-transformers")
            raise
        except Exception as e:
            self.logger.error(f"加载Sentence Transformers模型失败: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        对文本进行编码
        
        Args:
            texts: 单个文本或文本列表
            **kwargs: 其他参数
            
        Returns:
            embedding向量或向量列表
        """
        if self.model is None:
            raise RuntimeError("模型未正确初始化")
        
        # 设置编码参数
        encode_kwargs = {
            "batch_size": self.config.get("batch_size", 32),
            "normalize_embeddings": self.config.get("normalize_embeddings", True),
            "convert_to_numpy": True
        }
        encode_kwargs.update(kwargs)
        
        try:
            embeddings = self.model.encode(texts, **encode_kwargs)
            
            # 如果输入是单个文本，返回单个向量
            if isinstance(texts, str):
                return embeddings.tolist()
            else:
                return embeddings.tolist()
                
        except Exception as e:
            self.logger.error(f"文本编码失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """
        检查模型是否可用
        
        Returns:
            bool: 模型是否可用
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "provider": "sentence_transformers",
            "model_name": self.config.get("model_name"),
            "device": self.config.get("device"),
            "max_seq_length": self.config.get("max_seq_length", 512),
            "available": self.is_available()
        }


class EmbeddingFactory:
    """
    Embedding工厂类
    
    根据配置创建和管理不同的embedding提供商实例。
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化embedding工厂
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self._embedding_instance = None
    
    def create_embedding(self) -> EmbeddingInterface:
        """
        根据配置创建embedding实例
        
        Returns:
            EmbeddingInterface: embedding实例
            
        Raises:
            ValueError: 不支持的embedding提供商
            RuntimeError: 创建embedding实例失败
        """
        if self._embedding_instance is not None:
            return self._embedding_instance
        
        embedding_config = self.config_manager.get_embedding_config()
        provider = embedding_config.provider
        
        self.logger.info(f"创建embedding实例，提供商: {provider}")
        
        try:
            if provider == "sentence_transformers":
                self._embedding_instance = SentenceTransformersEmbedding(
                    embedding_config.sentence_transformers
                )
            elif provider == "ollama":
                # 从环境变量获取Ollama基础URL
                ollama_config = embedding_config.ollama.copy()
                base_url = os.getenv("OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434"))
                ollama_config["base_url"] = base_url
                
                self._embedding_instance = create_ollama_embedding_client(ollama_config)
            else:
                raise ValueError(f"不支持的embedding提供商: {provider}")
            
            self.logger.info(f"成功创建{provider} embedding实例")
            return self._embedding_instance
            
        except Exception as e:
            self.logger.error(f"创建embedding实例失败: {e}")
            raise RuntimeError(f"创建embedding实例失败: {e}")
    
    def get_embedding(self) -> EmbeddingInterface:
        """
        获取embedding实例（单例模式）
        
        Returns:
            EmbeddingInterface: embedding实例
        """
        if self._embedding_instance is None:
            self._embedding_instance = self.create_embedding()
        return self._embedding_instance
    
    def reset_embedding(self):
        """
        重置embedding实例
        
        用于配置更改后重新创建embedding实例。
        """
        self._embedding_instance = None
        self.logger.info("embedding实例已重置")


def create_embedding_factory(config_manager: ConfigManager) -> EmbeddingFactory:
    """
    创建embedding工厂实例
    
    Args:
        config_manager: 配置管理器实例
        
    Returns:
        EmbeddingFactory: embedding工厂实例
    """
    return EmbeddingFactory(config_manager)