#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

提供统一的配置管理功能，支持配置文件加载、验证、热更新等功能。
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import copy


@dataclass
class LLMProviderConfig:
    """
    LLM提供商配置
    
    Attributes:
        model: 模型名称
        api_key: API密钥
        base_url: API基础URL
        max_tokens: 最大token数
        temperature: 温度参数
        timeout: 超时时间（秒）
    """
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 4000
    temperature: float = 0.3
    timeout: int = 30


@dataclass
class KeywordExtractionConfig:
    """
    关键词提取配置
    
    Attributes:
        max_keywords_per_chunk: 每个chunk最大关键词数
        min_keyword_length: 关键词最小长度
        max_keyword_length: 关键词最大长度
        enable_synonyms: 是否启用同义词扩展
        enable_medical_terms: 是否启用医学术语识别
        keyword_prefix: 关键词前缀
        extraction_methods: 提取方法配置
    """
    max_keywords_per_chunk: int = 8
    min_keyword_length: int = 2
    max_keyword_length: int = 20
    enable_synonyms: bool = True
    enable_medical_terms: bool = True
    keyword_prefix: str = "#"
    extraction_methods: Dict[str, Any] = None


@dataclass
class ChunkProcessingConfig:
    """
    分块处理配置
    
    Attributes:
        chunk_boundary_marker: 分块边界标记
        target_chunk_size: 目标分块大小
        preserve_formatting: 是否保持格式
        add_keywords_at_beginning: 是否在开头添加关键词
        keyword_separator: 关键词分隔符
        max_keywords_display: 最大显示关键词数
    """
    chunk_boundary_marker: str = "[CHUNK_BOUNDARY]"
    target_chunk_size: int = 1000
    preserve_formatting: bool = True
    add_keywords_at_beginning: bool = True
    keyword_separator: str = " "
    max_keywords_display: int = 6


@dataclass
class OutputConfig:
    """
    输出配置
    
    Attributes:
        save_original: 是否保存原始文件
        output_suffix: 输出文件后缀
        create_backup: 是否创建备份
        log_level: 日志级别
    """
    save_original: bool = True
    output_suffix: str = "_with_keywords"
    create_backup: bool = True
    log_level: str = "INFO"


@dataclass
class EmbeddingConfig:
    """
    Embedding模型配置
    
    Attributes:
        provider: 模型提供商 (sentence_transformers, ollama)
        sentence_transformers: Sentence Transformers配置
        ollama: Ollama配置
    """
    provider: str = "sentence_transformers"
    sentence_transformers: Dict[str, Any] = None
    ollama: Dict[str, Any] = None


@dataclass
class ChatConfig:
    """
    Chat配置
    
    Attributes:
        provider: Chat提供商名称 (ollama, openai, anthropic)
        ollama: Ollama Chat配置
        openai: OpenAI Chat配置
        alternative_models: 备选模型列表
    """
    provider: str = "ollama"
    ollama: Dict[str, Any] = None
    openai: Dict[str, Any] = None
    alternative_models: Dict[str, List[str]] = None


@dataclass
class RerankConfig:
    """
    Rerank配置
    
    Attributes:
        provider: Rerank提供商名称 (ollama, cohere, jina)
        ollama: Ollama Rerank配置
        alternative_models: 备选模型列表
    """
    provider: str = "ollama"
    ollama: Dict[str, Any] = None
    alternative_models: Dict[str, List[str]] = None


@dataclass
class PathsConfig:
    """
    路径配置
    
    Attributes:
        input_directory: 输入目录
        output_directory: 输出目录
        config_directory: 配置目录
        logs_directory: 日志目录
        temp_directory: 临时目录
        auto_create_directories: 是否自动创建目录
        use_relative_paths: 是否使用相对路径
    """
    input_directory: str = "doc/To_be_processed"
    output_directory: str = "doc/processed_output"
    config_directory: str = "config"
    logs_directory: str = "logs"
    temp_directory: str = "temp"
    auto_create_directories: bool = True
    use_relative_paths: bool = True


class ConfigManager:
    """
    配置管理器
    
    负责加载、验证、管理项目配置
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        self.config_path = config_path
        self.config = {}
        self.default_config = self._get_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = copy.deepcopy(self.default_config)
            if config_path:
                logging.warning(f"配置文件不存在: {config_path}，使用默认配置")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置字典
        """
        return {
            "llm_config": {
                "default_provider": "glm",
                "providers": {
                    "glm": {
                        "model": "glm-4-flash",
                        "api_key": "",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                        "max_tokens": 4000,
                        "temperature": 0.3,
                        "timeout": 30
                    },
                    "deepseek": {
                        "model": "deepseek-chat",
                        "api_key": "",
                        "base_url": "https://api.deepseek.com/",
                        "max_tokens": 4000,
                        "temperature": 0.3,
                        "timeout": 30
                    },
                    "openai": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "",
                        "base_url": "https://api.openai.com/v1/",
                        "max_tokens": 4000,
                        "temperature": 0.3,
                        "timeout": 30
                    }
                }
            },
            "keyword_extraction": {
                "max_keywords_per_chunk": 8,
                "min_keyword_length": 2,
                "max_keyword_length": 20,
                "enable_synonyms": True,
                "enable_medical_terms": True,
                "keyword_prefix": "#",
                "extraction_methods": {
                    "local": {
                        "enabled": True,
                        "use_regex": True,
                        "use_frequency": True,
                        "use_medical_dict": True
                    },
                    "llm": {
                        "enabled": True,
                        "fallback_to_local": True,
                        "prompt_template": "请从以下医学文档片段中提取5-8个最重要的关键词，包括疾病名称、药物名称、治疗方法、症状等。每个关键词用逗号分隔：\n\n{chunk_content}"
                    }
                }
            },
            "medical_knowledge": {
                "enable_synonym_expansion": True,
                "synonym_sources": ["local_dict", "medical_terminology"],
                "disease_synonyms": {
                    "恶性肿瘤": ["癌症", "肿瘤", "癌", "恶性新生物"],
                    "肺癌": ["肺部恶性肿瘤", "支气管癌", "肺部癌症"],
                    "乳腺癌": ["乳癌", "乳房癌", "乳腺恶性肿瘤"],
                    "胃癌": ["胃部恶性肿瘤", "胃腺癌"],
                    "肝癌": ["肝细胞癌", "肝部恶性肿瘤", "原发性肝癌"]
                },
                "drug_synonyms": {
                    "阿司匹林": ["乙酰水杨酸", "ASA"],
                    "布洛芬": ["异丁苯丙酸"],
                    "对乙酰氨基酚": ["扑热息痛", "醋氨酚"],
                    "吗啡": ["硫酸吗啡", "盐酸吗啡"]
                },
                "symptom_synonyms": {
                    "发热": ["发烧", "体温升高", "热症"],
                    "疼痛": ["痛", "疼", "痛感", "疼痛感"],
                    "恶心": ["想吐", "恶心感"],
                    "呕吐": ["吐", "呕"]
                }
            },
            "chunk_processing": {
                "chunk_boundary_marker": "[CHUNK_BOUNDARY]",
                "target_chunk_size": 1000,
                "preserve_formatting": True,
                "add_keywords_at_beginning": True,
                "keyword_separator": " ",
                "max_keywords_display": 6
            },
            "output": {
                "save_original": True,
                "output_suffix": "_with_keywords",
                "create_backup": True,
                "log_level": "INFO"
            },
            "chunking_strategies": {
                "default_strategy": "semantic",
                "strategies": {
                    "token_based": {
                        "enabled": True,
                        "token_sizes": [512, 1024, 2048],
                        "overlap_ratio": 0.1,
                        "smart_boundary": True
                    },
                    "semantic": {
                        "enabled": True,
                        "min_chunk_size": 200,
                        "max_chunk_size": 2000,
                        "preserve_structure": True,
                        "merge_short_chunks": True
                    },
                    "hybrid": {
                        "enabled": True,
                        "primary_strategy": "semantic",
                        "fallback_strategy": "token_based",
                        "quality_threshold": 0.8
                    }
                }
            },
            "quality_evaluation": {
                "enabled": True,
                "metrics": {
                    "size_distribution": True,
                    "semantic_integrity": True,
                    "format_correctness": True,
                    "keyword_coverage": True
                },
                "thresholds": {
                    "min_quality_score": 70.0,
                    "max_chunk_size_variance": 0.5,
                    "min_keyword_coverage": 0.8
                }
            },
            "paths": {
                "input_directory": "doc/To_be_processed",
                "output_directory": "doc/processed_output",
                "config_directory": "config",
                "logs_directory": "logs",
                "temp_directory": "temp",
                "auto_create_directories": True,
                "use_relative_paths": True
            },
            "models": {
                "embedding": {
                    "provider": "sentence_transformers",
                    "sentence_transformers": {
                        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "device": "cpu",
                        "batch_size": 32,
                        "max_seq_length": 512,
                        "normalize_embeddings": True
                    },
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "nomic-embed-text",
                        "timeout": 30,
                        "max_retries": 3,
                        "batch_size": 16,
                        "normalize_embeddings": True,
                        "enable_cache": True
                    }
                },
                "alternative_models": {
                    "sentence_transformers": [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/distiluse-base-multilingual-cased",
                        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    ],
                    "ollama": [
                        "nomic-embed-text",
                        "mxbai-embed-large",
                        "all-minilm"
                    ]
                }
            },
            "chat": {
                "provider": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model_name": "qwen3:1.7b",
                    "timeout": 60,
                    "max_retries": 3,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.9,
                    "top_k": 40,
                    "stream": True,
                    "keep_alive": "5m",
                    "system_prompt": "你是一个专业的AI助手，请根据用户的问题提供准确、有用的回答。",
                    "context": {
                        "max_history": 10,
                        "max_context_tokens": 4096,
                        "enable_context": True
                    }
                },
                "openai": {
                    "api_key": "",
                    "base_url": "https://api.openai.com/v1",
                    "model_name": "gpt-3.5-turbo",
                    "timeout": 60,
                    "max_retries": 3,
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "alternative_models": {
                    "ollama": [
                        "qwen3:1.7b",
                        "llama3.2",
                        "gemma2:2b",
                        "phi3:mini"
                    ],
                    "openai": [
                        "gpt-3.5-turbo",
                        "gpt-4",
                        "gpt-4-turbo"
                    ]
                }
            },
            "rerank": {
                "provider": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model_name": "nomic-embed-text",
                    "timeout": 30,
                    "max_retries": 3,
                    "top_k": 10
                },
                "alternative_models": {
                    "ollama": [
                        "nomic-embed-text",
                        "mxbai-embed-large"
                    ]
                }
            }
        }
    
    def load_config(self, config_path: str):
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # 合并配置（用加载的配置覆盖默认配置）
            self.config = self._merge_configs(self.default_config, loaded_config)
            self.config_path = config_path
            
            # 验证配置
            self._validate_config()
            
            logging.info(f"配置文件加载成功: {config_path}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件JSON格式错误: {e}")
        except Exception as e:
            raise ValueError(f"配置文件加载失败: {e}")
    
    def save_config(self, config_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用当前路径
            
        Raises:
            IOError: 文件写入失败
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ValueError("未指定配置文件路径")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            logging.info(f"配置文件保存成功: {config_path}")
            
        except Exception as e:
            raise IOError(f"配置文件保存失败: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置字典
        
        Args:
            default: 默认配置
            loaded: 加载的配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        merged = copy.deepcopy(default)
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _validate_config(self):
        """
        验证配置的有效性
        
        Raises:
            ValueError: 配置无效
        """
        # 验证必需的配置项
        required_sections = ["llm_config", "keyword_extraction", "chunk_processing", "output"]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"缺少必需的配置节: {section}")
        
        # 验证LLM配置
        llm_config = self.config.get("llm_config", {})
        if "default_provider" not in llm_config:
            raise ValueError("LLM配置缺少default_provider")
        
        default_provider = llm_config["default_provider"]
        providers = llm_config.get("providers", {})
        
        if default_provider not in providers:
            raise ValueError(f"默认LLM提供商 '{default_provider}' 未在providers中定义")
        
        # 验证关键词提取配置
        keyword_config = self.config.get("keyword_extraction", {})
        max_keywords = keyword_config.get("max_keywords_per_chunk", 0)
        if max_keywords <= 0:
            raise ValueError("max_keywords_per_chunk必须大于0")
        
        # 验证分块配置
        chunk_config = self.config.get("chunk_processing", {})
        target_size = chunk_config.get("target_chunk_size", 0)
        if target_size <= 0:
            raise ValueError("target_chunk_size必须大于0")
        
        logging.info("配置验证通过")
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取完整配置
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return copy.deepcopy(self.config)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section: 配置节名称
            
        Returns:
            Dict[str, Any]: 配置节内容
            
        Raises:
            KeyError: 配置节不存在
        """
        if section not in self.config:
            raise KeyError(f"配置节不存在: {section}")
        
        return copy.deepcopy(self.config[section])
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点分隔的路径）
        
        Args:
            key_path: 配置键路径，如 "llm_config.default_provider"
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_value(self, key_path: str, value: Any):
        """
        设置配置值（支持点分隔的路径）
        
        Args:
            key_path: 配置键路径，如 "llm_config.default_provider"
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到父级字典
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
    
    def get_llm_config(self, provider: Optional[str] = None) -> LLMProviderConfig:
        """
        获取LLM配置
        
        Args:
            provider: LLM提供商名称，如果为None则使用默认提供商
            
        Returns:
            LLMProviderConfig: LLM配置对象
            
        Raises:
            KeyError: 提供商不存在
        """
        llm_config = self.get_section("llm_config")
        
        if provider is None:
            provider = llm_config["default_provider"]
        
        if provider not in llm_config["providers"]:
            raise KeyError(f"LLM提供商不存在: {provider}")
        
        provider_config = llm_config["providers"][provider]
        
        return LLMProviderConfig(**provider_config)
    
    def get_keyword_extraction_config(self) -> KeywordExtractionConfig:
        """
        获取关键词提取配置
        
        Returns:
            KeywordExtractionConfig: 关键词提取配置对象
        """
        config = self.get_section("keyword_extraction")
        return KeywordExtractionConfig(**config)
    
    def get_chunk_processing_config(self) -> ChunkProcessingConfig:
        """
        获取分块处理配置
        
        Returns:
            ChunkProcessingConfig: 分块处理配置对象
        """
        config = self.get_section("chunk_processing")
        return ChunkProcessingConfig(**config)
    
    def get_output_config(self) -> OutputConfig:
        """
        获取输出配置
        
        Returns:
            OutputConfig: 输出配置对象
        """
        config = self.get_section("output")
        return OutputConfig(**config)
    
    def get_paths_config(self) -> PathsConfig:
        """
        获取路径配置
        
        Returns:
            PathsConfig: 路径配置对象
        """
        config = self.get_section("paths")
        # 过滤掉description字段，因为PathsConfig不接受这个参数
        filtered_config = {k: v for k, v in config.items() if k != "description"}
        return PathsConfig(**filtered_config)
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """
        获取embedding模型配置
        
        Returns:
            EmbeddingConfig: embedding配置对象
        """
        models_config = self.get_section("models")
        embedding_config = models_config.get("embedding", {})
        
        return EmbeddingConfig(
            provider=embedding_config.get("provider", "sentence_transformers"),
            sentence_transformers=embedding_config.get("sentence_transformers", {}),
            ollama=embedding_config.get("ollama", {})
        )
    
    def get_chat_config(self) -> ChatConfig:
        """
        获取聊天模型配置
        
        Returns:
            ChatConfig: 聊天模型配置对象
        """
        chat_config = self.get_section("chat")
        return ChatConfig(
            provider=chat_config.get("provider", "ollama"),
            ollama=chat_config.get("ollama", {}),
            openai=chat_config.get("openai", {}),
            alternative_models=chat_config.get("alternative_models", {})
        )
    
    def get_rerank_config(self) -> RerankConfig:
        """
        获取重排序模型配置
        
        Returns:
            RerankConfig: 重排序模型配置对象
        """
        config = self.get_section("models")["rerank"]
        return RerankConfig(**config)
    
    def get_absolute_path(self, path_type: str) -> str:
        """
        获取绝对路径
        
        Args:
            path_type: 路径类型 (input_directory, output_directory, config_directory, logs_directory, temp_directory)
            
        Returns:
            str: 绝对路径
        """
        paths_config = self.get_paths_config()
        relative_path = getattr(paths_config, path_type, "")
        
        if paths_config.use_relative_paths and not os.path.isabs(relative_path):
            # 相对于项目根目录
            project_root = Path(__file__).parent.parent.parent
            return str(project_root / relative_path)
        else:
            return relative_path
    
    def ensure_directories_exist(self):
        """
        确保所有配置的目录存在
        """
        paths_config = self.get_paths_config()
        
        if not paths_config.auto_create_directories:
            return
            
        directories = [
            self.get_absolute_path("input_directory"),
            self.get_absolute_path("output_directory"),
            self.get_absolute_path("config_directory"),
            self.get_absolute_path("logs_directory"),
            self.get_absolute_path("temp_directory")
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logging.debug(f"确保目录存在: {directory}")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 更新的配置项
        """
        self.config = self._merge_configs(self.config, updates)
        self._validate_config()
    
    def reset_to_default(self):
        """
        重置为默认配置
        """
        self.config = copy.deepcopy(self.default_config)
        logging.info("配置已重置为默认值")
    
    def export_config_template(self, output_path: str):
        """
        导出配置模板
        
        Args:
            output_path: 输出文件路径
        """
        template = copy.deepcopy(self.default_config)
        
        # 清空敏感信息
        if "llm_config" in template and "providers" in template["llm_config"]:
            for provider in template["llm_config"]["providers"].values():
                provider["api_key"] = ""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
            
            logging.info(f"配置模板导出成功: {output_path}")
            
        except Exception as e:
            raise IOError(f"配置模板导出失败: {e}")


def main():
    """
    主函数，用于测试配置管理器
    """
    try:
        # 测试配置管理器
        config_manager = ConfigManager("config.json")
        
        print("✅ 配置管理器初始化成功")
        
        # 测试获取配置
        llm_config = config_manager.get_llm_config()
        print(f"默认LLM提供商: {llm_config.model}")
        
        keyword_config = config_manager.get_keyword_extraction_config()
        print(f"最大关键词数: {keyword_config.max_keywords_per_chunk}")
        
        # 测试配置值获取
        default_provider = config_manager.get_value("llm_config.default_provider")
        print(f"默认提供商: {default_provider}")
        
        # 测试导出配置模板
        config_manager.export_config_template("config_template.json")
        print("✅ 配置模板导出成功")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()