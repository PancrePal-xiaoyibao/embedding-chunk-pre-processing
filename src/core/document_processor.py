#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理核心模块

提供文档分块、关键词提取、质量评估等核心功能。
整合现有的分块和关键词提取功能，提供统一的文档处理接口。
"""

import os
import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from datetime import datetime

# 导入现有模块
from config.config_manager import ConfigManager
from config.keyword_extractor import MedicalKeywordExtractor
from core.chunk_evaluator import ChunkEvaluator
from core.preprocess_enhanced_v3 import MedicalDocumentProcessor as EnhancedProcessor


@dataclass
class DocumentChunk:
    """
    文档分块数据类
    
    Attributes:
        id: 分块唯一标识
        content: 分块内容
        keywords: 关键词列表
        start_position: 在原文档中的起始位置
        end_position: 在原文档中的结束位置
        size: 分块大小（字符数）
        quality_score: 质量评分
        metadata: 元数据
    """
    id: str
    content: str
    keywords: List[str] = None
    start_position: int = 0
    end_position: int = 0
    size: int = 0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.keywords is None:
            self.keywords = []
        if self.metadata is None:
            self.metadata = {}
        if self.size == 0:
            self.size = len(self.content)


@dataclass
class ProcessingResult:
    """
    处理结果数据类
    
    Attributes:
        success: 是否成功
        chunks: 分块列表
        total_chunks: 总分块数
        total_keywords: 总关键词数
        processing_time: 处理时间（秒）
        quality_metrics: 质量指标
        error_message: 错误信息
        metadata: 元数据
    """
    success: bool
    chunks: List[DocumentChunk] = None
    total_chunks: int = 0
    total_keywords: int = 0
    processing_time: float = 0.0
    quality_metrics: Dict[str, Any] = None
    error_message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.chunks is None:
            self.chunks = []
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.metadata is None:
            self.metadata = {}
        
        # 自动计算统计信息
        if self.chunks:
            self.total_chunks = len(self.chunks)
            self.total_keywords = sum(len(chunk.keywords) for chunk in self.chunks)


class DocumentProcessor:
    """
    文档处理器
    
    负责文档的分块、关键词提取、质量评估等核心功能
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化文档处理器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.keyword_extractor = None
        self.chunk_evaluator = None
        self.enhanced_processor = None  # 增强处理器实例
        
        # 初始化组件
        self._initialize_components()
        
        # 处理统计
        self.processing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_keywords": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0
        }
    
    def _initialize_components(self):
        """
        初始化处理组件
        
        Raises:
            ImportError: 组件初始化失败
        """
        try:
            # 初始化增强处理器（默认主逻辑）
            chunk_config = self.config_manager.get_chunk_processing_config()
            target_size = chunk_config.target_chunk_size
            max_size = getattr(chunk_config, 'max_chunk_size', 1500)
            
            self.enhanced_processor = EnhancedProcessor(
                target_chunk_size=target_size,
                max_chunk_size=max_size
            )
            
            # 初始化关键词提取器（LLM备份逻辑）
            self.keyword_extractor = MedicalKeywordExtractor()
            
            # 初始化分块评估器
            self.chunk_evaluator = ChunkEvaluator()
            
            logging.info("文档处理组件初始化成功（增强处理器为主逻辑）")
            
        except Exception as e:
            logging.error(f"文档处理组件初始化失败: {e}")
            raise ImportError(f"组件初始化失败: {e}")
    
    def process_document(self, 
                        content: str, 
                        document_path: Optional[str] = None,
                        strategy: str = "semantic",
                        extract_keywords: bool = True,
                        evaluate_quality: bool = True) -> ProcessingResult:
        """
        处理单个文档
        
        Args:
            content: 文档内容
            document_path: 文档路径（可选）
            strategy: 分块策略
            extract_keywords: 是否提取关键词
            evaluate_quality: 是否评估质量
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            # 预处理文档
            preprocessed_content = self._preprocess_content(content)
            
            # 文档分块
            chunks = self._chunk_document(preprocessed_content, strategy)
            
            # 关键词提取
            if extract_keywords:
                chunks = self._extract_keywords_for_chunks(chunks)
            
            # 质量评估
            if evaluate_quality:
                chunks = self._evaluate_chunks_quality(chunks)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 计算质量指标
            quality_metrics = self._calculate_quality_metrics(chunks)
            
            # 更新统计信息
            self._update_stats(chunks, processing_time, quality_metrics)
            
            # 创建处理结果
            result = ProcessingResult(
                success=True,
                chunks=chunks,
                processing_time=processing_time,
                quality_metrics=quality_metrics,
                metadata={
                    "document_path": document_path,
                    "strategy": strategy,
                    "original_size": len(content),
                    "preprocessed_size": len(preprocessed_content),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logging.info(f"文档处理成功: {len(chunks)}个分块, 耗时{processing_time:.2f}秒")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"文档处理失败: {e}"
            logging.error(error_msg)
            
            return ProcessingResult(
                success=False,
                processing_time=processing_time,
                error_message=error_msg,
                metadata={
                    "document_path": document_path,
                    "strategy": strategy,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def process_file(self, 
                    file_path: str,
                    strategy: str = "semantic",
                    extract_keywords: bool = True,
                    evaluate_quality: bool = True,
                    encoding: str = "utf-8") -> ProcessingResult:
        """
        处理文件
        
        Args:
            file_path: 文件路径
            strategy: 分块策略
            extract_keywords: 是否提取关键词
            evaluate_quality: 是否评估质量
            encoding: 文件编码
            
        Returns:
            ProcessingResult: 处理结果
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 处理文档
            result = self.process_document(
                content=content,
                document_path=file_path,
                strategy=strategy,
                extract_keywords=extract_keywords,
                evaluate_quality=evaluate_quality
            )
            
            return result
            
        except Exception as e:
            error_msg = f"文件处理失败: {e}"
            logging.error(error_msg)
            
            return ProcessingResult(
                success=False,
                error_message=error_msg,
                metadata={
                    "document_path": file_path,
                    "strategy": strategy,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def process_directory(self, 
                         directory_path: str,
                         file_pattern: str = "*.md",
                         strategy: str = "semantic",
                         extract_keywords: bool = True,
                         evaluate_quality: bool = True,
                         recursive: bool = True) -> List[ProcessingResult]:
        """
        批量处理目录中的文件
        
        Args:
            directory_path: 目录路径
            file_pattern: 文件匹配模式
            strategy: 分块策略
            extract_keywords: 是否提取关键词
            evaluate_quality: 是否评估质量
            recursive: 是否递归处理子目录
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        results = []
        
        try:
            # 获取文件列表
            if recursive:
                files = list(Path(directory_path).rglob(file_pattern))
            else:
                files = list(Path(directory_path).glob(file_pattern))
            
            logging.info(f"找到{len(files)}个文件待处理")
            
            # 逐个处理文件
            for file_path in files:
                result = self.process_file(
                    file_path=str(file_path),
                    strategy=strategy,
                    extract_keywords=extract_keywords,
                    evaluate_quality=evaluate_quality
                )
                results.append(result)
                
                # 记录进度
                if len(results) % 10 == 0:
                    logging.info(f"已处理{len(results)}/{len(files)}个文件")
            
            # 统计结果
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            logging.info(f"批量处理完成: 成功{successful}个, 失败{failed}个")
            
            return results
            
        except Exception as e:
            error_msg = f"目录处理失败: {e}"
            logging.error(error_msg)
            
            # 返回错误结果
            error_result = ProcessingResult(
                success=False,
                error_message=error_msg,
                metadata={
                    "directory_path": directory_path,
                    "file_pattern": file_pattern,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return [error_result]
    
    def _preprocess_content(self, content: str) -> str:
        """
        预处理文档内容
        
        Args:
            content: 原始内容
            
        Returns:
            str: 预处理后的内容
        """
        # 首先使用增强处理器进行预处理（默认主逻辑）
        if self.enhanced_processor and hasattr(self.enhanced_processor, 'preprocess_content'):
            try:
                content = self.enhanced_processor.preprocess_content(content)
                logging.info("使用增强处理器进行内容预处理")
            except Exception as e:
                logging.warning(f"增强处理器预处理失败，使用备用逻辑: {e}")
        
        # 备用预处理逻辑
        # 移除多余的空白字符
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # 移除行首行尾空白
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)
        
        # 标准化换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        return content.strip()
    
    def _chunk_document(self, content: str, strategy: str) -> List[DocumentChunk]:
        """
        文档分块 - 优先使用增强处理器，LLM作为备份
        
        Args:
            content: 文档内容
            strategy: 分块策略
            
        Returns:
            List[DocumentChunk]: 分块列表
        """
        # 首先尝试使用增强处理器（默认主逻辑）
        if self.enhanced_processor and strategy in ["semantic", "hybrid"]:
            try:
                # 创建临时文件用于增强处理器处理
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(content)
                    tmp_input_path = tmp_file.name
                
                # 创建输出临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
                    tmp_output_path = tmp_file.name
                
                try:
                    # 使用增强处理器处理文档
                    self.enhanced_processor.process_document(
                        input_path=tmp_input_path,
                        output_path=tmp_output_path,
                        include_metadata=False  # 只获取纯内容
                    )
                    
                    # 读取处理结果并转换为DocumentChunk格式
                    return self._convert_enhanced_chunks_to_document_chunks(tmp_output_path)
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_input_path):
                        os.unlink(tmp_input_path)
                    if os.path.exists(tmp_output_path):
                        os.unlink(tmp_output_path)
                        
            except Exception as e:
                logging.warning(f"增强处理器分块失败，使用LLM备份逻辑: {e}")
        
        # LLM备份逻辑
        logging.info("使用LLM备份分块逻辑")
        chunk_config = self.config_manager.get_chunk_processing_config()
        target_size = chunk_config.target_chunk_size
        
        if strategy == "semantic":
            return self._semantic_chunking(content, target_size)
        elif strategy == "token_based":
            return self._token_based_chunking(content, target_size)
        elif strategy == "hybrid":
            return self._hybrid_chunking(content, target_size)
        else:
            logging.warning(f"未知分块策略: {strategy}，使用语义分块")
            return self._semantic_chunking(content, target_size)
    
    def _semantic_chunking(self, content: str, target_size: int) -> List[DocumentChunk]:
        """
        语义分块
        
        Args:
            content: 文档内容
            target_size: 目标分块大小
            
        Returns:
            List[DocumentChunk]: 分块列表
        """
        chunks = []
        
        # 按段落分割
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        start_pos = 0
        chunk_id = 1
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 检查是否需要开始新分块
            if current_chunk and len(current_chunk) + len(paragraph) > target_size:
                # 创建当前分块
                chunk = DocumentChunk(
                    id=f"chunk_{chunk_id:04d}",
                    content=current_chunk.strip(),
                    start_position=start_pos,
                    end_position=start_pos + len(current_chunk),
                    metadata={"strategy": "semantic"}
                )
                chunks.append(chunk)
                
                # 开始新分块
                start_pos += len(current_chunk)
                current_chunk = paragraph
                chunk_id += 1
            else:
                # 添加到当前分块
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # 处理最后一个分块
        if current_chunk:
            chunk = DocumentChunk(
                id=f"chunk_{chunk_id:04d}",
                content=current_chunk.strip(),
                start_position=start_pos,
                end_position=start_pos + len(current_chunk),
                metadata={"strategy": "semantic"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _token_based_chunking(self, content: str, target_size: int) -> List[DocumentChunk]:
        """
        基于token的分块
        
        Args:
            content: 文档内容
            target_size: 目标分块大小
            
        Returns:
            List[DocumentChunk]: 分块列表
        """
        chunks = []
        chunk_id = 1
        
        # 简单按字符数分块（实际应用中可以使用tokenizer）
        for i in range(0, len(content), target_size):
            chunk_content = content[i:i + target_size]
            
            # 尝试在单词边界分割
            if i + target_size < len(content):
                # 寻找最近的空白字符
                last_space = chunk_content.rfind(' ')
                last_newline = chunk_content.rfind('\n')
                
                if last_space > target_size * 0.8 or last_newline > target_size * 0.8:
                    split_pos = max(last_space, last_newline)
                    chunk_content = chunk_content[:split_pos]
            
            chunk = DocumentChunk(
                id=f"chunk_{chunk_id:04d}",
                content=chunk_content.strip(),
                start_position=i,
                end_position=i + len(chunk_content),
                metadata={"strategy": "token_based"}
            )
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks
    
    def _convert_enhanced_chunks_to_document_chunks(self, enhanced_output_path: str) -> List[DocumentChunk]:
        """
        将增强处理器的输出转换为DocumentChunk格式
        
        Args:
            enhanced_output_path: 增强处理器输出文件路径
            
        Returns:
            List[DocumentChunk]: 转换后的分块列表
        """
        chunks = []
        chunk_id = 1
        current_content = []
        start_pos = 0
        
        try:
            with open(enhanced_output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.rstrip('\n\r')
                
                # 检测分块边界（增强处理器使用双换行作为分隔符）
                if line.strip() == '' and current_content:
                    # 创建分块
                    content = '\n'.join(current_content).strip()
                    if content:
                        chunk = DocumentChunk(
                            id=f"chunk_{chunk_id:04d}",
                            content=content,
                            start_position=start_pos,
                            end_position=start_pos + len(content),
                            metadata={"strategy": "enhanced", "source": "preprocess_enhanced_v3"}
                        )
                        chunks.append(chunk)
                        
                        # 更新位置和计数器
                        start_pos += len(content) + 2  # +2 用于换行符
                        chunk_id += 1
                        current_content = []
                elif line.strip():
                    current_content.append(line)
            
            # 处理最后一个分块
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    chunk = DocumentChunk(
                        id=f"chunk_{chunk_id:04d}",
                        content=content,
                        start_position=start_pos,
                        end_position=start_pos + len(content),
                        metadata={"strategy": "enhanced", "source": "preprocess_enhanced_v3"}
                    )
                    chunks.append(chunk)
            
            logging.info(f"成功转换 {len(chunks)} 个增强分块")
            return chunks
            
        except Exception as e:
            logging.error(f"转换增强分块失败: {e}")
            # 如果转换失败，返回空列表，让备用逻辑接管
            return []
    
    def _hybrid_chunking(self, content: str, target_size: int) -> List[DocumentChunk]:
        """
        混合分块策略
        
        Args:
            content: 文档内容
            target_size: 目标分块大小
            
        Returns:
            List[DocumentChunk]: 分块列表
        """
        # 首先尝试语义分块
        semantic_chunks = self._semantic_chunking(content, target_size)
        
        # 检查分块质量
        quality_threshold = 0.8
        final_chunks = []
        
        for chunk in semantic_chunks:
            # 简单的质量评估（实际应用中可以更复杂）
            size_score = min(1.0, chunk.size / target_size)
            if size_score < 0.3:  # 分块太小
                size_score = 0.3
            elif size_score > 2.0:  # 分块太大
                size_score = 0.5
            
            if size_score >= quality_threshold:
                chunk.metadata["strategy"] = "hybrid_semantic"
                final_chunks.append(chunk)
            else:
                # 使用token分块重新处理
                token_chunks = self._token_based_chunking(chunk.content, target_size)
                for token_chunk in token_chunks:
                    token_chunk.id = chunk.id + "_" + token_chunk.id
                    token_chunk.metadata["strategy"] = "hybrid_token"
                    final_chunks.append(token_chunk)
        
        return final_chunks
    
    def _extract_keywords_for_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        为分块提取关键词
        
        Args:
            chunks: 分块列表
            
        Returns:
            List[DocumentChunk]: 包含关键词的分块列表
        """
        if not self.keyword_extractor:
            logging.warning("关键词提取器未初始化，跳过关键词提取")
            return chunks
        
        for chunk in chunks:
            try:
                # 提取关键词
                extraction_result = self.keyword_extractor.extract_keywords_from_chunk(chunk.content, chunk.id)
                chunk.keywords = extraction_result.keywords
                
                # 更新元数据
                chunk.metadata["keyword_extraction"] = {
                    "method": "hybrid",
                    "count": len(extraction_result.keywords),
                    "timestamp": datetime.now().isoformat(),
                    "confidence": extraction_result.confidence if hasattr(extraction_result, 'confidence') else 0.8
                }
                
            except Exception as e:
                logging.error(f"分块{chunk.id}关键词提取失败: {e}")
                chunk.keywords = []
        
        return chunks
    
    def _evaluate_chunks_quality(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        评估分块质量
        
        Args:
            chunks: 分块列表
            
        Returns:
            List[DocumentChunk]: 包含质量评分的分块列表
        """
        if not self.chunk_evaluator:
            logging.warning("分块评估器未初始化，跳过质量评估")
            return chunks
        
        for chunk in chunks:
            try:
                # 简单的质量评估：基于分块大小和内容完整性
                quality_score = self._evaluate_single_chunk(chunk.content)
                chunk.quality_score = quality_score
                
                # 更新元数据
                chunk.metadata["quality_evaluation"] = {
                    "score": quality_score,
                    "timestamp": datetime.now().isoformat(),
                    "method": "simple_heuristic"
                }
                
            except Exception as e:
                logging.error(f"分块{chunk.id}质量评估失败: {e}")
                chunk.quality_score = 0.0
        
        return chunks
    
    def _evaluate_single_chunk(self, content: str) -> float:
        """
        评估单个分块的质量
        
        Args:
            content: 分块内容
            
        Returns:
            float: 质量评分 (0-100)
        """
        if not content or not content.strip():
            return 0.0
        
        score = 0.0
        content = content.strip()
        
        # 1. 长度评分 (30分)
        content_length = len(content)
        if 800 <= content_length <= 1200:  # 最优长度范围
            score += 30.0
        elif 500 <= content_length < 800 or 1200 < content_length <= 1500:  # 可接受范围
            score += 20.0
        elif content_length >= 200:  # 最小长度要求
            score += 10.0
        
        # 2. 结构完整性评分 (25分)
        # 检查是否有完整的句子结构（以中文标点结束）
        sentence_endings = content.count('。') + content.count('！') + content.count('？')
        if sentence_endings >= 2:  # 多个完整句子
            score += 25.0
        elif sentence_endings >= 1:  # 至少一个完整句子
            score += 15.0
        
        # 3. 内容连贯性评分 (25分)
        # 检查段落结构
        paragraphs = content.split('\n\n')
        if len(paragraphs) >= 2:  # 多个段落
            score += 25.0
        elif '\n' in content:  # 有换行结构
            score += 15.0
        
        # 4. 医学内容评分 (20分)
        # 检查是否包含医学相关关键词
        medical_keywords = ['治疗', '诊断', '症状', '药物', '手术', '检查', '患者', '医生']
        medical_score = sum(1 for keyword in medical_keywords if keyword in content) / len(medical_keywords)
        score += medical_score * 20.0
        
        return min(100.0, score)
    
    def _calculate_quality_metrics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        计算整体质量指标
        
        Args:
            chunks: 分块列表
            
        Returns:
            Dict[str, Any]: 质量指标
        """
        if not chunks:
            return {}
        
        # 基本统计
        sizes = [chunk.size for chunk in chunks]
        quality_scores = [chunk.quality_score for chunk in chunks if chunk.quality_score > 0]
        keyword_counts = [len(chunk.keywords) for chunk in chunks]
        
        metrics = {
            "chunk_count": len(chunks),
            "average_chunk_size": sum(sizes) / len(sizes) if sizes else 0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "size_variance": self._calculate_variance(sizes),
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "average_keyword_count": sum(keyword_counts) / len(keyword_counts) if keyword_counts else 0,
            "total_keywords": sum(keyword_counts),
            "chunks_with_keywords": sum(1 for count in keyword_counts if count > 0),
            "keyword_coverage": sum(1 for count in keyword_counts if count > 0) / len(chunks) if chunks else 0
        }
        
        return metrics
    
    def _calculate_variance(self, values: List[float]) -> float:
        """
        计算方差
        
        Args:
            values: 数值列表
            
        Returns:
            float: 方差
        """
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        return variance
    
    def _update_stats(self, chunks: List[DocumentChunk], processing_time: float, quality_metrics: Dict[str, Any]):
        """
        更新处理统计信息
        
        Args:
            chunks: 分块列表
            processing_time: 处理时间
            quality_metrics: 质量指标
        """
        self.processing_stats["total_documents"] += 1
        self.processing_stats["total_chunks"] += len(chunks)
        self.processing_stats["total_keywords"] += sum(len(chunk.keywords) for chunk in chunks)
        self.processing_stats["total_processing_time"] += processing_time
        
        # 更新平均质量评分
        if quality_metrics.get("average_quality_score", 0) > 0:
            current_avg = self.processing_stats["average_quality_score"]
            total_docs = self.processing_stats["total_documents"]
            new_score = quality_metrics["average_quality_score"]
            
            # 计算加权平均
            self.processing_stats["average_quality_score"] = (
                (current_avg * (total_docs - 1) + new_score) / total_docs
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.processing_stats.copy()
        
        # 计算平均处理时间
        if stats["total_documents"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_documents"]
        else:
            stats["average_processing_time"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.processing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_keywords": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0
        }
        
        logging.info("处理统计信息已重置")
    
    def export_results(self, results: List[ProcessingResult], output_path: str, format: str = "json"):
        """
        导出处理结果
        
        Args:
            results: 处理结果列表
            output_path: 输出文件路径
            format: 导出格式（json, csv）
            
        Raises:
            ValueError: 不支持的格式
            IOError: 文件写入失败
        """
        try:
            if format.lower() == "json":
                self._export_json(results, output_path)
            elif format.lower() == "csv":
                self._export_csv(results, output_path)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logging.info(f"处理结果导出成功: {output_path}")
            
        except Exception as e:
            raise IOError(f"结果导出失败: {e}")
    
    def _export_json(self, results: List[ProcessingResult], output_path: str):
        """
        导出JSON格式结果
        
        Args:
            results: 处理结果列表
            output_path: 输出文件路径
        """
        # 转换为可序列化的格式
        export_data = []
        
        for result in results:
            result_dict = asdict(result)
            
            # 转换分块数据
            if result_dict["chunks"]:
                result_dict["chunks"] = [asdict(chunk) for chunk in result.chunks]
            
            export_data.append(result_dict)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def _export_csv(self, results: List[ProcessingResult], output_path: str):
        """
        导出CSV格式结果
        
        Args:
            results: 处理结果列表
            output_path: 输出文件路径
        """
        import csv
        
        # 准备CSV数据
        csv_data = []
        
        for result in results:
            if result.success and result.chunks:
                for chunk in result.chunks:
                    row = {
                        "document_path": result.metadata.get("document_path", ""),
                        "chunk_id": chunk.id,
                        "chunk_size": chunk.size,
                        "keywords": "; ".join(chunk.keywords),
                        "keyword_count": len(chunk.keywords),
                        "quality_score": chunk.quality_score,
                        "strategy": chunk.metadata.get("strategy", ""),
                        "processing_time": result.processing_time
                    }
                    csv_data.append(row)
        
        # 写入CSV文件
        if csv_data:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)


def main():
    """
    主函数，用于测试文档处理器
    """
    try:
        # 初始化配置管理器
        config_manager = ConfigManager("config.json")
        
        # 初始化文档处理器
        processor = DocumentProcessor(config_manager)
        
        print("✅ 文档处理器初始化成功")
        
        # 测试文档处理
        test_content = """
        # 医学文档示例
        
        ## 疾病概述
        肺癌是一种常见的恶性肿瘤，主要分为小细胞肺癌和非小细胞肺癌两大类。
        
        ## 症状表现
        患者可能出现咳嗽、胸痛、呼吸困难等症状。
        
        ## 治疗方案
        治疗方案包括手术、化疗、放疗和靶向治疗等多种方法。
        """
        
        result = processor.process_document(test_content)
        
        if result.success:
            print(f"✅ 文档处理成功: {result.total_chunks}个分块, {result.total_keywords}个关键词")
            print(f"处理时间: {result.processing_time:.2f}秒")
            
            # 显示分块信息
            for chunk in result.chunks:
                print(f"分块 {chunk.id}: {len(chunk.content)}字符, {len(chunk.keywords)}个关键词")
                print(f"关键词: {', '.join(chunk.keywords)}")
                print(f"质量评分: {chunk.quality_score:.2f}")
                print("-" * 50)
        else:
            print(f"❌ 文档处理失败: {result.error_message}")
        
        # 显示统计信息
        stats = processor.get_processing_stats()
        print(f"处理统计: {stats}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()