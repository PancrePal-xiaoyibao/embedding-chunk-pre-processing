#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块策略模块

实现多种文档分块策略，包括基于token、语义、结构化和混合策略。
提供灵活的分块配置和优化算法。
"""

import re
import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# 尝试下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class ChunkingStrategy(Enum):
    """分块策略枚举"""
    TOKEN_BASED = "token_based"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"


@dataclass
class ChunkingConfig:
    """
    分块配置
    
    Attributes:
        strategy: 分块策略
        target_size: 目标分块大小
        min_size: 最小分块大小
        max_size: 最大分块大小
        overlap_ratio: 重叠比例
        preserve_structure: 是否保持结构
        smart_boundary: 是否智能边界检测
        language: 语言设置
        custom_params: 自定义参数
    """
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    target_size: int = 1000
    min_size: int = 200
    max_size: int = 2000
    overlap_ratio: float = 0.1
    preserve_structure: bool = True
    smart_boundary: bool = True
    language: str = "chinese"
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class ChunkInfo:
    """
    分块信息
    
    Attributes:
        id: 分块ID
        content: 分块内容
        start_pos: 起始位置
        end_pos: 结束位置
        size: 分块大小
        overlap_with_prev: 与前一个分块的重叠内容
        overlap_with_next: 与下一个分块的重叠内容
        structure_info: 结构信息
        quality_score: 质量评分
        metadata: 元数据
    """
    id: str
    content: str
    start_pos: int = 0
    end_pos: int = 0
    size: int = 0
    overlap_with_prev: str = ""
    overlap_with_next: str = ""
    structure_info: Dict[str, Any] = None
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.structure_info is None:
            self.structure_info = {}
        if self.metadata is None:
            self.metadata = {}
        if self.size == 0:
            self.size = len(self.content)


class BaseChunkingStrategy(ABC):
    """
    分块策略基类
    
    定义分块策略的通用接口
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        初始化分块策略
        
        Args:
            config: 分块配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def chunk(self, content: str) -> List[ChunkInfo]:
        """
        执行分块
        
        Args:
            content: 文档内容
            
        Returns:
            List[ChunkInfo]: 分块列表
        """
        pass
    
    def _calculate_overlap(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """
        计算分块间的重叠
        
        Args:
            chunks: 分块列表
            
        Returns:
            List[ChunkInfo]: 包含重叠信息的分块列表
        """
        if self.config.overlap_ratio <= 0:
            return chunks
        
        for i in range(len(chunks)):
            # 计算与前一个分块的重叠
            if i > 0:
                overlap_size = int(chunks[i-1].size * self.config.overlap_ratio)
                if overlap_size > 0:
                    prev_content = chunks[i-1].content
                    overlap_content = prev_content[-overlap_size:]
                    chunks[i].overlap_with_prev = overlap_content
            
            # 计算与下一个分块的重叠
            if i < len(chunks) - 1:
                overlap_size = int(chunks[i].size * self.config.overlap_ratio)
                if overlap_size > 0:
                    current_content = chunks[i].content
                    overlap_content = current_content[-overlap_size:]
                    chunks[i].overlap_with_next = overlap_content
        
        return chunks
    
    def _validate_chunk_size(self, chunk: ChunkInfo) -> bool:
        """
        验证分块大小是否合理
        
        Args:
            chunk: 分块信息
            
        Returns:
            bool: 是否合理
        """
        return self.config.min_size <= chunk.size <= self.config.max_size
    
    def _find_smart_boundary(self, content: str, position: int, direction: str = "backward") -> int:
        """
        寻找智能边界
        
        Args:
            content: 文档内容
            position: 当前位置
            direction: 搜索方向 ("backward" 或 "forward")
            
        Returns:
            int: 边界位置
        """
        if not self.config.smart_boundary:
            return position
        
        # 定义边界标记的优先级
        boundary_patterns = [
            r'\n\n',  # 段落分隔
            r'\n#',   # 标题
            r'\n\*',  # 列表项
            r'\n\d+\.', # 编号列表
            r'[。！？]',  # 句子结束
            r'[，；]',    # 句子内分隔
            r'\s',       # 空白字符
        ]
        
        search_range = min(100, len(content) - position) if direction == "forward" else min(100, position)
        
        if direction == "backward":
            search_start = max(0, position - search_range)
            search_content = content[search_start:position]
            
            for pattern in boundary_patterns:
                matches = list(re.finditer(pattern, search_content))
                if matches:
                    # 选择最接近目标位置的匹配
                    best_match = matches[-1]
                    return search_start + best_match.end()
        
        else:  # forward
            search_end = min(len(content), position + search_range)
            search_content = content[position:search_end]
            
            for pattern in boundary_patterns:
                match = re.search(pattern, search_content)
                if match:
                    return position + match.end()
        
        return position


class TokenBasedChunking(BaseChunkingStrategy):
    """
    基于Token的分块策略
    
    按照固定的token数量进行分块，支持智能边界检测
    """
    
    def chunk(self, content: str) -> List[ChunkInfo]:
        """
        执行基于token的分块
        
        Args:
            content: 文档内容
            
        Returns:
            List[ChunkInfo]: 分块列表
        """
        chunks = []
        chunk_id = 1
        
        # 简化的token计算（实际应用中可以使用专业的tokenizer）
        tokens = self._tokenize(content)
        
        for i in range(0, len(tokens), self.config.target_size):
            chunk_tokens = tokens[i:i + self.config.target_size]
            
            # 重构文本内容
            chunk_content = self._detokenize(chunk_tokens)
            
            # 智能边界调整
            if self.config.smart_boundary and i + self.config.target_size < len(tokens):
                # 寻找更好的分割点
                original_end = len(chunk_content)
                adjusted_end = self._find_smart_boundary(content, original_end, "backward")
                
                if adjusted_end != original_end:
                    # 重新计算chunk内容
                    chunk_content = content[i:adjusted_end] if i < len(content) else chunk_content
            
            chunk = ChunkInfo(
                id=f"token_chunk_{chunk_id:04d}",
                content=chunk_content.strip(),
                start_pos=i,
                end_pos=i + len(chunk_content),
                metadata={
                    "strategy": "token_based",
                    "token_count": len(chunk_tokens),
                    "target_tokens": self.config.target_size
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
        
        # 计算重叠
        chunks = self._calculate_overlap(chunks)
        
        self.logger.info(f"Token分块完成: {len(chunks)}个分块")
        
        return chunks
    
    def _tokenize(self, content: str) -> List[str]:
        """
        文本分词
        
        Args:
            content: 文档内容
            
        Returns:
            List[str]: token列表
        """
        if self.config.language == "chinese":
            # 中文分词（简化版）
            tokens = []
            for char in content:
                if char.strip():
                    tokens.append(char)
            return tokens
        else:
            # 英文分词
            return word_tokenize(content)
    
    def _detokenize(self, tokens: List[str]) -> str:
        """
        token重构为文本
        
        Args:
            tokens: token列表
            
        Returns:
            str: 重构的文本
        """
        if self.config.language == "chinese":
            return ''.join(tokens)
        else:
            return ' '.join(tokens)


class SemanticChunking(BaseChunkingStrategy):
    """
    语义分块策略
    
    基于文档的语义结构进行分块，保持语义完整性
    """
    
    def chunk(self, content: str) -> List[ChunkInfo]:
        """
        执行语义分块
        
        Args:
            content: 文档内容
            
        Returns:
            List[ChunkInfo]: 分块列表
        """
        chunks = []
        chunk_id = 1
        
        # 按段落分割
        paragraphs = self._split_paragraphs(content)
        
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph_content = paragraph["content"]
            paragraph_start = paragraph["start"]
            
            # 检查是否需要开始新分块
            if current_chunk and len(current_chunk) + len(paragraph_content) > self.config.target_size:
                # 创建当前分块
                chunk = self._create_chunk(
                    chunk_id, current_chunk, current_start, 
                    current_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # 开始新分块
                current_chunk = paragraph_content
                current_start = paragraph_start
                chunk_id += 1
            else:
                # 添加到当前分块
                if current_chunk:
                    current_chunk += "\n\n" + paragraph_content
                else:
                    current_chunk = paragraph_content
                    current_start = paragraph_start
        
        # 处理最后一个分块
        if current_chunk:
            chunk = self._create_chunk(
                chunk_id, current_chunk, current_start,
                current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        # 后处理：合并过小的分块
        chunks = self._merge_small_chunks(chunks)
        
        # 计算重叠
        chunks = self._calculate_overlap(chunks)
        
        self.logger.info(f"语义分块完成: {len(chunks)}个分块")
        
        return chunks
    
    def _split_paragraphs(self, content: str) -> List[Dict[str, Any]]:
        """
        分割段落
        
        Args:
            content: 文档内容
            
        Returns:
            List[Dict[str, Any]]: 段落信息列表
        """
        paragraphs = []
        current_pos = 0
        
        # 按双换行符分割段落
        parts = content.split('\n\n')
        
        for part in parts:
            part = part.strip()
            if part:
                paragraphs.append({
                    "content": part,
                    "start": current_pos,
                    "end": current_pos + len(part),
                    "type": self._classify_paragraph(part)
                })
            current_pos += len(part) + 2  # 加上分隔符长度
        
        return paragraphs
    
    def _classify_paragraph(self, paragraph: str) -> str:
        """
        分类段落类型
        
        Args:
            paragraph: 段落内容
            
        Returns:
            str: 段落类型
        """
        # 标题
        if paragraph.startswith('#'):
            return "heading"
        
        # 列表项
        if re.match(r'^\s*[-*+]\s', paragraph) or re.match(r'^\s*\d+\.\s', paragraph):
            return "list_item"
        
        # 代码块
        if paragraph.startswith('```') or paragraph.startswith('    '):
            return "code_block"
        
        # 引用
        if paragraph.startswith('>'):
            return "quote"
        
        # 普通段落
        return "paragraph"
    
    def _create_chunk(self, chunk_id: int, content: str, start_pos: int, end_pos: int) -> ChunkInfo:
        """
        创建分块
        
        Args:
            chunk_id: 分块ID
            content: 分块内容
            start_pos: 起始位置
            end_pos: 结束位置
            
        Returns:
            ChunkInfo: 分块信息
        """
        # 分析结构信息
        structure_info = self._analyze_structure(content)
        
        # 计算质量评分
        quality_score = self._calculate_semantic_quality(content, structure_info)
        
        return ChunkInfo(
            id=f"semantic_chunk_{chunk_id:04d}",
            content=content.strip(),
            start_pos=start_pos,
            end_pos=end_pos,
            structure_info=structure_info,
            quality_score=quality_score,
            metadata={
                "strategy": "semantic",
                "paragraph_count": len(content.split('\n\n')),
                "has_heading": any(line.startswith('#') for line in content.split('\n'))
            }
        )
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """
        分析文档结构
        
        Args:
            content: 分块内容
            
        Returns:
            Dict[str, Any]: 结构信息
        """
        lines = content.split('\n')
        
        structure = {
            "headings": [],
            "lists": [],
            "code_blocks": [],
            "quotes": [],
            "paragraphs": 0
        }
        
        in_code_block = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block:
                    structure["code_blocks"].append(i)
                continue
            
            if in_code_block:
                continue
            
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                structure["headings"].append({
                    "level": level,
                    "text": line.lstrip('# '),
                    "line": i
                })
            elif re.match(r'^\s*[-*+]\s', line) or re.match(r'^\s*\d+\.\s', line):
                structure["lists"].append(i)
            elif line.startswith('>'):
                structure["quotes"].append(i)
            elif line and not line.startswith(' '):
                structure["paragraphs"] += 1
        
        return structure
    
    def _calculate_semantic_quality(self, content: str, structure_info: Dict[str, Any]) -> float:
        """
        计算语义质量评分
        
        Args:
            content: 分块内容
            structure_info: 结构信息
            
        Returns:
            float: 质量评分 (0-100)
        """
        score = 50.0  # 基础分数
        
        # 大小评分
        size = len(content)
        if self.config.min_size <= size <= self.config.target_size:
            score += 20
        elif size <= self.config.max_size:
            score += 10
        
        # 结构完整性评分
        if structure_info["headings"]:
            score += 15  # 有标题结构
        
        if structure_info["paragraphs"] > 0:
            score += 10  # 有段落结构
        
        # 内容连贯性评分（简化版）
        sentences = sent_tokenize(content)
        if len(sentences) >= 2:
            score += 5  # 有多个句子
        
        return min(100.0, score)
    
    def _merge_small_chunks(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """
        合并过小的分块
        
        Args:
            chunks: 分块列表
            
        Returns:
            List[ChunkInfo]: 合并后的分块列表
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # 检查是否需要合并
            if (current_chunk.size < self.config.min_size and 
                i + 1 < len(chunks) and 
                current_chunk.size + chunks[i + 1].size <= self.config.max_size):
                
                # 合并当前分块和下一个分块
                next_chunk = chunks[i + 1]
                merged_content = current_chunk.content + "\n\n" + next_chunk.content
                
                merged_chunk = ChunkInfo(
                    id=f"merged_{current_chunk.id}_{next_chunk.id}",
                    content=merged_content,
                    start_pos=current_chunk.start_pos,
                    end_pos=next_chunk.end_pos,
                    structure_info=self._merge_structure_info(
                        current_chunk.structure_info, 
                        next_chunk.structure_info
                    ),
                    metadata={
                        "strategy": "semantic_merged",
                        "merged_from": [current_chunk.id, next_chunk.id]
                    }
                )
                
                merged_chunks.append(merged_chunk)
                i += 2  # 跳过下一个分块
            else:
                merged_chunks.append(current_chunk)
                i += 1
        
        return merged_chunks
    
    def _merge_structure_info(self, info1: Dict[str, Any], info2: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并结构信息
        
        Args:
            info1: 第一个分块的结构信息
            info2: 第二个分块的结构信息
            
        Returns:
            Dict[str, Any]: 合并后的结构信息
        """
        merged = {
            "headings": info1.get("headings", []) + info2.get("headings", []),
            "lists": info1.get("lists", []) + info2.get("lists", []),
            "code_blocks": info1.get("code_blocks", []) + info2.get("code_blocks", []),
            "quotes": info1.get("quotes", []) + info2.get("quotes", []),
            "paragraphs": info1.get("paragraphs", 0) + info2.get("paragraphs", 0)
        }
        
        return merged


class StructuralChunking(BaseChunkingStrategy):
    """
    结构化分块策略
    
    基于文档的结构标记（如标题、章节）进行分块
    """
    
    def chunk(self, content: str) -> List[ChunkInfo]:
        """
        执行结构化分块
        
        Args:
            content: 文档内容
            
        Returns:
            List[ChunkInfo]: 分块列表
        """
        chunks = []
        
        # 识别结构标记
        structure_markers = self._identify_structure_markers(content)
        
        if not structure_markers:
            # 如果没有结构标记，回退到语义分块
            semantic_chunker = SemanticChunking(self.config)
            return semantic_chunker.chunk(content)
        
        # 按结构标记分块
        for i, marker in enumerate(structure_markers):
            start_pos = marker["start"]
            end_pos = structure_markers[i + 1]["start"] if i + 1 < len(structure_markers) else len(content)
            
            chunk_content = content[start_pos:end_pos].strip()
            
            if chunk_content:
                chunk = ChunkInfo(
                    id=f"struct_chunk_{i + 1:04d}",
                    content=chunk_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    structure_info={
                        "marker_type": marker["type"],
                        "marker_level": marker.get("level", 0),
                        "marker_text": marker["text"]
                    },
                    metadata={
                        "strategy": "structural",
                        "section_title": marker["text"]
                    }
                )
                chunks.append(chunk)
        
        # 分割过大的分块
        chunks = self._split_large_chunks(chunks)
        
        # 计算重叠
        chunks = self._calculate_overlap(chunks)
        
        self.logger.info(f"结构化分块完成: {len(chunks)}个分块")
        
        return chunks
    
    def _identify_structure_markers(self, content: str) -> List[Dict[str, Any]]:
        """
        识别结构标记
        
        Args:
            content: 文档内容
            
        Returns:
            List[Dict[str, Any]]: 结构标记列表
        """
        markers = []
        lines = content.split('\n')
        current_pos = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Markdown标题
            if line_stripped.startswith('#'):
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                text = line_stripped.lstrip('# ').strip()
                
                markers.append({
                    "type": "heading",
                    "level": level,
                    "text": text,
                    "start": current_pos,
                    "line": line
                })
            
            # 其他结构标记可以在这里添加
            
            current_pos += len(line) + 1  # +1 for newline
        
        return markers
    
    def _split_large_chunks(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """
        分割过大的分块
        
        Args:
            chunks: 分块列表
            
        Returns:
            List[ChunkInfo]: 分割后的分块列表
        """
        result_chunks = []
        
        for chunk in chunks:
            if chunk.size <= self.config.max_size:
                result_chunks.append(chunk)
            else:
                # 使用语义分块策略分割大分块
                semantic_config = ChunkingConfig(
                    strategy=ChunkingStrategy.SEMANTIC,
                    target_size=self.config.target_size,
                    min_size=self.config.min_size,
                    max_size=self.config.max_size
                )
                semantic_chunker = SemanticChunking(semantic_config)
                sub_chunks = semantic_chunker.chunk(chunk.content)
                
                # 更新子分块的ID和元数据
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.id = f"{chunk.id}_sub_{i + 1:02d}"
                    sub_chunk.metadata.update({
                        "parent_chunk": chunk.id,
                        "parent_strategy": "structural"
                    })
                    result_chunks.append(sub_chunk)
        
        return result_chunks


class HybridChunking(BaseChunkingStrategy):
    """
    混合分块策略
    
    结合多种分块策略的优点，根据内容特点选择最佳策略
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        初始化混合分块策略
        
        Args:
            config: 分块配置
        """
        super().__init__(config)
        
        # 初始化子策略
        self.structural_chunker = StructuralChunking(config)
        self.semantic_chunker = SemanticChunking(config)
        self.token_chunker = TokenBasedChunking(config)
    
    def chunk(self, content: str) -> List[ChunkInfo]:
        """
        执行混合分块
        
        Args:
            content: 文档内容
            
        Returns:
            List[ChunkInfo]: 分块列表
        """
        # 分析文档特征
        doc_features = self._analyze_document_features(content)
        
        # 选择主要策略
        primary_strategy = self._select_primary_strategy(doc_features)
        
        # 执行主要策略分块
        if primary_strategy == "structural":
            chunks = self.structural_chunker.chunk(content)
        elif primary_strategy == "semantic":
            chunks = self.semantic_chunker.chunk(content)
        else:
            chunks = self.token_chunker.chunk(content)
        
        # 质量评估和优化
        chunks = self._optimize_chunks(chunks, doc_features)
        
        self.logger.info(f"混合分块完成: {len(chunks)}个分块, 主策略: {primary_strategy}")
        
        return chunks
    
    def _analyze_document_features(self, content: str) -> Dict[str, Any]:
        """
        分析文档特征
        
        Args:
            content: 文档内容
            
        Returns:
            Dict[str, Any]: 文档特征
        """
        lines = content.split('\n')
        
        features = {
            "total_length": len(content),
            "line_count": len(lines),
            "paragraph_count": len(content.split('\n\n')),
            "heading_count": sum(1 for line in lines if line.strip().startswith('#')),
            "list_count": sum(1 for line in lines if re.match(r'^\s*[-*+]\s', line.strip())),
            "code_block_count": content.count('```'),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "structure_ratio": 0.0,
            "complexity_score": 0.0
        }
        
        # 计算结构化程度
        structure_elements = features["heading_count"] + features["list_count"] + features["code_block_count"]
        features["structure_ratio"] = structure_elements / features["line_count"] if features["line_count"] > 0 else 0
        
        # 计算复杂度评分
        features["complexity_score"] = self._calculate_complexity_score(features)
        
        return features
    
    def _calculate_complexity_score(self, features: Dict[str, Any]) -> float:
        """
        计算文档复杂度评分
        
        Args:
            features: 文档特征
            
        Returns:
            float: 复杂度评分 (0-1)
        """
        score = 0.0
        
        # 长度复杂度
        if features["total_length"] > 10000:
            score += 0.3
        elif features["total_length"] > 5000:
            score += 0.2
        else:
            score += 0.1
        
        # 结构复杂度
        score += min(0.4, features["structure_ratio"] * 2)
        
        # 内容复杂度
        if features["avg_line_length"] > 100:
            score += 0.2
        elif features["avg_line_length"] > 50:
            score += 0.1
        
        # 多样性复杂度
        element_types = sum(1 for count in [
            features["heading_count"],
            features["list_count"],
            features["code_block_count"]
        ] if count > 0)
        
        score += element_types * 0.1
        
        return min(1.0, score)
    
    def _select_primary_strategy(self, features: Dict[str, Any]) -> str:
        """
        选择主要分块策略
        
        Args:
            features: 文档特征
            
        Returns:
            str: 策略名称
        """
        # 如果有丰富的结构标记，使用结构化分块
        if features["structure_ratio"] > 0.1 and features["heading_count"] > 2:
            return "structural"
        
        # 如果文档较长且复杂度适中，使用语义分块
        if features["total_length"] > 2000 and features["complexity_score"] < 0.8:
            return "semantic"
        
        # 其他情况使用token分块
        return "token_based"
    
    def _optimize_chunks(self, chunks: List[ChunkInfo], features: Dict[str, Any]) -> List[ChunkInfo]:
        """
        优化分块结果
        
        Args:
            chunks: 分块列表
            features: 文档特征
            
        Returns:
            List[ChunkInfo]: 优化后的分块列表
        """
        optimized_chunks = []
        
        for chunk in chunks:
            # 检查分块质量
            if self._is_chunk_quality_acceptable(chunk, features):
                optimized_chunks.append(chunk)
            else:
                # 重新分块
                if chunk.size > self.config.max_size:
                    # 分割大分块
                    sub_chunks = self._split_chunk(chunk)
                    optimized_chunks.extend(sub_chunks)
                elif chunk.size < self.config.min_size and optimized_chunks:
                    # 合并小分块
                    last_chunk = optimized_chunks[-1]
                    if last_chunk.size + chunk.size <= self.config.max_size:
                        merged_chunk = self._merge_chunks(last_chunk, chunk)
                        optimized_chunks[-1] = merged_chunk
                    else:
                        optimized_chunks.append(chunk)
                else:
                    optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _is_chunk_quality_acceptable(self, chunk: ChunkInfo, features: Dict[str, Any]) -> bool:
        """
        检查分块质量是否可接受
        
        Args:
            chunk: 分块信息
            features: 文档特征
            
        Returns:
            bool: 是否可接受
        """
        # 大小检查
        if not (self.config.min_size <= chunk.size <= self.config.max_size):
            return False
        
        # 内容完整性检查
        if chunk.content.strip() == "":
            return False
        
        # 结构完整性检查（如果有结构信息）
        if chunk.structure_info:
            # 检查是否有不完整的结构元素
            if chunk.content.count('```') % 2 != 0:  # 不完整的代码块
                return False
        
        return True
    
    def _split_chunk(self, chunk: ChunkInfo) -> List[ChunkInfo]:
        """
        分割分块
        
        Args:
            chunk: 要分割的分块
            
        Returns:
            List[ChunkInfo]: 分割后的分块列表
        """
        # 使用语义分块策略分割
        sub_chunks = self.semantic_chunker.chunk(chunk.content)
        
        # 更新子分块的ID和元数据
        for i, sub_chunk in enumerate(sub_chunks):
            sub_chunk.id = f"{chunk.id}_split_{i + 1:02d}"
            sub_chunk.metadata.update({
                "parent_chunk": chunk.id,
                "split_reason": "size_optimization"
            })
        
        return sub_chunks
    
    def _merge_chunks(self, chunk1: ChunkInfo, chunk2: ChunkInfo) -> ChunkInfo:
        """
        合并两个分块
        
        Args:
            chunk1: 第一个分块
            chunk2: 第二个分块
            
        Returns:
            ChunkInfo: 合并后的分块
        """
        merged_content = chunk1.content + "\n\n" + chunk2.content
        
        return ChunkInfo(
            id=f"merged_{chunk1.id}_{chunk2.id}",
            content=merged_content,
            start_pos=chunk1.start_pos,
            end_pos=chunk2.end_pos,
            metadata={
                "strategy": "hybrid_merged",
                "merged_from": [chunk1.id, chunk2.id],
                "merge_reason": "size_optimization"
            }
        )


class ChunkingStrategyFactory:
    """
    分块策略工厂
    
    根据配置创建相应的分块策略实例
    """
    
    @staticmethod
    def create_strategy(config: ChunkingConfig) -> BaseChunkingStrategy:
        """
        创建分块策略
        
        Args:
            config: 分块配置
            
        Returns:
            BaseChunkingStrategy: 分块策略实例
            
        Raises:
            ValueError: 不支持的策略类型
        """
        if config.strategy == ChunkingStrategy.TOKEN_BASED:
            return TokenBasedChunking(config)
        elif config.strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunking(config)
        elif config.strategy == ChunkingStrategy.STRUCTURAL:
            return StructuralChunking(config)
        elif config.strategy == ChunkingStrategy.HYBRID:
            return HybridChunking(config)
        else:
            raise ValueError(f"不支持的分块策略: {config.strategy}")


def main():
    """
    主函数，用于测试分块策略
    """
    # 测试文档
    test_content = """
    # 医学文档分块测试
    
    ## 概述
    这是一个用于测试文档分块功能的示例文档。文档包含多个段落、列表和代码块。
    
    ## 疾病介绍
    
    ### 肺癌
    肺癌是最常见的恶性肿瘤之一，根据组织学特点可分为：
    
    - 小细胞肺癌 (SCLC)
    - 非小细胞肺癌 (NSCLC)
      - 腺癌
      - 鳞状细胞癌
      - 大细胞癌
    
    ### 症状表现
    患者可能出现以下症状：
    1. 持续性咳嗽
    2. 胸痛
    3. 呼吸困难
    4. 咯血
    
    ## 治疗方案
    
    ### 手术治疗
    对于早期肺癌患者，手术切除是首选治疗方法。手术方式包括：
    - 肺叶切除术
    - 全肺切除术
    - 楔形切除术
    
    ### 化疗
    化疗是肺癌的重要治疗手段，常用药物包括顺铂、卡铂等。
    
    ```python
    # 示例代码：药物剂量计算
    def calculate_dose(weight, drug_name):
        dose_per_kg = {
            'cisplatin': 75,  # mg/m2
            'carboplatin': 400  # mg/m2
        }
        return weight * dose_per_kg.get(drug_name, 0)
    ```
    
    ### 放疗
    放射治疗适用于不能手术的患者或术后辅助治疗。
    
    ## 预后评估
    肺癌的预后与多个因素相关，包括分期、组织学类型、患者年龄等。
    早期发现和治疗是改善预后的关键。
    """
    
    try:
        # 测试不同的分块策略
        strategies = [
            ChunkingStrategy.TOKEN_BASED,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.STRUCTURAL,
            ChunkingStrategy.HYBRID
        ]
        
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"测试策略: {strategy.value}")
            print(f"{'='*50}")
            
            config = ChunkingConfig(
                strategy=strategy,
                target_size=500,
                min_size=100,
                max_size=1000
            )
            
            chunker = ChunkingStrategyFactory.create_strategy(config)
            chunks = chunker.chunk(test_content)
            
            print(f"分块数量: {len(chunks)}")
            
            for i, chunk in enumerate(chunks):
                print(f"\n分块 {i+1} ({chunk.id}):")
                print(f"  大小: {chunk.size} 字符")
                print(f"  质量评分: {chunk.quality_score:.2f}")
                print(f"  内容预览: {chunk.content[:100]}...")
                
                if chunk.structure_info:
                    print(f"  结构信息: {chunk.structure_info}")
        
        print(f"\n{'='*50}")
        print("所有测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()