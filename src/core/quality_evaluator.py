#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量评估模块

基于现有chunk_evaluator.py优化，提供更全面的文档分块质量评估功能。
包括语义完整性、结构一致性、内容质量、可读性等多维度评估。
"""

import re
import os
import math
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
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


class QualityLevel(Enum):
    """质量等级枚举"""
    EXCELLENT = "excellent"  # 90-100分
    GOOD = "good"           # 75-89分
    FAIR = "fair"           # 60-74分
    POOR = "poor"           # 40-59分
    VERY_POOR = "very_poor" # 0-39分


class EvaluationMetric(Enum):
    """评估指标枚举"""
    SEMANTIC_INTEGRITY = "semantic_integrity"
    STRUCTURAL_CONSISTENCY = "structural_consistency"
    CONTENT_QUALITY = "content_quality"
    READABILITY = "readability"
    SIZE_DISTRIBUTION = "size_distribution"
    BOUNDARY_QUALITY = "boundary_quality"
    MEDICAL_TERMINOLOGY = "medical_terminology"
    OVERALL = "overall"


@dataclass
class MetricScore:
    """
    指标评分
    
    Attributes:
        value: 评分值 (0-100)
        weight: 权重
        details: 详细信息
        suggestions: 改进建议
    """
    value: float
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityEvaluationResult:
    """
    质量评估结果
    
    Attributes:
        file_path: 文件路径
        total_chunks: 总分块数量
        total_chars: 总字符数
        avg_chunk_size: 平均分块大小
        chunk_size_distribution: 分块大小分布
        metrics: 各项指标评分
        overall_score: 总体评分
        quality_level: 质量等级
        recommendations: 优化建议
        detailed_analysis: 详细分析
        processing_time: 处理时间
    """
    file_path: str = ""
    total_chunks: int = 0
    total_chars: int = 0
    avg_chunk_size: int = 0
    chunk_size_distribution: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[EvaluationMetric, MetricScore] = field(default_factory=dict)
    overall_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.POOR
    recommendations: List[str] = field(default_factory=list)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class ChunkAnalysis:
    """
    单个分块分析结果
    
    Attributes:
        id: 分块ID
        content: 分块内容
        size: 分块大小
        semantic_score: 语义评分
        structural_score: 结构评分
        readability_score: 可读性评分
        issues: 发现的问题
        strengths: 优点
    """
    id: str
    content: str
    size: int
    semantic_score: float = 0.0
    structural_score: float = 0.0
    readability_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


class QualityEvaluator:
    """
    质量评估器
    
    提供全面的文档分块质量评估功能，包括多维度评估和详细的改进建议。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化质量评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 默认配置
        self.target_chunk_size = self.config.get('target_chunk_size', 1000)
        self.min_chunk_size = self.config.get('min_chunk_size', 200)
        self.max_chunk_size = self.config.get('max_chunk_size', 2000)
        self.chunk_boundary_marker = self.config.get('chunk_boundary_marker', '[CHUNK_BOUNDARY]')
        self.language = self.config.get('language', 'chinese')
        
        # 评估权重配置
        self.metric_weights = self.config.get('metric_weights', {
            EvaluationMetric.SEMANTIC_INTEGRITY: 0.25,
            EvaluationMetric.STRUCTURAL_CONSISTENCY: 0.20,
            EvaluationMetric.CONTENT_QUALITY: 0.20,
            EvaluationMetric.READABILITY: 0.15,
            EvaluationMetric.SIZE_DISTRIBUTION: 0.10,
            EvaluationMetric.BOUNDARY_QUALITY: 0.10
        })
        
        # 医学文档特定模式
        self.medical_patterns = {
            'drug_names': r'[A-Za-z\u4e00-\u9fff]+(?:酸|素|胺|醇|酮|肽|霉素|西林|沙星|替尼|单抗)',
            'medical_terms': r'(?:治疗|诊断|症状|并发症|副作用|禁忌|适应症|病理|生理|解剖)',
            'dosage_info': r'\d+(?:\.\d+)?(?:mg|g|ml|μg|IU|U)(?:/(?:日|次|kg|m²))?',
            'medical_procedures': r'(?:手术|检查|治疗|护理|康复|预防|筛查)',
            'anatomical_terms': r'(?:心脏|肺部|肝脏|肾脏|大脑|血管|神经|肌肉|骨骼)',
            'disease_names': r'(?:癌症|肿瘤|炎症|感染|综合征|病|症)',
            'clinical_indicators': r'(?:血压|心率|体温|血糖|血脂|白细胞|红细胞|血小板)'
        }
        
        # 结构模式
        self.structural_patterns = {
            'headings': r'^#{1,6}\s+.+$',
            'numbered_lists': r'^\d+[\.、]\s+.+$',
            'bullet_lists': r'^[•·\-\*]\s+.+$',
            'code_blocks': r'```[\s\S]*?```',
            'tables': r'\|.*\|',
            'quotes': r'^>\s+.+$',
            'emphasis': r'\*\*.*?\*\*|__.*?__|_.*?_|\*.*?\*'
        }
    
    def evaluate_file(self, file_path: str) -> QualityEvaluationResult:
        """
        评估文件质量
        
        Args:
            file_path: 文件路径
            
        Returns:
            QualityEvaluationResult: 评估结果
            
        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 文件编码错误
        """
        import time
        start_time = time.time()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"文件编码错误: {e}")
        
        result = self.evaluate_content(content)
        result.file_path = file_path
        result.processing_time = time.time() - start_time
        
        self.logger.info(f"文件评估完成: {file_path}, 总体评分: {result.overall_score:.2f}")
        
        return result
    
    def evaluate_content(self, content: str) -> QualityEvaluationResult:
        """
        评估内容质量
        
        Args:
            content: 文档内容
            
        Returns:
            QualityEvaluationResult: 评估结果
        """
        # 分割分块
        chunks = self._split_chunks(content)
        
        # 基础统计
        total_chunks = len(chunks)
        total_chars = len(content.replace(self.chunk_boundary_marker, ''))
        avg_chunk_size = total_chars // total_chunks if total_chunks > 0 else 0
        
        # 分块大小分布
        size_distribution = self._analyze_size_distribution(chunks)
        
        # 分析每个分块
        chunk_analyses = self._analyze_chunks(chunks)
        
        # 计算各项指标
        metrics = {}
        
        # 语义完整性
        metrics[EvaluationMetric.SEMANTIC_INTEGRITY] = self._evaluate_semantic_integrity(
            chunks, chunk_analyses
        )
        
        # 结构一致性
        metrics[EvaluationMetric.STRUCTURAL_CONSISTENCY] = self._evaluate_structural_consistency(
            content, chunks, chunk_analyses
        )
        
        # 内容质量
        metrics[EvaluationMetric.CONTENT_QUALITY] = self._evaluate_content_quality(
            chunks, chunk_analyses
        )
        
        # 可读性
        metrics[EvaluationMetric.READABILITY] = self._evaluate_readability(
            chunks, chunk_analyses
        )
        
        # 大小分布
        metrics[EvaluationMetric.SIZE_DISTRIBUTION] = self._evaluate_size_distribution(
            size_distribution, total_chunks
        )
        
        # 边界质量
        metrics[EvaluationMetric.BOUNDARY_QUALITY] = self._evaluate_boundary_quality(
            content, chunks
        )
        
        # 计算总体评分
        overall_score = self._calculate_overall_score(metrics)
        
        # 确定质量等级
        quality_level = self._determine_quality_level(overall_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            metrics, size_distribution, chunk_analyses
        )
        
        # 详细分析
        detailed_analysis = self._generate_detailed_analysis(
            chunks, chunk_analyses, metrics
        )
        
        return QualityEvaluationResult(
            total_chunks=total_chunks,
            total_chars=total_chars,
            avg_chunk_size=avg_chunk_size,
            chunk_size_distribution=size_distribution,
            metrics=metrics,
            overall_score=overall_score,
            quality_level=quality_level,
            recommendations=recommendations,
            detailed_analysis=detailed_analysis
        )
    
    def _split_chunks(self, content: str) -> List[str]:
        """
        分割分块
        
        Args:
            content: 文档内容
            
        Returns:
            List[str]: 分块列表
        """
        if self.chunk_boundary_marker in content:
            chunks = content.split(self.chunk_boundary_marker)
        else:
            # 如果没有边界标记，按段落分割
            chunks = content.split('\n\n')
        
        # 移除空分块
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _analyze_size_distribution(self, chunks: List[str]) -> Dict[str, int]:
        """
        分析分块大小分布
        
        Args:
            chunks: 分块列表
            
        Returns:
            Dict[str, int]: 大小分布统计
        """
        distribution = {
            'very_small': 0,    # < min_size
            'small': 0,         # min_size - target_size*0.8
            'optimal': 0,       # target_size*0.8 - target_size*1.2
            'large': 0,         # target_size*1.2 - max_size
            'very_large': 0     # > max_size
        }
        
        for chunk in chunks:
            size = len(chunk)
            if size < self.min_chunk_size:
                distribution['very_small'] += 1
            elif size < self.target_chunk_size * 0.8:
                distribution['small'] += 1
            elif size <= self.target_chunk_size * 1.2:
                distribution['optimal'] += 1
            elif size <= self.max_chunk_size:
                distribution['large'] += 1
            else:
                distribution['very_large'] += 1
        
        return distribution
    
    def _analyze_chunks(self, chunks: List[str]) -> List[ChunkAnalysis]:
        """
        分析每个分块
        
        Args:
            chunks: 分块列表
            
        Returns:
            List[ChunkAnalysis]: 分块分析结果列表
        """
        analyses = []
        
        for i, chunk in enumerate(chunks):
            analysis = ChunkAnalysis(
                id=f"chunk_{i+1:04d}",
                content=chunk,
                size=len(chunk)
            )
            
            # 语义分析
            analysis.semantic_score = self._analyze_chunk_semantics(chunk)
            
            # 结构分析
            analysis.structural_score = self._analyze_chunk_structure(chunk)
            
            # 可读性分析
            analysis.readability_score = self._analyze_chunk_readability(chunk)
            
            # 问题检测
            analysis.issues = self._detect_chunk_issues(chunk)
            
            # 优点识别
            analysis.strengths = self._identify_chunk_strengths(chunk)
            
            analyses.append(analysis)
        
        return analyses
    
    def _analyze_chunk_semantics(self, chunk: str) -> float:
        """
        分析分块语义完整性
        
        Args:
            chunk: 分块内容
            
        Returns:
            float: 语义评分 (0-100)
        """
        score = 0.0
        
        # 句子完整性检查
        sentences = sent_tokenize(chunk)
        complete_sentences = [s for s in sentences if len(s.strip()) > 10 and s.strip().endswith(('.', '。', '!', '！', '?', '？'))]
        
        if sentences:
            sentence_completeness = len(complete_sentences) / len(sentences)
            score += sentence_completeness * 30
        
        # 医学术语完整性
        medical_term_score = 0
        term_count = 0
        
        for pattern_name, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            for match in matches:
                if not self._is_term_truncated(match, chunk):
                    medical_term_score += 1
                term_count += 1
        
        if term_count > 0:
            score += (medical_term_score / term_count) * 25
        else:
            score += 15  # 没有医学术语也不扣分太多
        
        # 段落完整性
        if self._has_complete_paragraphs(chunk):
            score += 20
        
        # 上下文连贯性
        coherence_score = self._calculate_coherence(chunk)
        score += coherence_score * 25
        
        return min(100.0, score)
    
    def _analyze_chunk_structure(self, chunk: str) -> float:
        """
        分析分块结构质量
        
        Args:
            chunk: 分块内容
            
        Returns:
            float: 结构评分 (0-100)
        """
        score = 0.0
        
        # 标题结构
        headings = re.findall(self.structural_patterns['headings'], chunk, re.MULTILINE)
        if headings:
            score += 20
        
        # 列表结构
        numbered_lists = re.findall(self.structural_patterns['numbered_lists'], chunk, re.MULTILINE)
        bullet_lists = re.findall(self.structural_patterns['bullet_lists'], chunk, re.MULTILINE)
        
        if numbered_lists or bullet_lists:
            # 检查列表完整性
            if self._has_complete_lists(chunk):
                score += 25
            else:
                score += 10
        
        # 代码块结构
        code_blocks = re.findall(self.structural_patterns['code_blocks'], chunk, re.DOTALL)
        if code_blocks:
            # 检查代码块完整性
            if chunk.count('```') % 2 == 0:
                score += 15
            else:
                score -= 10  # 不完整的代码块扣分
        
        # 表格结构
        tables = re.findall(self.structural_patterns['tables'], chunk)
        if tables:
            score += 10
        
        # 格式一致性
        if self._check_format_consistency(chunk):
            score += 20
        
        # 结构层次
        structure_depth = self._calculate_structure_depth(chunk)
        if 1 <= structure_depth <= 3:  # 合理的结构深度
            score += 10
        
        return min(100.0, score)
    
    def _analyze_chunk_readability(self, chunk: str) -> float:
        """
        分析分块可读性
        
        Args:
            chunk: 分块内容
            
        Returns:
            float: 可读性评分 (0-100)
        """
        score = 0.0
        
        # 句子长度分析
        sentences = sent_tokenize(chunk)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # 理想句子长度 15-25 词
            if 15 <= avg_sentence_length <= 25:
                score += 25
            elif 10 <= avg_sentence_length <= 30:
                score += 20
            else:
                score += 10
        
        # 段落长度分析
        paragraphs = chunk.split('\n\n')
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            
            # 理想段落长度 50-150 词
            if 50 <= avg_paragraph_length <= 150:
                score += 25
            elif 30 <= avg_paragraph_length <= 200:
                score += 20
            else:
                score += 10
        
        # 词汇复杂度
        words = chunk.split()
        if words:
            # 计算长词比例（>6个字符）
            long_words = [w for w in words if len(w) > 6]
            long_word_ratio = len(long_words) / len(words)
            
            if 0.1 <= long_word_ratio <= 0.3:  # 合理的复杂词比例
                score += 20
            else:
                score += 10
        
        # 标点符号使用
        punctuation_score = self._analyze_punctuation_usage(chunk)
        score += punctuation_score * 15
        
        # 格式清晰度
        if self._is_format_clear(chunk):
            score += 15
        
        return min(100.0, score)
    
    def _detect_chunk_issues(self, chunk: str) -> List[str]:
        """
        检测分块问题
        
        Args:
            chunk: 分块内容
            
        Returns:
            List[str]: 问题列表
        """
        issues = []
        
        # 大小问题
        size = len(chunk)
        if size < self.min_chunk_size:
            issues.append(f"分块过小 ({size} < {self.min_chunk_size})")
        elif size > self.max_chunk_size:
            issues.append(f"分块过大 ({size} > {self.max_chunk_size})")
        
        # 内容问题
        if not chunk.strip():
            issues.append("分块为空")
        
        if chunk.startswith(('，', '。', '！', '？', '；', '：')):
            issues.append("分块以标点符号开始")
        
        # 结构问题
        if chunk.count('```') % 2 != 0:
            issues.append("代码块不完整")
        
        # 医学术语截断
        truncated_terms = []
        for pattern_name, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            for match in matches:
                if self._is_term_truncated(match, chunk):
                    truncated_terms.append(match)
        
        if truncated_terms:
            issues.append(f"医学术语可能被截断: {', '.join(truncated_terms[:3])}")
        
        # 句子完整性
        sentences = sent_tokenize(chunk)
        incomplete_sentences = [s for s in sentences if not s.strip().endswith(('.', '。', '!', '！', '?', '？'))]
        
        if len(incomplete_sentences) > len(sentences) * 0.3:
            issues.append("存在较多不完整句子")
        
        return issues
    
    def _identify_chunk_strengths(self, chunk: str) -> List[str]:
        """
        识别分块优点
        
        Args:
            chunk: 分块内容
            
        Returns:
            List[str]: 优点列表
        """
        strengths = []
        
        # 大小合适
        size = len(chunk)
        if self.target_chunk_size * 0.8 <= size <= self.target_chunk_size * 1.2:
            strengths.append("分块大小适中")
        
        # 结构完整
        if self._has_complete_structure(chunk):
            strengths.append("结构完整")
        
        # 医学术语丰富
        medical_terms = 0
        for pattern in self.medical_patterns.values():
            medical_terms += len(re.findall(pattern, chunk, re.IGNORECASE))
        
        if medical_terms >= 5:
            strengths.append("医学术语丰富")
        
        # 格式规范
        if self._check_format_consistency(chunk):
            strengths.append("格式规范")
        
        # 内容连贯
        coherence = self._calculate_coherence(chunk)
        if coherence > 0.7:
            strengths.append("内容连贯")
        
        return strengths
    
    def _evaluate_semantic_integrity(self, chunks: List[str], analyses: List[ChunkAnalysis]) -> MetricScore:
        """
        评估语义完整性
        
        Args:
            chunks: 分块列表
            analyses: 分块分析结果
            
        Returns:
            MetricScore: 语义完整性评分
        """
        if not analyses:
            return MetricScore(value=0.0, suggestions=["无分块数据"])
        
        # 计算平均语义评分
        semantic_scores = [analysis.semantic_score for analysis in analyses]
        avg_score = sum(semantic_scores) / len(semantic_scores)
        
        # 详细信息
        details = {
            "avg_semantic_score": avg_score,
            "score_distribution": {
                "excellent": len([s for s in semantic_scores if s >= 90]),
                "good": len([s for s in semantic_scores if 75 <= s < 90]),
                "fair": len([s for s in semantic_scores if 60 <= s < 75]),
                "poor": len([s for s in semantic_scores if s < 60])
            },
            "problematic_chunks": [
                analysis.id for analysis in analyses 
                if analysis.semantic_score < 60
            ]
        }
        
        # 生成建议
        suggestions = []
        if avg_score < 70:
            suggestions.append("语义完整性较低，建议在语义边界处分割")
        if details["problematic_chunks"]:
            suggestions.append(f"有{len(details['problematic_chunks'])}个分块语义质量较差，需要重新处理")
        
        return MetricScore(
            value=avg_score,
            details=details,
            suggestions=suggestions
        )
    
    def _evaluate_structural_consistency(self, content: str, chunks: List[str], analyses: List[ChunkAnalysis]) -> MetricScore:
        """
        评估结构一致性
        
        Args:
            content: 原始内容
            chunks: 分块列表
            analyses: 分块分析结果
            
        Returns:
            MetricScore: 结构一致性评分
        """
        if not analyses:
            return MetricScore(value=0.0, suggestions=["无分块数据"])
        
        # 计算平均结构评分
        structural_scores = [analysis.structural_score for analysis in analyses]
        avg_score = sum(structural_scores) / len(structural_scores)
        
        # 检查结构元素保持
        structure_preservation = self._check_structure_preservation(content, chunks)
        
        # 综合评分
        final_score = (avg_score + structure_preservation * 100) / 2
        
        details = {
            "avg_structural_score": avg_score,
            "structure_preservation": structure_preservation,
            "chunks_with_headings": len([a for a in analyses if "标题" in a.strengths]),
            "chunks_with_lists": len([a for a in analyses if "列表" in a.strengths])
        }
        
        suggestions = []
        if final_score < 70:
            suggestions.append("结构一致性较低，建议保持文档结构完整性")
        if structure_preservation < 0.8:
            suggestions.append("结构元素保持不佳，检查分块边界")
        
        return MetricScore(
            value=final_score,
            details=details,
            suggestions=suggestions
        )
    
    def _evaluate_content_quality(self, chunks: List[str], analyses: List[ChunkAnalysis]) -> MetricScore:
        """
        评估内容质量
        
        Args:
            chunks: 分块列表
            analyses: 分块分析结果
            
        Returns:
            MetricScore: 内容质量评分
        """
        if not chunks:
            return MetricScore(value=0.0, suggestions=["无分块数据"])
        
        score = 0.0
        
        # 内容丰富度
        total_words = sum(len(chunk.split()) for chunk in chunks)
        avg_words_per_chunk = total_words / len(chunks)
        
        if avg_words_per_chunk >= 100:
            score += 30
        elif avg_words_per_chunk >= 50:
            score += 20
        else:
            score += 10
        
        # 医学术语密度
        medical_term_density = self._calculate_medical_term_density(chunks)
        score += medical_term_density * 30
        
        # 信息完整性
        info_completeness = self._calculate_information_completeness(chunks)
        score += info_completeness * 40
        
        details = {
            "avg_words_per_chunk": avg_words_per_chunk,
            "medical_term_density": medical_term_density,
            "information_completeness": info_completeness,
            "total_medical_terms": sum(
                len(re.findall('|'.join(self.medical_patterns.values()), chunk, re.IGNORECASE))
                for chunk in chunks
            )
        }
        
        suggestions = []
        if medical_term_density < 0.1:
            suggestions.append("医学术语密度较低，检查是否为医学文档")
        if info_completeness < 0.7:
            suggestions.append("信息完整性不足，建议增加分块大小")
        
        return MetricScore(
            value=min(100.0, score),
            details=details,
            suggestions=suggestions
        )
    
    def _evaluate_readability(self, chunks: List[str], analyses: List[ChunkAnalysis]) -> MetricScore:
        """
        评估可读性
        
        Args:
            chunks: 分块列表
            analyses: 分块分析结果
            
        Returns:
            MetricScore: 可读性评分
        """
        if not analyses:
            return MetricScore(value=0.0, suggestions=["无分块数据"])
        
        # 计算平均可读性评分
        readability_scores = [analysis.readability_score for analysis in analyses]
        avg_score = sum(readability_scores) / len(readability_scores)
        
        details = {
            "avg_readability_score": avg_score,
            "highly_readable_chunks": len([s for s in readability_scores if s >= 80]),
            "poorly_readable_chunks": len([s for s in readability_scores if s < 50])
        }
        
        suggestions = []
        if avg_score < 60:
            suggestions.append("可读性较低，建议优化句子和段落长度")
        if details["poorly_readable_chunks"] > len(chunks) * 0.2:
            suggestions.append("存在较多可读性差的分块，需要重新组织")
        
        return MetricScore(
            value=avg_score,
            details=details,
            suggestions=suggestions
        )
    
    def _evaluate_size_distribution(self, distribution: Dict[str, int], total_chunks: int) -> MetricScore:
        """
        评估大小分布
        
        Args:
            distribution: 大小分布
            total_chunks: 总分块数
            
        Returns:
            MetricScore: 大小分布评分
        """
        if total_chunks == 0:
            return MetricScore(value=0.0, suggestions=["无分块数据"])
        
        score = 0.0
        
        # 理想分布：大部分分块应该在optimal范围内
        optimal_ratio = distribution['optimal'] / total_chunks
        score += optimal_ratio * 60
        
        # 惩罚极端大小的分块
        very_small_ratio = distribution['very_small'] / total_chunks
        very_large_ratio = distribution['very_large'] / total_chunks
        
        penalty = (very_small_ratio + very_large_ratio) * 30
        score = max(0, score - penalty)
        
        # 奖励合理的分布
        reasonable_ratio = (distribution['small'] + distribution['optimal'] + distribution['large']) / total_chunks
        score += reasonable_ratio * 40
        
        details = {
            "distribution_percentages": {
                k: (v / total_chunks * 100) for k, v in distribution.items()
            },
            "optimal_ratio": optimal_ratio,
            "extreme_size_ratio": very_small_ratio + very_large_ratio
        }
        
        suggestions = []
        if optimal_ratio < 0.5:
            suggestions.append("最优大小分块比例较低，建议调整分块策略")
        if very_small_ratio > 0.2:
            suggestions.append("过多小分块，建议合并相邻分块")
        if very_large_ratio > 0.1:
            suggestions.append("存在过大分块，建议进一步分割")
        
        return MetricScore(
            value=min(100.0, score),
            details=details,
            suggestions=suggestions
        )
    
    def _evaluate_boundary_quality(self, content: str, chunks: List[str]) -> MetricScore:
        """
        评估边界质量
        
        Args:
            content: 原始内容
            chunks: 分块列表
            
        Returns:
            MetricScore: 边界质量评分
        """
        score = 0.0
        
        # 检查边界标记的正确性
        if self.chunk_boundary_marker in content:
            boundary_count = content.count(self.chunk_boundary_marker)
            expected_boundaries = len(chunks) - 1
            
            if boundary_count == expected_boundaries:
                score += 40
            elif abs(boundary_count - expected_boundaries) <= 1:
                score += 30
            else:
                score += 10
        else:
            # 如果没有边界标记，检查自然边界
            score += 20
        
        # 检查分块开始和结束的质量
        well_bounded_chunks = 0
        for chunk in chunks:
            if self._is_well_bounded(chunk):
                well_bounded_chunks += 1
        
        boundary_ratio = well_bounded_chunks / len(chunks) if chunks else 0
        score += boundary_ratio * 60
        
        details = {
            "boundary_marker_accuracy": boundary_count == expected_boundaries if self.chunk_boundary_marker in content else None,
            "well_bounded_ratio": boundary_ratio,
            "boundary_issues": self._detect_boundary_issues(chunks)
        }
        
        suggestions = []
        if boundary_ratio < 0.8:
            suggestions.append("边界质量较低，建议在自然边界处分割")
        if details["boundary_issues"]:
            suggestions.append("发现边界问题，需要调整分块策略")
        
        return MetricScore(
            value=min(100.0, score),
            details=details,
            suggestions=suggestions
        )
    
    def _calculate_overall_score(self, metrics: Dict[EvaluationMetric, MetricScore]) -> float:
        """
        计算总体评分
        
        Args:
            metrics: 各项指标评分
            
        Returns:
            float: 总体评分
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in metrics.items():
            weight = self.metric_weights.get(metric, 1.0)
            total_score += score.value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """
        确定质量等级
        
        Args:
            score: 评分
            
        Returns:
            QualityLevel: 质量等级
        """
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 75:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.FAIR
        elif score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _generate_recommendations(self, metrics: Dict[EvaluationMetric, MetricScore], 
                                distribution: Dict[str, int], 
                                analyses: List[ChunkAnalysis]) -> List[str]:
        """
        生成优化建议
        
        Args:
            metrics: 各项指标评分
            distribution: 大小分布
            analyses: 分块分析结果
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 收集各指标的建议
        for metric, score in metrics.items():
            recommendations.extend(score.suggestions)
        
        # 基于分析结果的建议
        problematic_chunks = [a for a in analyses if len(a.issues) > 2]
        if problematic_chunks:
            recommendations.append(f"有{len(problematic_chunks)}个分块存在多个问题，建议重点优化")
        
        # 基于质量等级的总体建议
        overall_score = self._calculate_overall_score(metrics)
        quality_level = self._determine_quality_level(overall_score)
        
        if quality_level == QualityLevel.EXCELLENT:
            recommendations.append("分块质量优秀，可直接用于RAG系统")
        elif quality_level == QualityLevel.GOOD:
            recommendations.append("分块质量良好，建议微调后使用")
        elif quality_level == QualityLevel.FAIR:
            recommendations.append("分块质量一般，建议优化后使用")
        else:
            recommendations.append("分块质量较差，建议重新处理")
        
        # 去重并排序
        recommendations = list(set(recommendations))
        
        return recommendations if recommendations else ["评估完成，未发现明显问题"]
    
    def _generate_detailed_analysis(self, chunks: List[str], analyses: List[ChunkAnalysis], 
                                  metrics: Dict[EvaluationMetric, MetricScore]) -> Dict[str, Any]:
        """
        生成详细分析
        
        Args:
            chunks: 分块列表
            analyses: 分块分析结果
            metrics: 各项指标评分
            
        Returns:
            Dict[str, Any]: 详细分析结果
        """
        return {
            "chunk_count_by_quality": {
                "excellent": len([a for a in analyses if (a.semantic_score + a.structural_score + a.readability_score) / 3 >= 90]),
                "good": len([a for a in analyses if 75 <= (a.semantic_score + a.structural_score + a.readability_score) / 3 < 90]),
                "fair": len([a for a in analyses if 60 <= (a.semantic_score + a.structural_score + a.readability_score) / 3 < 75]),
                "poor": len([a for a in analyses if (a.semantic_score + a.structural_score + a.readability_score) / 3 < 60])
            },
            "common_issues": self._identify_common_issues(analyses),
            "common_strengths": self._identify_common_strengths(analyses),
            "metric_details": {metric.value: score.details for metric, score in metrics.items()},
            "processing_statistics": {
                "total_words": sum(len(chunk.split()) for chunk in chunks),
                "avg_words_per_chunk": sum(len(chunk.split()) for chunk in chunks) / len(chunks) if chunks else 0,
                "total_sentences": sum(len(sent_tokenize(chunk)) for chunk in chunks),
                "avg_sentences_per_chunk": sum(len(sent_tokenize(chunk)) for chunk in chunks) / len(chunks) if chunks else 0
            }
        }
    
    # 辅助方法
    def _is_term_truncated(self, term: str, chunk: str) -> bool:
        """检查术语是否被截断"""
        term_index = chunk.find(term)
        if term_index == -1:
            return True
        
        before_char = chunk[term_index - 1] if term_index > 0 else ' '
        after_char = chunk[term_index + len(term)] if term_index + len(term) < len(chunk) else ' '
        
        return (before_char.isalnum() or '\u4e00' <= before_char <= '\u9fff') and \
               (after_char.isalnum() or '\u4e00' <= after_char <= '\u9fff')
    
    def _has_complete_paragraphs(self, chunk: str) -> bool:
        """检查是否有完整段落"""
        lines = chunk.strip().split('\n')
        if not lines:
            return False
        
        for line in lines:
            if line.strip().endswith(('。', '！', '？', '.', '!', '?')):
                return True
        
        return False
    
    def _has_complete_lists(self, chunk: str) -> bool:
        """检查是否有完整列表"""
        lines = chunk.strip().split('\n')
        list_items = []
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+[\.、]', line) or re.match(r'^[•·\-\*]', line):
                list_items.append(line)
        
        return len(list_items) >= 2 or len(list_items) == 0
    
    def _calculate_coherence(self, chunk: str) -> float:
        """计算内容连贯性"""
        sentences = sent_tokenize(chunk)
        if len(sentences) < 2:
            return 1.0
        
        # 简化的连贯性计算：检查句子间的词汇重叠
        coherence_score = 0.0
        
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                coherence_score += overlap
        
        return coherence_score / (len(sentences) - 1) if len(sentences) > 1 else 1.0
    
    def _check_format_consistency(self, chunk: str) -> bool:
        """检查格式一致性"""
        # 检查标题格式一致性
        headings = re.findall(r'^(#{1,6})\s+(.+)$', chunk, re.MULTILINE)
        if headings:
            # 检查同级标题格式是否一致
            heading_levels = {}
            for level, text in headings:
                if level not in heading_levels:
                    heading_levels[level] = []
                heading_levels[level].append(text)
            
            # 简单检查：同级标题长度不应该差异太大
            for level, texts in heading_levels.items():
                if len(texts) > 1:
                    lengths = [len(text) for text in texts]
                    if max(lengths) - min(lengths) > 50:
                        return False
        
        return True
    
    def _calculate_structure_depth(self, chunk: str) -> int:
        """计算结构深度"""
        headings = re.findall(r'^(#{1,6})', chunk, re.MULTILINE)
        if not headings:
            return 0
        
        levels = [len(h) for h in headings]
        return max(levels) - min(levels) + 1 if levels else 0
    
    def _analyze_punctuation_usage(self, chunk: str) -> float:
        """分析标点符号使用"""
        if not chunk:
            return 0.0
        
        # 计算标点符号密度
        punctuation = '，。！？；：、'
        punct_count = sum(chunk.count(p) for p in punctuation)
        
        # 理想密度约为每100字符5-15个标点
        density = punct_count / len(chunk) * 100
        
        if 5 <= density <= 15:
            return 1.0
        elif 3 <= density <= 20:
            return 0.8
        else:
            return 0.5
    
    def _is_format_clear(self, chunk: str) -> bool:
        """检查格式是否清晰"""
        # 检查是否有合理的段落分隔
        paragraphs = chunk.split('\n\n')
        if len(paragraphs) > 1:
            return True
        
        # 检查是否有列表或标题
        if re.search(r'^#{1,6}\s+', chunk, re.MULTILINE):
            return True
        
        if re.search(r'^\d+[\.、]|^[•·\-\*]', chunk, re.MULTILINE):
            return True
        
        return False
    
    def _has_complete_structure(self, chunk: str) -> bool:
        """检查是否有完整结构"""
        # 检查代码块完整性
        if chunk.count('```') % 2 != 0:
            return False
        
        # 检查列表完整性
        if not self._has_complete_lists(chunk):
            return False
        
        # 检查段落完整性
        if not self._has_complete_paragraphs(chunk):
            return False
        
        return True
    
    def _check_structure_preservation(self, content: str, chunks: List[str]) -> float:
        """检查结构保持程度"""
        # 统计原文档的结构元素
        original_headings = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        original_lists = len(re.findall(r'^\d+[\.、]|^[•·\-\*]', content, re.MULTILINE))
        original_code_blocks = content.count('```') // 2
        
        # 统计分块后的结构元素
        chunk_headings = sum(len(re.findall(r'^#{1,6}\s+', chunk, re.MULTILINE)) for chunk in chunks)
        chunk_lists = sum(len(re.findall(r'^\d+[\.、]|^[•·\-\*]', chunk, re.MULTILINE)) for chunk in chunks)
        chunk_code_blocks = sum(chunk.count('```') // 2 for chunk in chunks)
        
        # 计算保持率
        preservation_scores = []
        
        if original_headings > 0:
            preservation_scores.append(min(1.0, chunk_headings / original_headings))
        
        if original_lists > 0:
            preservation_scores.append(min(1.0, chunk_lists / original_lists))
        
        if original_code_blocks > 0:
            preservation_scores.append(min(1.0, chunk_code_blocks / original_code_blocks))
        
        return sum(preservation_scores) / len(preservation_scores) if preservation_scores else 1.0
    
    def _calculate_medical_term_density(self, chunks: List[str]) -> float:
        """计算医学术语密度"""
        total_words = sum(len(chunk.split()) for chunk in chunks)
        if total_words == 0:
            return 0.0
        
        total_medical_terms = 0
        for chunk in chunks:
            for pattern in self.medical_patterns.values():
                total_medical_terms += len(re.findall(pattern, chunk, re.IGNORECASE))
        
        return min(1.0, total_medical_terms / total_words * 10)  # 归一化到0-1
    
    def _calculate_information_completeness(self, chunks: List[str]) -> float:
        """计算信息完整性"""
        # 简化的信息完整性评估
        completeness_indicators = [
            r'(?:定义|概念|介绍)',  # 定义性信息
            r'(?:症状|表现|特征)',  # 症状信息
            r'(?:治疗|疗法|药物)',  # 治疗信息
            r'(?:诊断|检查|检测)',  # 诊断信息
            r'(?:预防|护理|康复)',  # 预防护理信息
        ]
        
        found_indicators = set()
        for chunk in chunks:
            for i, pattern in enumerate(completeness_indicators):
                if re.search(pattern, chunk, re.IGNORECASE):
                    found_indicators.add(i)
        
        return len(found_indicators) / len(completeness_indicators)
    
    def _is_well_bounded(self, chunk: str) -> bool:
        """检查分块边界是否良好"""
        chunk = chunk.strip()
        if not chunk:
            return False
        
        # 检查开始
        if chunk[0] in '，。！？；：':
            return False
        
        # 检查结束
        if not chunk.endswith(('。', '！', '？', '.', '!', '?', '\n')):
            # 如果不是以句号结尾，检查是否是完整的结构元素
            if not (chunk.endswith('```') or re.search(r'\n\s*$', chunk)):
                return False
        
        return True
    
    def _detect_boundary_issues(self, chunks: List[str]) -> List[str]:
        """检测边界问题"""
        issues = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1}"
            
            if not self._is_well_bounded(chunk):
                issues.append(f"{chunk_id}: 边界不良")
            
            if chunk.strip().startswith(('，', '。', '！', '？')):
                issues.append(f"{chunk_id}: 以标点开始")
            
            if i > 0 and chunk.strip() and chunks[i-1].strip():
                # 检查与前一个分块的连接
                prev_end = chunks[i-1].strip()[-10:]
                curr_start = chunk.strip()[:10]
                
                # 简单的连接性检查
                if not prev_end.endswith(('。', '！', '？', '.', '!', '?')) and \
                   not curr_start.startswith(('#', '##', '###')):
                    issues.append(f"{chunk_id}: 与前一分块连接不自然")
        
        return issues
    
    def _identify_common_issues(self, analyses: List[ChunkAnalysis]) -> List[str]:
        """识别常见问题"""
        issue_counts = {}
        
        for analysis in analyses:
            for issue in analysis.issues:
                issue_type = issue.split('(')[0].strip()  # 提取问题类型
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # 返回出现频率较高的问题
        common_issues = [
            issue for issue, count in issue_counts.items()
            if count > len(analyses) * 0.1  # 超过10%的分块有此问题
        ]
        
        return common_issues
    
    def _identify_common_strengths(self, analyses: List[ChunkAnalysis]) -> List[str]:
        """识别常见优点"""
        strength_counts = {}
        
        for analysis in analyses:
            for strength in analysis.strengths:
                strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        # 返回出现频率较高的优点
        common_strengths = [
            strength for strength, count in strength_counts.items()
            if count > len(analyses) * 0.3  # 超过30%的分块有此优点
        ]
        
        return common_strengths
    
    def print_evaluation_report(self, result: QualityEvaluationResult):
        """
        打印评估报告
        
        Args:
            result: 评估结果
        """
        print("=" * 80)
        print("文档分块质量评估报告")
        print("=" * 80)
        
        if result.file_path:
            print(f"文件路径: {result.file_path}")
            print("-" * 80)
        
        # 基础统计
        print("基础统计:")
        print(f"  总分块数量: {result.total_chunks}")
        print(f"  总字符数: {result.total_chars:,}")
        print(f"  平均分块大小: {result.avg_chunk_size} 字符")
        print(f"  处理时间: {result.processing_time:.2f} 秒")
        print()
        
        # 分块大小分布
        print("分块大小分布:")
        for size_range, count in result.chunk_size_distribution.items():
            percentage = (count / result.total_chunks * 100) if result.total_chunks > 0 else 0
            print(f"  {size_range}: {count} ({percentage:.1f}%)")
        print()
        
        # 各项指标评分
        print("评估指标:")
        for metric, score in result.metrics.items():
            print(f"  {metric.value}: {score.value:.1f}/100")
        print()
        
        # 总体评分和质量等级
        print(f"总体评分: {result.overall_score:.1f}/100")
        print(f"质量等级: {result.quality_level.value.upper()}")
        print()
        
        # 优化建议
        print("优化建议:")
        for i, recommendation in enumerate(result.recommendations, 1):
            print(f"  {i}. {recommendation}")
        print()
        
        # 详细分析
        if result.detailed_analysis:
            print("详细分析:")
            analysis = result.detailed_analysis
            
            if "chunk_count_by_quality" in analysis:
                quality_dist = analysis["chunk_count_by_quality"]
                print(f"  分块质量分布:")
                for level, count in quality_dist.items():
                    print(f"    {level}: {count}")
            
            if "common_issues" in analysis and analysis["common_issues"]:
                print(f"  常见问题: {', '.join(analysis['common_issues'])}")
            
            if "common_strengths" in analysis and analysis["common_strengths"]:
                print(f"  常见优点: {', '.join(analysis['common_strengths'])}")
        
        print("=" * 80)


def main():
    """
    主函数，用于测试质量评估器
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建评估器
    config = {
        'target_chunk_size': 1000,
        'min_chunk_size': 200,
        'max_chunk_size': 2000,
        'language': 'chinese'
    }
    
    evaluator = QualityEvaluator(config)
    
    # 测试内容
    test_content = """
    # 肺癌诊疗指南
    
    ## 概述
    肺癌是最常见的恶性肿瘤之一，根据组织学特点可分为小细胞肺癌和非小细胞肺癌。
    
    [CHUNK_BOUNDARY]
    
    ## 症状表现
    患者可能出现以下症状：
    1. 持续性咳嗽
    2. 胸痛
    3. 呼吸困难
    4. 咯血
    
    早期肺癌症状不明显，容易被忽视。
    
    [CHUNK_BOUNDARY]
    
    ## 治疗方案
    
    ### 手术治疗
    对于早期肺癌患者，手术切除是首选治疗方法。常用的手术方式包括：
    - 肺叶切除术
    - 全肺切除术
    - 楔形切除术
    
    ### 化疗
    化疗是肺癌的重要治疗手段，常用药物包括顺铂75mg/m²、卡铂400mg/m²等。
    
    ```python
    # 药物剂量计算示例
    def calculate_dose(weight, drug_name):
        dose_per_kg = {
            'cisplatin': 75,  # mg/m2
            'carboplatin': 400  # mg/m2
        }
        return weight * dose_per_kg.get(drug_name, 0)
    ```
    """
    
    try:
        # 执行评估
        result = evaluator.evaluate_content(test_content)
        
        # 打印报告
        evaluator.print_evaluation_report(result)
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()