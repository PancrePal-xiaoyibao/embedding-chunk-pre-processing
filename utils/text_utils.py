#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本处理工具模块

提供文本清理、格式化、分析和处理等实用功能。
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import jieba
import jieba.posseg as pseg

# 可选导入textstat
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    flesch_reading_ease = None
    flesch_kincaid_grade = None

from .logger import get_logger


class TextUtils:
    """
    文本处理工具类
    
    提供各种文本处理和分析功能。
    """
    
    def __init__(self, logger_name: str = "TextUtils"):
        """
        初始化文本工具
        
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
        
        # 初始化jieba
        jieba.setLogLevel(20)  # 减少jieba日志输出
        
        # 常用停用词
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '它', '他', '她', '们', '这个', '那个', '什么', '怎么',
            '为什么', '因为', '所以', '但是', '然后', '如果', '虽然', '虽然', '虽然'
        }
        
        # 标点符号
        self.punctuation = set(string.punctuation + '，。！？；：""''（）【】《》〈〉「」『』〔〕')
    
    def clean_text(
        self,
        text: str,
        remove_extra_whitespace: bool = True,
        remove_special_chars: bool = False,
        normalize_unicode: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True
    ) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            remove_extra_whitespace: 移除多余空白
            remove_special_chars: 移除特殊字符
            normalize_unicode: 标准化Unicode
            remove_urls: 移除URL
            remove_emails: 移除邮箱
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
        
        cleaned_text = text
        
        # 标准化Unicode
        if normalize_unicode:
            cleaned_text = unicodedata.normalize('NFKC', cleaned_text)
        
        # 移除URL
        if remove_urls:
            url_pattern = r'https?://[^\s<>"{}|\\^`[\]]*'
            cleaned_text = re.sub(url_pattern, '', cleaned_text)
        
        # 移除邮箱
        if remove_emails:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            cleaned_text = re.sub(email_pattern, '', cleaned_text)
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        if remove_special_chars:
            pattern = r'[^\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f\w\s.,!?;:()[]{}"""''（）【】《》〈〉「」『』〔〕，。！？；：]'
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 移除多余空白
        if remove_extra_whitespace:
            # 移除多个连续空格
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            # 移除行首行尾空白
            cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        提取句子
        
        Args:
            text: 文本内容
            
        Returns:
            List[str]: 句子列表
        """
        if not text:
            return []
        
        # 中英文句子分割模式
        sentence_pattern = r'[.!?。！？；;]+\s*'
        sentences = re.split(sentence_pattern, text)
        
        # 清理空句子和过短句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:  # 过滤过短的句子
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """
        提取段落
        
        Args:
            text: 文本内容
            
        Returns:
            List[str]: 段落列表
        """
        if not text:
            return []
        
        # 按双换行符分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 清理空段落
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                cleaned_paragraphs.append(paragraph)
        
        return cleaned_paragraphs
    
    def tokenize_chinese(self, text: str, use_pos: bool = False) -> List[str]:
        """
        中文分词
        
        Args:
            text: 中文文本
            use_pos: 是否使用词性标注
            
        Returns:
            List[str]: 分词结果
        """
        if not text:
            return []
        
        if use_pos:
            words = pseg.cut(text)
            return [(word, flag) for word, flag in words if word.strip()]
        else:
            words = jieba.cut(text)
            return [word for word in words if word.strip()]
    
    def remove_stop_words(self, words: List[str], custom_stop_words: Set[str] = None) -> List[str]:
        """
        移除停用词
        
        Args:
            words: 词语列表
            custom_stop_words: 自定义停用词集合
            
        Returns:
            List[str]: 过滤后的词语列表
        """
        stop_words = self.stop_words.copy()
        if custom_stop_words:
            stop_words.update(custom_stop_words)
        
        return [word for word in words if word not in stop_words and word not in self.punctuation]
    
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        min_length: int = 2,
        use_pos_filter: bool = True
    ) -> List[Tuple[str, float]]:
        """
        提取关键词
        
        Args:
            text: 文本内容
            top_k: 返回前k个关键词
            min_length: 最小词长
            use_pos_filter: 是否使用词性过滤
            
        Returns:
            List[Tuple[str, float]]: 关键词及其权重
        """
        if not text:
            return []
        
        # 分词和词性标注
        words_with_pos = self.tokenize_chinese(text, use_pos=True)
        
        # 词性过滤（保留名词、动词、形容词等）
        if use_pos_filter:
            valid_pos = {'n', 'nr', 'ns', 'nt', 'nw', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an'}
            filtered_words = [
                word for word, pos in words_with_pos
                if len(word) >= min_length and any(pos.startswith(p) for p in valid_pos)
            ]
        else:
            filtered_words = [word for word, _ in words_with_pos if len(word) >= min_length]
        
        # 移除停用词
        filtered_words = self.remove_stop_words(filtered_words)
        
        # 计算词频
        word_freq = Counter(filtered_words)
        
        # 计算TF权重
        total_words = len(filtered_words)
        keywords = []
        for word, freq in word_freq.most_common(top_k):
            tf_weight = freq / total_words
            keywords.append((word, tf_weight))
        
        return keywords
    
    def calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        计算文本统计信息
        
        Args:
            text: 文本内容
            
        Returns:
            Dict[str, Any]: 文本统计信息
        """
        if not text:
            return {}
        
        # 基本统计
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        # 句子和段落
        sentences = self.extract_sentences(text)
        paragraphs = self.extract_paragraphs(text)
        
        # 中文分词
        words = self.tokenize_chinese(text)
        words_no_stop = self.remove_stop_words(words)
        
        # 词频统计
        word_freq = Counter(words_no_stop)
        
        # 可读性分析（仅适用于英文）
        readability = {}
        try:
            # 检查是否包含英文且textstat可用
            if HAS_TEXTSTAT and re.search(r'[a-zA-Z]', text):
                readability = {
                    'flesch_reading_ease': flesch_reading_ease(text),
                    'flesch_kincaid_grade': flesch_kincaid_grade(text)
                }
        except:
            pass
        
        statistics = {
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'word_count': len(words),
            'unique_word_count': len(set(words)),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'average_sentence_length': char_count / len(sentences) if sentences else 0,
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'most_common_words': word_freq.most_common(10),
            'readability': readability
        }
        
        return statistics
    
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            str: 语言代码
        """
        if not text:
            return 'unknown'
        
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.3:
            return 'zh'
        elif english_ratio > 0.5:
            return 'en'
        else:
            return 'mixed'
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取命名实体
        
        Args:
            text: 文本内容
            
        Returns:
            Dict[str, List[str]]: 实体类型和实体列表
        """
        entities = {
            'emails': [],
            'urls': [],
            'phone_numbers': [],
            'dates': [],
            'numbers': []
        }
        
        # 邮箱
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # URL
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]*'
        entities['urls'] = re.findall(url_pattern, text)
        
        # 电话号码
        phone_pattern = r'1[3-9]\d{9}|(?:\+86[-\s]?)?(?:\d{3,4}[-\s]?)?\d{7,8}'
        entities['phone_numbers'] = re.findall(phone_pattern, text)
        
        # 日期
        date_patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?',
            r'\d{1,2}[-/月]\d{1,2}[-/日]?',
            r'\d{4}年\d{1,2}月\d{1,2}日'
        ]
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text))
        
        # 数字
        number_pattern = r'\d+(?:\.\d+)?'
        entities['numbers'] = re.findall(number_pattern, text)
        
        return entities
    
    def format_text(
        self,
        text: str,
        line_length: int = 80,
        indent: str = '',
        preserve_paragraphs: bool = True
    ) -> str:
        """
        格式化文本
        
        Args:
            text: 原始文本
            line_length: 行长度
            indent: 缩进字符
            preserve_paragraphs: 保留段落结构
            
        Returns:
            str: 格式化后的文本
        """
        if not text:
            return ""
        
        if preserve_paragraphs:
            paragraphs = self.extract_paragraphs(text)
            formatted_paragraphs = []
            
            for paragraph in paragraphs:
                # 简单的行包装
                words = paragraph.split()
                lines = []
                current_line = indent
                
                for word in words:
                    if len(current_line + word) <= line_length:
                        current_line += word + ' '
                    else:
                        if current_line.strip():
                            lines.append(current_line.rstrip())
                        current_line = indent + word + ' '
                
                if current_line.strip():
                    lines.append(current_line.rstrip())
                
                formatted_paragraphs.append('\n'.join(lines))
            
            return '\n\n'.join(formatted_paragraphs)
        else:
            # 简单格式化
            words = text.split()
            lines = []
            current_line = indent
            
            for word in words:
                if len(current_line + word) <= line_length:
                    current_line += word + ' '
                else:
                    if current_line.strip():
                        lines.append(current_line.rstrip())
                    current_line = indent + word + ' '
            
            if current_line.strip():
                lines.append(current_line.rstrip())
            
            return '\n'.join(lines)
    
    def similarity_score(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度分数（0-1）
        """
        if not text1 or not text2:
            return 0.0
        
        # 分词
        words1 = set(self.tokenize_chinese(text1))
        words2 = set(self.tokenize_chinese(text2))
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def truncate_text(
        self,
        text: str,
        max_length: int,
        truncate_at: str = 'word',
        suffix: str = '...'
    ) -> str:
        """
        截断文本
        
        Args:
            text: 原始文本
            max_length: 最大长度
            truncate_at: 截断位置（'char', 'word', 'sentence'）
            suffix: 后缀
            
        Returns:
            str: 截断后的文本
        """
        if not text or len(text) <= max_length:
            return text
        
        if truncate_at == 'char':
            return text[:max_length - len(suffix)] + suffix
        
        elif truncate_at == 'word':
            words = self.tokenize_chinese(text)
            truncated_words = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + len(suffix) <= max_length:
                    truncated_words.append(word)
                    current_length += len(word)
                else:
                    break
            
            if truncated_words:
                return ''.join(truncated_words) + suffix
            else:
                return text[:max_length - len(suffix)] + suffix
        
        elif truncate_at == 'sentence':
            sentences = self.extract_sentences(text)
            truncated_sentences = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) + len(suffix) <= max_length:
                    truncated_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            if truncated_sentences:
                return '。'.join(truncated_sentences) + '。' + suffix
            else:
                return text[:max_length - len(suffix)] + suffix
        
        return text[:max_length - len(suffix)] + suffix


def main():
    """
    测试文本工具功能
    """
    # 创建文本工具实例
    text_utils = TextUtils()
    
    # 测试文本
    test_text = """
    这是一个测试文档，包含中文和English混合内容。
    
    文档中包含多个段落，每个段落都有不同的内容。我们需要测试各种文本处理功能，
    包括分词、关键词提取、文本清理等。
    
    联系方式：test@example.com，电话：13800138000
    网站：https://www.example.com
    
    这个文档用于测试文本处理工具的各种功能。
    """
    
    print("原始文本:")
    print(test_text)
    print("\n" + "="*50 + "\n")
    
    # 测试文本清理
    cleaned_text = text_utils.clean_text(test_text)
    print("清理后的文本:")
    print(cleaned_text)
    print("\n" + "="*50 + "\n")
    
    # 测试分词
    words = text_utils.tokenize_chinese(cleaned_text)
    print("分词结果:")
    print(words[:20])  # 显示前20个词
    print("\n" + "="*50 + "\n")
    
    # 测试关键词提取
    keywords = text_utils.extract_keywords(cleaned_text, top_k=10)
    print("关键词:")
    for word, weight in keywords:
        print(f"{word}: {weight:.3f}")
    print("\n" + "="*50 + "\n")
    
    # 测试文本统计
    stats = text_utils.calculate_text_statistics(cleaned_text)
    print("文本统计:")
    for key, value in stats.items():
        if key != 'most_common_words':
            print(f"{key}: {value}")
    print("\n最常用词:")
    for word, count in stats.get('most_common_words', []):
        print(f"{word}: {count}")
    print("\n" + "="*50 + "\n")
    
    # 测试实体提取
    entities = text_utils.extract_entities(test_text)
    print("实体提取:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"{entity_type}: {entity_list}")
    print("\n" + "="*50 + "\n")
    
    # 测试语言检测
    language = text_utils.detect_language(test_text)
    print(f"检测到的语言: {language}")
    print("\n" + "="*50 + "\n")
    
    # 测试文本格式化
    formatted_text = text_utils.format_text(cleaned_text, line_length=40)
    print("格式化文本:")
    print(formatted_text)
    print("\n" + "="*50 + "\n")
    
    # 测试文本截断
    truncated_text = text_utils.truncate_text(cleaned_text, max_length=100, truncate_at='sentence')
    print("截断文本:")
    print(truncated_text)
    
    print("\n文本工具测试完成")


if __name__ == "__main__":
    main()