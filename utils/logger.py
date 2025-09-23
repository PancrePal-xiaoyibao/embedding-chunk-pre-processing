#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志管理模块

提供统一的日志记录功能，支持多种输出格式和级别控制。
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器
    
    为不同级别的日志添加颜色标识，提升可读性。
    """
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        """
        格式化日志记录
        
        Args:
            record: 日志记录对象
            
        Returns:
            str: 格式化后的日志字符串
        """
        # 获取原始格式化结果
        log_message = super().format(record)
        
        # 添加颜色
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            log_message = f"{color}{log_message}{reset}"
        
        return log_message


class JSONFormatter(logging.Formatter):
    """
    JSON格式化器
    
    将日志记录格式化为JSON格式，便于结构化处理。
    """
    
    def format(self, record):
        """
        格式化日志记录为JSON
        
        Args:
            record: 日志记录对象
            
        Returns:
            str: JSON格式的日志字符串
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


class Logger:
    """
    日志管理器
    
    提供统一的日志记录接口，支持多种输出方式和格式。
    """
    
    def __init__(self, name: str = "EmbeddingEnhancer", log_dir: str = "logs"):
        """
        初始化日志管理器
        
        Args:
            name: 日志器名称
            log_dir: 日志文件目录
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """
        设置日志处理器
        """
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 使用彩色格式化器
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # 文件处理器（普通日志）
        log_file = self.log_dir / f"{self.name.lower()}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 错误日志处理器
        error_file = self.log_dir / f"{self.name.lower()}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # JSON日志处理器
        json_file = self.log_dir / f"{self.name.lower()}.json"
        json_handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONFormatter())
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(json_handler)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        记录调试信息
        
        Args:
            message: 日志消息
            extra_data: 额外数据
        """
        self._log(logging.DEBUG, message, extra_data)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        记录信息
        
        Args:
            message: 日志消息
            extra_data: 额外数据
        """
        self._log(logging.INFO, message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        记录警告信息
        
        Args:
            message: 日志消息
            extra_data: 额外数据
        """
        self._log(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """
        记录错误信息
        
        Args:
            message: 日志消息
            extra_data: 额外数据
            exc_info: 是否包含异常信息
        """
        self._log(logging.ERROR, message, extra_data, exc_info)
    
    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """
        记录严重错误信息
        
        Args:
            message: 日志消息
            extra_data: 额外数据
            exc_info: 是否包含异常信息
        """
        self._log(logging.CRITICAL, message, extra_data, exc_info)
    
    def _log(self, level: int, message: str, extra_data: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """
        内部日志记录方法
        
        Args:
            level: 日志级别
            message: 日志消息
            extra_data: 额外数据
            exc_info: 是否包含异常信息
        """
        # 创建日志记录
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            __file__,
            0,
            message,
            (),
            None if not exc_info else sys.exc_info()
        )
        
        # 添加额外数据
        if extra_data:
            record.extra_data = extra_data
        
        # 处理日志记录
        self.logger.handle(record)
    
    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None, result: Any = None, duration: float = None):
        """
        记录函数调用信息
        
        Args:
            func_name: 函数名称
            args: 位置参数
            kwargs: 关键字参数
            result: 返回结果
            duration: 执行时间
        """
        extra_data = {
            'function': func_name,
            'args_count': len(args) if args else 0,
            'kwargs_count': len(kwargs) if kwargs else 0,
            'has_result': result is not None,
            'duration_ms': round(duration * 1000, 2) if duration else None
        }
        
        message = f"函数调用: {func_name}"
        if duration:
            message += f" (耗时: {duration:.3f}s)"
        
        self.info(message, extra_data)
    
    def log_processing_stats(self, stats: Dict[str, Any]):
        """
        记录处理统计信息
        
        Args:
            stats: 统计数据
        """
        message = "处理统计: " + ", ".join([f"{k}={v}" for k, v in stats.items()])
        self.info(message, {'stats': stats})
    
    def set_level(self, level: str):
        """
        设置日志级别
        
        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'无效的日志级别: {level}')
        
        self.logger.setLevel(numeric_level)
        
        # 更新控制台处理器级别
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(numeric_level)
    
    def get_logger(self) -> logging.Logger:
        """
        获取底层日志器对象
        
        Returns:
            logging.Logger: 日志器对象
        """
        return self.logger


def setup_logging(
    name: str = "EmbeddingEnhancer",
    level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    json_output: bool = True
) -> Logger:
    """
    设置日志系统
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        json_output: 是否输出JSON格式
        
    Returns:
        Logger: 配置好的日志器
    """
    logger = Logger(name, log_dir)
    logger.set_level(level)
    
    # 根据参数调整处理器
    if not console_output:
        # 移除控制台处理器
        for handler in logger.logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                logger.logger.removeHandler(handler)
    
    if not file_output:
        # 移除文件处理器
        for handler in logger.logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                if 'json' not in str(handler.baseFilename):
                    logger.logger.removeHandler(handler)
    
    if not json_output:
        # 移除JSON处理器
        for handler in logger.logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                if 'json' in str(handler.baseFilename):
                    logger.logger.removeHandler(handler)
    
    return logger


# 全局日志器实例
_global_logger = None


def get_logger(name: str = None) -> Logger:
    """
    获取全局日志器实例
    
    Args:
        name: 日志器名称
        
    Returns:
        Logger: 日志器实例
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_logging(name or "EmbeddingEnhancer")
    
    return _global_logger


def main():
    """
    测试日志功能
    """
    # 创建日志器
    logger = setup_logging("TestLogger", "DEBUG")
    
    # 测试各种日志级别
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")
    
    # 测试带额外数据的日志
    logger.info("处理文档", {
        'file_name': 'test.md',
        'file_size': 1024,
        'chunks': 5
    })
    
    # 测试函数调用日志
    logger.log_function_call(
        'process_document',
        args=('test.md',),
        kwargs={'chunk_size': 1000},
        result={'chunks': 5},
        duration=1.23
    )
    
    # 测试统计日志
    logger.log_processing_stats({
        'total_files': 10,
        'total_chunks': 50,
        'total_keywords': 200,
        'processing_time': 15.6
    })
    
    print("日志测试完成，请检查logs目录中的日志文件")


if __name__ == "__main__":
    main()