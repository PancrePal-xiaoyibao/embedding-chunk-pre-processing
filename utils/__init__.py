#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块包

提供日志、错误处理、性能监控等通用工具功能。
"""

from .logger import Logger, setup_logging
from .error_handler import ErrorHandler, DocumentProcessingError, ConfigurationError
from .performance_monitor import PerformanceMonitor, performance_monitor
from .file_utils import FileUtils
from .text_utils import TextUtils

__version__ = "1.0.0"
__author__ = "Embedding增强项目团队"

__all__ = [
    'Logger',
    'setup_logging',
    'ErrorHandler',
    'DocumentProcessingError',
    'ConfigurationError',
    'PerformanceMonitor',
    'performance_monitor',
    'FileUtils',
    'TextUtils'
]