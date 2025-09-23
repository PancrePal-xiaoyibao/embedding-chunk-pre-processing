#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理模块

提供统一的异常处理机制，包括自定义异常类型和错误恢复策略。
"""

import sys
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime
from pathlib import Path
import json

from .logger import get_logger


class BaseEmbeddingError(Exception):
    """
    基础异常类
    
    所有自定义异常的基类，提供统一的错误信息格式。
    """
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 错误信息字典
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """
        字符串表示
        
        Returns:
            str: 错误信息字符串
        """
        return f"[{self.error_code}] {self.message}"


class DocumentProcessingError(BaseEmbeddingError):
    """
    文档处理异常
    
    在文档处理过程中发生的错误。
    """
    pass


class ChunkingError(DocumentProcessingError):
    """
    分块处理异常
    
    在文档分块过程中发生的错误。
    """
    pass


class KeywordExtractionError(DocumentProcessingError):
    """
    关键词提取异常
    
    在关键词提取过程中发生的错误。
    """
    pass


class QualityEvaluationError(DocumentProcessingError):
    """
    质量评估异常
    
    在质量评估过程中发生的错误。
    """
    pass


class ConfigurationError(BaseEmbeddingError):
    """
    配置异常
    
    配置文件或配置参数相关的错误。
    """
    pass


class FileOperationError(BaseEmbeddingError):
    """
    文件操作异常
    
    文件读写操作相关的错误。
    """
    pass


class APIError(BaseEmbeddingError):
    """
    API调用异常
    
    外部API调用相关的错误。
    """
    pass


class ValidationError(BaseEmbeddingError):
    """
    验证异常
    
    数据验证相关的错误。
    """
    pass


class ErrorHandler:
    """
    错误处理器
    
    提供统一的错误处理、记录和恢复机制。
    """
    
    def __init__(self, logger_name: str = "ErrorHandler"):
        """
        初始化错误处理器
        
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
        self.error_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        
        # 注册默认恢复策略
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """
        注册默认的错误恢复策略
        """
        self.recovery_strategies.update({
            FileNotFoundError: self._handle_file_not_found,
            PermissionError: self._handle_permission_error,
            ConfigurationError: self._handle_configuration_error,
            DocumentProcessingError: self._handle_processing_error,
        })
    
    def register_strategy(self, exception_type: Type[Exception], strategy: Callable):
        """
        注册错误恢复策略
        
        Args:
            exception_type: 异常类型
            strategy: 恢复策略函数
        """
        self.recovery_strategies[exception_type] = strategy
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            
        Returns:
            Optional[Any]: 恢复结果
        """
        # 记录错误
        self._log_error(error, context)
        
        # 更新错误统计
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # 尝试恢复
        recovery_result = self._attempt_recovery(error, context)
        
        return recovery_result
    
    def _log_error(self, error: Exception, context: Dict[str, Any] = None):
        """
        记录错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文信息
        """
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        # 添加到历史记录
        self.error_history.append(error_info)
        
        # 记录到日志
        if isinstance(error, BaseEmbeddingError):
            self.logger.error(
                f"应用错误: {error}",
                extra_data=error.to_dict(),
                exc_info=True
            )
        else:
            self.logger.error(
                f"系统错误: {error}",
                extra_data=error_info,
                exc_info=True
            )
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        尝试错误恢复
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            
        Returns:
            Optional[Any]: 恢复结果
        """
        # 查找匹配的恢复策略
        for exception_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exception_type):
                try:
                    self.logger.info(f"尝试恢复错误: {type(error).__name__}")
                    result = strategy(error, context)
                    self.logger.info("错误恢复成功")
                    return result
                except Exception as recovery_error:
                    self.logger.error(f"错误恢复失败: {recovery_error}")
                    break
        
        return None
    
    def _handle_file_not_found(self, error: FileNotFoundError, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        处理文件未找到错误
        
        Args:
            error: 文件未找到异常
            context: 错误上下文
            
        Returns:
            Optional[Any]: 恢复结果
        """
        if context and 'file_path' in context:
            file_path = Path(context['file_path'])
            
            # 尝试在常见位置查找文件
            search_paths = [
                file_path.parent,
                Path.cwd(),
                Path.cwd() / 'data',
                Path.cwd() / 'docs',
                Path.cwd() / 'inputs'
            ]
            
            for search_path in search_paths:
                candidate = search_path / file_path.name
                if candidate.exists():
                    self.logger.info(f"在 {candidate} 找到文件")
                    return str(candidate)
        
        return None
    
    def _handle_permission_error(self, error: PermissionError, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        处理权限错误
        
        Args:
            error: 权限异常
            context: 错误上下文
            
        Returns:
            Optional[Any]: 恢复结果
        """
        if context and 'file_path' in context:
            file_path = Path(context['file_path'])
            
            # 尝试创建备用文件路径
            backup_path = file_path.parent / f"backup_{file_path.name}"
            try:
                # 测试写入权限
                backup_path.touch()
                backup_path.unlink()
                return str(backup_path)
            except Exception:
                pass
        
        return None
    
    def _handle_configuration_error(self, error: ConfigurationError, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        处理配置错误
        
        Args:
            error: 配置异常
            context: 错误上下文
            
        Returns:
            Optional[Any]: 恢复结果
        """
        # 尝试使用默认配置
        if context and 'config_manager' in context:
            config_manager = context['config_manager']
            try:
                config_manager.reset_to_default()
                self.logger.info("已重置为默认配置")
                return config_manager.get_config()
            except Exception:
                pass
        
        return None
    
    def _handle_processing_error(self, error: DocumentProcessingError, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        处理文档处理错误
        
        Args:
            error: 文档处理异常
            context: 错误上下文
            
        Returns:
            Optional[Any]: 恢复结果
        """
        # 尝试使用简化的处理策略
        if context and 'fallback_strategy' in context:
            fallback_strategy = context['fallback_strategy']
            try:
                result = fallback_strategy()
                self.logger.info("使用备用策略处理成功")
                return result
            except Exception:
                pass
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        Returns:
            Dict[str, Any]: 错误统计数据
        """
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def clear_history(self):
        """
        清空错误历史记录
        """
        self.error_history.clear()
        self.error_counts.clear()
        self.logger.info("错误历史记录已清空")
    
    def export_error_report(self, file_path: str):
        """
        导出错误报告
        
        Args:
            file_path: 报告文件路径
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'error_history': self.error_history
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"错误报告已导出到: {file_path}")


def error_handler(
    exceptions: Union[Type[Exception], tuple] = Exception,
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False
):
    """
    错误处理装饰器
    
    Args:
        exceptions: 要捕获的异常类型
        default_return: 默认返回值
        log_error: 是否记录错误
        reraise: 是否重新抛出异常
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    logger = get_logger()
                    logger.error(f"函数 {func.__name__} 执行失败: {e}", exc_info=True)
                
                if reraise:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> Any:
    """
    安全执行函数
    
    Args:
        func: 要执行的函数
        *args: 位置参数
        default_return: 默认返回值
        max_retries: 最大重试次数
        retry_delay: 重试延迟
        **kwargs: 关键字参数
        
    Returns:
        Any: 函数执行结果
    """
    import time
    
    logger = get_logger()
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"函数 {func.__name__} 执行失败，已达最大重试次数: {e}")
                return default_return
            else:
                logger.warning(f"函数 {func.__name__} 执行失败，第 {attempt + 1} 次重试: {e}")
                time.sleep(retry_delay)


def main():
    """
    测试错误处理功能
    """
    # 创建错误处理器
    error_handler = ErrorHandler()
    
    # 测试自定义异常
    try:
        raise DocumentProcessingError(
            "文档处理失败",
            error_code="DOC_001",
            details={'file': 'test.md', 'line': 42}
        )
    except DocumentProcessingError as e:
        error_handler.handle_error(e, {'operation': 'chunk_document'})
    
    # 测试系统异常
    try:
        open('nonexistent_file.txt', 'r')
    except FileNotFoundError as e:
        result = error_handler.handle_error(e, {'file_path': 'nonexistent_file.txt'})
        print(f"恢复结果: {result}")
    
    # 测试装饰器
    @error_handler(FileNotFoundError, default_return="文件未找到")
    def read_file(filename):
        with open(filename, 'r') as f:
            return f.read()
    
    result = read_file('nonexistent.txt')
    print(f"装饰器结果: {result}")
    
    # 测试安全执行
    def risky_function():
        raise ValueError("测试异常")
    
    result = safe_execute(risky_function, default_return="安全返回值")
    print(f"安全执行结果: {result}")
    
    # 显示错误统计
    stats = error_handler.get_error_statistics()
    print(f"错误统计: {stats}")
    
    # 导出错误报告
    error_handler.export_error_report('error_report.json')
    
    print("错误处理测试完成")


if __name__ == "__main__":
    main()