#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控模块

提供系统性能监控、函数执行时间统计和资源使用情况跟踪。
"""

import time
import psutil
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import gc

from .logger import get_logger


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类
    
    记录函数或操作的性能指标。
    """
    name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta(self) -> float:
        """
        内存变化量
        
        Returns:
            float: 内存变化量（MB）
        """
        return self.memory_after - self.memory_before
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 性能指标字典
        """
        return {
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_peak': self.memory_peak,
            'memory_delta': self.memory_delta,
            'cpu_percent': self.cpu_percent,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class SystemMetrics:
    """
    系统指标数据类
    
    记录系统资源使用情况。
    """
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: float
    memory_available: float
    disk_usage: float
    network_sent: int = 0
    network_recv: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 系统指标字典
        """
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used': self.memory_used,
            'memory_available': self.memory_available,
            'disk_usage': self.disk_usage,
            'network_sent': self.network_sent,
            'network_recv': self.network_recv
        }


class PerformanceMonitor:
    """
    性能监控器
    
    提供函数执行时间监控、系统资源监控和性能分析功能。
    """
    
    def __init__(self, logger_name: str = "PerformanceMonitor"):
        """
        初始化性能监控器
        
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
        self.metrics_history: List[PerformanceMetrics] = []
        self.system_metrics: deque = deque(maxlen=1000)
        self.function_stats: Dict[str, List[float]] = defaultdict(list)
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # 监控间隔（秒）
        
        # 性能阈值
        self.thresholds = {
            'execution_time': 10.0,  # 执行时间阈值（秒）
            'memory_usage': 100.0,   # 内存使用阈值（MB）
            'cpu_usage': 80.0,       # CPU使用率阈值（%）
            'memory_percent': 90.0   # 内存使用率阈值（%）
        }
        
    def start_timer(self, operation: str = None) -> float:
        """
        开始计时器
        
        Args:
            operation: 操作名称（可选）
            
        Returns:
            float: 开始时间戳
        """
        start_time = time.time()
        if operation:
            self.logger.debug(f"开始计时: {operation}")
        return start_time
        
    def end_timer(self, start_time: float, operation: str = None) -> float:
        """
        结束计时器并计算耗时
        
        Args:
            start_time: 开始时间戳
            operation: 操作名称（可选）
            
        Returns:
            float: 耗时（秒）
        """
        end_time = time.time()
        duration = end_time - start_time
        
        if operation:
            self.logger.info(f"操作耗时: {operation} - {duration:.3f}秒")
        
        return duration
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始系统监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_active:
            self.logger.warning("监控已在运行中")
            return
        
        self.monitor_interval = interval
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"开始系统监控，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """
        停止系统监控
        """
        if not self.monitoring_active:
            self.logger.warning("监控未在运行")
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("系统监控已停止")
    
    def _monitor_system(self):
        """
        系统监控线程函数
        """
        while self.monitoring_active:
            try:
                # 获取系统指标
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 网络统计（可选）
                network = psutil.net_io_counters()
                
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used=memory.used / 1024 / 1024,  # MB
                    memory_available=memory.available / 1024 / 1024,  # MB
                    disk_usage=disk.percent,
                    network_sent=network.bytes_sent,
                    network_recv=network.bytes_recv
                )
                
                self.system_metrics.append(metrics)
                
                # 检查阈值
                self._check_thresholds(metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"系统监控错误: {e}")
                time.sleep(self.monitor_interval)
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """
        检查性能阈值
        
        Args:
            metrics: 系统指标
        """
        if metrics.cpu_percent > self.thresholds['cpu_usage']:
            self.logger.warning(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            self.logger.warning(f"内存使用率过高: {metrics.memory_percent:.1f}%")
    
    def measure_function(self, func: Callable, *args, **kwargs) -> tuple:
        """
        测量函数执行性能
        
        Args:
            func: 要测量的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            tuple: (函数结果, 性能指标)
        """
        # 获取初始状态
        start_time = time.time()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 强制垃圾回收
        gc.collect()
        
        success = True
        error_message = ""
        result = None
        memory_peak = memory_before
        
        try:
            # 执行函数
            result = func(*args, **kwargs)
            
            # 监控内存峰值
            memory_current = process.memory_info().rss / 1024 / 1024
            memory_peak = max(memory_peak, memory_current)
            
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(f"函数执行失败: {func.__name__}: {e}")
        
        # 获取结束状态
        end_time = time.time()
        duration = end_time - start_time
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = psutil.cpu_percent()
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            name=func.__name__,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=memory_peak,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message,
            metadata={
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
        )
        
        # 记录指标
        self.metrics_history.append(metrics)
        self.function_stats[func.__name__].append(duration)
        
        # 检查性能阈值
        if duration > self.thresholds['execution_time']:
            self.logger.warning(f"函数执行时间过长: {func.__name__}: {duration:.2f}秒")
        
        if abs(metrics.memory_delta) > self.thresholds['memory_usage']:
            self.logger.warning(f"函数内存使用异常: {func.__name__}: {metrics.memory_delta:.2f}MB")
        
        return result, metrics
    
    def get_function_statistics(self, function_name: str = None) -> Dict[str, Any]:
        """
        获取函数性能统计
        
        Args:
            function_name: 函数名称，为None时返回所有函数统计
            
        Returns:
            Dict[str, Any]: 函数性能统计
        """
        if function_name:
            if function_name not in self.function_stats:
                return {}
            
            durations = self.function_stats[function_name]
            return {
                'function_name': function_name,
                'call_count': len(durations),
                'total_time': sum(durations),
                'average_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'recent_calls': durations[-10:]
            }
        else:
            stats = {}
            for func_name, durations in self.function_stats.items():
                stats[func_name] = {
                    'call_count': len(durations),
                    'total_time': sum(durations),
                    'average_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations)
                }
            return stats
    
    def get_system_statistics(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """
        获取系统性能统计
        
        Args:
            duration_minutes: 统计时间范围（分钟）
            
        Returns:
            Dict[str, Any]: 系统性能统计
        """
        if not self.system_metrics:
            return {}
        
        # 过滤指定时间范围内的数据
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # 计算统计值
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        memory_used_values = [m.memory_used for m in recent_metrics]
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory': {
                'average_percent': sum(memory_values) / len(memory_values),
                'min_percent': min(memory_values),
                'max_percent': max(memory_values),
                'current_percent': memory_values[-1] if memory_values else 0,
                'average_used_mb': sum(memory_used_values) / len(memory_used_values),
                'current_used_mb': memory_used_values[-1] if memory_used_values else 0
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取完整性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'total_function_calls': len(self.metrics_history),
            'function_statistics': self.get_function_statistics(),
            'system_statistics': self.get_system_statistics(),
            'thresholds': self.thresholds.copy(),
            'recent_metrics': [m.to_dict() for m in self.metrics_history[-10:]]
        }
    
    def export_report(self, file_path: str):
        """
        导出性能报告
        
        Args:
            file_path: 报告文件路径
        """
        report = self.get_performance_report()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"性能报告已导出到: {file_path}")
    
    def clear_history(self):
        """
        清空性能历史记录
        """
        self.metrics_history.clear()
        self.function_stats.clear()
        self.system_metrics.clear()
        self.logger.info("性能历史记录已清空")
    
    def set_thresholds(self, **thresholds):
        """
        设置性能阈值
        
        Args:
            **thresholds: 阈值参数
        """
        for key, value in thresholds.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                self.logger.info(f"阈值已更新: {key} = {value}")
            else:
                self.logger.warning(f"未知阈值参数: {key}")


def performance_monitor(
    log_result: bool = True,
    include_args: bool = False,
    threshold_seconds: float = None
):
    """
    性能监控装饰器
    
    Args:
        log_result: 是否记录结果
        include_args: 是否包含参数信息
        threshold_seconds: 性能阈值（秒）
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            result, metrics = monitor.measure_function(func, *args, **kwargs)
            
            if log_result:
                logger = get_logger()
                log_data = {
                    'function': func.__name__,
                    'duration': metrics.duration,
                    'memory_delta': metrics.memory_delta,
                    'success': metrics.success
                }
                
                if include_args:
                    log_data['args_count'] = len(args)
                    log_data['kwargs_count'] = len(kwargs)
                
                if threshold_seconds and metrics.duration > threshold_seconds:
                    logger.warning(f"函数执行超时: {func.__name__}", extra_data=log_data)
                else:
                    logger.debug(f"函数执行完成: {func.__name__}", extra_data=log_data)
            
            return result
        
        return wrapper
    return decorator


def benchmark(func: Callable, iterations: int = 100, *args, **kwargs) -> Dict[str, Any]:
    """
    函数基准测试
    
    Args:
        func: 要测试的函数
        iterations: 测试迭代次数
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        Dict[str, Any]: 基准测试结果
    """
    monitor = PerformanceMonitor()
    durations = []
    memory_deltas = []
    success_count = 0
    
    logger = get_logger()
    logger.info(f"开始基准测试: {func.__name__}, 迭代次数: {iterations}")
    
    for i in range(iterations):
        try:
            result, metrics = monitor.measure_function(func, *args, **kwargs)
            durations.append(metrics.duration)
            memory_deltas.append(metrics.memory_delta)
            if metrics.success:
                success_count += 1
        except Exception as e:
            logger.error(f"基准测试第 {i+1} 次迭代失败: {e}")
    
    if not durations:
        return {'error': '所有迭代都失败了'}
    
    # 计算统计值
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    avg_memory = sum(memory_deltas) / len(memory_deltas)
    
    result = {
        'function_name': func.__name__,
        'iterations': iterations,
        'success_rate': success_count / iterations,
        'duration': {
            'average': avg_duration,
            'min': min_duration,
            'max': max_duration,
            'total': sum(durations)
        },
        'memory': {
            'average_delta': avg_memory,
            'min_delta': min(memory_deltas),
            'max_delta': max(memory_deltas)
        },
        'performance_score': 1.0 / avg_duration if avg_duration > 0 else 0
    }
    
    logger.info(f"基准测试完成: {func.__name__}, 平均耗时: {avg_duration:.4f}秒")
    return result


def main():
    """
    测试性能监控功能
    """
    # 创建性能监控器
    monitor = PerformanceMonitor()
    
    # 开始系统监控
    monitor.start_monitoring(interval=0.5)
    
    # 测试函数性能监控
    @performance_monitor(log_result=True, include_args=True)
    def test_function(n: int):
        """测试函数"""
        time.sleep(0.1)
        return sum(range(n))
    
    # 执行测试
    result = test_function(1000)
    print(f"测试函数结果: {result}")
    
    # 直接测量函数
    def slow_function():
        time.sleep(0.2)
        return "完成"
    
    result, metrics = monitor.measure_function(slow_function)
    print(f"慢函数结果: {result}")
    print(f"性能指标: {metrics.to_dict()}")
    
    # 基准测试
    def quick_function(x):
        return x * x
    
    benchmark_result = benchmark(quick_function, iterations=1000, x=42)
    print(f"基准测试结果: {benchmark_result}")
    
    # 等待一段时间收集系统指标
    time.sleep(2)
    
    # 获取统计信息
    func_stats = monitor.get_function_statistics()
    print(f"函数统计: {func_stats}")
    
    system_stats = monitor.get_system_statistics(duration_minutes=1)
    print(f"系统统计: {system_stats}")
    
    # 导出报告
    monitor.export_report('performance_report.json')
    
    # 停止监控
    monitor.stop_monitoring()
    
    print("性能监控测试完成")


if __name__ == "__main__":
    main()