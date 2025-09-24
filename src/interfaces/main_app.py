#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown文档Embedding增强项目 - 主入口文件

本项目通过智能预处理，对指定目录的Markdown文档进行优化，
便于后续的Embedding效果增强，显著提升RAG系统的检索效果。

主要功能：
- 智能分块：基于语义和结构的多策略chunk分割
- 关键词增强：医学专业术语提取和同义词扩展
- 质量保证：全面的chunk质量评估和优化建议
- 用户友好：直观的Web界面和命令行工具
- 高性能：本地化优先，LLM备用的混合处理模式

作者: Embedding增强项目团队
版本: 1.0
更新日期: 2024
"""

import sys
import os
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 添加项目根目录到Python路径
# 当从根目录的main.py调用时，__file__指向src/interfaces/main_app.py
# 需要上溯两级目录到达项目根目录
project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
utils_root = project_root / "utils"

# 添加必要的路径到sys.path
for path in [str(project_root), str(src_root), str(utils_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入项目模块
# 初始化全局变量
setup_logging = None
ErrorHandler = None
PerformanceMonitor = None
ConfigManager = None
DocumentProcessor = None
QualityEvaluator = None
WebInterface = None
CLIInterface = None

# 逐个尝试导入模块，使用fallback机制
modules_imported = []

# 尝试导入ConfigManager
try:
    from config.config_manager import ConfigManager
    modules_imported.append("ConfigManager")
except ImportError:
    try:
        from src.config.config_manager import ConfigManager
        modules_imported.append("ConfigManager")
    except ImportError:
        print("导入错误: No module named 'config_manager'")
        ConfigManager = None

# 尝试导入DocumentProcessor
try:
    from core.document_processor import DocumentProcessor
    modules_imported.append("DocumentProcessor")
except ImportError:
    try:
        from src.core.document_processor import DocumentProcessor
        modules_imported.append("DocumentProcessor")
    except ImportError:
        print("导入错误: No module named 'document_processor'")
        DocumentProcessor = None

# 尝试导入QualityEvaluator
try:
    from core.quality_evaluator import QualityEvaluator
    modules_imported.append("QualityEvaluator")
except ImportError:
    try:
        from src.core.quality_evaluator import QualityEvaluator
        modules_imported.append("QualityEvaluator")
    except ImportError:
        print("导入错误: No module named 'quality_evaluator'")
        QualityEvaluator = None

# 尝试导入WebInterface
try:
    from interfaces.web_interface import WebInterface
    modules_imported.append("WebInterface")
except ImportError:
    try:
        from src.interfaces.web_interface import WebInterface
        modules_imported.append("WebInterface")
    except ImportError:
        print("导入错误: No module named 'web_interface'")
        WebInterface = None

# 尝试导入CLIInterface
try:
    from interfaces.cli_interface import CLIInterface
    modules_imported.append("CLIInterface")
except ImportError:
    try:
        from src.interfaces.cli_interface import CLIInterface
        modules_imported.append("CLIInterface")
    except ImportError:
        print("导入错误: No module named 'cli_interface'")
        CLIInterface = None

# 尝试导入setup_logging
try:
    from logger import setup_logging
    modules_imported.append("setup_logging")
except ImportError:
    try:
        from utils.logger import setup_logging
        modules_imported.append("setup_logging")
    except ImportError:
        print("导入错误: No module named 'logger'")
        setup_logging = None

# 尝试导入ErrorHandler
try:
    from error_handler import ErrorHandler
    modules_imported.append("ErrorHandler")
except ImportError:
    try:
        from utils.error_handler import ErrorHandler
        modules_imported.append("ErrorHandler")
    except ImportError:
        print("导入错误: No module named 'error_handler'")
        ErrorHandler = None

# 尝试导入PerformanceMonitor
try:
    from performance_monitor import PerformanceMonitor
    modules_imported.append("PerformanceMonitor")
except ImportError:
    try:
        from utils.performance_monitor import PerformanceMonitor
        modules_imported.append("PerformanceMonitor")
    except ImportError:
        print("导入错误: No module named 'performance_monitor'")
        PerformanceMonitor = None

# 检查是否有模块导入失败
if len(modules_imported) < 8:
    print(f"警告: 只有 {len(modules_imported)}/8 个模块成功导入: {', '.join(modules_imported)}")
    print("使用最小化fallback模式运行...")
    
    # 创建fallback函数和类
    def setup_logging(level="INFO", log_file=None, log_dir=None, **kwargs):
        """简单的日志设置函数作为fallback"""
        import logging
        import os
        
        # 创建日志目录
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志配置
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if log_file and log_dir:
            log_path = os.path.join(log_dir, log_file)
            logging.basicConfig(
                level=getattr(logging, level.upper(), logging.INFO),
                format=log_format,
                filename=log_path,
                filemode='a'
            )
        else:
            logging.basicConfig(
                level=getattr(logging, level.upper(), logging.INFO),
                format=log_format
            )
        return logging.getLogger("EmbeddingEnhancer")
    
    class ErrorHandler:
        """简单的错误处理器fallback"""
        def __init__(self, name):
            self.name = name
        
        def handle_error(self, error, context=""):
            print(f"错误 [{self.name}]: {error}")
            if context:
                print(f"上下文: {context}")
    
    class PerformanceMonitor:
        """简单的性能监控器fallback"""
        def __init__(self, name):
            self.name = name
        
        def start_timer(self, operation):
            import time
            return time.time()
        
        def end_timer(self, start_time, operation):
            import time
            end_time = time.time()
            duration = end_time - start_time
            print(f"性能监控 [{self.name}]: {operation} 耗时 {duration:.3f}秒")
            return duration
    
    class ConfigManager:
        """简单的配置管理器fallback"""
        def __init__(self, config_path=None):
            self.config_path = config_path
            self.config = {
                "output": {
                    "log_level": "INFO"
                },
                "processing": {
                    "chunk_size": 1000,
                    "overlap": 200
                }
            }
        
        def get_config(self):
            return self.config
    
    class DocumentProcessor:
            """简单的文档处理器fallback"""
            def __init__(self, config_manager=None):
                self.config_manager = config_manager
            
            async def process_document(self, input_file, output_file=None):
                """简单的文档处理fallback"""
                print(f"处理文档: {input_file}")
                if output_file:
                    print(f"输出到: {output_file}")
                
                # 模拟处理结果
                return {
                    "status": "success",
                    "chunks": ["模拟分块1", "模拟分块2"],
                    "keywords": ["关键词1", "关键词2"],
                    "quality_score": 0.8
                }
        
    class QualityEvaluator:
        """简单的质量评估器fallback"""
        def __init__(self, config=None):
            self.config = config
        
        def evaluate_quality(self, document_data):
            return 0.8  # 默认质量分数
        
    class WebInterface:
        """简单的Web界面fallback"""
        def __init__(self, app, **kwargs):
            self.app = app
        
        def run(self, host="127.0.0.1", port=8000):
            print(f"Web界面fallback模式 - 无法启动服务器在 {host}:{port}")
            print("请安装完整的依赖包以使用Web界面")
        
    class CLIInterface:
        """简单的CLI界面fallback"""
        def __init__(self, app, config_manager=None, logger=None, **kwargs):
            self.app = app
            self.config_manager = config_manager
            self.logger = logger
        
        def run(self):
            print("CLI界面fallback模式")
            print("基本功能可用，但某些高级功能可能受限")
            print("请安装完整的依赖包以获得完整功能")
            
            while True:
                try:
                    user_input = input("\n请输入文件路径 (或 'quit' 退出): ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("退出程序")
                        break
                    
                    if user_input:
                        print(f"处理文件: {user_input}")
                        # 这里可以调用app的处理方法
                        print("处理完成 (fallback模式)")
                    else:
                        print("请输入有效的文件路径")
                except KeyboardInterrupt:
                    print("\n\n程序被用户中断")
                    break
                except Exception as e:
                    print(f"发生错误: {e}")


@dataclass
class ProcessingResult:
    """
    文档处理结果数据类
    
    Attributes:
        success: 处理是否成功
        input_file: 输入文件路径
        output_file: 输出文件路径
        chunks_count: 生成的chunk数量
        keywords_count: 提取的关键词数量
        quality_score: 质量评分
        processing_time: 处理时间（秒）
        error_message: 错误信息（如果有）
    """
    success: bool
    input_file: str
    output_file: Optional[str] = None
    chunks_count: int = 0
    keywords_count: int = 0
    quality_score: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None


class EmbeddingEnhancementApp:
    """
    Embedding增强项目主应用类
    
    负责协调各个模块，提供统一的处理接口
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化应用
        
        Args:
            config_path: 配置文件路径
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        self.config_path = config_path
        self.config_manager = None
        self.document_processor = None
        self.quality_evaluator = None
        self.logger = None
        self.error_handler = None
        self.performance_monitor = None
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self):
        """
        初始化各个组件
        
        Raises:
            Exception: 组件初始化失败
        """
        try:
            # 初始化配置管理器
            if os.path.exists(self.config_path):
                self.config_manager = ConfigManager(self.config_path)
            else:
                print(f"警告: 配置文件 {self.config_path} 不存在，使用默认配置")
                self.config_manager = ConfigManager()
            
            # 确保配置的目录存在
            self.config_manager.ensure_directories_exist()
            
            # 初始化日志系统
            log_config = self.config_manager.get_config().get("output", {})
            logs_dir = self.config_manager.get_absolute_path("logs_directory")
            self.logger = setup_logging(
                level=log_config.get("log_level", "INFO"),
                log_dir=logs_dir
            )
            
            # 初始化错误处理器
            self.error_handler = ErrorHandler("EmbeddingEnhancementApp")
            
            # 初始化性能监控器
            self.performance_monitor = PerformanceMonitor("EmbeddingEnhancementApp")
            
            # 初始化文档处理器
            self.document_processor = DocumentProcessor(
                config_manager=self.config_manager
            )
            
            # 初始化质量评估器
            self.quality_evaluator = QualityEvaluator(
                config=self.config_manager.get_config()
            )
            
            self.logger.info("应用组件初始化完成")
            
        except Exception as e:
            error_msg = f"组件初始化失败: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            raise
    
    async def process_single_file(self, input_file: str, output_dir: Optional[str] = None) -> ProcessingResult:
        """
        处理单个文件
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录（可选）
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = self.performance_monitor.start_timer()
        
        try:
            self.logger.info(f"开始处理文件: {input_file}")
            
            # 验证输入文件
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"输入文件不存在: {input_file}")
            
            # 确定输出目录
            if output_dir is None:
                # 使用配置中的默认输出目录
                output_dir = self.config_manager.get_absolute_path("output_directory")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理文档
            processing_result = self.document_processor.process_file(
                file_path=input_file,
                strategy="semantic",
                extract_keywords=True,
                evaluate_quality=True
            )
            
            # 质量评估
            if processing_result.success:
                quality_result = self.quality_evaluator.evaluate_file(input_file)
                quality_score = quality_result.overall_score
            else:
                quality_score = 0.0
            
            # 记录处理时间
            processing_time = self.performance_monitor.end_timer(start_time)
            
            # 生成输出文件路径
            output_file = os.path.join(output_dir, f"{os.path.basename(input_file).split('.')[0]}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            # 保存处理结果到输出文件
            if processing_result.success and processing_result.chunks:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(processing_result.chunks, 1):
                        f.write(f"=== 分块 {i} ===\n")
                        f.write(f"大小: {chunk.size} 字符\n")
                        f.write(f"关键词: {', '.join(chunk.keywords)}\n")
                        f.write(f"质量评分: {chunk.quality_score:.2f}\n")
                        f.write(f"内容:\n{chunk.content}\n\n")
                self.logger.info(f"文件处理完成: {input_file}, 耗时: {processing_time:.2f}秒, 输出文件: {output_file}")
            else:
                output_file = None
                self.logger.info(f"文件处理完成: {input_file}, 耗时: {processing_time:.2f}秒")
            
            # 创建返回结果
            return ProcessingResult(
                success=processing_result.success,
                input_file=input_file,
                output_file=output_file,  # 添加输出文件路径
                chunks_count=processing_result.total_chunks,
                keywords_count=processing_result.total_keywords,
                quality_score=quality_score,
                processing_time=processing_time,
                error_message=processing_result.error_message if not processing_result.success else None
            )
            
        except Exception as e:
            error_msg = f"处理文件失败 {input_file}: {e}"
            self.logger.error(error_msg)
            
            processing_time = self.performance_monitor.end_timer(start_time)
            
            return ProcessingResult(
                success=False,
                input_file=input_file,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def process_directory(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> List[ProcessingResult]:
        """
        批量处理目录中的文件
        
        Args:
            input_dir: 输入目录路径（可选，默认使用配置中的输入目录）
            output_dir: 输出目录（可选，默认使用配置中的输出目录）
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        try:
            # 确定输入目录
            if input_dir is None:
                input_dir = self.config_manager.get_absolute_path("input_directory")
            
            self.logger.info(f"开始批量处理目录: {input_dir}")
            
            # 查找Markdown文件
            input_path = Path(input_dir)
            if not input_path.exists():
                self.logger.error(f"输入目录不存在: {input_dir}")
                return []
                
            md_files = list(input_path.glob("*.md"))
            
            if not md_files:
                self.logger.warning(f"目录中未找到Markdown文件: {input_dir}")
                return []
            
            # 确定输出目录
            if output_dir is None:
                output_dir = self.config_manager.get_absolute_path("output_directory")
            
            # 并发处理文件
            tasks = []
            for md_file in md_files:
                task = self.process_single_file(str(md_file), output_dir)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = ProcessingResult(
                        success=False,
                        input_file=str(md_files[i]),
                        error_message=str(result)
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            # 统计结果
            success_count = sum(1 for r in processed_results if r.success)
            total_count = len(processed_results)
            
            self.logger.info(f"批量处理完成: {success_count}/{total_count} 文件处理成功")
            
            return processed_results
            
        except Exception as e:
            error_msg = f"批量处理失败: {e}"
            self.logger.error(error_msg)
            raise
    
    def start_web_interface(self, host: str = "127.0.0.1", port: int = 8000):
        """
        启动Web界面
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
        """
        try:
            self.logger.info(f"启动Web界面: http://{host}:{port}")
            
            web_interface = WebInterface(
                config_path=self.config_path
            )
            
            web_interface.run(host=host, port=port)
            
        except Exception as e:
            error_msg = f"Web界面启动失败: {e}"
            self.logger.error(error_msg)
            raise
    
    def start_cli_interface(self):
        """
        启动命令行界面
        """
        try:
            self.logger.info("启动命令行界面")
            
            cli_interface = CLIInterface(
                app=self,
                config_manager=self.config_manager,
                logger=self.logger
            )
            
            cli_interface.run()
            
        except Exception as e:
            error_msg = f"命令行界面启动失败: {e}"
            self.logger.error(error_msg)
            raise


def main():
    """
    主函数，解析命令行参数并启动应用
    
    命令行用法：
        python main_app.py --mode web                    # 启动Web界面
        python main_app.py --mode cli                    # 启动命令行界面
        python main_app.py --file input.md              # 处理单个文件
        python main_app.py --dir input_dir              # 批量处理目录
    """
    parser = argparse.ArgumentParser(
        description="Markdown文档Embedding增强项目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --mode web                    # 启动Web界面
  %(prog)s --mode cli                    # 启动命令行界面
  %(prog)s --file input.md              # 处理单个文件
  %(prog)s --dir To_be_processed        # 批量处理目录
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["web", "cli"], 
        help="运行模式：web界面或命令行界面"
    )
    
    parser.add_argument(
        "--file", 
        help="处理单个文件的路径"
    )
    
    parser.add_argument(
        "--dir", 
        help="批量处理目录的路径"
    )
    
    parser.add_argument(
        "--output", 
        help="输出目录路径（可选）"
    )
    
    parser.add_argument(
        "--config", 
        default="config.json",
        help="配置文件路径（默认: config.json）"
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Web服务器主机地址（默认: 127.0.0.1）"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Web服务器端口（默认: 8000）"
    )
    
    args = parser.parse_args()
    
    # 如果没有提供任何参数，显示帮助信息
    if not any([args.mode, args.file, args.dir]):
        parser.print_help()
        return
    
    try:
        # 初始化应用
        app = EmbeddingEnhancementApp(config_path=args.config)
        
        # 根据参数执行相应操作
        if args.mode == "web":
            app.start_web_interface(host=args.host, port=args.port)
            
        elif args.mode == "cli":
            app.start_cli_interface()
            
        elif args.file:
            # 处理单个文件
            result = asyncio.run(app.process_single_file(args.file, args.output))
            
            if result.success:
                print(f"✅ 文件处理成功!")
                print(f"   输入文件: {result.input_file}")
                print(f"   输出文件: {result.output_file}")
                print(f"   质量评分: {result.quality_score:.1f}")
                print(f"   处理时间: {result.processing_time:.1f}s")
                print(f"   分块数量: {result.chunks_count}")
                print(f"   关键词数量: {result.keywords_count}")
            else:
                print(f"❌ 文件处理失败!")
                print(f"   输入文件: {result.input_file}")
                print(f"   错误信息: {result.error_message}")
                sys.exit(1)
                
        elif args.dir:
            # 批量处理目录
            results = asyncio.run(app.process_directory(args.dir, args.output))
            
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            
            print(f"\n📊 批量处理完成:")
            print(f"   总文件数: {total_count}")
            print(f"   成功处理: {success_count}")
            print(f"   失败处理: {total_count - success_count}")
            
            # 显示详细结果
            for result in results:
                status = "✅" if result.success else "❌"
                filename = os.path.basename(result.input_file)
                if result.success:
                    print(f"   {status} {filename} - {result.quality_score:.1f}分 ({result.processing_time:.1f}s)")
                else:
                    print(f"   {status} {filename} - {result.error_message}")
            
            if success_count < total_count:
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ 应用运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()