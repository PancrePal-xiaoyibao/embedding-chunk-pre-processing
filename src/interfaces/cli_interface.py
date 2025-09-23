#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行界面模块

提供完整的命令行界面功能，支持文档处理、配置管理、质量评估等操作。
包含丰富的命令行参数、进度显示、结果输出等功能。
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

# 导入项目模块
try:
    # 新的src目录结构导入
    from core.document_processor import DocumentProcessor, ProcessingResult
    from config.config_manager import ConfigManager
    from core.quality_evaluator import QualityEvaluator, QualityEvaluationResult
    from core.chunk_strategies import BaseChunkingStrategy, ChunkingConfig
except ImportError:
    # 向后兼容的导入
    try:
        from document_processor import DocumentProcessor, ProcessingResult
        from config_manager import ConfigManager
        from quality_evaluator import QualityEvaluator, QualityEvaluationResult
        from chunk_strategies import BaseChunkingStrategy, ChunkingConfig
    except ImportError as e:
        print(f"导入错误: {e}")
        sys.exit(1)


class CLIInterface:
    """
    命令行界面类
    
    提供完整的命令行操作功能，包括文档处理、配置管理、质量评估等。
    """
    
    def __init__(self, config_path: str = "config.json", app=None, config_manager=None, logger=None):
        """
        初始化命令行界面
        
        Args:
            config_path: 配置文件路径
            app: 主应用实例（可选）
            config_manager: 配置管理器实例（可选）
            logger: 日志记录器（可选）
        """
        self.config_path = config_path
        self.app = app
        
        # 使用提供的config_manager或创建新的
        if config_manager:
            self.config_manager = config_manager
        else:
            self.config_manager = ConfigManager(config_path)
        
        # 使用提供的logger或创建新的
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化处理器
        self.document_processor = DocumentProcessor(self.config_manager)
        self.quality_evaluator = QualityEvaluator(self.config_manager.config)
        
        # 创建输出目录
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """
        创建命令行参数解析器
        
        Returns:
            argparse.ArgumentParser: 参数解析器
        """
        parser = argparse.ArgumentParser(
            description="Embedding增强系统 - 命令行界面",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  %(prog)s process document.md                    # 处理单个文档
  %(prog)s process docs/ -r                       # 递归处理目录
  %(prog)s evaluate document.md                   # 评估文档质量
  %(prog)s config --show                          # 显示当前配置
  %(prog)s config --set chunk_size=1000           # 设置配置项
  %(prog)s batch files.txt                        # 批量处理文件列表
            """
        )
        
        # 全局参数
        parser.add_argument(
            '--config', '-c',
            default='config.json',
            help='配置文件路径 (默认: config.json)'
        )
        
        parser.add_argument(
            '--output', '-o',
            default='outputs',
            help='输出目录 (默认: outputs)'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='详细输出模式'
        )
        
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='静默模式，只输出错误信息'
        )
        
        parser.add_argument(
            '--format',
            choices=['text', 'json', 'csv'],
            default='text',
            help='输出格式 (默认: text)'
        )
        
        # 子命令
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 处理命令
        process_parser = subparsers.add_parser(
            'process',
            help='处理文档',
            description='处理Markdown文档，进行分块和关键词提取'
        )
        
        process_parser.add_argument(
            'input',
            help='输入文件或目录路径'
        )
        
        process_parser.add_argument(
            '--recursive', '-r',
            action='store_true',
            help='递归处理目录中的所有文件'
        )
        
        process_parser.add_argument(
            '--strategy',
            choices=['token', 'semantic', 'structured', 'hybrid'],
            help='分块策略'
        )
        
        process_parser.add_argument(
            '--chunk-size',
            type=int,
            help='目标分块大小'
        )
        
        process_parser.add_argument(
            '--keywords',
            type=int,
            help='每个分块的最大关键词数'
        )
        
        process_parser.add_argument(
            '--no-evaluation',
            action='store_true',
            help='跳过质量评估'
        )
        
        process_parser.add_argument(
            '--save-intermediate',
            action='store_true',
            help='保存中间处理结果'
        )
        
        # 评估命令
        evaluate_parser = subparsers.add_parser(
            'evaluate',
            help='评估文档质量',
            description='评估文档分块质量并提供优化建议'
        )
        
        evaluate_parser.add_argument(
            'input',
            help='输入文件路径'
        )
        
        evaluate_parser.add_argument(
            '--detailed',
            action='store_true',
            help='显示详细评估结果'
        )
        
        evaluate_parser.add_argument(
            '--export',
            help='导出评估结果到文件'
        )
        
        # 配置命令
        config_parser = subparsers.add_parser(
            'config',
            help='配置管理',
            description='管理系统配置参数'
        )
        
        config_group = config_parser.add_mutually_exclusive_group(required=True)
        
        config_group.add_argument(
            '--show',
            action='store_true',
            help='显示当前配置'
        )
        
        config_group.add_argument(
            '--set',
            action='append',
            metavar='KEY=VALUE',
            help='设置配置项 (可多次使用)'
        )
        
        config_group.add_argument(
            '--reset',
            action='store_true',
            help='重置为默认配置'
        )
        
        config_group.add_argument(
            '--validate',
            action='store_true',
            help='验证配置文件'
        )
        
        # 批量处理命令
        batch_parser = subparsers.add_parser(
            'batch',
            help='批量处理',
            description='批量处理多个文件'
        )
        
        batch_parser.add_argument(
            'filelist',
            help='包含文件路径列表的文本文件'
        )
        
        batch_parser.add_argument(
            '--parallel',
            type=int,
            default=1,
            help='并行处理数量 (默认: 1)'
        )
        
        batch_parser.add_argument(
            '--continue-on-error',
            action='store_true',
            help='遇到错误时继续处理其他文件'
        )
        
        # 统计命令
        stats_parser = subparsers.add_parser(
            'stats',
            help='统计信息',
            description='显示处理统计信息'
        )
        
        stats_parser.add_argument(
            'input',
            nargs='?',
            help='输入目录或结果文件'
        )
        
        stats_parser.add_argument(
            '--summary',
            action='store_true',
            help='显示汇总统计'
        )
        
        # 工具命令
        tools_parser = subparsers.add_parser(
            'tools',
            help='实用工具',
            description='提供各种实用工具'
        )
        
        tools_subparsers = tools_parser.add_subparsers(dest='tool', help='可用工具')
        
        # 清理工具
        clean_parser = tools_subparsers.add_parser('clean', help='清理临时文件')
        clean_parser.add_argument('--all', action='store_true', help='清理所有文件')
        
        # 转换工具
        convert_parser = tools_subparsers.add_parser('convert', help='格式转换')
        convert_parser.add_argument('input', help='输入文件')
        convert_parser.add_argument('--to', choices=['json', 'csv', 'txt'], required=True, help='目标格式')
        
        return parser
    
    def setup_logging(self, verbose: bool = False, quiet: bool = False):
        """
        设置日志配置
        
        Args:
            verbose: 详细模式
            quiet: 静默模式
        """
        if quiet:
            level = logging.ERROR
        elif verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def print_progress(self, current: int, total: int, message: str = ""):
        """
        打印进度条
        
        Args:
            current: 当前进度
            total: 总数
            message: 附加消息
        """
        if total == 0:
            return
        
        percent = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current // total)
        
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r进度: |{bar}| {percent:.1f}% ({current}/{total}) {message}', end='', flush=True)
        
        if current == total:
            print()  # 换行
    
    def format_output(self, data: Any, format_type: str = 'text') -> str:
        """
        格式化输出数据
        
        Args:
            data: 要格式化的数据
            format_type: 输出格式
            
        Returns:
            str: 格式化后的字符串
        """
        if format_type == 'json':
            return json.dumps(data, ensure_ascii=False, indent=2, default=str)
        elif format_type == 'csv':
            # 简单的CSV格式化
            if isinstance(data, dict):
                lines = ['key,value']
                for k, v in data.items():
                    lines.append(f'"{k}","{v}"')
                return '\n'.join(lines)
            else:
                return str(data)
        else:
            # 文本格式
            if isinstance(data, dict):
                lines = []
                for k, v in data.items():
                    if isinstance(v, dict):
                        lines.append(f"{k}:")
                        for sub_k, sub_v in v.items():
                            lines.append(f"  {sub_k}: {sub_v}")
                    else:
                        lines.append(f"{k}: {v}")
                return '\n'.join(lines)
            else:
                return str(data)
    
    def process_command(self, args) -> int:
        """
        处理process命令
        
        Args:
            args: 命令行参数
            
        Returns:
            int: 退出码
        """
        try:
            input_path = Path(args.input)
            
            if not input_path.exists():
                print(f"❌ 错误: 文件或目录不存在: {input_path}")
                return 1
            
            # 更新配置
            if args.strategy:
                self.config_manager.set_value('chunking_strategies.default_strategy', args.strategy)
            if args.chunk_size:
                self.config_manager.set_value('chunk_processing.target_chunk_size', args.chunk_size)
            if args.keywords:
                self.config_manager.set_value('keyword_extraction.max_keywords_per_chunk', args.keywords)
            
            # 重新初始化处理器
            self.document_processor = DocumentProcessor(self.config_manager)
            
            files_to_process = []
            
            if input_path.is_file():
                files_to_process.append(input_path)
            elif input_path.is_dir():
                if args.recursive:
                    # 递归查找所有支持的文件
                    extensions = {'.md', '.markdown', '.txt'}
                    for ext in extensions:
                        files_to_process.extend(input_path.rglob(f'*{ext}'))
                else:
                    # 只处理当前目录的文件
                    extensions = {'.md', '.markdown', '.txt'}
                    for file_path in input_path.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in extensions:
                            files_to_process.append(file_path)
            
            if not files_to_process:
                print("❌ 错误: 没有找到可处理的文件")
                return 1
            
            print(f"📝 找到 {len(files_to_process)} 个文件待处理")
            
            # 处理文件
            results = []
            for i, file_path in enumerate(files_to_process, 1):
                self.print_progress(i-1, len(files_to_process), f"处理 {file_path.name}")
                
                try:
                    # 处理文档
                    result = self.document_processor.process_file(str(file_path))
                    
                    # 生成输出文件
                    output_base = self.output_dir / f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # 保存分块结果
                    chunks_file = output_base.with_suffix('.chunks.txt')
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        for j, chunk in enumerate(result.chunks, 1):
                            f.write(f"=== 分块 {j} ===\n")
                            f.write(f"大小: {len(chunk.content)} 字符\n")
                            f.write(f"关键词: {', '.join(chunk.keywords)}\n")
                            f.write(f"质量评分: {chunk.quality_score:.2f}\n")
                            f.write(f"内容:\n{chunk.content}\n\n")
                    
                    # 保存关键词结果
                    if args.save_intermediate:
                        keywords_file = output_base.with_suffix('.keywords.json')
                        keywords_data = {
                            'total_keywords': result.total_keywords,
                            'keywords_by_chunk': [
                                {'chunk_id': j+1, 'keywords': chunk.keywords}
                                for j, chunk in enumerate(result.chunks)
                            ],
                            'keyword_frequency': {}  # ProcessingResult没有keyword_frequency属性
                        }
                        
                        with open(keywords_file, 'w', encoding='utf-8') as f:
                            json.dump(keywords_data, f, ensure_ascii=False, indent=2)
                    
                    # 质量评估
                    if not args.no_evaluation:
                        evaluation_result = self.quality_evaluator.evaluate_file(str(file_path))
                        
                        if args.save_intermediate:
                            eval_file = output_base.with_suffix('.evaluation.json')
                            with open(eval_file, 'w', encoding='utf-8') as f:
                                json.dump(asdict(evaluation_result), f, ensure_ascii=False, indent=2, default=str)
                    
                    results.append({
                        'file': str(file_path),
                        'chunks': len(result.chunks),
                        'keywords': result.total_keywords,
                        'processing_time': result.processing_time,
                        'output_file': str(chunks_file)
                    })
                    
                except Exception as e:
                    print(f"\n❌ 处理文件 {file_path} 时出错: {e}")
                    if not args.continue_on_error:
                        return 1
            
            self.print_progress(len(files_to_process), len(files_to_process), "完成")
            
            # 输出结果摘要
            print(f"\n✅ 处理完成!")
            print(f"📊 处理统计:")
            print(f"  - 文件数量: {len(results)}")
            print(f"  - 总分块数: {sum(r['chunks'] for r in results)}")
            print(f"  - 总关键词数: {sum(r['keywords'] for r in results)}")
            print(f"  - 平均处理时间: {sum(r['processing_time'] for r in results) / len(results):.2f}秒")
            
            # 输出详细结果
            if args.format != 'text':
                print(f"\n{self.format_output(results, args.format)}")
            
            return 0
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def evaluate_command(self, args) -> int:
        """
        处理evaluate命令
        
        Args:
            args: 命令行参数
            
        Returns:
            int: 退出码
        """
        try:
            input_path = Path(args.input)
            
            if not input_path.exists():
                print(f"❌ 错误: 文件不存在: {input_path}")
                return 1
            
            print(f"🔍 评估文档: {input_path}")
            
            # 执行评估
            result = self.quality_evaluator.evaluate_file(str(input_path))
            
            # 显示结果
            print(f"\n📊 评估结果:")
            print(f"  - 总体评分: {result.overall_score:.2f}")
            print(f"  - 质量等级: {result.quality_level.value}")
            print(f"  - 分块数量: {len(result.chunk_scores)}")
            print(f"  - 平均分块大小: {result.avg_chunk_size:.0f} 字符")
            
            if args.detailed:
                print(f"\n📈 详细指标:")
                for metric, score in result.detailed_metrics.items():
                    print(f"  - {metric}: {score:.2f}")
                
                print(f"\n💡 优化建议:")
                for suggestion in result.suggestions:
                    print(f"  • {suggestion}")
            
            # 导出结果
            if args.export:
                export_path = Path(args.export)
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)
                print(f"\n💾 结果已导出到: {export_path}")
            
            return 0
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def config_command(self, args) -> int:
        """
        处理config命令
        
        Args:
            args: 命令行参数
            
        Returns:
            int: 退出码
        """
        try:
            if args.show:
                # 显示当前配置
                config = self.config_manager.get_config()
                print("⚙️  当前配置:")
                print(self.format_output(config, args.format))
                
            elif args.set:
                # 设置配置项
                for setting in args.set:
                    if '=' not in setting:
                        print(f"❌ 错误: 无效的设置格式: {setting}")
                        return 1
                    
                    key, value = setting.split('=', 1)
                    
                    # 尝试转换值类型
                    try:
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif value.isdigit():
                            value = int(value)
                        elif '.' in value and value.replace('.', '').isdigit():
                            value = float(value)
                    except ValueError:
                        pass  # 保持字符串类型
                    
                    self.config_manager.set_config(key, value)
                    print(f"✅ 设置 {key} = {value}")
                
                # 保存配置
                self.config_manager.save_config()
                print("💾 配置已保存")
                
            elif args.reset:
                # 重置配置
                self.config_manager.reset_to_default()
                self.config_manager.save_config()
                print("🔄 配置已重置为默认值")
                
            elif args.validate:
                # 验证配置
                is_valid, errors = self.config_manager.validate_config()
                if is_valid:
                    print("✅ 配置文件有效")
                else:
                    print("❌ 配置文件无效:")
                    for error in errors:
                        print(f"  • {error}")
                    return 1
            
            return 0
            
        except Exception as e:
            print(f"❌ 配置操作失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def batch_command(self, args) -> int:
        """
        处理batch命令
        
        Args:
            args: 命令行参数
            
        Returns:
            int: 退出码
        """
        try:
            filelist_path = Path(args.filelist)
            
            if not filelist_path.exists():
                print(f"❌ 错误: 文件列表不存在: {filelist_path}")
                return 1
            
            # 读取文件列表
            with open(filelist_path, 'r', encoding='utf-8') as f:
                file_paths = [line.strip() for line in f if line.strip()]
            
            # 验证文件存在性
            valid_files = []
            for file_path in file_paths:
                path = Path(file_path)
                if path.exists():
                    valid_files.append(path)
                else:
                    print(f"⚠️  警告: 文件不存在: {file_path}")
            
            if not valid_files:
                print("❌ 错误: 没有有效的文件可处理")
                return 1
            
            print(f"📝 批量处理 {len(valid_files)} 个文件")
            
            # 处理文件
            success_count = 0
            error_count = 0
            
            for i, file_path in enumerate(valid_files, 1):
                self.print_progress(i-1, len(valid_files), f"处理 {file_path.name}")
                
                try:
                    # 处理文档
                    result = self.document_processor.process_file(str(file_path))
                    
                    # 生成输出文件
                    output_file = self.output_dir / f"{file_path.stem}_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.chunks.txt"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for j, chunk in enumerate(result.chunks, 1):
                            f.write(f"=== 分块 {j} ===\n")
                            f.write(f"大小: {len(chunk.content)} 字符\n")
                            f.write(f"关键词: {', '.join(chunk.keywords)}\n")
                            f.write(f"质量评分: {chunk.quality_score:.2f}\n")
                            f.write(f"内容:\n{chunk.content}\n\n")
                    
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if not args.continue_on_error:
                        print(f"\n❌ 处理文件 {file_path} 时出错: {e}")
                        return 1
            
            self.print_progress(len(valid_files), len(valid_files), "完成")
            
            print(f"\n✅ 批量处理完成!")
            print(f"📊 处理统计:")
            print(f"  - 成功: {success_count}")
            print(f"  - 失败: {error_count}")
            print(f"  - 总计: {len(valid_files)}")
            
            return 0 if error_count == 0 else 1
            
        except Exception as e:
            print(f"❌ 批量处理失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def stats_command(self, args) -> int:
        """
        处理stats命令
        
        Args:
            args: 命令行参数
            
        Returns:
            int: 退出码
        """
        try:
            if args.input:
                input_path = Path(args.input)
                if not input_path.exists():
                    print(f"❌ 错误: 路径不存在: {input_path}")
                    return 1
            else:
                input_path = self.output_dir
            
            # 统计输出文件
            chunk_files = list(input_path.glob('*.chunks.txt'))
            keyword_files = list(input_path.glob('*.keywords.json'))
            eval_files = list(input_path.glob('*.evaluation.json'))
            
            print(f"📊 统计信息 ({input_path}):")
            print(f"  - 分块文件: {len(chunk_files)}")
            print(f"  - 关键词文件: {len(keyword_files)}")
            print(f"  - 评估文件: {len(eval_files)}")
            
            if args.summary and chunk_files:
                # 详细统计
                total_chunks = 0
                total_size = 0
                
                for chunk_file in chunk_files:
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            chunks = content.count('=== 分块')
                            total_chunks += chunks
                            total_size += len(content)
                    except Exception:
                        continue
                
                print(f"\n📈 详细统计:")
                print(f"  - 总分块数: {total_chunks}")
                print(f"  - 平均每文件分块数: {total_chunks / len(chunk_files):.1f}")
                print(f"  - 总文件大小: {total_size / 1024:.1f} KB")
                print(f"  - 平均文件大小: {total_size / len(chunk_files) / 1024:.1f} KB")
            
            return 0
            
        except Exception as e:
            print(f"❌ 统计失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def tools_command(self, args) -> int:
        """
        处理tools命令
        
        Args:
            args: 命令行参数
            
        Returns:
            int: 退出码
        """
        try:
            if args.tool == 'clean':
                # 清理临时文件
                if args.all:
                    # 清理所有输出文件
                    files_to_clean = list(self.output_dir.glob('*'))
                else:
                    # 只清理临时文件
                    files_to_clean = list(self.output_dir.glob('*.tmp'))
                    files_to_clean.extend(self.output_dir.glob('*.temp'))
                
                if not files_to_clean:
                    print("✅ 没有需要清理的文件")
                    return 0
                
                print(f"🧹 清理 {len(files_to_clean)} 个文件...")
                
                for file_path in files_to_clean:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            import shutil
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"⚠️  无法删除 {file_path}: {e}")
                
                print("✅ 清理完成")
                
            elif args.tool == 'convert':
                # 格式转换
                input_path = Path(args.input)
                if not input_path.exists():
                    print(f"❌ 错误: 文件不存在: {input_path}")
                    return 1
                
                output_path = input_path.with_suffix(f'.{args.to}')
                
                # 读取输入文件
                with open(input_path, 'r', encoding='utf-8') as f:
                    if input_path.suffix == '.json':
                        data = json.load(f)
                    else:
                        data = f.read()
                
                # 转换并保存
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(self.format_output(data, args.to))
                
                print(f"✅ 转换完成: {output_path}")
            
            return 0
            
        except Exception as e:
            print(f"❌ 工具操作失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def run(self, argv: Optional[List[str]] = None) -> int:
        """
        运行命令行界面
        
        Args:
            argv: 命令行参数列表
            
        Returns:
            int: 退出码
        """
        parser = self.create_parser()
        args = parser.parse_args(argv)
        
        # 设置日志
        self.setup_logging(args.verbose, args.quiet)
        
        # 更新配置路径
        if args.config != 'config.json':
            self.config_path = args.config
            self.config_manager = ConfigManager(self.config_path)
            self.document_processor = DocumentProcessor(self.config_manager)
            self.quality_evaluator = QualityEvaluator(self.config_manager.config)
        
        # 更新输出目录
        if args.output != 'outputs':
            self.output_dir = Path(args.output)
            self.output_dir.mkdir(exist_ok=True)
        
        # 执行命令
        if args.command == 'process':
            return self.process_command(args)
        elif args.command == 'evaluate':
            return self.evaluate_command(args)
        elif args.command == 'config':
            return self.config_command(args)
        elif args.command == 'batch':
            return self.batch_command(args)
        elif args.command == 'stats':
            return self.stats_command(args)
        elif args.command == 'tools':
            return self.tools_command(args)
        else:
            parser.print_help()
            return 0


def main():
    """
    主函数，用于运行命令行界面
    """
    try:
        cli = CLIInterface()
        exit_code = cli.run()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n👋 操作已取消")
        sys.exit(130)
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()