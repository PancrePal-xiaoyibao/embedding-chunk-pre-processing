#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web界面模块

提供基于Flask的Web界面，用户可以通过浏览器进行文档处理、分块策略配置、
质量评估等操作。界面美观、响应式，支持文件上传、实时处理状态显示等功能。
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# 导入项目模块
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config.config_manager import ConfigManager
    from src.core.quality_evaluator import QualityEvaluator, QualityEvaluationResult
    from src.core.chunk_strategies import BaseChunkingStrategy, ChunkingConfig
except ImportError:
    # 向后兼容
    from config_manager import ConfigManager
    from quality_evaluator import QualityEvaluator, QualityEvaluationResult
    from chunk_strategies import BaseChunkingStrategy, ChunkingConfig


class WebInterface:
    """
    Web界面类
    
    提供完整的Web界面功能，包括文件上传、处理配置、结果展示等。
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化Web界面
        
        Args:
            config_path: 配置文件路径
        """
        self.app = Flask(__name__)
        self.app.secret_key = 'embedding_enhancement_project_2024'
        self.app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
        
        # 配置上传文件夹
        self.upload_folder = 'uploads'
        self.output_folder = 'outputs'
        self.app.config['UPLOAD_FOLDER'] = self.upload_folder
        self.app.config['OUTPUT_FOLDER'] = self.output_folder
        
        # 创建必要的目录
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static/css', exist_ok=True)
        os.makedirs('static/js', exist_ok=True)
        
        # 初始化组件
        self.config_manager = ConfigManager(config_path)
        self.document_processor = DocumentProcessor(self.config_manager)
        self.quality_evaluator = QualityEvaluator(self.config_manager.config)
        
        # 处理状态管理
        self.processing_status = {}
        self.processing_lock = threading.Lock()
        
        # 允许的文件扩展名
        self.allowed_extensions = {'txt', 'md', 'markdown', 'doc', 'docx', 'pdf'}
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 注册路由
        self._register_routes()
        
        # 创建模板和静态文件
        self._create_templates()
        self._create_static_files()
    
    def _register_routes(self):
        """
        注册Flask路由
        """
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('index.html')
        
        @self.app.route('/upload', methods=['GET', 'POST'])
        def upload_file():
            """文件上传页面"""
            if request.method == 'POST':
                return self._handle_file_upload()
            return render_template('upload.html')
        
        @self.app.route('/config', methods=['GET', 'POST'])
        def config_page():
            """配置页面"""
            if request.method == 'POST':
                return self._handle_config_update()
            
            config = self.config_manager.get_config()
            return render_template('config.html', config=config)
        
        @self.app.route('/process/<task_id>')
        def process_status(task_id):
            """处理状态页面"""
            with self.processing_lock:
                status = self.processing_status.get(task_id, {})
            
            if not status:
                flash('任务不存在或已过期', 'error')
                return redirect(url_for('index'))
            
            return render_template('process_status.html', task_id=task_id, status=status)
        
        @self.app.route('/api/status/<task_id>')
        def api_status(task_id):
            """获取处理状态API"""
            with self.processing_lock:
                status = self.processing_status.get(task_id, {})
            # 序列化状态数据以处理枚举类型
            return jsonify(status, default=str)
        
        @self.app.route('/api/process', methods=['POST'])
        def api_process():
            """处理文档API"""
            return self._handle_api_process()
        
        @self.app.route('/results/<task_id>')
        def results_page(task_id):
            """结果展示页面"""
            with self.processing_lock:
                status = self.processing_status.get(task_id, {})
            
            if not status or status.get('status') != 'completed':
                flash('任务未完成或不存在', 'error')
                return redirect(url_for('index'))
            
            return render_template('results.html', task_id=task_id, results=status.get('results'))
        
        @self.app.route('/download/<task_id>/<file_type>')
        def download_file(task_id, file_type):
            """下载结果文件"""
            return self._handle_file_download(task_id, file_type)
        
        @self.app.route('/evaluate', methods=['GET', 'POST'])
        def evaluate_page():
            """质量评估页面"""
            if request.method == 'POST':
                return self._handle_evaluation()
            return render_template('evaluate.html')
        
        @self.app.errorhandler(RequestEntityTooLarge)
        def handle_file_too_large(e):
            """处理文件过大错误"""
            flash('文件过大，请上传小于50MB的文件', 'error')
            return redirect(url_for('upload_file'))
        
        @self.app.errorhandler(404)
        def handle_404(e):
            """处理404错误"""
            return render_template('error.html', error_code=404, error_message='页面不存在'), 404
        
        @self.app.errorhandler(500)
        def handle_500(e):
            """处理500错误"""
            return render_template('error.html', error_code=500, error_message='服务器内部错误'), 500
    
    def _handle_file_upload(self):
        """
        处理文件上传
        
        Returns:
            Flask响应
        """
        if 'file' not in request.files:
            flash('请选择文件', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('请选择文件', 'error')
            return redirect(request.url)
        
        if file and self._allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # 创建处理任务
                task_id = self._create_processing_task(filepath, filename)
                
                flash('文件上传成功，开始处理...', 'success')
                return redirect(url_for('process_status', task_id=task_id))
                
            except Exception as e:
                self.logger.error(f"文件上传失败: {e}")
                flash(f'文件上传失败: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('不支持的文件类型', 'error')
            return redirect(request.url)
    
    def _handle_config_update(self):
        """
        处理配置更新
        
        Returns:
            Flask响应
        """
        try:
            # 获取表单数据
            config_data = {}
            
            # LLM配置
            config_data['llm_config'] = {
                'default_provider': request.form.get('default_provider', 'glm'),
                'providers': {
                    'glm': {
                        'model': request.form.get('glm_model', 'glm-4-flash'),
                        'api_key': request.form.get('glm_api_key', ''),
                        'base_url': request.form.get('glm_base_url', 'https://open.bigmodel.cn/api/paas/v4/'),
                        'max_tokens': int(request.form.get('glm_max_tokens', 4000)),
                        'temperature': float(request.form.get('glm_temperature', 0.3)),
                        'timeout': int(request.form.get('glm_timeout', 30))
                    },
                    'deepseek': {
                        'model': request.form.get('deepseek_model', 'deepseek-chat'),
                        'api_key': request.form.get('deepseek_api_key', ''),
                        'base_url': request.form.get('deepseek_base_url', 'https://api.deepseek.com/'),
                        'max_tokens': int(request.form.get('deepseek_max_tokens', 4000)),
                        'temperature': float(request.form.get('deepseek_temperature', 0.3)),
                        'timeout': int(request.form.get('deepseek_timeout', 30))
                    },
                    'openai': {
                        'model': request.form.get('openai_model', 'gpt-3.5-turbo'),
                        'api_key': request.form.get('openai_api_key', ''),
                        'base_url': request.form.get('openai_base_url', 'https://api.openai.com/v1/'),
                        'max_tokens': int(request.form.get('openai_max_tokens', 4000)),
                        'temperature': float(request.form.get('openai_temperature', 0.3)),
                        'timeout': int(request.form.get('openai_timeout', 30))
                    }
                }
            }
            
            # 关键词提取配置
            config_data['keyword_extraction'] = {
                'max_keywords_per_chunk': int(request.form.get('max_keywords_per_chunk', 8)),
                'min_keyword_length': int(request.form.get('min_keyword_length', 2)),
                'max_keyword_length': int(request.form.get('max_keyword_length', 20)),
                'keyword_prefix': request.form.get('keyword_prefix', '#'),
                'enable_synonyms': request.form.get('enable_synonyms') == 'on',
                'enable_medical_terms': request.form.get('enable_medical_terms') == 'on',
                'extraction_methods': {
                    'local': {
                        'enabled': request.form.get('local_enabled') == 'on',
                        'use_regex': request.form.get('use_regex') == 'on',
                        'use_frequency': request.form.get('use_frequency') == 'on',
                        'use_medical_dict': request.form.get('use_medical_dict') == 'on'
                    },
                    'llm': {
                        'enabled': request.form.get('llm_enabled') == 'on',
                        'fallback_to_local': request.form.get('fallback_to_local') == 'on'
                    }
                }
            }
            
            # 分块处理配置
            config_data['chunk_processing'] = {
                'target_chunk_size': int(request.form.get('target_chunk_size', 1000)),
                'chunk_boundary_marker': request.form.get('chunk_boundary_marker', '[CHUNK_BOUNDARY]'),
                'max_keywords_display': int(request.form.get('max_keywords_display', 6)),
                'keyword_separator': request.form.get('keyword_separator', ' '),
                'preserve_formatting': request.form.get('preserve_formatting') == 'on',
                'add_keywords_at_beginning': request.form.get('add_keywords_at_beginning') == 'on'
            }
            
            # 医学知识配置
            config_data['medical_knowledge'] = {
                'enable_medical_terms': request.form.get('medical_enable_medical_terms') == 'on',
                'medical_dict_path': request.form.get('medical_dict_path', 'data/medical_terms.json'),
                'enable_drug_names': request.form.get('enable_drug_names') == 'on',
                'enable_disease_names': request.form.get('enable_disease_names') == 'on',
                'enable_symptom_names': request.form.get('enable_symptom_names') == 'on',
                'enable_anatomy_terms': request.form.get('enable_anatomy_terms') == 'on',
                'enable_procedure_names': request.form.get('enable_procedure_names') == 'on',
                'custom_medical_terms': request.form.get('custom_medical_terms', '').split(',') if request.form.get('custom_medical_terms') else []
            }
            
            # 输出配置
            config_data['output'] = {
                'output_suffix': request.form.get('output_suffix', '_with_keywords'),
                'log_level': request.form.get('log_level', 'INFO'),
                'save_original': request.form.get('save_original') == 'on',
                'create_backup': request.form.get('create_backup') == 'on'
            }
            
            # 更新配置
            self.config_manager.update_config(config_data)
            self.config_manager.save_config()
            
            # 重新初始化处理器
            self.document_processor = DocumentProcessor(self.config_manager)
            self.quality_evaluator = QualityEvaluator(self.config_manager.config)
            
            flash('配置更新成功', 'success')
            
        except Exception as e:
            self.logger.error(f"配置更新失败: {e}")
            flash(f'配置更新失败: {str(e)}', 'error')
        
        return redirect(url_for('config_page'))
    
    def _handle_api_process(self):
        """
        处理API处理请求
        
        Returns:
            JSON响应
        """
        try:
            data = request.get_json()
            if not data or 'file_path' not in data:
                return jsonify({'error': '缺少文件路径'}), 400
            
            file_path = data['file_path']
            if not os.path.exists(file_path):
                return jsonify({'error': '文件不存在'}), 404
            
            # 创建处理任务
            task_id = self._create_processing_task(file_path, os.path.basename(file_path))
            
            return jsonify({'task_id': task_id, 'status': 'started'})
            
        except Exception as e:
            self.logger.error(f"API处理失败: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_evaluation(self):
        """
        处理质量评估
        
        Returns:
            Flask响应
        """
        if 'file' not in request.files:
            flash('请选择文件', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('请选择文件', 'error')
            return redirect(request.url)
        
        if file and self._allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"eval_{timestamp}_{filename}"
            filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # 执行评估
                result = self.quality_evaluator.evaluate_file(filepath)
                
                # 保存评估结果
                result_data = self._serialize_evaluation_result(result)
                result_file = os.path.join(self.app.config['OUTPUT_FOLDER'], f"evaluation_{timestamp}.json")
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
                
                flash('评估完成', 'success')
                return render_template('evaluation_results.html', result=result)
                
            except Exception as e:
                self.logger.error(f"评估失败: {e}")
                flash(f'评估失败: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('不支持的文件类型', 'error')
            return redirect(request.url)
    
    def _handle_file_download(self, task_id: str, file_type: str):
        """
        处理文件下载
        
        Args:
            task_id: 任务ID
            file_type: 文件类型 (chunks, keywords, evaluation, summary)
            
        Returns:
            Flask响应
        """
        try:
            with self.processing_lock:
                if task_id not in self.processing_status:
                    flash('任务不存在', 'error')
                    return redirect(url_for('index'))
                
                status = self.processing_status[task_id]
                if status['status'] != 'completed':
                    flash('任务未完成', 'error')
                    return redirect(url_for('index'))
                
                results = status.get('results', {})
                
                # 根据文件类型获取文件路径
                file_mapping = {
                    'chunks': results.get('output_file'),
                    'keywords': results.get('keywords_file'),
                    'evaluation': results.get('evaluation_file'),
                    'summary': results.get('summary_file')  # 添加摘要文件支持
                }
                
                file_path = file_mapping.get(file_type)
                if not file_path or not os.path.exists(file_path):
                    flash('文件不存在', 'error')
                    return redirect(url_for('results_page', task_id=task_id))
                
                # 获取文件名
                filename = os.path.basename(file_path)
                
                return send_file(
                    file_path, 
                    as_attachment=True,
                    download_name=filename,
                    mimetype='application/octet-stream'
                )
                
        except Exception as e:
            self.logger.error(f"文件下载失败: {e}")
            flash(f'下载失败: {str(e)}', 'error')
            return redirect(url_for('results_page', task_id=task_id))
    
    def _create_processing_task(self, file_path: str, filename: str) -> str:
        """
        创建处理任务
        
        Args:
            file_path: 文件路径
            filename: 文件名
            
        Returns:
            str: 任务ID
        """
        task_id = f"task_{int(time.time())}_{hash(file_path) % 10000}"
        
        with self.processing_lock:
            self.processing_status[task_id] = {
                'task_id': task_id,
                'filename': filename,
                'file_path': file_path,
                'status': 'pending',
                'progress': 0,
                'message': '等待处理...',
                'start_time': datetime.now().isoformat(),
                'results': {}
            }
        
        # 启动后台处理线程
        thread = threading.Thread(target=self._process_document_background, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _process_document_background(self, task_id: str):
        """
        后台处理文档
        
        Args:
            task_id: 任务ID
        """
        try:
            with self.processing_lock:
                status = self.processing_status[task_id]
                file_path = status['file_path']
                filename = status['filename']
            
            # 更新状态：开始处理
            self._update_task_status(task_id, 'processing', 10, '开始处理文档...')
            
            # 处理文档
            result = self.document_processor.process_file(file_path)
            
            # 更新状态：处理完成
            self._update_task_status(task_id, 'processing', 70, '文档处理完成，生成输出文件...')
            
            # 生成输出文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(filename)[0]
            
            # 保存分块结果
            chunks_file = os.path.join(self.app.config['OUTPUT_FOLDER'], f"{base_name}_chunks_{timestamp}.txt")
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(result.chunks, 1):
                    f.write(f"=== 分块 {i} ===\n")
                    f.write(f"大小: {len(chunk.content)} 字符\n")
                    f.write(f"关键词: {', '.join(chunk.keywords)}\n")
                    f.write(f"质量评分: {chunk.quality_score:.2f}\n")
                    f.write(f"内容:\n{chunk.content}\n\n")
            
            # 保存关键词结果
            keywords_file = os.path.join(self.app.config['OUTPUT_FOLDER'], f"{base_name}_keywords_{timestamp}.json")
            keywords_data = {
                'total_keywords': result.total_keywords,
                'keywords_by_chunk': [
                    {'chunk_id': i+1, 'keywords': chunk.keywords}
                    for i, chunk in enumerate(result.chunks)
                ],
                'keyword_frequency': {}  # ProcessingResult没有keyword_frequency属性
            }
            
            with open(keywords_file, 'w', encoding='utf-8') as f:
                json.dump(keywords_data, f, ensure_ascii=False, indent=2)
            
            # 更新状态：质量评估
            self._update_task_status(task_id, 'processing', 90, '执行质量评估...')
            
            # 执行质量评估
            evaluation_result = self.quality_evaluator.evaluate_file(file_path)
            
            # 保存评估结果
            evaluation_file = os.path.join(self.app.config['OUTPUT_FOLDER'], f"{base_name}_evaluation_{timestamp}.json")
            evaluation_data = self._serialize_evaluation_result(evaluation_result)
            
            with open(evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2, default=str)
            
            # 生成摘要文件
            summary_file = os.path.join(self.app.config['OUTPUT_FOLDER'], f"{base_name}_summary_{timestamp}.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"文档处理摘要报告\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"文件名: {filename}\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理耗时: {result.processing_time:.2f}秒\n\n")
                
                f.write(f"处理统计:\n")
                f.write(f"- 总分块数: {len(result.chunks)}\n")
                f.write(f"- 总关键词数: {result.total_keywords}\n")
                f.write(f"- 平均分块大小: {sum(len(chunk.content) for chunk in result.chunks) // len(result.chunks) if result.chunks else 0} 字符\n")
                f.write(f"- 整体质量评分: {evaluation_result.overall_score:.2f}/10\n\n")
                
                f.write(f"质量评估详情:\n")
                for metric, score in evaluation_result.metrics.items():
                    f.write(f"- {metric}: {score:.2f}/10\n")
                
                if hasattr(evaluation_result, 'suggestions') and evaluation_result.suggestions:
                    f.write(f"\n改进建议:\n")
                    for suggestion in evaluation_result.suggestions:
                        f.write(f"- {suggestion}\n")
            
            # 更新状态：完成
            results = {
                'processing_result': asdict(result),
                'evaluation_result': evaluation_data,
                'output_file': chunks_file,
                'keywords_file': keywords_file,
                'evaluation_file': evaluation_file,
                'summary_file': summary_file,
                'statistics': {
                    'total_chunks': len(result.chunks),
                    'total_keywords': result.total_keywords,
                    'avg_chunk_size': sum(len(chunk.content) for chunk in result.chunks) // len(result.chunks) if result.chunks else 0,
                    'overall_quality': evaluation_result.overall_score,
                    'processing_time': result.processing_time
                }
            }
            
            self._update_task_status(task_id, 'completed', 100, '处理完成', results)
            
        except Exception as e:
            self.logger.error(f"后台处理失败: {e}")
            self._update_task_status(task_id, 'failed', 0, f'处理失败: {str(e)}')
    
    def _update_task_status(self, task_id: str, status: str, progress: int, 
                          message: str, results: Optional[Dict] = None):
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 状态
            progress: 进度
            message: 消息
            results: 结果数据
        """
        with self.processing_lock:
            if task_id in self.processing_status:
                self.processing_status[task_id].update({
                    'status': status,
                    'progress': progress,
                    'message': message,
                    'update_time': datetime.now().isoformat()
                })
                
                if results:
                    self.processing_status[task_id]['results'] = results
                
                if status in ['completed', 'failed']:
                    self.processing_status[task_id]['end_time'] = datetime.now().isoformat()
    
    def _serialize_evaluation_result(self, evaluation_result) -> Dict[str, Any]:
        """
        序列化评估结果，处理枚举类型
        
        Args:
            evaluation_result: QualityEvaluationResult对象
            
        Returns:
            Dict[str, Any]: 可序列化的字典
        """
        from quality_evaluator import EvaluationMetric, QualityLevel
        
        # 转换为字典
        data = asdict(evaluation_result)
        
        # 处理metrics字典中的枚举键
        if 'metrics' in data and data['metrics']:
            serialized_metrics = {}
            for metric, score in data['metrics'].items():
                # 如果metric是枚举类型，转换为字符串
                if isinstance(metric, EvaluationMetric):
                    key = metric.value
                else:
                    key = str(metric)
                serialized_metrics[key] = score
            data['metrics'] = serialized_metrics
        
        # 处理quality_level枚举
        if 'quality_level' in data and hasattr(data['quality_level'], 'value'):
            data['quality_level'] = data['quality_level'].value
        
        return data
    
    def _serialize_processing_result(self, processing_result) -> Dict[str, Any]:
        """
        序列化处理结果
        
        Args:
            processing_result: ProcessingResult对象
            
        Returns:
            Dict[str, Any]: 可序列化的字典
        """
        return asdict(processing_result)
    
    def _allowed_file(self, filename: str) -> bool:
        """
        检查文件是否允许上传
        
        Args:
            filename: 文件名
        
        Returns:
            bool: 是否允许上传
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _create_templates(self):
        """
        创建HTML模板文件
        """
        # 基础模板
        base_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Embedding增强系统{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css" rel="stylesheet">
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="bi bi-file-text"></i> 文档增强系统
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="{{ url_for('index') }}">首页</a>
                <a class="nav-link" href="{{ url_for('upload_file') }}">文档处理</a>
                <a class="nav-link" href="{{ url_for('evaluate_page') }}">质量评估</a>
                <a class="nav-link" href="{{ url_for('config_page') }}">系统配置</a>
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </main>

    <footer class="bg-light mt-5 py-4">
        <div class="container text-center">
            <p class="text-muted">&copy; 2024 Embedding增强系统</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
        
        # 主页模板
        index_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="text-center mb-5">
            <h1 class="display-4">Embedding增强系统</h1>
            <p class="lead">智能文档分块、关键词提取和质量评估一体化平台</p>
        </div>

        <div class="row g-4">
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="bi bi-file-earmark-text display-1 text-primary"></i>
                        <h5 class="card-title mt-3">文档处理</h5>
                        <p class="card-text">上传文档，智能分块和提取关键词</p>
                        <a href="{{ url_for('upload_file') }}" class="btn btn-primary">开始处理</a>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="bi bi-graph-up display-1 text-success"></i>
                        <h5 class="card-title mt-3">质量评估</h5>
                        <p class="card-text">评估文档分块质量，提供优化建议</p>
                        <a href="{{ url_for('evaluate_page') }}" class="btn btn-success">质量评估</a>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="bi bi-gear display-1 text-warning"></i>
                        <h5 class="card-title mt-3">系统配置</h5>
                        <p class="card-text">配置处理参数和模型设置</p>
                        <a href="{{ url_for('config_page') }}" class="btn btn-warning">系统配置</a>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="bi bi-info-circle display-1 text-info"></i>
                        <h5 class="card-title mt-3">使用说明</h5>
                        <p class="card-text">查看系统功能和使用方法</p>
                        <button class="btn btn-info" data-bs-toggle="modal" data-bs-target="#helpModal">查看帮助</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- 帮助模态框 -->
<div class="modal fade" id="helpModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">系统使用说明</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <h6>主要功能：</h6>
                <ul>
                    <li><strong>文档处理：</strong>支持Markdown、TXT等格式，智能分块和关键词提取</li>
                    <li><strong>质量评估：</strong>多维度评估分块质量，提供优化建议</li>
                    <li><strong>系统配置：</strong>灵活配置处理参数和模型设置</li>
                </ul>
                
                <h6>支持的文件格式：</h6>
                <p>TXT, MD, MARKDOWN, DOC, DOCX, PDF</p>
                
                <h6>处理流程：</h6>
                <ol>
                    <li>上传文档文件</li>
                    <li>系统自动进行分块处理</li>
                    <li>提取关键词和评估质量</li>
                    <li>下载处理结果</li>
                </ol>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
        
        # 上传页面模板
        upload_template = '''{% extends "base.html" %}

{% block title %}文档上传 - {{ super() }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="card">
            <div class="card-header">
                <h4><i class="bi bi-cloud-upload"></i> 文档上传处理</h4>
            </div>
            <div class="card-body">
                <form method="post" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="file" class="form-label">选择文档文件</label>
                        <input type="file" class="form-control" id="file" name="file" 
                               accept=".txt,.md,.markdown,.doc,.docx,.pdf" required>
                        <div class="form-text">
                            支持格式：TXT, MD, MARKDOWN, DOC, DOCX, PDF（最大50MB）
                        </div>
                    </div>
                    
                    <div class="d-grid">
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="bi bi-upload"></i> 上传并处理
                        </button>
                    </div>
                </form>
            </div>
        </div>
        
        <div class="mt-4">
            <h5>处理说明：</h5>
            <ul class="list-group list-group-flush">
                <li class="list-group-item">
                    <i class="bi bi-1-circle text-primary"></i>
                    <strong>文档分块：</strong>根据语义和结构进行智能分块
                </li>
                <li class="list-group-item">
                    <i class="bi bi-2-circle text-primary"></i>
                    <strong>关键词提取：</strong>提取每个分块的关键词和医学术语
                </li>
                <li class="list-group-item">
                    <i class="bi bi-3-circle text-primary"></i>
                    <strong>质量评估：</strong>评估分块质量并提供优化建议
                </li>
                <li class="list-group-item">
                    <i class="bi bi-4-circle text-primary"></i>
                    <strong>结果导出：</strong>生成处理结果文件供下载
                </li>
            </ul>
        </div>
    </div>
</div>
{% endblock %}'''
        
        # 保存模板文件
        templates = {
            'base.html': base_template,
            'index.html': index_template,
            'upload.html': upload_template
        }
        
        for filename, content in templates.items():
            with open(f'templates/{filename}', 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _create_static_files(self):
        """
        创建CSS和JavaScript文件
        """
        # CSS样式
        css_content = '''
/* 自定义样式 */
.card {
    transition: transform 0.2s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.progress-container {
    margin: 20px 0;
}

.status-badge {
    font-size: 0.9em;
    padding: 0.5em 1em;
}

.file-info {
    background-color: #f8f9fa;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 1rem 0;
}

.metric-card {
    border-left: 4px solid #007bff;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #007bff;
}

.quality-excellent { border-left-color: #28a745; }
.quality-good { border-left-color: #17a2b8; }
.quality-fair { border-left-color: #ffc107; }
.quality-poor { border-left-color: #fd7e14; }
.quality-very-poor { border-left-color: #dc3545; }

.processing-animation {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.result-section {
    margin: 2rem 0;
    padding: 1.5rem;
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    background-color: #fff;
}

.keyword-tag {
    display: inline-block;
    background-color: #e9ecef;
    color: #495057;
    padding: 0.25rem 0.5rem;
    margin: 0.125rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}

.chunk-preview {
    max-height: 200px;
    overflow-y: auto;
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.375rem;
    font-family: monospace;
    font-size: 0.875rem;
}
'''
        
        # JavaScript代码
        js_content = '''
// 应用JavaScript功能
document.addEventListener('DOMContentLoaded', function() {
    // 自动刷新处理状态
    if (window.location.pathname.includes('/process/')) {
        const taskId = window.location.pathname.split('/').pop();
        const statusInterval = setInterval(function() {
            fetch(`/api/status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    updateProcessStatus(data);
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(statusInterval);
                        if (data.status === 'completed') {
                            setTimeout(() => {
                                window.location.href = `/results/${taskId}`;
                            }, 2000);
                        }
                    }
                })
                .catch(error => {
                    console.error('状态更新失败:', error);
                });
        }, 2000);
    }
    
    // 文件上传进度
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                const fileInfo = document.createElement('div');
                fileInfo.className = 'alert alert-info mt-2';
                fileInfo.innerHTML = `
                    <i class="bi bi-file-earmark"></i>
                    已选择文件: ${file.name} (${fileSize} MB)
                `;
                
                const existingInfo = document.querySelector('.file-upload-info');
                if (existingInfo) {
                    existingInfo.remove();
                }
                
                fileInfo.className += ' file-upload-info';
                fileInput.parentNode.appendChild(fileInfo);
            }
        });
    }
    
    // 配置表单验证
    const configForm = document.querySelector('form[action*="config"]');
    if (configForm) {
        configForm.addEventListener('submit', function(e) {
            const targetSize = parseInt(document.getElementById('target_chunk_size')?.value || 0);
            const minSize = parseInt(document.getElementById('min_chunk_size')?.value || 0);
            const maxSize = parseInt(document.getElementById('max_chunk_size')?.value || 0);
            
            if (minSize >= targetSize || targetSize >= maxSize) {
                e.preventDefault();
                alert('分块大小配置错误：最小 < 目标 < 最大');
                return false;
            }
        });
    }
});

function updateProcessStatus(data) {
    const progressBar = document.querySelector('.progress-bar');
    const statusBadge = document.querySelector('.status-badge');
    const messageElement = document.querySelector('.status-message');
    
    if (progressBar) {
        progressBar.style.width = data.progress + '%';
        progressBar.setAttribute('aria-valuenow', data.progress);
        progressBar.textContent = data.progress + '%';
    }
    
    if (statusBadge) {
        statusBadge.className = 'badge status-badge';
        switch(data.status) {
            case 'pending':
                statusBadge.className += ' bg-secondary';
                break;
            case 'processing':
                statusBadge.className += ' bg-primary processing-animation';
                break;
            case 'completed':
                statusBadge.className += ' bg-success';
                break;
            case 'failed':
                statusBadge.className += ' bg-danger';
                break;
        }
        statusBadge.textContent = getStatusText(data.status);
    }
    
    if (messageElement) {
        messageElement.textContent = data.message || '';
    }
}

function getStatusText(status) {
    const statusMap = {
        'pending': '等待中',
        'processing': '处理中',
        'completed': '已完成',
        'failed': '失败'
    };
    return statusMap[status] || status;
}

// 工具函数
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
}
'''
        
        # 保存静态文件
        with open('static/css/style.css', 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        with open('static/js/app.js', 'w', encoding='utf-8') as f:
            f.write(js_content)
    
    def run(self, host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
        """
        运行Web应用
        
        Args:
            host: 主机地址
            port: 端口号
            debug: 调试模式
        """
        self.logger.info(f"启动Web界面: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """
    主函数，用于测试Web界面
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建Web界面
        web_app = WebInterface()
        
        print("🚀 启动Web界面...")
        print("📝 访问地址: http://127.0.0.1:5000")
        print("⚡ 按 Ctrl+C 停止服务")
        
        # 运行应用
        web_app.run(debug=True)
        
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()