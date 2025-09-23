#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件工具模块

提供文件操作、路径处理和文件格式检测等实用功能。
"""

import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
import json
import yaml
import csv
from datetime import datetime
import tempfile
import zipfile
import tarfile

from .logger import get_logger


class FileUtils:
    """
    文件工具类
    
    提供各种文件操作的实用方法。
    """
    
    def __init__(self, logger_name: str = "FileUtils"):
        """
        初始化文件工具
        
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
        
        # 支持的文档格式
        self.supported_formats = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.rst': 'text/x-rst',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.xml': 'text/xml',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.csv': 'text/csv',
            '.tsv': 'text/tab-separated-values'
        }
    
    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """
        确保目录存在
        
        Args:
            path: 目录路径
            
        Returns:
            Path: 目录路径对象
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"确保目录存在: {path}")
        return path
    
    def safe_filename(self, filename: str, max_length: int = 255) -> str:
        """
        生成安全的文件名
        
        Args:
            filename: 原始文件名
            max_length: 最大长度
            
        Returns:
            str: 安全的文件名
        """
        # 移除或替换不安全的字符
        unsafe_chars = '<>:"/\\|?*'
        safe_name = filename
        
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')
        
        # 移除连续的下划线
        while '__' in safe_name:
            safe_name = safe_name.replace('__', '_')
        
        # 移除开头和结尾的下划线和点
        safe_name = safe_name.strip('_.')
        
        # 限制长度
        if len(safe_name) > max_length:
            name, ext = os.path.splitext(safe_name)
            max_name_length = max_length - len(ext)
            safe_name = name[:max_name_length] + ext
        
        # 确保不为空
        if not safe_name:
            safe_name = "unnamed_file"
        
        return safe_name
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        path = Path(file_path)
        
        if not path.exists():
            return {'error': '文件不存在'}
        
        stat = path.stat()
        
        # 获取MIME类型
        mime_type, encoding = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = self.supported_formats.get(path.suffix.lower(), 'unknown')
        
        info = {
            'path': str(path.absolute()),
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'size_human': self.format_size(stat.st_size),
            'mime_type': mime_type,
            'encoding': encoding,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'is_supported': path.suffix.lower() in self.supported_formats
        }
        
        # 计算文件哈希（小文件）
        if stat.st_size < 10 * 1024 * 1024:  # 10MB以下
            try:
                info['md5'] = self.calculate_hash(path, 'md5')
                info['sha256'] = self.calculate_hash(path, 'sha256')
            except Exception as e:
                self.logger.warning(f"计算文件哈希失败: {e}")
        
        return info
    
    def calculate_hash(self, file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        计算文件哈希值
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法（md5, sha1, sha256等）
            
        Returns:
            str: 哈希值
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def format_size(self, size_bytes: int) -> str:
        """
        格式化文件大小
        
        Args:
            size_bytes: 字节数
            
        Returns:
            str: 格式化的大小
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    def find_files(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
        include_dirs: bool = False
    ) -> List[Path]:
        """
        查找文件
        
        Args:
            directory: 搜索目录
            pattern: 文件模式
            recursive: 是否递归搜索
            include_dirs: 是否包含目录
            
        Returns:
            List[Path]: 找到的文件列表
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"目录不存在: {directory}")
            return []
        
        if recursive:
            glob_pattern = f"**/{pattern}"
        else:
            glob_pattern = pattern
        
        files = []
        for path in directory.glob(glob_pattern):
            if path.is_file() or (include_dirs and path.is_dir()):
                files.append(path)
        
        self.logger.debug(f"在 {directory} 中找到 {len(files)} 个文件")
        return sorted(files)
    
    def find_supported_documents(self, directory: Union[str, Path]) -> List[Path]:
        """
        查找支持的文档文件
        
        Args:
            directory: 搜索目录
            
        Returns:
            List[Path]: 支持的文档文件列表
        """
        all_files = self.find_files(directory, "*", recursive=True)
        supported_files = [
            f for f in all_files
            if f.suffix.lower() in self.supported_formats
        ]
        
        self.logger.info(f"找到 {len(supported_files)} 个支持的文档文件")
        return supported_files
    
    def copy_file(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        preserve_metadata: bool = True
    ) -> bool:
        """
        复制文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            preserve_metadata: 是否保留元数据
            
        Returns:
            bool: 是否成功
        """
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            # 确保目标目录存在
            self.ensure_directory(dst_path.parent)
            
            if preserve_metadata:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
            
            self.logger.debug(f"文件复制成功: {src} -> {dst}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件复制失败: {e}")
            return False
    
    def move_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """
        移动文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            # 确保目标目录存在
            self.ensure_directory(dst_path.parent)
            
            shutil.move(src_path, dst_path)
            self.logger.debug(f"文件移动成功: {src} -> {dst}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件移动失败: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                self.logger.debug(f"文件删除成功: {file_path}")
                return True
            else:
                self.logger.warning(f"文件不存在: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"文件删除失败: {e}")
            return False
    
    def read_text_file(
        self,
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        fallback_encodings: List[str] = None
    ) -> Optional[str]:
        """
        读取文本文件
        
        Args:
            file_path: 文件路径
            encoding: 编码格式
            fallback_encodings: 备用编码列表
            
        Returns:
            Optional[str]: 文件内容
        """
        if fallback_encodings is None:
            fallback_encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        
        encodings_to_try = [encoding] + [enc for enc in fallback_encodings if enc != encoding]
        
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                self.logger.debug(f"文件读取成功: {file_path} (编码: {enc})")
                return content
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"文件读取失败: {e}")
                return None
        
        self.logger.error(f"无法使用任何编码读取文件: {file_path}")
        return None
    
    def write_text_file(
        self,
        file_path: Union[str, Path],
        content: str,
        encoding: str = 'utf-8',
        backup: bool = False
    ) -> bool:
        """
        写入文本文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 编码格式
            backup: 是否备份原文件
            
        Returns:
            bool: 是否成功
        """
        try:
            path = Path(file_path)
            
            # 备份原文件
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                self.copy_file(path, backup_path)
            
            # 确保目录存在
            self.ensure_directory(path.parent)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            self.logger.debug(f"文件写入成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件写入失败: {e}")
            return False
    
    def read_json_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        读取JSON文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[Dict[str, Any]]: JSON数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.debug(f"JSON文件读取成功: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"JSON文件读取失败: {e}")
            return None
    
    def write_json_file(
        self,
        file_path: Union[str, Path],
        data: Dict[str, Any],
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> bool:
        """
        写入JSON文件
        
        Args:
            file_path: 文件路径
            data: JSON数据
            indent: 缩进
            ensure_ascii: 是否确保ASCII
            
        Returns:
            bool: 是否成功
        """
        try:
            path = Path(file_path)
            self.ensure_directory(path.parent)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
            
            self.logger.debug(f"JSON文件写入成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"JSON文件写入失败: {e}")
            return False
    
    def read_yaml_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        读取YAML文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[Dict[str, Any]]: YAML数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.logger.debug(f"YAML文件读取成功: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"YAML文件读取失败: {e}")
            return None
    
    def write_yaml_file(
        self,
        file_path: Union[str, Path],
        data: Dict[str, Any],
        default_flow_style: bool = False
    ) -> bool:
        """
        写入YAML文件
        
        Args:
            file_path: 文件路径
            data: YAML数据
            default_flow_style: 默认流样式
            
        Returns:
            bool: 是否成功
        """
        try:
            path = Path(file_path)
            self.ensure_directory(path.parent)
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=default_flow_style, allow_unicode=True)
            
            self.logger.debug(f"YAML文件写入成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"YAML文件写入失败: {e}")
            return False
    
    def create_archive(
        self,
        source_path: Union[str, Path],
        archive_path: Union[str, Path],
        format: str = 'zip'
    ) -> bool:
        """
        创建压缩包
        
        Args:
            source_path: 源路径
            archive_path: 压缩包路径
            format: 压缩格式（zip, tar, tar.gz, tar.bz2）
            
        Returns:
            bool: 是否成功
        """
        try:
            source = Path(source_path)
            archive = Path(archive_path)
            
            self.ensure_directory(archive.parent)
            
            if format == 'zip':
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
                    if source.is_file():
                        zf.write(source, source.name)
                    else:
                        for file_path in source.rglob('*'):
                            if file_path.is_file():
                                zf.write(file_path, file_path.relative_to(source.parent))
            
            elif format.startswith('tar'):
                mode = 'w'
                if format == 'tar.gz':
                    mode = 'w:gz'
                elif format == 'tar.bz2':
                    mode = 'w:bz2'
                
                with tarfile.open(archive, mode) as tf:
                    tf.add(source, arcname=source.name)
            
            else:
                raise ValueError(f"不支持的压缩格式: {format}")
            
            self.logger.info(f"压缩包创建成功: {archive}")
            return True
            
        except Exception as e:
            self.logger.error(f"压缩包创建失败: {e}")
            return False
    
    def extract_archive(
        self,
        archive_path: Union[str, Path],
        extract_path: Union[str, Path]
    ) -> bool:
        """
        解压压缩包
        
        Args:
            archive_path: 压缩包路径
            extract_path: 解压路径
            
        Returns:
            bool: 是否成功
        """
        try:
            archive = Path(archive_path)
            extract_dir = Path(extract_path)
            
            self.ensure_directory(extract_dir)
            
            if archive.suffix == '.zip':
                with zipfile.ZipFile(archive, 'r') as zf:
                    zf.extractall(extract_dir)
            
            elif archive.suffix in ['.tar', '.gz', '.bz2'] or '.tar.' in archive.name:
                with tarfile.open(archive, 'r:*') as tf:
                    tf.extractall(extract_dir)
            
            else:
                raise ValueError(f"不支持的压缩格式: {archive.suffix}")
            
            self.logger.info(f"压缩包解压成功: {archive} -> {extract_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"压缩包解压失败: {e}")
            return False
    
    def get_temp_file(self, suffix: str = '', prefix: str = 'tmp') -> str:
        """
        获取临时文件路径
        
        Args:
            suffix: 文件后缀
            prefix: 文件前缀
            
        Returns:
            str: 临时文件路径
        """
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # 关闭文件描述符
        return path
    
    def get_temp_dir(self, prefix: str = 'tmp') -> str:
        """
        获取临时目录路径
        
        Args:
            prefix: 目录前缀
            
        Returns:
            str: 临时目录路径
        """
        return tempfile.mkdtemp(prefix=prefix)
    
    def cleanup_temp_files(self, temp_paths: List[str]):
        """
        清理临时文件
        
        Args:
            temp_paths: 临时文件路径列表
        """
        for temp_path in temp_paths:
            try:
                path = Path(temp_path)
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    self.logger.debug(f"临时文件清理成功: {temp_path}")
            except Exception as e:
                self.logger.warning(f"临时文件清理失败: {temp_path}: {e}")


def main():
    """
    测试文件工具功能
    """
    # 创建文件工具实例
    file_utils = FileUtils()
    
    # 测试目录创建
    test_dir = file_utils.ensure_directory("test_files")
    print(f"测试目录: {test_dir}")
    
    # 测试文件写入
    test_file = test_dir / "test.txt"
    content = "这是一个测试文件\n包含中文内容"
    success = file_utils.write_text_file(test_file, content)
    print(f"文件写入: {success}")
    
    # 测试文件读取
    read_content = file_utils.read_text_file(test_file)
    print(f"文件内容: {read_content}")
    
    # 测试文件信息
    file_info = file_utils.get_file_info(test_file)
    print(f"文件信息: {file_info}")
    
    # 测试JSON文件
    json_data = {"name": "测试", "value": 123, "list": [1, 2, 3]}
    json_file = test_dir / "test.json"
    file_utils.write_json_file(json_file, json_data)
    loaded_data = file_utils.read_json_file(json_file)
    print(f"JSON数据: {loaded_data}")
    
    # 测试文件查找
    found_files = file_utils.find_files(test_dir, "*.txt")
    print(f"找到的文件: {found_files}")
    
    # 测试压缩包
    archive_path = test_dir / "test.zip"
    file_utils.create_archive(test_file, archive_path, 'zip')
    print(f"压缩包创建: {archive_path.exists()}")
    
    # 测试临时文件
    temp_file = file_utils.get_temp_file(suffix='.tmp', prefix='test_')
    print(f"临时文件: {temp_file}")
    
    # 清理
    file_utils.cleanup_temp_files([temp_file])
    
    print("文件工具测试完成")


if __name__ == "__main__":
    main()