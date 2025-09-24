#!/usr/bin/env python3
"""
AI Services 模块安装脚本

提供统一的AI服务接口，支持Chat、Embedding和Rerank功能。
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    """读取README文件内容"""
    readme_path = os.path.join(os.path.dirname(__file__), 'docs', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AI Services - 统一的AI服务接口模块"

# 读取requirements文件
def read_requirements():
    """读取requirements文件"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if line and not line.startswith('#'):
                    # 只包含核心依赖，不包含开发依赖
                    if not any(dev_keyword in line.lower() for dev_keyword in 
                              ['pytest', 'black', 'flake8', 'mypy', 'sphinx', 'memory-profiler', 'line-profiler']):
                        requirements.append(line)
    
    return requirements

# 可选依赖
extras_require = {
    'local': [
        'sentence-transformers>=2.2.0',
        'torch>=1.12.0',
        'transformers>=4.20.0',
    ],
    'ml': [
        'scikit-learn>=1.1.0',
        'numpy>=1.21.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.0.0',
        'black>=22.0.0',
        'flake8>=5.0.0',
        'mypy>=0.991',
    ],
    'docs': [
        'sphinx>=5.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'profiling': [
        'memory-profiler>=0.60.0',
        'line-profiler>=4.0.0',
    ]
}

# 添加all选项，包含所有可选依赖
extras_require['all'] = []
for deps in extras_require.values():
    extras_require['all'].extend(deps)

setup(
    name="ai-services",
    version="1.0.0",
    author="AI Services Team",
    author_email="ai-services@example.com",
    description="统一的AI服务接口模块，支持Chat、Embedding和Rerank功能",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/ai-services",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        'ai_services': [
            'docs/*.md',
            'examples/*.py',
            'tests/*.py',
        ],
    },
    entry_points={
        'console_scripts': [
            'ai-services-test=ai_services.tests.test_ai_services:run_all_tests',
            'ai-services-example=ai_services.examples.basic_usage:main',
        ],
    },
    keywords=[
        'ai', 'artificial-intelligence', 'chat', 'embedding', 'rerank',
        'ollama', 'machine-learning', 'nlp', 'natural-language-processing'
    ],
    project_urls={
        'Documentation': 'https://github.com/example/ai-services/docs',
        'Source': 'https://github.com/example/ai-services',
        'Tracker': 'https://github.com/example/ai-services/issues',
    },
)