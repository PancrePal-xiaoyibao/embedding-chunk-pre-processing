# Embedding增强项目

一个基于深度学习的文档处理和向量化系统，提供智能文档分块、质量评估和向量检索功能。

## 🚀 项目特性

- **智能文档分块**: 支持多种分块策略（语义分块、Token分块、混合分块等）
- **质量评估**: 全面的文档质量评估体系，包括语义完整性、格式正确性等
- **多格式支持**: 支持PDF、Word、Excel、纯文本等多种文档格式
- **向量化处理**: 集成多种预训练模型，支持中英文文档向量化
- **Web界面**: 提供直观的Web操作界面
- **命令行工具**: 支持批量处理和自动化操作
- **配置管理**: 灵活的配置系统，支持多种配置格式
- **性能监控**: 内置性能监控和日志系统

## 📁 项目结构

```
Embedding增强项目/
├── main_app.py                 # 主入口文件
├── config_manager.py           # 配置管理模块
├── document_processor.py       # 文档处理核心模块
├── chunk_strategies.py         # 分块策略模块
├── quality_evaluator.py        # 质量评估模块
├── web_interface.py            # Web界面模块
├── cli_interface.py            # 命令行界面模块
├── utils/                      # 工具模块目录
│   ├── __init__.py
│   ├── logger.py              # 日志管理
│   ├── error_handler.py       # 错误处理
│   ├── performance_monitor.py # 性能监控
│   ├── file_utils.py          # 文件工具
│   └── text_utils.py          # 文本处理工具
├── config/                     # 配置文件目录
├── data/                       # 数据目录
├── logs/                       # 日志目录
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明
```

## 🛠️ 安装和配置

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd Embedding增强项目
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **下载NLTK数据**（首次运行时）
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. **配置设置**
   ```bash
   # 复制配置模板
   cp config/config.template.yaml config/config.yaml
   # 根据需要修改配置文件
   ```

## 🚀 快速开始

### 1. Web界面使用

启动Web服务：
```bash
python main_app.py --mode web --port 5000
```

访问 `http://localhost:5000` 使用Web界面。

### 2. 命令行使用

#### 处理单个文档
```bash
python main_app.py --mode cli process --input document.pdf --output output/
```

#### 批量处理文档
```bash
python main_app.py --mode cli batch --input-dir documents/ --output-dir output/
```

#### 质量评估
```bash
python main_app.py --mode cli evaluate --input document.pdf --report-format json
```

#### 配置管理
```bash
# 查看当前配置
python main_app.py --mode cli config --show

# 设置配置项
python main_app.py --mode cli config --set chunk_size=512

# 重置配置
python main_app.py --mode cli config --reset
```

### 3. Python API使用

```python
from document_processor import DocumentProcessor
from config_manager import ConfigManager

# 初始化配置
config_manager = ConfigManager()
config = config_manager.load_config()

# 创建文档处理器
processor = DocumentProcessor(config)

# 处理文档
result = processor.process_file("document.pdf")

# 查看结果
print(f"处理了 {len(result.chunks)} 个分块")
print(f"平均质量分数: {result.average_quality_score:.2f}")
```

## ⚙️ 配置说明

### 主要配置项

```yaml
# 分块配置
chunking:
  strategy: "hybrid"          # 分块策略: semantic, token_based, hybrid
  chunk_size: 512            # 分块大小
  overlap_size: 50           # 重叠大小
  min_chunk_size: 100        # 最小分块大小
  max_chunk_size: 1000       # 最大分块大小

# 模型配置
models:
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  similarity_threshold: 0.7   # 语义相似度阈值

# 质量评估配置
quality:
  enable_semantic_check: true
  enable_format_check: true
  min_quality_score: 0.6

# 输出配置
output:
  format: "json"             # 输出格式: json, csv, excel
  include_metadata: true     # 包含元数据
  save_embeddings: false     # 保存向量
```

## 📊 功能模块详解

### 1. 文档处理模块 (document_processor.py)

- **多格式支持**: PDF、Word、Excel、TXT等
- **智能预处理**: 文本清理、格式标准化
- **分块处理**: 多种分块策略可选
- **向量化**: 支持多种预训练模型
- **质量评估**: 实时质量监控

### 2. 分块策略模块 (chunk_strategies.py)

- **Token分块**: 基于Token数量的固定分块
- **语义分块**: 基于语义相似度的智能分块
- **结构化分块**: 基于文档结构的分块
- **混合分块**: 结合多种策略的优化分块

### 3. 质量评估模块 (quality_evaluator.py)

- **语义完整性**: 评估分块的语义连贯性
- **格式正确性**: 检查格式和结构问题
- **大小分布**: 分析分块大小的合理性
- **重复检测**: 识别重复和冗余内容

### 4. Web界面模块 (web_interface.py)

- **文件上传**: 支持拖拽上传
- **实时处理**: 显示处理进度
- **结果展示**: 可视化处理结果
- **配置管理**: 在线配置调整

### 5. 命令行界面 (cli_interface.py)

- **批量处理**: 支持目录批量处理
- **进度显示**: 实时进度条
- **多种输出**: 支持多种输出格式
- **配置管理**: 命令行配置操作

## 🔧 高级功能

### 1. 自定义分块策略

```python
from chunk_strategies import ChunkStrategy, ChunkConfig

class CustomChunkStrategy(ChunkStrategy):
    def chunk_text(self, text: str, config: ChunkConfig) -> List[str]:
        # 实现自定义分块逻辑
        pass

# 注册自定义策略
strategy_factory.register_strategy("custom", CustomChunkStrategy)
```

### 2. 自定义质量评估器

```python
from quality_evaluator import QualityEvaluator

class CustomQualityEvaluator(QualityEvaluator):
    def evaluate_custom_metric(self, chunk: str) -> float:
        # 实现自定义质量指标
        pass
```

### 3. 插件系统

项目支持插件扩展，可以通过插件添加新的功能模块。

## 📈 性能优化

### 1. 内存优化

- 流式处理大文件
- 分批处理向量化
- 智能缓存机制

### 2. 速度优化

- 多进程并行处理
- GPU加速（可选）
- 预计算缓存

### 3. 存储优化

- 压缩存储向量
- 增量更新机制
- 清理临时文件

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 启用流式处理
   - 增加虚拟内存

2. **模型下载失败**
   - 检查网络连接
   - 使用镜像源
   - 手动下载模型

3. **文件格式不支持**
   - 检查文件格式
   - 转换为支持格式
   - 添加自定义解析器

### 日志分析

查看日志文件：
```bash
tail -f logs/app.log
```

调整日志级别：
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目维护者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目链接: [项目URL]

## 🙏 致谢

特别感谢 **小x宝社区** ❤️ 的贡献和支持！我们盼望更多技术爱好者加入社区，共同推动癌症/罕见病患者的公益力量。

### 技术支持
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - 向量化模型
- [jieba](https://github.com/fxsjy/jieba) - 中文分词
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具

### 社区贡献
- **小x宝社区** - 为癌症/罕见病患者提供技术支持和公益服务
- 感谢所有为医疗AI技术发展贡献力量的开发者和志愿者

## 📚 更多文档

- [API文档](docs/api.md)
- [配置指南](docs/configuration.md)
- [开发指南](docs/development.md)
- [部署指南](docs/deployment.md)

---

**注意**: 这是一个开发中的项目，功能可能会有变化。建议在生产环境使用前进行充分测试。