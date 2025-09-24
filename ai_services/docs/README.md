# AI Services 模块使用指南

一个统一的AI服务模块，提供Chat、Embedding和Rerank功能，支持多种提供商（Ollama、本地模型等）。

## 🚀 快速开始

### 安装依赖

```bash
pip install requests numpy sentence-transformers transformers torch
```

### 基本使用

```python
from ai_services import AIServiceFactory

# 创建服务工厂
factory = AIServiceFactory.create_default()

# 使用Chat服务
chat_service = factory.create_service("chat")
response = chat_service.chat("你好，请介绍一下自己")
print(response.content)

# 使用Embedding服务
embedding_service = factory.create_service("embedding")
embeddings = embedding_service.embed(["文本1", "文本2"])
print(f"嵌入维度: {len(embeddings.vectors[0])}")

# 使用Rerank服务
rerank_service = factory.create_service("rerank")
results = rerank_service.rerank("查询文本", ["候选文档1", "候选文档2"])
print(f"最佳匹配: {results.results[0].document}")
```

### 配置管理

AI Services 支持灵活的配置管理：

```bash
# 1. 复制配置模板
cp config.example.yaml config.yaml

# 2. 验证配置
python validate_config.py config.yaml

# 3. 运行测试
python quick_start.py
```

**配置文件说明：**
- `config.template.yaml` - 完整配置模板（包含所有选项）
- `config.example.yaml` - 简化配置示例（快速开始）
- `CONFIG.md` - 详细配置指南

**配置验证：**
```bash
# 验证指定配置文件
python validate_config.py config.yaml --verbose

# 验证默认配置
python validate_config.py --default
```

## 📋 目录结构

```
ai_services/
├── __init__.py              # 模块入口
├── core/                    # 核心组件
│   ├── __init__.py
│   ├── factory.py          # AI服务工厂
│   ├── interfaces.py       # 基础接口定义
│   ├── exceptions.py       # 异常类定义
│   └── config_manager.py   # 配置管理器
├── services/               # 具体服务实现
│   ├── __init__.py
│   ├── models.py          # 数据模型
│   ├── chat_service.py    # Chat服务
│   ├── embedding_service.py # Embedding服务
│   └── rerank_service.py  # Rerank服务
├── config/                # 配置管理
│   ├── __init__.py
│   └── config.py         # 配置工具
├── examples/             # 示例代码
├── tests/               # 测试代码
└── docs/               # 文档
    └── README.md       # 本文档
```

## ⚙️ 配置管理

### 1. 使用默认配置

```python
from ai_services import AIServiceFactory

# 使用默认配置创建工厂
factory = AIServiceFactory.create_default()
```

### 2. 从配置文件加载

```python
# 创建配置模板
from ai_services.config import create_config_template
create_config_template("config.yaml", format="yaml")

# 从配置文件加载
factory = AIServiceFactory.from_config_file("config.yaml")
```

### 3. 从字典配置

```python
config = {
    "services": {
        "chat": {
            "default_provider": "ollama",
            "providers": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model_name": "llama2"
                }
            }
        }
    }
}

factory = AIServiceFactory.from_config(config)
```

### 4. 环境变量配置

支持以下环境变量：

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_CHAT_MODEL="llama2"
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
export OLLAMA_RERANK_MODEL="llama2"
export LOG_LEVEL="INFO"
```

## 🔧 服务详细使用

### Chat服务

```python
from ai_services import create_chat_service
from ai_services.services.models import create_user_message, create_system_message

# 创建Chat服务
chat_service = create_chat_service()

# 单轮对话
response = chat_service.chat("你好")
print(response.content)

# 多轮对话
messages = [
    create_system_message("你是一个有用的助手"),
    create_user_message("请介绍一下Python"),
    create_user_message("它有什么优势？")
]
response = chat_service.chat(messages)
print(response.content)

# 异步对话
import asyncio

async def async_chat():
    response = await chat_service.chat_async("异步消息")
    print(response.content)

asyncio.run(async_chat())

# 流式对话
for chunk in chat_service.chat_stream("流式消息"):
    print(chunk.content, end="", flush=True)
```

### Embedding服务

```python
from ai_services import create_embedding_service

# 创建Embedding服务
embedding_service = create_embedding_service()

# 单个文本嵌入
result = embedding_service.embed("这是一个测试文本")
print(f"嵌入向量维度: {len(result.vectors[0])}")

# 批量文本嵌入
texts = ["文本1", "文本2", "文本3"]
result = embedding_service.embed(texts)
print(f"处理了 {len(result.vectors)} 个文本")

# 异步嵌入
async def async_embed():
    result = await embedding_service.embed_async(texts)
    return result

# 计算相似度
similarity = embedding_service.compute_similarity(
    result.vectors[0], 
    result.vectors[1]
)
print(f"相似度: {similarity}")
```

### Rerank服务

```python
from ai_services import create_rerank_service

# 创建Rerank服务
rerank_service = create_rerank_service()

# 重排序文档
query = "Python编程语言"
documents = [
    "Python是一种高级编程语言",
    "Java是面向对象的编程语言", 
    "Python具有简洁的语法",
    "JavaScript用于Web开发"
]

result = rerank_service.rerank(query, documents)

# 查看排序结果
for i, item in enumerate(result.results):
    print(f"{i+1}. 分数: {item.score:.3f} - {item.document}")

# 异步重排序
async def async_rerank():
    result = await rerank_service.rerank_async(query, documents)
    return result
```

## 🎯 高级功能

### 1. 服务健康检查

```python
# 检查服务是否可用
is_healthy = chat_service.health_check()
print(f"Chat服务状态: {'正常' if is_healthy else '异常'}")

# 检查所有服务
factory = AIServiceFactory.create_default()
status = factory.health_check()
for service_type, is_healthy in status.items():
    print(f"{service_type}: {'正常' if is_healthy else '异常'}")
```

### 2. 获取可用提供商

```python
# 获取Chat服务的可用提供商
providers = factory.get_available_providers("chat")
print(f"可用的Chat提供商: {providers}")

# 测试提供商连接
connection_ok = factory.test_provider_connection("chat", "ollama")
print(f"Ollama连接状态: {'正常' if connection_ok else '异常'}")
```

### 3. 模型管理（Ollama）

```python
# 获取可用模型
models = chat_service.get_available_models()
print(f"可用模型: {models}")

# 拉取新模型
success = chat_service.pull_model("llama2:7b")
print(f"模型拉取: {'成功' if success else '失败'}")
```

### 4. 错误处理

```python
from ai_services.core.exceptions import (
    AIServiceError, 
    ConnectionError, 
    ModelNotFoundError
)

try:
    response = chat_service.chat("测试消息")
except ConnectionError as e:
    print(f"连接错误: {e}")
except ModelNotFoundError as e:
    print(f"模型未找到: {e}")
except AIServiceError as e:
    print(f"AI服务错误: {e}")
```

## 🔌 支持的提供商

### Chat服务
- **Ollama**: 本地大语言模型服务
  - 支持流式响应
  - 支持多种开源模型
  - 支持自定义参数

### Embedding服务
- **Ollama**: 使用Ollama的嵌入模型
- **Local**: 本地sentence-transformers模型
  - 支持CPU/GPU计算
  - 支持多种预训练模型

### Rerank服务
- **Embedding-based**: 基于嵌入向量的重排序
- **Ollama**: 使用Ollama进行生成式重排序
- **Cross-encoder**: 交叉编码器模型重排序

## 📊 性能优化

### 1. 批处理

```python
# Embedding批处理
texts = ["文本1", "文本2", "文本3"]
result = embedding_service.embed(texts)  # 自动批处理

# Rerank批处理
documents = ["文档1", "文档2", "文档3"]
result = rerank_service.rerank("查询", documents)  # 自动批处理
```

### 2. 异步处理

```python
import asyncio

async def process_multiple():
    tasks = [
        chat_service.chat_async("消息1"),
        chat_service.chat_async("消息2"),
        embedding_service.embed_async("文本1")
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### 3. 连接池和重试

```python
# 配置重试和超时
config = {
    "services": {
        "chat": {
            "providers": {
                "ollama": {
                    "timeout": 60.0,      # 超时时间
                    "max_retries": 5,     # 最大重试次数
                    "base_url": "http://localhost:11434"
                }
            }
        }
    }
}
```

## 🧪 测试

```python
# 运行测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_chat_service.py

# 运行集成测试
python -m pytest tests/test_integration.py
```

## 🐛 故障排除

### 常见问题

1. **Ollama连接失败**
   ```bash
   # 检查Ollama是否运行
   curl http://localhost:11434/api/tags
   
   # 启动Ollama
   ollama serve
   ```

2. **模型未找到**
   ```bash
   # 拉取所需模型
   ollama pull llama2
   ollama pull nomic-embed-text
   ```

3. **依赖缺失**
   ```bash
   # 安装所有依赖
   pip install requests numpy sentence-transformers transformers torch
   ```

### 日志配置

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 或在配置中设置
config = {
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持Chat、Embedding、Rerank服务
- 支持Ollama、本地模型提供商
- 完整的配置管理系统
- 异步和流式支持

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个模块！

## 📄 许可证

MIT License