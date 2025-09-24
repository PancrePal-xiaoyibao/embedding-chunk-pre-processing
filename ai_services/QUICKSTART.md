# AI Services 快速开始指南

## 🚀 快速配置

### 1. 生成配置文件

```bash
# 生成最小化配置（推荐新手）
python generate_config.py config.yaml minimal

# 生成示例配置（包含常用设置）
python generate_config.py config.yaml example

# 生成完整配置（包含所有选项）
python generate_config.py config.yaml full
```

### 2. 验证配置

```bash
# 验证配置文件
python validate_config.py config.yaml

# 详细验证信息
python validate_config.py config.yaml --verbose
```

### 3. 运行测试

```bash
# 运行完整测试
python quick_start.py

# 使用自定义配置
python quick_start.py --config config.yaml
```

## 📋 配置文件说明

| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `config.template.yaml` | 完整配置模板 | 了解所有配置选项 |
| `config.example.yaml` | 简化示例配置 | 快速开始使用 |
| `CONFIG.md` | 详细配置指南 | 深入了解配置 |

## 🛠️ 常用命令

```bash
# 查看配置生成帮助
python generate_config.py --help

# 查看配置验证帮助
python validate_config.py --help

# 强制覆盖已存在的配置文件
python generate_config.py config.yaml minimal --force
```

## 🔧 环境准备

### Ollama 服务

```bash
# 启动Ollama服务
ollama serve

# 安装所需模型
ollama pull qwen3:1.7b
ollama pull nomic-embed-text:latest

# 查看已安装模型
ollama list
```

### Python 依赖

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用poetry
poetry install
```

## 🎯 使用示例

### 基本使用

```python
from ai_services import AIServiceFactory

# 使用默认配置
factory = AIServiceFactory.create_default()

# 使用自定义配置
factory = AIServiceFactory.from_config_file("config.yaml")

# 创建服务
chat_service = factory.create_service("chat")
embedding_service = factory.create_service("embedding")
rerank_service = factory.create_service("rerank")
```

### 异步使用

```python
import asyncio
from ai_services import AIServiceFactory

async def main():
    factory = AIServiceFactory.create_default()
    
    # 异步Chat
    chat_service = factory.create_service("chat")
    response = await chat_service.chat_async("你好")
    
    # 异步Embedding
    embedding_service = factory.create_service("embedding")
    embeddings = await embedding_service.embed_async(["文本1", "文本2"])
    
    # 异步Rerank
    rerank_service = factory.create_service("rerank")
    results = await rerank_service.rerank_async("查询", ["文档1", "文档2"])

asyncio.run(main())
```

## 🐛 故障排除

### 常见问题

1. **连接错误**
   ```bash
   # 检查Ollama服务状态
   curl http://localhost:11434/api/tags
   ```

2. **模型未找到**
   ```bash
   # 安装缺失的模型
   ollama pull <model_name>
   ```

3. **配置验证失败**
   ```bash
   # 使用详细模式查看错误
   python validate_config.py config.yaml --verbose
   ```

### 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在配置文件中设置
# logging:
#   level: "DEBUG"
```

## 📚 更多资源

- [详细配置指南](CONFIG.md)
- [API文档](docs/README.md)
- [示例代码](examples/)

## 💡 提示

- 首次使用建议从 `minimal` 配置开始
- 生产环境使用前请仔细验证配置
- 定期更新模型以获得更好性能
- 根据实际需求调整超时和重试参数