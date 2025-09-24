# AI Services 配置指南

## 概述

AI Services 模块支持灵活的配置管理，允许您自定义各种服务提供商和参数。

## 配置文件

### 1. 配置模板文件
- **`config.template.yaml`** - 完整的配置模板，包含所有可用选项和详细注释
- **`config.example.yaml`** - 简化的配置示例，适合快速开始

### 2. 使用配置文件

```python
from ai_services.core.config_manager import ConfigManager

# 从文件加载配置
config_manager = ConfigManager("config.yaml")

# 或使用默认配置
config_manager = ConfigManager()
```

## 配置结构

### 服务配置

#### Chat 服务
```yaml
services:
  chat:
    default_provider: "ollama"  # 默认提供商
    providers:
      ollama:
        base_url: "http://localhost:11434"
        model_name: "qwen3:1.7b"
        timeout: 30.0
        max_retries: 3
        stream: false
        options:
          temperature: 0.7    # 生成温度 (0.0-1.0)
          top_p: 0.9         # Top-p 采样
          top_k: 40          # Top-k 采样
```

#### Embedding 服务
```yaml
services:
  embedding:
    default_provider: "ollama"
    providers:
      ollama:
        base_url: "http://localhost:11434"
        model_name: "nomic-embed-text:latest"
        timeout: 30.0
        max_retries: 3
      local:
        model_name: "all-MiniLM-L6-v2"
        device: "cpu"  # 或 "cuda"
```

#### Rerank 服务
```yaml
services:
  rerank:
    default_provider: "ollama"  # 或 "embedding_based"
    providers:
      ollama:
        base_url: "http://localhost:11434"
        model_name: "qwen3:1.7b"
        timeout: 30.0
        max_retries: 3
      embedding_based:
        similarity_method: "cosine"  # cosine, dot, euclidean
        normalize_scores: true
      cross_encoder:
        model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        device: "cpu"
        batch_size: 32
```

### 日志配置
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null  # 设置文件路径以保存到文件
```

## 快速开始

### 1. 复制配置模板
```bash
cp config.example.yaml config.yaml
```

### 2. 修改配置
根据您的需求编辑 `config.yaml` 文件：

- 确保 Ollama 服务正在运行 (`ollama serve`)
- 确认模型已安装 (`ollama list`)
- 根据需要调整服务地址和端口

### 3. 验证配置
```python
from ai_services.config.config import validate_config
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

errors = validate_config(config)
if errors:
    print("配置错误:")
    for error in errors:
        print(f"  - {error}")
else:
    print("配置验证通过!")
```

## 环境变量支持

您也可以通过环境变量覆盖配置：

```bash
export AI_SERVICES_CHAT_MODEL="llama3:8b"
export AI_SERVICES_EMBEDDING_MODEL="nomic-embed-text:latest"
export AI_SERVICES_OLLAMA_URL="http://localhost:11434"
```

## 常见配置场景

### 1. 仅使用 Ollama 服务
```yaml
services:
  chat:
    default_provider: "ollama"
  embedding:
    default_provider: "ollama"
  rerank:
    default_provider: "ollama"
```

### 2. 混合使用不同提供商
```yaml
services:
  chat:
    default_provider: "ollama"
  embedding:
    default_provider: "local"  # 使用本地模型
  rerank:
    default_provider: "embedding_based"  # 基于嵌入的重排序
```

### 3. 生产环境配置
```yaml
services:
  chat:
    providers:
      ollama:
        timeout: 60.0      # 增加超时时间
        max_retries: 5     # 增加重试次数
        options:
          temperature: 0.3  # 降低温度以获得更一致的结果

logging:
  level: "WARNING"         # 减少日志输出
  file: "/var/log/ai_services.log"  # 保存到文件
```

## 故障排除

### 常见问题

1. **模型不存在错误**
   - 检查模型是否已安装: `ollama list`
   - 安装所需模型: `ollama pull qwen3:1.7b`

2. **连接超时**
   - 确认 Ollama 服务正在运行: `ollama serve`
   - 检查服务地址和端口配置

3. **配置验证失败**
   - 使用 `validate_config()` 函数检查配置
   - 参考模板文件确认配置格式

### 调试模式

启用调试日志以获取更多信息：
```yaml
logging:
  level: "DEBUG"
```

## 配置文件生成

您可以使用内置功能生成配置文件：

```python
from ai_services.config.config import create_config_template

# 生成 YAML 模板
create_config_template("my_config.yaml", "yaml")

# 生成 JSON 模板
create_config_template("my_config.json", "json")
```