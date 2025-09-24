# GLM-4.5 配置指南

本指南详细介绍如何在AI Services中配置和使用GLM-4.5模型。

## 📋 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API密钥设置](#api密钥设置)
- [模型选择](#模型选择)
- [使用示例](#使用示例)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)

## 🚀 快速开始

### 1. 获取API密钥

1. 访问 [智谱AI开放平台](https://open.bigmodel.cn/)
2. 注册账号并完成认证
3. 创建API密钥
4. 记录您的API密钥

### 2. 设置环境变量

```bash
# 方法1: 直接设置环境变量
export GLM_API_KEY="your_api_key_here"

# 方法2: 创建.env文件
echo "GLM_API_KEY=your_api_key_here" > .env
```

### 3. 使用GLM-4.5配置

```bash
# 复制GLM-4.5配置模板
cp config.glm4.yaml my_glm4_config.yaml

# 验证配置
python validate_config.py my_glm4_config.yaml

# 运行示例
python examples/glm4_example.py
```

## ⚙️ 配置说明

### 完整配置结构

```yaml
version: "1.0"

services:
  # 聊天服务配置
  chat:
    default_provider: "glm4"
    providers:
      glm4:
        base_url: "https://open.bigmodel.cn/api/paas/v4/"
        api_key: "${GLM_API_KEY}"
        model_name: "glm-4-plus"
        timeout: 60.0
        max_retries: 3
        stream: false
        options:
          temperature: 0.7      # 生成温度 (0.01-0.99)
          top_p: 0.9           # Top-p采样 (0.01-0.99)
          max_tokens: 4096     # 最大输出token数
          do_sample: true      # 是否启用采样

  # 嵌入服务配置
  embedding:
    default_provider: "glm4"
    providers:
      glm4:
        base_url: "https://open.bigmodel.cn/api/paas/v4/"
        api_key: "${GLM_API_KEY}"
        model_name: "embedding-2"
        timeout: 60.0
        max_retries: 3
        batch_size: 100      # 批处理大小

  # 重排序服务配置
  rerank:
    default_provider: "glm4"
    providers:
      glm4:
        base_url: "https://open.bigmodel.cn/api/paas/v4/"
        api_key: "${GLM_API_KEY}"
        model_name: "glm-4-plus"
        timeout: 60.0
        max_retries: 3
        batch_size: 10       # 重排序批处理大小

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: None
```

### 配置参数详解

#### 聊天服务参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_url` | string | `https://open.bigmodel.cn/api/paas/v4/` | GLM API基础URL |
| `api_key` | string | - | API密钥，建议使用环境变量 |
| `model_name` | string | `glm-4-plus` | 模型名称 |
| `timeout` | float | 60.0 | 请求超时时间（秒） |
| `max_retries` | int | 3 | 最大重试次数 |
| `stream` | bool | false | 是否使用流式响应 |
| `temperature` | float | 0.7 | 生成温度，控制随机性 |
| `top_p` | float | 0.9 | Top-p采样参数 |
| `max_tokens` | int | 4096 | 最大输出token数 |
| `do_sample` | bool | true | 是否启用采样 |

#### 嵌入服务参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | string | `embedding-2` | GLM嵌入模型名称 |
| `batch_size` | int | 100 | 批处理大小 |

#### 重排序服务参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | 10 | 重排序批处理大小 |

## 🔑 API密钥设置

### 环境变量方式（推荐）

```bash
# Linux/macOS
export GLM_API_KEY="your_api_key_here"

# Windows
set GLM_API_KEY=your_api_key_here
```

### .env文件方式

创建 `.env` 文件：

```env
GLM_API_KEY=your_api_key_here
```

### 配置文件方式（不推荐）

```yaml
glm4:
  api_key: "your_api_key_here"  # 不推荐直接写在配置文件中
```

## 🎯 模型选择

### 聊天模型

| 模型名称 | 说明 | 适用场景 |
|----------|------|----------|
| `glm-4-plus` | GLM-4.5最新版本 | 通用对话、文本生成 |
| `glm-4` | GLM-4标准版本 | 基础对话任务 |
| `glm-4-air` | GLM-4轻量版 | 快速响应场景 |
| `glm-4-airx` | GLM-4增强轻量版 | 平衡性能和速度 |

### 嵌入模型

| 模型名称 | 说明 | 维度 |
|----------|------|------|
| `embedding-2` | GLM嵌入模型v2 | 1024 |

## 💻 使用示例

### 基础聊天示例

```python
import asyncio
from src.chat.chat_service import ChatService
from config.config import load_config

async def chat_example():
    config = load_config("config.glm4.yaml")
    chat_service = ChatService(config)
    
    messages = [
        {"role": "user", "content": "你好，GLM-4.5！"}
    ]
    
    response = await chat_service.chat(messages)
    print(f"回复: {response}")

# 运行示例
asyncio.run(chat_example())
```

### 嵌入向量示例

```python
import asyncio
from src.embedding.embedding_service import EmbeddingService
from config.config import load_config

async def embedding_example():
    config = load_config("config.glm4.yaml")
    embedding_service = EmbeddingService(config)
    
    texts = ["人工智能", "机器学习", "深度学习"]
    embeddings = await embedding_service.embed(texts)
    
    print(f"嵌入向量维度: {len(embeddings[0])}")

# 运行示例
asyncio.run(embedding_example())
```

### 重排序示例

```python
import asyncio
from src.rerank.rerank_service import RerankService
from config.config import load_config

async def rerank_example():
    config = load_config("config.glm4.yaml")
    rerank_service = RerankService(config)
    
    query = "什么是人工智能？"
    documents = [
        "人工智能是计算机科学的一个分支",
        "机器学习是AI的核心技术",
        "深度学习使用神经网络"
    ]
    
    results = await rerank_service.rerank(query, documents)
    for doc, score in results:
        print(f"分数: {score:.4f} - {doc}")

# 运行示例
asyncio.run(rerank_example())
```

## ❓ 常见问题

### Q1: API密钥错误

**问题**: `Authentication failed` 或 `Invalid API key`

**解决方案**:
1. 检查API密钥是否正确
2. 确认环境变量设置正确
3. 验证API密钥是否有效且未过期

### Q2: 请求超时

**问题**: `Request timeout` 或连接超时

**解决方案**:
1. 增加 `timeout` 参数值
2. 检查网络连接
3. 确认GLM服务状态

### Q3: 配额不足

**问题**: `Quota exceeded` 或 `Rate limit exceeded`

**解决方案**:
1. 检查API配额使用情况
2. 升级API套餐
3. 实现请求频率控制

### Q4: 模型不存在

**问题**: `Model not found` 或 `Invalid model`

**解决方案**:
1. 检查模型名称是否正确
2. 确认账号是否有权限使用该模型
3. 参考最新的模型列表

## 🎯 最佳实践

### 1. 安全性

- ✅ 使用环境变量存储API密钥
- ✅ 不要在代码中硬编码密钥
- ✅ 定期轮换API密钥
- ❌ 不要将密钥提交到版本控制

### 2. 性能优化

- ✅ 合理设置批处理大小
- ✅ 使用连接池
- ✅ 实现请求重试机制
- ✅ 监控API使用情况

### 3. 错误处理

- ✅ 实现完整的异常处理
- ✅ 记录详细的错误日志
- ✅ 提供友好的错误提示
- ✅ 实现降级策略

### 4. 配置管理

- ✅ 使用配置文件管理参数
- ✅ 支持环境变量覆盖
- ✅ 验证配置的有效性
- ✅ 提供配置模板

## 📚 相关资源

- [GLM-4.5 官方文档](https://open.bigmodel.cn/dev/api)
- [智谱AI开放平台](https://open.bigmodel.cn/)
- [API参考文档](https://open.bigmodel.cn/dev/api#overview)
- [定价信息](https://open.bigmodel.cn/pricing)

## 🆘 获取帮助

如果您遇到问题，可以：

1. 查看本指南的常见问题部分
2. 运行配置验证工具：`python validate_config.py config.glm4.yaml`
3. 查看详细的错误日志
4. 参考官方文档和API说明

---

**注意**: 请确保您的API密钥安全，不要在公开的代码仓库中暴露密钥信息。