# OpenAI兼容格式配置指南

本指南详细介绍如何在AI Services中配置和使用OpenAI兼容格式的API服务，包括OpenAI官方API、StepFun、DeepSeek等提供商。

## 📋 目录

- [快速开始](#快速开始)
- [支持的服务提供商](#支持的服务提供商)
- [配置说明](#配置说明)
- [API密钥设置](#api密钥设置)
- [模型选择](#模型选择)
- [使用示例](#使用示例)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)
- [相关资源](#相关资源)

## 🚀 快速开始

### 1. 设置API密钥

```bash
# OpenAI API密钥
export OPENAI_API_KEY="your_openai_api_key_here"

# StepFun API密钥
export STEPFUN_API_KEY="your_stepfun_api_key_here"

# DeepSeek API密钥
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

### 2. 复制配置文件

```bash
cp config.openai.yaml config.yaml
```

### 3. 验证配置

```bash
python validate_config.py config.yaml
```

### 4. 运行示例

```bash
# 测试所有OpenAI兼容服务
python examples/openai_example.py

# 测试特定提供商
python examples/openai_example.py openai
python examples/openai_example.py stepfun
python examples/openai_example.py deepseek
```

## 🏢 支持的服务提供商

### OpenAI 官方

- **API地址**: `https://api.openai.com/v1`
- **支持服务**: 聊天、嵌入、重排序
- **推荐模型**:
  - 聊天: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
  - 嵌入: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`

### StepFun (阶跃星辰)

- **API地址**: `https://api.stepfun.com/v1`
- **支持服务**: 聊天、嵌入、重排序
- **推荐模型**:
  - 聊天: `step-1v-8k`, `step-1v-32k`, `step-2-16k`
  - 嵌入: `step-1v-embedding`

### DeepSeek

- **API地址**: `https://api.deepseek.com/v1`
- **支持服务**: 聊天、重排序
- **推荐模型**:
  - 聊天: `deepseek-chat`, `deepseek-coder`

### 其他兼容提供商

支持任何遵循OpenAI API格式的服务提供商，只需修改`base_url`和相应的API密钥即可。

## ⚙️ 配置说明

### 聊天服务配置

```yaml
chat:
  default_provider: "openai"  # 默认提供商
  providers:
    openai:
      base_url: "https://api.openai.com/v1"  # API基础URL
      api_key: "${OPENAI_API_KEY}"           # API密钥
      model_name: "gpt-3.5-turbo"           # 模型名称
      timeout: 60.0                         # 超时时间(秒)
      max_retries: 3                        # 最大重试次数
      stream: false                         # 是否启用流式响应
      options:                              # 模型参数
        temperature: 0.7                    # 生成温度 (0.0-2.0)
        top_p: 1.0                         # Top-p采样 (0.0-1.0)
        max_tokens: 4096                   # 最大输出token数
        frequency_penalty: 0.0             # 频率惩罚 (-2.0-2.0)
        presence_penalty: 0.0              # 存在惩罚 (-2.0-2.0)
```

### 嵌入服务配置

```yaml
embedding:
  default_provider: "openai"
  providers:
    openai:
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
      model_name: "text-embedding-ada-002"
      timeout: 60.0
      max_retries: 3
      batch_size: 100                      # 批处理大小
```

### 重排序服务配置

```yaml
rerank:
  default_provider: "openai"
  providers:
    openai:
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
      model_name: "gpt-3.5-turbo"
      timeout: 60.0
      max_retries: 3
      batch_size: 10                       # 批处理大小
```

## 🔑 API密钥设置

### 环境变量方式 (推荐)

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export OPENAI_API_KEY="sk-..."
export STEPFUN_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."

# 重新加载配置
source ~/.bashrc  # 或 source ~/.zshrc
```

### .env 文件方式

创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-...
STEPFUN_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

### 配置文件直接设置 (不推荐)

```yaml
api_key: "sk-your-actual-api-key-here"
```

⚠️ **安全提醒**: 不要将API密钥直接写入配置文件并提交到版本控制系统。

## 🎯 模型选择

### OpenAI 模型

| 服务 | 模型名称 | 特点 | 适用场景 |
|------|----------|------|----------|
| 聊天 | gpt-3.5-turbo | 快速、经济 | 一般对话、简单任务 |
| 聊天 | gpt-4 | 高质量、复杂推理 | 复杂任务、专业分析 |
| 聊天 | gpt-4-turbo | 平衡性能和成本 | 大多数应用场景 |
| 嵌入 | text-embedding-ada-002 | 通用嵌入 | 语义搜索、相似度计算 |
| 嵌入 | text-embedding-3-small | 小型高效 | 轻量级应用 |
| 嵌入 | text-embedding-3-large | 大型精确 | 高精度要求 |

### StepFun 模型

| 服务 | 模型名称 | 特点 | 适用场景 |
|------|----------|------|----------|
| 聊天 | step-1v-8k | 8K上下文 | 短对话 |
| 聊天 | step-1v-32k | 32K上下文 | 长文档处理 |
| 聊天 | step-2-16k | 新一代模型 | 平衡性能 |
| 嵌入 | step-1v-embedding | 中文优化 | 中文语义理解 |

### DeepSeek 模型

| 服务 | 模型名称 | 特点 | 适用场景 |
|------|----------|------|----------|
| 聊天 | deepseek-chat | 通用对话 | 日常对话 |
| 聊天 | deepseek-coder | 代码专用 | 编程辅助 |

## 💻 使用示例

### Python 代码示例

```python
import asyncio
from src.chat.chat_service import ChatService
from src.embedding.embedding_service import EmbeddingService
from src.rerank.rerank_service import RerankService
from config.config_loader import ConfigLoader

async def main():
    # 加载配置
    config = ConfigLoader.load_config("config.openai.yaml")
    
    # 1. 聊天服务
    chat_service = ChatService(config, provider="openai")
    messages = [{"role": "user", "content": "你好！"}]
    response = await chat_service.chat(messages)
    print(f"聊天回复: {response['content']}")
    
    # 2. 嵌入服务
    embedding_service = EmbeddingService(config, provider="openai")
    texts = ["人工智能", "机器学习", "深度学习"]
    embeddings = await embedding_service.embed(texts)
    print(f"嵌入向量维度: {len(embeddings[0])}")
    
    # 3. 重排序服务
    rerank_service = RerankService(config, provider="openai")
    query = "什么是AI？"
    docs = ["AI是人工智能", "ML是机器学习", "DL是深度学习"]
    ranked = await rerank_service.rerank(query, docs)
    print(f"重排序结果: {ranked}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 命令行使用

```bash
# 使用OpenAI进行聊天
python -c "
from src.chat.chat_service import ChatService
from config.config_loader import ConfigLoader
config = ConfigLoader.load_config('config.openai.yaml')
service = ChatService(config, provider='openai')
result = service.chat_sync([{'role': 'user', 'content': '你好'}])
print(result['content'])
"
```

## ❓ 常见问题

### Q1: API密钥无效错误

**问题**: `Invalid API key provided`

**解决方案**:
1. 检查API密钥是否正确设置
2. 确认API密钥有效且未过期
3. 验证环境变量是否正确加载

```bash
# 检查环境变量
echo $OPENAI_API_KEY
echo $STEPFUN_API_KEY
echo $DEEPSEEK_API_KEY
```

### Q2: 网络连接超时

**问题**: `Connection timeout`

**解决方案**:
1. 检查网络连接
2. 增加超时时间配置
3. 使用代理服务器

```yaml
timeout: 120.0  # 增加到120秒
```

### Q3: 模型不存在错误

**问题**: `Model not found`

**解决方案**:
1. 检查模型名称是否正确
2. 确认API密钥有权限访问该模型
3. 查看提供商的最新模型列表

### Q4: 请求频率限制

**问题**: `Rate limit exceeded`

**解决方案**:
1. 增加重试次数和延迟
2. 使用批处理减少请求频率
3. 升级API套餐

```yaml
max_retries: 5
batch_size: 50  # 减少批处理大小
```

### Q5: 响应格式错误

**问题**: `Invalid response format`

**解决方案**:
1. 检查API地址是否正确
2. 确认提供商API兼容OpenAI格式
3. 查看错误日志获取详细信息

## 🎯 最佳实践

### 1. 安全性

- ✅ 使用环境变量存储API密钥
- ✅ 定期轮换API密钥
- ✅ 限制API密钥权限
- ❌ 不要在代码中硬编码密钥
- ❌ 不要将密钥提交到版本控制

### 2. 性能优化

- ✅ 合理设置批处理大小
- ✅ 使用连接池和重试机制
- ✅ 缓存常用的嵌入向量
- ✅ 选择合适的模型和参数

### 3. 成本控制

- ✅ 选择性价比高的模型
- ✅ 设置合理的token限制
- ✅ 监控API使用量
- ✅ 使用本地模型作为备选

### 4. 错误处理

- ✅ 实现完善的重试机制
- ✅ 记录详细的错误日志
- ✅ 提供降级方案
- ✅ 监控服务可用性

### 5. 配置管理

```yaml
# 推荐的配置结构
chat:
  default_provider: "openai"
  fallback_provider: "stepfun"  # 备用提供商
  providers:
    openai:
      # 生产环境配置
      timeout: 60.0
      max_retries: 3
      options:
        temperature: 0.7
        max_tokens: 4096
```

## 📚 相关资源

### 官方文档

- [OpenAI API 文档](https://platform.openai.com/docs)
- [StepFun API 文档](https://platform.stepfun.com/docs)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)

### 示例代码

- `examples/openai_example.py` - 完整使用示例
- `config.openai.yaml` - 配置文件示例
- `config.template.yaml` - 配置模板

### 相关工具

- `validate_config.py` - 配置验证工具
- `generate_config.py` - 配置生成工具

### 社区资源

- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Awesome OpenAI](https://github.com/humanloop/awesome-chatgpt)

## 🔄 更新日志

### v1.0.0 (2024-01-XX)

- ✅ 添加OpenAI官方API支持
- ✅ 添加StepFun API支持
- ✅ 添加DeepSeek API支持
- ✅ 实现聊天、嵌入、重排序服务
- ✅ 提供完整的配置示例和文档

---

## 📞 技术支持

如果您在使用过程中遇到问题，请：

1. 查看本文档的常见问题部分
2. 检查配置文件和环境变量
3. 查看日志文件获取详细错误信息
4. 参考示例代码进行调试

**注意**: 请确保您的API密钥安全，不要在公开场所分享。