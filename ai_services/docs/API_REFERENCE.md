# AI Services API 参考文档

## 核心类和接口

### AIServiceFactory

统一的AI服务工厂类，用于创建和管理各种AI服务。

#### 类方法

##### `create_default() -> AIServiceFactory`
创建使用默认配置的工厂实例。

**返回值:**
- `AIServiceFactory`: 工厂实例

**示例:**
```python
factory = AIServiceFactory.create_default()
```

##### `from_config(config: Dict[str, Any]) -> AIServiceFactory`
从配置字典创建工厂实例。

**参数:**
- `config`: 配置字典

**返回值:**
- `AIServiceFactory`: 工厂实例

**示例:**
```python
config = {"services": {"chat": {...}}}
factory = AIServiceFactory.from_config(config)
```

##### `from_config_file(config_path: str) -> AIServiceFactory`
从配置文件创建工厂实例。

**参数:**
- `config_path`: 配置文件路径

**返回值:**
- `AIServiceFactory`: 工厂实例

**异常:**
- `FileNotFoundError`: 配置文件不存在
- `ConfigurationError`: 配置格式错误

**示例:**
```python
factory = AIServiceFactory.from_config_file("config.yaml")
```

#### 实例方法

##### `create_service(service_type: str, provider: str = None) -> BaseService`
创建指定类型的服务实例。

**参数:**
- `service_type`: 服务类型 ("chat", "embedding", "rerank")
- `provider`: 提供商名称（可选，使用默认提供商）

**返回值:**
- `BaseService`: 服务实例

**异常:**
- `ServiceNotAvailableError`: 服务不可用
- `ConfigurationError`: 配置错误

**示例:**
```python
chat_service = factory.create_service("chat")
embedding_service = factory.create_service("embedding", "local")
```

##### `get_available_providers(service_type: str) -> List[str]`
获取指定服务类型的可用提供商列表。

**参数:**
- `service_type`: 服务类型

**返回值:**
- `List[str]`: 提供商名称列表

##### `test_provider_connection(service_type: str, provider: str) -> bool`
测试指定提供商的连接状态。

**参数:**
- `service_type`: 服务类型
- `provider`: 提供商名称

**返回值:**
- `bool`: 连接是否正常

##### `health_check() -> Dict[str, bool]`
检查所有服务的健康状态。

**返回值:**
- `Dict[str, bool]`: 服务类型到健康状态的映射

## 服务接口

### ChatService

聊天服务的抽象基类。

#### 抽象方法

##### `chat(messages: Union[str, List[ChatMessage]], **kwargs) -> ChatResponse`
发送聊天消息并获取响应。

**参数:**
- `messages`: 消息内容（字符串或消息列表）
- `**kwargs`: 额外参数

**返回值:**
- `ChatResponse`: 聊天响应

##### `chat_async(messages: Union[str, List[ChatMessage]], **kwargs) -> ChatResponse`
异步发送聊天消息。

##### `chat_stream(messages: Union[str, List[ChatMessage]], **kwargs) -> Iterator[ChatResponse]`
流式聊天，返回响应流。

#### 实现类

##### OllamaChatService

基于Ollama的聊天服务实现。

**额外方法:**
- `get_available_models() -> List[str]`: 获取可用模型列表
- `pull_model(model_name: str) -> bool`: 拉取指定模型

### EmbeddingService

嵌入服务的抽象基类。

#### 抽象方法

##### `embed(texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse`
生成文本嵌入向量。

**参数:**
- `texts`: 文本内容（字符串或字符串列表）
- `**kwargs`: 额外参数

**返回值:**
- `EmbeddingResponse`: 嵌入响应

##### `embed_async(texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse`
异步生成嵌入向量。

##### `compute_similarity(vector1: List[float], vector2: List[float], method: str = "cosine") -> float`
计算两个向量的相似度。

**参数:**
- `vector1`: 第一个向量
- `vector2`: 第二个向量
- `method`: 相似度计算方法 ("cosine", "dot", "euclidean")

**返回值:**
- `float`: 相似度分数

#### 实现类

##### OllamaEmbeddingService
基于Ollama的嵌入服务。

##### LocalEmbeddingService
基于本地模型的嵌入服务。

### RerankService

重排序服务的抽象基类。

#### 抽象方法

##### `rerank(query: str, documents: List[str], top_k: int = None, **kwargs) -> RerankResponse`
对文档进行重排序。

**参数:**
- `query`: 查询文本
- `documents`: 候选文档列表
- `top_k`: 返回前K个结果（可选）
- `**kwargs`: 额外参数

**返回值:**
- `RerankResponse`: 重排序响应

##### `rerank_async(query: str, documents: List[str], top_k: int = None, **kwargs) -> RerankResponse`
异步重排序。

#### 实现类

##### EmbeddingBasedRerankService
基于嵌入向量的重排序服务。

##### OllamaRerankService
基于Ollama的重排序服务。

##### CrossEncoderRerankService
基于交叉编码器的重排序服务。

## 数据模型

### ChatMessage

聊天消息数据类。

**属性:**
- `role: MessageRole`: 消息角色
- `content: str`: 消息内容
- `timestamp: Optional[datetime]`: 时间戳

**工厂方法:**
- `create_system_message(content: str) -> ChatMessage`
- `create_user_message(content: str) -> ChatMessage`
- `create_assistant_message(content: str) -> ChatMessage`

### ChatResponse

聊天响应数据类。

**属性:**
- `content: str`: 响应内容
- `model: str`: 使用的模型
- `usage: Optional[ChatUsage]`: 使用统计
- `finish_reason: Optional[str]`: 完成原因
- `timestamp: datetime`: 响应时间戳

### EmbeddingResponse

嵌入响应数据类。

**属性:**
- `vectors: List[List[float]]`: 嵌入向量列表
- `model: str`: 使用的模型
- `usage: Optional[EmbeddingUsage]`: 使用统计
- `timestamp: datetime`: 响应时间戳

### RerankResponse

重排序响应数据类。

**属性:**
- `results: List[RerankResult]`: 排序结果列表
- `model: str`: 使用的模型
- `usage: Optional[RerankUsage]`: 使用统计
- `timestamp: datetime`: 响应时间戳

### RerankResult

重排序结果数据类。

**属性:**
- `index: int`: 原始索引
- `document: str`: 文档内容
- `score: float`: 相关性分数

## 配置管理

### 配置函数

##### `get_default_config() -> Dict[str, Any]`
获取默认配置字典。

##### `create_config_template(output_path: str, format: str = "yaml") -> None`
创建配置模板文件。

**参数:**
- `output_path`: 输出文件路径
- `format`: 配置格式 ("yaml" 或 "json")

##### `validate_config(config: Dict[str, Any]) -> List[str]`
验证配置的有效性。

**参数:**
- `config`: 配置字典

**返回值:**
- `List[str]`: 验证错误列表（空列表表示验证通过）

##### `load_config_from_env() -> Dict[str, Any]`
从环境变量加载配置。

##### `merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]`
合并两个配置字典。

## 异常类

### AIServiceError
所有AI服务异常的基类。

### ConfigurationError
配置相关错误。

### ServiceNotAvailableError
服务不可用错误。

### ConnectionError
连接错误。

### AuthenticationError
认证错误。

### RateLimitError
速率限制错误。

### ValidationError
验证错误。

### ModelNotFoundError
模型未找到错误。

### ServiceTimeoutError
服务超时错误。

## 便捷函数

### `create_chat_service(provider: str = None, **kwargs) -> ChatService`
创建聊天服务的便捷函数。

### `create_embedding_service(provider: str = None, **kwargs) -> EmbeddingService`
创建嵌入服务的便捷函数。

### `create_rerank_service(provider: str = None, **kwargs) -> RerankService`
创建重排序服务的便捷函数。

## 使用示例

### 基本使用

```python
from ai_services import AIServiceFactory

# 创建工厂
factory = AIServiceFactory.create_default()

# 创建服务
chat = factory.create_service("chat")
embedding = factory.create_service("embedding")
rerank = factory.create_service("rerank")

# 使用服务
response = chat.chat("Hello")
vectors = embedding.embed(["text1", "text2"])
ranked = rerank.rerank("query", ["doc1", "doc2"])
```

### 异步使用

```python
import asyncio

async def main():
    factory = AIServiceFactory.create_default()
    chat = factory.create_service("chat")
    
    # 异步聊天
    response = await chat.chat_async("Hello")
    print(response.content)

asyncio.run(main())
```

### 流式聊天

```python
factory = AIServiceFactory.create_default()
chat = factory.create_service("chat")

# 流式响应
for chunk in chat.chat_stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### 错误处理

```python
from ai_services.core.exceptions import AIServiceError

try:
    response = chat.chat("Hello")
except AIServiceError as e:
    print(f"服务错误: {e}")
```