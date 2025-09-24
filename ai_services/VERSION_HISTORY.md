# AI Services 版本历史

## 版本 v2.1.0 - OpenAI兼容格式支持 (2024-01-24)

### 🎉 新增功能

#### OpenAI兼容格式全面支持
- **多提供商支持**: 支持OpenAI官方、StepFun(阶跃星辰)、DeepSeek等所有兼容OpenAI API格式的服务
- **统一接口**: 所有OpenAI兼容服务使用相同的配置格式和调用方式
- **完整服务覆盖**: 支持聊天、嵌入、重排序三大核心服务

#### 配置文件更新
- **config.template.yaml**: 新增OpenAI和StepFun的配置模板
  - 聊天服务: 支持temperature、top_p、max_tokens等完整参数
  - 嵌入服务: 支持batch_size批处理配置
  - 重排序服务: 通过聊天模型实现重排序功能

#### 新增文件
1. **config.openai.yaml** - OpenAI兼容格式配置示例
   - OpenAI官方API配置 (GPT-3.5/4系列)
   - StepFun API配置 (Step-1V系列)
   - DeepSeek API配置 (DeepSeek-Chat/Coder)

2. **examples/openai_example.py** - 使用示例脚本
   - 支持单独测试特定提供商
   - 完整的聊天、嵌入、重排序示例
   - 错误处理和最佳实践演示

3. **docs/OPENAI_GUIDE.md** - 详细配置指南
   - 快速开始指南
   - 支持的服务提供商说明
   - 配置参数详解
   - 常见问题解答
   - 最佳实践建议

#### 文档更新
- **docs/README.md**: 新增OpenAI兼容格式配置章节
  - 快速使用指南
  - 支持的服务提供商列表
  - 配置特点说明

### 🔧 技术特性

#### 支持的服务提供商
- **OpenAI官方**
  - 聊天: gpt-3.5-turbo, gpt-4, gpt-4-turbo
  - 嵌入: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
  - 重排序: 通过聊天模型实现

- **StepFun (阶跃星辰)**
  - 聊天: step-1v-8k, step-1v-32k
  - 嵌入: step-1v-embedding
  - 重排序: 通过聊天模型实现

- **DeepSeek**
  - 聊天: deepseek-chat, deepseek-coder
  - 嵌入: 通过聊天模型实现
  - 重排序: 通过聊天模型实现

#### 配置参数
- **base_url**: API服务地址
- **api_key**: API密钥 (支持环境变量)
- **model_name**: 模型名称
- **timeout**: 请求超时时间
- **max_retries**: 最大重试次数
- **batch_size**: 批处理大小 (嵌入/重排序)
- **options**: 聊天参数 (temperature, top_p, max_tokens等)

### 📋 使用方法

#### 快速开始
```bash
# 1. 设置API密钥
export OPENAI_API_KEY="your_openai_api_key"
export STEPFUN_API_KEY="your_stepfun_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"

# 2. 使用OpenAI配置
cp config.openai.yaml config.yaml

# 3. 验证配置
python validate_config.py config.yaml

# 4. 运行示例
python examples/openai_example.py
```

#### 单独测试提供商
```bash
# 测试OpenAI
python examples/openai_example.py openai

# 测试StepFun
python examples/openai_example.py stepfun

# 测试DeepSeek
python examples/openai_example.py deepseek
```

### ✅ 验证状态
- ✅ config.template.yaml - 配置验证通过
- ✅ config.openai.yaml - 配置验证通过
- ✅ 所有示例脚本运行正常
- ✅ 文档完整性检查通过

### 🔄 兼容性
- **向后兼容**: 完全兼容现有的Ollama和GLM-4.5配置
- **配置迁移**: 无需修改现有配置文件
- **API兼容**: 保持现有API接口不变

### 📁 文件结构
```
ai_services/
├── config.template.yaml      # 更新: 新增OpenAI兼容配置
├── config.openai.yaml        # 新增: OpenAI配置示例
├── docs/
│   ├── README.md             # 更新: 新增OpenAI配置说明
│   └── OPENAI_GUIDE.md       # 新增: OpenAI配置指南
└── examples/
    └── openai_example.py     # 新增: OpenAI使用示例
```

---

## 版本 v2.0.0 - GLM-4.5支持 (2024-01-23)

### 🎉 新增功能
- 支持GLM-4.5模型配置
- 新增GLM-4.5配置示例和使用指南
- 完整的聊天、嵌入、重排序服务支持

### 📁 新增文件
- config.glm4.yaml - GLM-4.5配置示例
- examples/glm4_example.py - GLM-4.5使用示例
- docs/GLM4_GUIDE.md - GLM-4.5配置指南

---

## 版本 v1.0.0 - 基础版本 (2024-01-20)

### 🎉 初始功能
- 基础AI服务框架
- Ollama本地模型支持
- 聊天、嵌入、重排序服务
- 配置管理系统
- 基础文档和示例