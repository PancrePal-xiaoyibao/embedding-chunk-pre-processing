# 更新日志

## [v2.1.0] - 2024-01-24

### 🎉 新增功能
- **OpenAI兼容格式全面支持**: 支持OpenAI官方、StepFun、DeepSeek等所有兼容OpenAI API格式的服务提供商
- **统一配置接口**: 所有OpenAI兼容服务使用相同的配置格式，便于管理和切换
- **完整服务覆盖**: 聊天、嵌入、重排序三大服务全面支持OpenAI格式

### 📁 新增文件
- `config.openai.yaml` - OpenAI兼容格式配置示例
- `examples/openai_example.py` - OpenAI兼容服务使用示例
- `docs/OPENAI_GUIDE.md` - OpenAI兼容格式详细配置指南
- `VERSION_HISTORY.md` - 完整版本历史记录
- `CHANGELOG.md` - 更新日志

### 🔧 配置更新
- **config.template.yaml**: 新增OpenAI和StepFun的配置模板
  - 聊天服务: 支持temperature、top_p、max_tokens等参数
  - 嵌入服务: 支持batch_size批处理配置
  - 重排序服务: 通过聊天模型实现

### 📖 文档更新
- **docs/README.md**: 新增OpenAI兼容格式配置章节
- 新增快速使用指南和配置说明

### 🔄 兼容性
- ✅ 完全向后兼容现有Ollama和GLM-4.5配置
- ✅ 无需修改现有配置文件
- ✅ 保持现有API接口不变

### 🧪 测试验证
- ✅ 所有配置文件验证通过
- ✅ 示例脚本运行正常
- ✅ 文档完整性检查通过

---

## [v2.0.0] - 2024-01-23

### 🎉 新增功能
- **GLM-4.5模型支持**: 完整支持GLM-4.5聊天、嵌入、重排序服务
- **GLM-4.5配置示例**: 提供开箱即用的配置文件
- **详细使用指南**: 包含配置说明和最佳实践

### 📁 新增文件
- `config.glm4.yaml` - GLM-4.5配置示例
- `examples/glm4_example.py` - GLM-4.5使用示例
- `docs/GLM4_GUIDE.md` - GLM-4.5配置指南

---

## [v1.0.0] - 2024-01-20

### 🎉 初始版本
- **基础AI服务框架**: 核心服务架构和接口定义
- **Ollama本地模型支持**: 完整的本地模型集成
- **三大核心服务**: 聊天、嵌入、重排序服务
- **配置管理系统**: 灵活的配置管理和验证
- **基础文档**: 使用指南和API参考