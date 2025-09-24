#!/usr/bin/env python3
"""
OpenAI兼容格式使用示例

本脚本演示如何使用OpenAI兼容的API服务，包括：
- OpenAI官方API
- StepFun API
- DeepSeek API
- 其他兼容OpenAI格式的API服务

功能包括：
1. 聊天对话
2. 文本嵌入
3. 文档重排序

使用前请确保：
1. 已安装依赖：pip install -r requirements.txt
2. 设置API密钥环境变量
3. 配置文件正确
"""

import os
import sys
import asyncio
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat.chat_service import ChatService
from src.embedding.embedding_service import EmbeddingService
from src.rerank.rerank_service import RerankService
from config.config_loader import ConfigLoader


def setup_environment():
    """
    设置环境变量
    
    请根据需要设置相应的API密钥：
    - OPENAI_API_KEY: OpenAI API密钥
    - STEPFUN_API_KEY: StepFun API密钥
    - DEEPSEEK_API_KEY: DeepSeek API密钥
    """
    # 检查必要的环境变量
    required_keys = {
        'OPENAI_API_KEY': 'OpenAI API密钥',
        'STEPFUN_API_KEY': 'StepFun API密钥',
        'DEEPSEEK_API_KEY': 'DeepSeek API密钥'
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} ({description})")
    
    if missing_keys:
        print("⚠️  警告：以下环境变量未设置：")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n请设置相应的API密钥环境变量，例如：")
        print("export OPENAI_API_KEY='your_openai_api_key'")
        print("export STEPFUN_API_KEY='your_stepfun_api_key'")
        print("export DEEPSEEK_API_KEY='your_deepseek_api_key'")
        print()


async def test_chat_service(provider: str = "openai"):
    """
    测试聊天服务
    
    Args:
        provider: 服务提供商 (openai, stepfun, deepseek)
    """
    print(f"\n🤖 测试 {provider.upper()} 聊天服务")
    print("-" * 50)
    
    try:
        # 加载配置
        config = ConfigLoader.load_config("config.openai.yaml")
        
        # 创建聊天服务
        chat_service = ChatService(config, provider=provider)
        
        # 测试消息
        messages = [
            {"role": "user", "content": "你好！请简单介绍一下人工智能。"}
        ]
        
        print(f"📤 发送消息: {messages[0]['content']}")
        
        # 发送聊天请求
        response = await chat_service.chat(messages)
        
        print(f"📥 {provider.upper()} 回复:")
        print(f"   {response['content']}")
        print(f"   模型: {response.get('model', 'unknown')}")
        print(f"   用时: {response.get('response_time', 0):.2f}秒")
        
    except Exception as e:
        print(f"❌ {provider.upper()} 聊天服务测试失败: {str(e)}")


async def test_embedding_service(provider: str = "openai"):
    """
    测试嵌入服务
    
    Args:
        provider: 服务提供商 (openai, stepfun)
    """
    print(f"\n🔢 测试 {provider.upper()} 嵌入服务")
    print("-" * 50)
    
    try:
        # 加载配置
        config = ConfigLoader.load_config("config.openai.yaml")
        
        # 创建嵌入服务
        embedding_service = EmbeddingService(config, provider=provider)
        
        # 测试文本
        texts = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的重要组成部分",
            "深度学习是机器学习的一个子领域"
        ]
        
        print(f"📤 处理文本数量: {len(texts)}")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. {text}")
        
        # 获取嵌入向量
        embeddings = await embedding_service.embed(texts)
        
        print(f"📥 {provider.upper()} 嵌入结果:")
        print(f"   向量数量: {len(embeddings)}")
        print(f"   向量维度: {len(embeddings[0]) if embeddings else 0}")
        print(f"   第一个向量前5维: {embeddings[0][:5] if embeddings else []}")
        
    except Exception as e:
        print(f"❌ {provider.upper()} 嵌入服务测试失败: {str(e)}")


async def test_rerank_service(provider: str = "openai"):
    """
    测试重排序服务
    
    Args:
        provider: 服务提供商 (openai, stepfun)
    """
    print(f"\n📊 测试 {provider.upper()} 重排序服务")
    print("-" * 50)
    
    try:
        # 加载配置
        config = ConfigLoader.load_config("config.openai.yaml")
        
        # 创建重排序服务
        rerank_service = RerankService(config, provider=provider)
        
        # 测试查询和文档
        query = "什么是机器学习？"
        documents = [
            "机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习。",
            "深度学习使用神经网络来模拟人脑的学习过程。",
            "自然语言处理是计算机科学和人工智能的一个分支。",
            "计算机视觉是让计算机能够理解和解释视觉信息的技术。"
        ]
        
        print(f"📤 查询: {query}")
        print(f"📤 文档数量: {len(documents)}")
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc}")
        
        # 执行重排序
        ranked_docs = await rerank_service.rerank(query, documents)
        
        print(f"📥 {provider.upper()} 重排序结果:")
        for i, (doc, score) in enumerate(ranked_docs, 1):
            print(f"   {i}. [分数: {score:.4f}] {doc}")
        
    except Exception as e:
        print(f"❌ {provider.upper()} 重排序服务测试失败: {str(e)}")


def test_sync_chat(provider: str = "openai"):
    """
    测试同步聊天服务
    
    Args:
        provider: 服务提供商 (openai, stepfun, deepseek)
    """
    print(f"\n🔄 测试 {provider.upper()} 同步聊天服务")
    print("-" * 50)
    
    try:
        # 加载配置
        config = ConfigLoader.load_config("config.openai.yaml")
        
        # 创建聊天服务
        chat_service = ChatService(config, provider=provider)
        
        # 测试消息
        messages = [
            {"role": "user", "content": "请用一句话解释什么是深度学习。"}
        ]
        
        print(f"📤 发送消息: {messages[0]['content']}")
        
        # 发送同步聊天请求
        response = chat_service.chat_sync(messages)
        
        print(f"📥 {provider.upper()} 回复:")
        print(f"   {response['content']}")
        print(f"   模型: {response.get('model', 'unknown')}")
        
    except Exception as e:
        print(f"❌ {provider.upper()} 同步聊天服务测试失败: {str(e)}")


async def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 OpenAI兼容格式 AI Services 综合测试")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 测试提供商列表
    providers = ["openai", "stepfun", "deepseek"]
    
    # 测试聊天服务
    for provider in providers:
        await test_chat_service(provider)
    
    # 测试嵌入服务 (只测试支持的提供商)
    embedding_providers = ["openai", "stepfun"]
    for provider in embedding_providers:
        await test_embedding_service(provider)
    
    # 测试重排序服务
    rerank_providers = ["openai", "stepfun"]
    for provider in rerank_providers:
        await test_rerank_service(provider)
    
    # 测试同步聊天
    print(f"\n🔄 同步服务测试")
    print("=" * 30)
    for provider in providers:
        test_sync_chat(provider)


async def run_single_provider_test(provider: str):
    """
    运行单个提供商测试
    
    Args:
        provider: 服务提供商名称
    """
    print(f"🚀 {provider.upper()} AI Services 测试")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 测试聊天服务
    await test_chat_service(provider)
    
    # 测试嵌入服务 (如果支持)
    if provider in ["openai", "stepfun"]:
        await test_embedding_service(provider)
    
    # 测试重排序服务 (如果支持)
    if provider in ["openai", "stepfun"]:
        await test_rerank_service(provider)
    
    # 测试同步聊天
    test_sync_chat(provider)


if __name__ == "__main__":
    """
    主函数
    
    使用方法：
    1. 运行所有测试：python openai_example.py
    2. 测试特定提供商：python openai_example.py openai
    3. 测试特定提供商：python openai_example.py stepfun
    4. 测试特定提供商：python openai_example.py deepseek
    """
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        if provider in ["openai", "stepfun", "deepseek"]:
            asyncio.run(run_single_provider_test(provider))
        else:
            print(f"❌ 不支持的提供商: {provider}")
            print("支持的提供商: openai, stepfun, deepseek")
            sys.exit(1)
    else:
        # 运行综合测试
        asyncio.run(run_comprehensive_test())
    
    print("\n✅ 测试完成！")
    print("\n💡 提示：")
    print("1. 确保已设置相应的API密钥环境变量")
    print("2. 检查网络连接和API服务可用性")
    print("3. 根据需要调整配置文件中的参数")
    print("4. 查看日志文件获取详细信息")