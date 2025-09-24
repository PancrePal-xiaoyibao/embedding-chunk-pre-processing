#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-4.5 使用示例脚本
演示如何使用GLM-4.5进行聊天、嵌入和重排序
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chat.chat_service import ChatService
from src.embedding.embedding_service import EmbeddingService
from src.rerank.rerank_service import RerankService
from config.config import load_config


def setup_environment():
    """
    设置环境变量
    请在运行前设置您的GLM API密钥
    """
    # 检查API密钥是否已设置
    if not os.getenv('GLM_API_KEY'):
        print("⚠️  请设置GLM_API_KEY环境变量")
        print("   export GLM_API_KEY='your_api_key_here'")
        print("   或者在.env文件中设置GLM_API_KEY=your_api_key_here")
        return False
    return True


async def chat_example():
    """
    GLM-4.5 聊天服务示例
    """
    print("\n🤖 GLM-4.5 聊天服务示例")
    print("=" * 50)
    
    try:
        # 加载配置
        config = load_config("config.glm4.yaml")
        
        # 创建聊天服务
        chat_service = ChatService(config)
        
        # 测试消息
        messages = [
            {"role": "user", "content": "你好，请介绍一下GLM-4.5模型的特点"}
        ]
        
        print(f"📤 发送消息: {messages[0]['content']}")
        
        # 发送聊天请求
        response = await chat_service.chat(messages)
        
        print(f"📥 GLM-4.5 回复: {response}")
        
    except Exception as e:
        print(f"❌ 聊天服务错误: {e}")


async def embedding_example():
    """
    GLM 嵌入服务示例
    """
    print("\n🔍 GLM 嵌入服务示例")
    print("=" * 50)
    
    try:
        # 加载配置
        config = load_config("config.glm4.yaml")
        
        # 创建嵌入服务
        embedding_service = EmbeddingService(config)
        
        # 测试文本
        texts = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的核心技术",
            "深度学习是机器学习的重要方法"
        ]
        
        print(f"📤 计算嵌入向量，文本数量: {len(texts)}")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. {text}")
        
        # 计算嵌入向量
        embeddings = await embedding_service.embed(texts)
        
        print(f"📥 嵌入向量维度: {len(embeddings[0])}")
        print(f"📥 向量数量: {len(embeddings)}")
        
        # 显示前几个维度的值
        print(f"📥 第一个向量的前5个维度: {embeddings[0][:5]}")
        
    except Exception as e:
        print(f"❌ 嵌入服务错误: {e}")


async def rerank_example():
    """
    GLM-4.5 重排序服务示例
    """
    print("\n📊 GLM-4.5 重排序服务示例")
    print("=" * 50)
    
    try:
        # 加载配置
        config = load_config("config.glm4.yaml")
        
        # 创建重排序服务
        rerank_service = RerankService(config)
        
        # 测试查询和文档
        query = "什么是人工智能？"
        documents = [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统",
            "机器学习是人工智能的一个子集，使计算机能够从数据中学习而无需明确编程",
            "深度学习使用神经网络来模拟人脑的工作方式",
            "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互",
            "计算机视觉是人工智能的一个领域，使计算机能够解释和理解视觉世界"
        ]
        
        print(f"📤 查询: {query}")
        print(f"📤 文档数量: {len(documents)}")
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc[:50]}...")
        
        # 执行重排序
        ranked_results = await rerank_service.rerank(query, documents)
        
        print(f"📥 重排序结果:")
        for i, (doc, score) in enumerate(ranked_results, 1):
            print(f"   {i}. [分数: {score:.4f}] {doc[:50]}...")
        
    except Exception as e:
        print(f"❌ 重排序服务错误: {e}")


async def main():
    """
    主函数：运行所有示例
    """
    print("🚀 GLM-4.5 AI Services 使用示例")
    print("=" * 60)
    
    # 检查环境设置
    if not setup_environment():
        return
    
    try:
        # 运行聊天示例
        await chat_example()
        
        # 运行嵌入示例
        await embedding_example()
        
        # 运行重排序示例
        await rerank_example()
        
        print("\n✅ 所有示例运行完成！")
        
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        print("\n💡 请检查:")
        print("   1. GLM_API_KEY 环境变量是否正确设置")
        print("   2. 网络连接是否正常")
        print("   3. API密钥是否有效且有足够的配额")


def sync_main():
    """
    同步版本的主函数
    """
    print("🚀 GLM-4.5 AI Services 同步使用示例")
    print("=" * 60)
    
    # 检查环境设置
    if not setup_environment():
        return
    
    try:
        from src.chat.chat_service import ChatService
        from src.embedding.embedding_service import EmbeddingService
        from src.rerank.rerank_service import RerankService
        
        # 加载配置
        config = load_config("config.glm4.yaml")
        
        # 聊天示例
        print("\n🤖 GLM-4.5 聊天服务示例 (同步)")
        chat_service = ChatService(config)
        messages = [{"role": "user", "content": "你好，GLM-4.5！"}]
        response = chat_service.chat_sync(messages)
        print(f"📥 回复: {response}")
        
        print("\n✅ 同步示例运行完成！")
        
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GLM-4.5 使用示例")
    parser.add_argument(
        "--sync", 
        action="store_true", 
        help="运行同步版本的示例"
    )
    
    args = parser.parse_args()
    
    if args.sync:
        sync_main()
    else:
        asyncio.run(main())