#!/usr/bin/env python3
"""
AI Services 基本使用示例

演示如何使用AI服务模块进行Chat、Embedding和Rerank操作。
"""

import asyncio
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_services import AIServiceFactory, create_chat_service, create_embedding_service, create_rerank_service
from ai_services.services.models import create_user_message, create_system_message
from ai_services.core.exceptions import AIServiceError


def basic_chat_example():
    """基本聊天示例"""
    print("=== 基本聊天示例 ===")
    
    try:
        # 创建聊天服务
        chat_service = create_chat_service()
        
        # 检查服务健康状态
        if not chat_service.health_check():
            print("❌ 聊天服务不可用，请检查Ollama是否运行")
            return
        
        print("✅ 聊天服务已就绪")
        
        # 单轮对话
        print("\n--- 单轮对话 ---")
        response = chat_service.chat("你好，请简单介绍一下自己")
        print(f"助手: {response.content}")
        print(f"模型: {response.model}")
        
        # 多轮对话
        print("\n--- 多轮对话 ---")
        messages = [
            create_system_message("你是一个Python编程专家"),
            create_user_message("什么是列表推导式？"),
            create_user_message("请给一个简单的例子")
        ]
        
        response = chat_service.chat(messages)
        print(f"专家: {response.content}")
        
    except AIServiceError as e:
        print(f"❌ 聊天服务错误: {e}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")


def basic_embedding_example():
    """基本嵌入示例"""
    print("\n=== 基本嵌入示例 ===")
    
    try:
        # 创建嵌入服务
        embedding_service = create_embedding_service()
        
        # 检查服务健康状态
        if not embedding_service.health_check():
            print("❌ 嵌入服务不可用，请检查配置")
            return
        
        print("✅ 嵌入服务已就绪")
        
        # 单个文本嵌入
        print("\n--- 单个文本嵌入 ---")
        text = "人工智能是计算机科学的一个分支"
        result = embedding_service.embed(text)
        
        print(f"文本: {text}")
        print(f"嵌入维度: {len(result.vectors[0])}")
        print(f"模型: {result.model}")
        
        # 批量文本嵌入
        print("\n--- 批量文本嵌入 ---")
        texts = [
            "机器学习是人工智能的核心技术",
            "深度学习使用神经网络进行学习",
            "自然语言处理处理人类语言",
            "计算机视觉让机器理解图像"
        ]
        
        result = embedding_service.embed(texts)
        print(f"处理了 {len(result.vectors)} 个文本")
        
        # 计算相似度
        print("\n--- 相似度计算 ---")
        similarity = embedding_service.compute_similarity(
            result.vectors[0], 
            result.vectors[1],
            method="cosine"
        )
        print(f"文本1和文本2的相似度: {similarity:.4f}")
        
        # 找到最相似的文本对
        max_similarity = 0
        best_pair = (0, 0)
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = embedding_service.compute_similarity(
                    result.vectors[i], 
                    result.vectors[j]
                )
                if sim > max_similarity:
                    max_similarity = sim
                    best_pair = (i, j)
        
        print(f"\n最相似的文本对 (相似度: {max_similarity:.4f}):")
        print(f"  文本{best_pair[0] + 1}: {texts[best_pair[0]]}")
        print(f"  文本{best_pair[1] + 1}: {texts[best_pair[1]]}")
        
    except AIServiceError as e:
        print(f"❌ 嵌入服务错误: {e}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")


def basic_rerank_example():
    """基本重排序示例"""
    print("\n=== 基本重排序示例 ===")
    
    try:
        # 创建重排序服务
        rerank_service = create_rerank_service()
        
        # 检查服务健康状态
        if not rerank_service.health_check():
            print("❌ 重排序服务不可用，请检查配置")
            return
        
        print("✅ 重排序服务已就绪")
        
        # 重排序示例
        print("\n--- 文档重排序 ---")
        query = "Python编程语言的特点"
        documents = [
            "Java是一种面向对象的编程语言，具有跨平台特性",
            "Python是一种简洁易读的编程语言，语法清晰",
            "JavaScript主要用于Web前端开发",
            "Python支持多种编程范式，包括面向对象和函数式编程",
            "C++是一种高性能的系统编程语言",
            "Python拥有丰富的第三方库生态系统"
        ]
        
        print(f"查询: {query}")
        print(f"候选文档数量: {len(documents)}")
        
        # 执行重排序
        result = rerank_service.rerank(query, documents, top_k=3)
        
        print(f"\n重排序结果 (Top 3):")
        for i, item in enumerate(result.results):
            print(f"  {i + 1}. 分数: {item.score:.4f}")
            print(f"     文档: {item.document}")
            print(f"     原始索引: {item.index}")
            print()
        
    except AIServiceError as e:
        print(f"❌ 重排序服务错误: {e}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")


async def async_example():
    """异步操作示例"""
    print("\n=== 异步操作示例 ===")
    
    try:
        # 创建服务
        factory = AIServiceFactory.create_default()
        chat_service = factory.create_service("chat")
        embedding_service = factory.create_service("embedding")
        
        print("开始异步操作...")
        
        # 并发执行多个任务
        tasks = [
            chat_service.chat_async("什么是异步编程？"),
            embedding_service.embed_async("异步编程提高程序效率"),
            embedding_service.embed_async("Python的asyncio库支持异步编程")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        print("\n异步结果:")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  任务{i + 1}: 错误 - {result}")
            else:
                if hasattr(result, 'content'):  # ChatResponse
                    print(f"  任务{i + 1}: 聊天响应 - {result.content[:50]}...")
                elif hasattr(result, 'vectors'):  # EmbeddingResponse
                    print(f"  任务{i + 1}: 嵌入响应 - 维度 {len(result.vectors[0])}")
        
    except Exception as e:
        print(f"❌ 异步操作错误: {e}")


def stream_chat_example():
    """流式聊天示例"""
    print("\n=== 流式聊天示例 ===")
    
    try:
        chat_service = create_chat_service()
        
        if not chat_service.health_check():
            print("❌ 聊天服务不可用")
            return
        
        print("开始流式聊天...")
        print("助手: ", end="", flush=True)
        
        # 流式响应
        for chunk in chat_service.chat_stream("请讲一个关于人工智能的简短故事"):
            print(chunk.content, end="", flush=True)
        
        print("\n")  # 换行
        
    except AIServiceError as e:
        print(f"❌ 流式聊天错误: {e}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")


def factory_example():
    """工厂模式示例"""
    print("\n=== 工厂模式示例 ===")
    
    try:
        # 创建工厂
        factory = AIServiceFactory.create_default()
        
        # 检查所有服务健康状态
        print("检查服务健康状态...")
        health_status = factory.health_check()
        
        for service_type, is_healthy in health_status.items():
            status = "✅ 正常" if is_healthy else "❌ 异常"
            print(f"  {service_type}: {status}")
        
        # 获取可用提供商
        print("\n可用提供商:")
        for service_type in ["chat", "embedding", "rerank"]:
            providers = factory.get_available_providers(service_type)
            print(f"  {service_type}: {providers}")
        
        # 测试连接
        print("\n测试连接:")
        for service_type in ["chat", "embedding"]:
            connection_ok = factory.test_provider_connection(service_type, "ollama")
            status = "✅ 连接正常" if connection_ok else "❌ 连接失败"
            print(f"  {service_type} (ollama): {status}")
        
    except Exception as e:
        print(f"❌ 工厂示例错误: {e}")


def main():
    """主函数"""
    print("🚀 AI Services 基本使用示例")
    print("=" * 50)
    
    # 基本功能示例
    basic_chat_example()
    basic_embedding_example()
    basic_rerank_example()
    
    # 高级功能示例
    factory_example()
    stream_chat_example()
    
    # 异步示例
    print("\n运行异步示例...")
    asyncio.run(async_example())
    
    print("\n🎉 所有示例运行完成！")


if __name__ == "__main__":
    main()