#!/usr/bin/env python3
"""
AI Services 高级功能示例

演示AI服务模块的高级功能，包括批处理、异步操作、错误处理、健康检查等。
"""

import asyncio
import os
import sys
import time
from typing import List

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_services import AIServiceFactory, create_chat_service, create_embedding_service, create_rerank_service
from ai_services.services.models import create_user_message, create_system_message
from ai_services.core.exceptions import (
    AIServiceError, 
    ServiceNotAvailableError, 
    ConnectionError,
    ModelNotFoundError
)


async def health_check_example():
    """健康检查示例"""
    print("=== 健康检查示例 ===")
    
    factory = AIServiceFactory.create_default()
    
    # 检查所有服务的健康状态
    services = ["chat", "embedding", "rerank"]
    
    for service_name in services:
        print(f"\n检查 {service_name} 服务:")
        try:
            service = factory.create_service(service_name)
            is_healthy = await service.health_check()
            
            if is_healthy:
                print(f"  ✅ {service_name} 服务健康")
            else:
                print(f"  ❌ {service_name} 服务不健康")
                
        except Exception as e:
            print(f"  ❌ {service_name} 服务检查失败: {e}")
    
    # 测试连接
    print(f"\n测试工厂连接:")
    try:
        connection_results = await factory.test_connections()
        for service_name, result in connection_results.items():
            status = "✅ 连接成功" if result else "❌ 连接失败"
            print(f"  {service_name}: {status}")
    except Exception as e:
        print(f"  ❌ 连接测试失败: {e}")


async def batch_processing_example():
    """批处理示例"""
    print("\n=== 批处理示例 ===")
    
    # 创建嵌入服务
    embedding_service = create_embedding_service()
    
    # 准备批量文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习是机器学习的一个子领域",
        "神经网络是深度学习的基础",
        "自然语言处理是AI的重要应用",
        "计算机视觉让机器能够理解图像",
        "强化学习通过试错来学习最优策略",
        "大语言模型改变了NLP的发展方向"
    ]
    
    print(f"批量处理 {len(texts)} 个文本...")
    
    start_time = time.time()
    
    try:
        # 检查服务是否支持批量处理
        if hasattr(embedding_service, 'embed_batch'):
            print("使用批量嵌入方法...")
            embeddings = await embedding_service.embed_batch(texts)
        else:
            print("使用单个嵌入方法...")
            embeddings = []
            for text in texts:
                result = await embedding_service.embed(text)
                embeddings.append(result.vectors[0])
        
        end_time = time.time()
        
        print(f"✅ 批量处理完成")
        print(f"  处理时间: {end_time - start_time:.2f} 秒")
        print(f"  平均每个文本: {(end_time - start_time) / len(texts):.3f} 秒")
        print(f"  嵌入维度: {len(embeddings[0]) if embeddings else 0}")
        
        # 计算文本相似度
        if len(embeddings) >= 2:
            from ai_services.services.embedding_service import cosine_similarity
            similarity = cosine_similarity(embeddings[0], embeddings[1])
            print(f"  前两个文本相似度: {similarity:.3f}")
            
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")


async def async_concurrent_example():
    """异步并发示例"""
    print("\n=== 异步并发示例 ===")
    
    factory = AIServiceFactory.create_default()
    
    # 创建多个服务
    chat_service = factory.create_service("chat")
    embedding_service = factory.create_service("embedding")
    rerank_service = factory.create_service("rerank")
    
    # 准备并发任务
    async def chat_task():
        """聊天任务"""
        messages = [
            create_system_message("你是一个有用的AI助手。"),
            create_user_message("请简单介绍一下机器学习。")
        ]
        response = await chat_service.chat_async(messages)
        return f"Chat: {response.message.content[:50]}..."
    
    async def embedding_task():
        """嵌入任务"""
        text = "机器学习是人工智能的重要分支"
        response = await embedding_service.embed(text)
        return f"Embedding: 维度={len(response.vectors[0])}"
    
    async def rerank_task():
        """重排序任务"""
        query = "什么是机器学习"
        documents = [
            "机器学习是人工智能的一个分支",
            "深度学习是机器学习的子领域",
            "今天天气很好"
        ]
        response = await rerank_service.rerank(query, documents)
        return f"Rerank: 最相关文档得分={response.results[0].score:.3f}"
    
    print("启动并发任务...")
    start_time = time.time()
    
    try:
        # 并发执行所有任务
        results = await asyncio.gather(
            chat_task(),
            embedding_task(),
            rerank_task(),
            return_exceptions=True
        )
        
        end_time = time.time()
        
        print(f"✅ 并发任务完成 (耗时: {end_time - start_time:.2f} 秒)")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  任务 {i+1}: ❌ {result}")
            else:
                print(f"  任务 {i+1}: ✅ {result}")
                
    except Exception as e:
        print(f"❌ 并发任务失败: {e}")


async def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 测试不同类型的错误
    error_scenarios = [
        {
            "name": "服务不可用",
            "action": lambda: create_chat_service(provider="nonexistent")
        },
        {
            "name": "连接错误",
            "action": lambda: create_chat_service(
                provider="ollama",
                base_url="http://invalid-host:11434"
            )
        },
        {
            "name": "模型不存在",
            "action": lambda: create_chat_service(
                provider="ollama",
                model_name="nonexistent-model"
            )
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n测试: {scenario['name']}")
        try:
            service = scenario['action']()
            
            # 尝试使用服务
            if hasattr(service, 'health_check'):
                await service.health_check()
            
            print(f"  ⚠️  意外成功")
            
        except ServiceNotAvailableError as e:
            print(f"  ✅ 捕获服务不可用错误: {e}")
        except ConnectionError as e:
            print(f"  ✅ 捕获连接错误: {e}")
        except ModelNotFoundError as e:
            print(f"  ✅ 捕获模型不存在错误: {e}")
        except AIServiceError as e:
            print(f"  ✅ 捕获AI服务错误: {e}")
        except Exception as e:
            print(f"  ❌ 未预期的错误: {type(e).__name__}: {e}")


async def streaming_example():
    """流式处理示例"""
    print("\n=== 流式处理示例 ===")
    
    try:
        chat_service = create_chat_service()
        
        messages = [
            create_system_message("你是一个有用的AI助手。"),
            create_user_message("请详细解释什么是深度学习，包括其原理和应用。")
        ]
        
        print("开始流式聊天...")
        print("回复: ", end="", flush=True)
        
        full_response = ""
        async for chunk in chat_service.chat_stream(messages):
            if chunk.message and chunk.message.content:
                content = chunk.message.content
                print(content, end="", flush=True)
                full_response += content
        
        print(f"\n\n✅ 流式聊天完成")
        print(f"总字符数: {len(full_response)}")
        
    except Exception as e:
        print(f"❌ 流式处理失败: {e}")


async def model_management_example():
    """模型管理示例"""
    print("\n=== 模型管理示例 ===")
    
    try:
        # 创建Ollama服务
        chat_service = create_chat_service(provider="ollama")
        
        if hasattr(chat_service, 'list_models'):
            print("获取可用模型列表...")
            models = await chat_service.list_models()
            
            print(f"✅ 找到 {len(models)} 个模型:")
            for model in models[:5]:  # 只显示前5个
                print(f"  - {model}")
            
            if len(models) > 5:
                print(f"  ... 还有 {len(models) - 5} 个模型")
        
        # 测试模型拉取（仅演示，不实际执行）
        if hasattr(chat_service, 'pull_model'):
            print(f"\n模型拉取功能可用")
            print(f"  可以使用 chat_service.pull_model('model_name') 拉取新模型")
        
    except Exception as e:
        print(f"❌ 模型管理失败: {e}")


async def performance_monitoring_example():
    """性能监控示例"""
    print("\n=== 性能监控示例 ===")
    
    # 监控不同操作的性能
    operations = [
        {
            "name": "Chat响应",
            "action": lambda: create_chat_service().chat_async([
                create_user_message("Hello, how are you?")
            ])
        },
        {
            "name": "文本嵌入",
            "action": lambda: create_embedding_service().embed("Hello world")
        },
        {
            "name": "文档重排序",
            "action": lambda: create_rerank_service().rerank(
                "machine learning",
                ["AI is the future", "Machine learning is a subset of AI", "Weather is nice"]
            )
        }
    ]
    
    performance_results = []
    
    for operation in operations:
        print(f"\n测试: {operation['name']}")
        
        # 执行多次测试
        times = []
        success_count = 0
        
        for i in range(3):  # 执行3次
            try:
                start_time = time.time()
                await operation['action']()
                end_time = time.time()
                
                duration = end_time - start_time
                times.append(duration)
                success_count += 1
                
                print(f"  第{i+1}次: {duration:.3f}秒")
                
            except Exception as e:
                print(f"  第{i+1}次: ❌ {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            performance_results.append({
                "operation": operation['name'],
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_rate": success_count / 3
            })
            
            print(f"  ✅ 平均耗时: {avg_time:.3f}秒")
            print(f"  📊 范围: {min_time:.3f}s - {max_time:.3f}s")
            print(f"  📈 成功率: {success_count}/3")
    
    # 性能总结
    if performance_results:
        print(f"\n📊 性能总结:")
        for result in performance_results:
            print(f"  {result['operation']}: {result['avg_time']:.3f}s (成功率: {result['success_rate']:.1%})")


async def main():
    """主函数"""
    print("🚀 AI Services 高级功能示例")
    print("=" * 50)
    
    # 执行各种高级功能示例
    await health_check_example()
    await batch_processing_example()
    await async_concurrent_example()
    await error_handling_example()
    await streaming_example()
    await model_management_example()
    await performance_monitoring_example()
    
    print("\n🎉 所有高级功能示例运行完成！")
    print("\n💡 高级功能总结:")
    print("  - ✅ 健康检查和连接测试")
    print("  - ⚡ 批处理和异步并发")
    print("  - 🛡️  完善的错误处理")
    print("  - 🌊 流式处理支持")
    print("  - 🔧 模型管理功能")
    print("  - 📊 性能监控能力")


if __name__ == "__main__":
    asyncio.run(main())