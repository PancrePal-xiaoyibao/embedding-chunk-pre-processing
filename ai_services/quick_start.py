#!/usr/bin/env python3
"""
AI Services 快速开始脚本

这个脚本演示了如何快速开始使用AI Services模块。
运行此脚本可以验证模块是否正确安装和配置。
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.factory import AIServiceFactory, create_chat_service, create_embedding_service, create_rerank_service
from core.exceptions import AIServiceError as ServiceError, ConnectionError
from services.models import create_user_message, create_system_message
from config.config import get_default_config


def print_banner():
    """打印欢迎横幅"""
    print("🚀 AI Services 快速开始")
    print("=" * 50)
    print("这个脚本将帮助您快速验证AI Services模块的功能。")
    print("请确保您已经启动了Ollama服务 (http://localhost:11434)")
    print("=" * 50)


def check_prerequisites():
    """检查前置条件"""
    print("\n🔍 检查前置条件...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro}")
        print("   需要Python 3.8或更高版本")
        return False
    
    # 检查必需的包
    required_packages = ['requests', 'aiohttp', 'pydantic', 'yaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


async def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        # 创建工厂
        print("创建AI服务工厂...")
        factory = AIServiceFactory.create_default()
        print("✅ 工厂创建成功")
        
        # 获取可用提供商
        print("\n获取可用提供商...")
        from core.interfaces import ServiceType
        chat_providers = factory.get_available_providers(ServiceType.CHAT)
        embedding_providers = factory.get_available_providers(ServiceType.EMBEDDING)
        rerank_providers = factory.get_available_providers(ServiceType.RERANK)
        
        print(f"  Chat提供商: {chat_providers}")
        print(f"  Embedding提供商: {embedding_providers}")
        print(f"  Rerank提供商: {rerank_providers}")
        
        # 创建服务
        print("\n创建AI服务...")
        from core.interfaces import ServiceProvider
        chat_service = factory.create_service(ServiceType.CHAT, ServiceProvider.OLLAMA)
        embedding_service = factory.create_service(ServiceType.EMBEDDING, ServiceProvider.OLLAMA)
        rerank_service = factory.create_service(ServiceType.RERANK, ServiceProvider.OLLAMA)
        print("✅ 所有服务创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        # 显示更详细的错误信息
        if hasattr(e, 'original_error') and e.original_error:
            print(f"原始错误: {e.original_error}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
        return False


async def test_health_checks():
    """测试健康检查"""
    print("\n🏥 测试服务健康检查...")
    
    try:
        factory = AIServiceFactory.create_default()
        
        # 测试连接
        print("测试服务连接...")
        connection_results = await factory.test_connections()
        
        for service_name, is_healthy in connection_results.items():
            status = "✅ 健康" if is_healthy else "❌ 不健康"
            print(f"  {service_name}: {status}")
        
        # 检查是否有健康的服务
        healthy_services = [name for name, status in connection_results.items() if status]
        if healthy_services:
            print(f"\n✅ {len(healthy_services)} 个服务健康")
            return True
        else:
            print("\n⚠️  没有健康的服务")
            print("请检查Ollama是否正在运行: http://localhost:11434")
            return False
            
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


async def test_chat_service():
    """测试Chat服务"""
    print("\n💬 测试Chat服务...")
    
    try:
        chat_service = create_chat_service()
        
        # 检查健康状态
        is_healthy = await chat_service.health_check()
        if not is_healthy:
            print("⚠️  Chat服务不健康，跳过测试")
            return False
        
        # 发送简单消息
        messages = [
            create_system_message("你是一个有用的AI助手。请用中文回答。"),
            create_user_message("你好！请简单介绍一下你自己。")
        ]
        
        print("发送聊天消息...")
        response = await chat_service.chat_async(messages)
        
        print(f"✅ Chat服务响应:")
        print(f"  内容: {response.message.content[:100]}...")
        print(f"  模型: {response.model}")
        
        if response.usage:
            print(f"  用量: {response.usage.total_tokens} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat服务测试失败: {e}")
        return False


async def test_embedding_service():
    """测试Embedding服务"""
    print("\n🔢 测试Embedding服务...")
    
    try:
        embedding_service = create_embedding_service()
        
        # 检查健康状态
        is_healthy = await embedding_service.health_check()
        if not is_healthy:
            print("⚠️  Embedding服务不健康，跳过测试")
            return False
        
        # 生成嵌入
        text = "人工智能是计算机科学的一个分支"
        print(f"为文本生成嵌入: '{text}'")
        
        response = await embedding_service.embed_async(text)
        
        print(f"✅ Embedding服务响应:")
        print(f"  向量维度: {len(response.embeddings[0].vector)}")
        print(f"  模型: {response.model}")
        
        if response.usage:
            print(f"  用量: {response.usage.total_tokens} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding服务测试失败: {e}")
        return False


async def test_rerank_service():
    """测试Rerank服务"""
    print("\n📊 测试Rerank服务...")
    
    try:
        rerank_service = create_rerank_service()
        
        # 检查健康状态
        is_healthy = await rerank_service.health_check()
        if not is_healthy:
            print("⚠️  Rerank服务不健康，跳过测试")
            return False
        
        # 重排序文档
        query = "什么是机器学习"
        documents = [
            "机器学习是人工智能的一个分支",
            "今天天气很好",
            "深度学习是机器学习的子领域",
            "我喜欢吃苹果"
        ]
        
        print(f"查询: '{query}'")
        print(f"重排序 {len(documents)} 个文档...")
        
        response = await rerank_service.rerank_async(query, documents)
        
        print(f"✅ Rerank服务响应:")
        for i, result in enumerate(response.results[:3]):  # 显示前3个结果
            print(f"  {i+1}. 得分: {result.score:.3f} - {documents[result.index][:50]}...")
        
        if response.usage:
            print(f"  用量: {response.usage.total_tokens} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Rerank服务测试失败: {e}")
        return False


async def run_comprehensive_test():
    """运行综合测试"""
    print("\n🎯 运行综合测试...")
    
    try:
        # 创建工厂
        factory = AIServiceFactory.create_default()
        
        # 并发测试多个服务
        print("并发测试多个服务...")
        
        async def quick_chat():
            from core.interfaces import ServiceProvider, ServiceType
            chat_service = factory.create_service(ServiceType.CHAT, ServiceProvider.OLLAMA)
            messages = [create_user_message("Hello!")]
            response = await chat_service.chat_async(messages)
            return f"Chat: {len(response.message.content)} 字符"
        
        async def quick_embedding():
            from core.interfaces import ServiceProvider, ServiceType
            embedding_service = factory.create_service(ServiceType.EMBEDDING, ServiceProvider.OLLAMA)
            response = await embedding_service.embed_async("Hello world")
            return f"Embedding: {len(response.embeddings[0].vector)} 维度"
        
        async def quick_rerank():
            from core.interfaces import ServiceProvider, ServiceType
            rerank_service = factory.create_service(ServiceType.RERANK, ServiceProvider.OLLAMA)
            response = await rerank_service.rerank_async("test", ["doc1", "doc2"])
            return f"Rerank: {len(response.results)} 结果"
        
        # 并发执行
        results = await asyncio.gather(
            quick_chat(),
            quick_embedding(),
            quick_rerank(),
            return_exceptions=True
        )
        
        print("✅ 并发测试结果:")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  任务 {i+1}: ❌ {result}")
            else:
                print(f"  任务 {i+1}: ✅ {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 综合测试失败: {e}")
        return False


def print_next_steps():
    """打印后续步骤"""
    print("\n🎉 快速开始完成！")
    print("\n📚 后续步骤:")
    print("1. 查看文档: docs/README.md")
    print("2. 运行示例: python examples/basic_usage.py")
    print("3. 查看配置: python examples/config_example.py")
    print("4. 高级功能: python examples/advanced_features.py")
    print("5. 运行测试: python tests/test_ai_services.py")
    
    print("\n💡 提示:")
    print("- 确保Ollama服务正在运行")
    print("- 可以通过配置文件自定义服务设置")
    print("- 支持异步和同步两种使用方式")
    print("- 查看API文档了解更多功能")


async def main():
    """主函数"""
    print_banner()
    
    # 检查前置条件
    if not check_prerequisites():
        print("\n❌ 前置条件检查失败，请解决问题后重试")
        return
    
    # 测试基本功能
    if not await test_basic_functionality():
        print("\n❌ 基本功能测试失败")
        return
    
    # 测试健康检查
    health_ok = await test_health_checks()
    
    # 如果健康检查通过，运行功能测试
    if health_ok:
        print("\n🚀 运行功能测试...")
        
        # 测试各个服务
        chat_ok = await test_chat_service()
        embedding_ok = await test_embedding_service()
        rerank_ok = await test_rerank_service()
        
        # 如果所有服务都正常，运行综合测试
        if chat_ok and embedding_ok and rerank_ok:
            await run_comprehensive_test()
    
    # 打印后续步骤
    print_next_steps()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        sys.exit(1)