#!/usr/bin/env python3
"""
StepFun模型测试脚本

本脚本用于测试StepFun模型的各项功能，包括：
1. 聊天服务
2. 嵌入服务  
3. 重排序服务

使用前请确保：
1. 已设置STEPFUN_API_KEY环境变量
2. config.openai.yaml配置文件正确
3. 网络连接正常
"""

import os
import sys
import asyncio
import yaml
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.factory import create_chat_service, create_embedding_service, create_rerank_service


class StepFunTester:
    """StepFun模型测试器"""
    
    def __init__(self, config_path: str = "config.template.yaml"):
        """初始化测试器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.test_results = {}
        self.chat_service = None
        self.embedding_service = None
        self.rerank_service = None
        
    def setup_environment(self):
        """
        设置环境变量和检查前置条件
        """
        print("🔧 设置StepFun环境变量...")
        
        # 检查API密钥
        api_key = os.getenv("STEPFUN_API_KEY")
        if not api_key:
            print("❌ 未设置STEPFUN_API_KEY环境变量")
            print("请设置：export STEPFUN_API_KEY='your_stepfun_api_key'")
            return False
            
        print("✅ StepFun API密钥已设置")
        return True
        
    def load_config(self):
        """
        加载配置并创建服务实例
        """
        print(f"📋 加载配置文件: {self.config_path}")
        
        try:
            # 使用便捷函数创建服务实例
            self.chat_service = create_chat_service(config_path=self.config_path)
            self.embedding_service = create_embedding_service(config_path=self.config_path)
            self.rerank_service = create_rerank_service(config_path=self.config_path)
            
            print("✅ 配置加载成功，服务实例创建完成")
            return True
            
        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
            return False
            
    async def test_chat_service(self):
        """
        测试StepFun聊天服务
        """
        print("\n💬 测试StepFun聊天服务...")
        
        try:
            # 创建测试消息
            messages = [
                {"role": "user", "content": "你好！请简单介绍一下人工智能的发展历程。"}
            ]
            
            print("📤 发送聊天消息...")
            print(f"   用户: {messages[0]['content']}")
            
            # 发送聊天请求
            response = self.chat_service.chat(messages)
            
            print("📥 StepFun回复:")
            print(f"   内容: {response.message.content[:200]}...")
            print(f"   模型: {response.model or 'unknown'}")
            print(f"   用时: {response.response_time or 0:.2f}秒")
            
            self.test_results['chat'] = True
            print("✅ 聊天服务测试通过")
            
            return True
            
        except Exception as e:
            print(f"❌ 聊天服务测试失败: {e}")
            self.test_results['chat'] = False
            return False
            
    async def test_embedding_service(self):
        """
        测试StepFun嵌入服务
        """
        print("\n🔢 测试StepFun嵌入服务...")
        
        try:
            text = "这是一个测试文本，用于验证嵌入服务。"
            print(f"📤 生成文本嵌入: '{text}'")
            
            # 生成嵌入
            response = self.embedding_service.embed(text)
            
            print("📥 嵌入结果:")
            print(f"   维度: {len(response.embeddings[0].vector)}")
            print(f"   模型: {response.model}")
            print(f"   前5个值: {response.embeddings[0].vector[:5]}")
            
            self.test_results['embedding'] = True
            print("✅ 嵌入服务测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 嵌入服务测试失败: {e}")
            self.test_results['embedding'] = False
            return False

    async def test_rerank_service(self):
        """
        测试StepFun重排序服务
        """
        print("\n📊 测试StepFun重排序服务...")
        
        try:
            query = "人工智能的发展趋势"
            documents = [
                "人工智能技术正在快速发展，深度学习是其核心驱动力。",
                "机器学习算法在各个领域都有广泛应用。",
                "自然语言处理技术让计算机能够理解人类语言。",
                "计算机视觉技术使机器能够识别和理解图像内容。"
            ]
            
            print(f"📤 查询: '{query}'")
            print(f"📤 重排序 {len(documents)} 个文档...")
            
            # 执行重排序
            response = self.rerank_service.rerank(query, documents)
            
            print("📥 重排序结果:")
            for i, result in enumerate(response.results):
                print(f"   {i+1}. 分数: {result.score:.4f}, 内容: {result.document[:50]}...")
            print(f"   用时: {response.response_time:.2f}秒")
            
            self.test_results['rerank'] = True
            print("✅ 重排序服务测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 重排序服务测试失败: {e}")
            self.test_results['rerank'] = False
            return False
            
    def print_config_info(self):
        """
        打印配置信息
        """
        print("\n📋 StepFun配置信息:")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 聊天服务配置
            chat_config = config.get('services', {}).get('chat', {}).get('providers', {}).get('stepfun', {})
            if chat_config:
                print(f"   聊天服务:")
                print(f"     - 基础URL: {chat_config.get('base_url')}")
                print(f"     - 模型: {chat_config.get('model_name')}")
                print(f"     - 超时: {chat_config.get('timeout')}秒")
                
            # 嵌入服务配置
            embed_config = config.get('services', {}).get('embedding', {}).get('providers', {}).get('stepfun', {})
            if embed_config:
                print(f"   嵌入服务:")
                print(f"     - 基础URL: {embed_config.get('base_url')}")
                print(f"     - 模型: {embed_config.get('model_name')}")
                print(f"     - 批次大小: {embed_config.get('batch_size')}")
                
            # 重排序服务配置
            rerank_config = config.get('services', {}).get('rerank', {}).get('providers', {}).get('stepfun', {})
            if rerank_config:
                print(f"   重排序服务:")
                print(f"     - 基础URL: {rerank_config.get('base_url')}")
                print(f"     - 模型: {rerank_config.get('model_name')}")
                print(f"     - 批次大小: {rerank_config.get('batch_size')}")
                
        except Exception as e:
            print(f"❌ 无法读取配置信息: {e}")
            
    async def run_all_tests(self):
        """
        运行所有测试
        """
        print("🚀 开始StepFun模型测试")
        print("=" * 50)
        
        # 设置环境
        if not self.setup_environment():
            return False
            
        # 加载配置
        if not self.load_config():
            return False
            
        # 打印配置信息
        self.print_config_info()
        
        # 运行测试
        test_results = []
        
        # 测试聊天服务
        chat_result = await self.test_chat_service()
        test_results.append(("聊天服务", chat_result))
        
        # 测试嵌入服务
        embed_result = await self.test_embedding_service()
        test_results.append(("嵌入服务", embed_result))
        
        # 测试重排序服务
        rerank_result = await self.test_rerank_service()
        test_results.append(("重排序服务", rerank_result))
        
        # 打印测试结果
        print("\n" + "=" * 50)
        print("📊 测试结果汇总:")
        
        all_passed = True
        for service_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {service_name}: {status}")
            if not result:
                all_passed = False
                
        if all_passed:
            print("\n🎉 所有StepFun服务测试通过！")
        else:
            print("\n⚠️  部分测试失败，请检查配置和网络连接")
            
        return all_passed


async def main():
    """
    主函数
    """
    tester = StepFunTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())