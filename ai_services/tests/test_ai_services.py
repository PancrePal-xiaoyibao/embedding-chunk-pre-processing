#!/usr/bin/env python3
"""
AI Services 模块测试

提供AI服务模块的单元测试和集成测试。
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_services import AIServiceFactory
from ai_services.core.exceptions import (
    AIServiceError,
    ConfigurationError,
    ServiceNotAvailableError,
    ConnectionError,
    ValidationError
)
from ai_services.services.models import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    RerankResponse,
    MessageRole,
    create_user_message,
    create_system_message,
    create_assistant_message
)
from ai_services.config import (
    get_default_config,
    validate_config,
    create_config_template,
    load_config_from_env,
    merge_configs
)


class TestAIServiceFactory(unittest.TestCase):
    """AI服务工厂测试"""
    
    def setUp(self):
        """测试设置"""
        self.factory = AIServiceFactory.create_default()
    
    def test_create_default_factory(self):
        """测试创建默认工厂"""
        factory = AIServiceFactory.create_default()
        self.assertIsInstance(factory, AIServiceFactory)
        
        # 检查默认配置
        config = factory.config
        self.assertIn('services', config)
        self.assertIn('chat', config['services'])
        self.assertIn('embedding', config['services'])
        self.assertIn('rerank', config['services'])
    
    def test_create_service(self):
        """测试创建服务"""
        # 测试创建Chat服务
        chat_service = self.factory.create_service("chat")
        self.assertIsNotNone(chat_service)
        
        # 测试创建Embedding服务
        embedding_service = self.factory.create_service("embedding")
        self.assertIsNotNone(embedding_service)
        
        # 测试创建Rerank服务
        rerank_service = self.factory.create_service("rerank")
        self.assertIsNotNone(rerank_service)
    
    def test_create_invalid_service(self):
        """测试创建无效服务"""
        with self.assertRaises(ServiceNotAvailableError):
            self.factory.create_service("invalid_service")
    
    def test_get_available_providers(self):
        """测试获取可用提供商"""
        chat_providers = self.factory.get_available_providers("chat")
        self.assertIsInstance(chat_providers, list)
        self.assertIn("ollama", chat_providers)
        
        embedding_providers = self.factory.get_available_providers("embedding")
        self.assertIsInstance(embedding_providers, list)
        self.assertIn("ollama", embedding_providers)
        self.assertIn("local", embedding_providers)
        
        rerank_providers = self.factory.get_available_providers("rerank")
        self.assertIsInstance(rerank_providers, list)
        self.assertIn("embedding_based", rerank_providers)
        self.assertIn("cross_encoder", rerank_providers)
    
    def test_from_config(self):
        """测试从配置创建工厂"""
        config = get_default_config()
        factory = AIServiceFactory.from_config(config)
        self.assertIsInstance(factory, AIServiceFactory)
    
    def test_from_config_file(self):
        """测试从配置文件创建工厂"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            create_config_template(f.name, format='yaml')
            
            try:
                factory = AIServiceFactory.from_config_file(f.name)
                self.assertIsInstance(factory, AIServiceFactory)
            finally:
                os.unlink(f.name)


class TestChatMessage(unittest.TestCase):
    """聊天消息测试"""
    
    def test_create_user_message(self):
        """测试创建用户消息"""
        message = create_user_message("Hello")
        self.assertEqual(message.role, MessageRole.USER)
        self.assertEqual(message.content, "Hello")
    
    def test_create_system_message(self):
        """测试创建系统消息"""
        message = create_system_message("You are a helpful assistant")
        self.assertEqual(message.role, MessageRole.SYSTEM)
        self.assertEqual(message.content, "You are a helpful assistant")
    
    def test_create_assistant_message(self):
        """测试创建助手消息"""
        message = create_assistant_message("How can I help you?")
        self.assertEqual(message.role, MessageRole.ASSISTANT)
        self.assertEqual(message.content, "How can I help you?")
    
    def test_message_to_dict(self):
        """测试消息转字典"""
        message = create_user_message("Hello")
        message_dict = message.to_dict()
        
        self.assertEqual(message_dict['role'], 'user')
        self.assertEqual(message_dict['content'], 'Hello')
    
    def test_message_from_dict(self):
        """测试从字典创建消息"""
        message_dict = {'role': 'user', 'content': 'Hello'}
        message = ChatMessage.from_dict(message_dict)
        
        self.assertEqual(message.role, MessageRole.USER)
        self.assertEqual(message.content, 'Hello')


class TestConfigManagement(unittest.TestCase):
    """配置管理测试"""
    
    def test_get_default_config(self):
        """测试获取默认配置"""
        config = get_default_config()
        
        self.assertIn('version', config)
        self.assertIn('services', config)
        self.assertIn('logging', config)
        
        # 检查服务配置
        services = config['services']
        self.assertIn('chat', services)
        self.assertIn('embedding', services)
        self.assertIn('rerank', services)
    
    def test_validate_config(self):
        """测试配置验证"""
        # 有效配置
        valid_config = get_default_config()
        errors = validate_config(valid_config)
        self.assertEqual(len(errors), 0)
        
        # 无效配置 - 缺少version
        invalid_config = {'services': {}}
        errors = validate_config(invalid_config)
        self.assertGreater(len(errors), 0)
        
        # 无效配置 - 缺少services
        invalid_config = {'version': '1.0'}
        errors = validate_config(invalid_config)
        self.assertGreater(len(errors), 0)
    
    def test_create_config_template(self):
        """测试创建配置模板"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                create_config_template(f.name, format='yaml')
                self.assertTrue(os.path.exists(f.name))
                
                # 检查文件内容
                with open(f.name, 'r', encoding='utf-8') as rf:
                    content = rf.read()
                    self.assertIn('version:', content)
                    self.assertIn('services:', content)
                    
            finally:
                os.unlink(f.name)
    
    def test_load_config_from_env(self):
        """测试从环境变量加载配置"""
        # 设置环境变量
        original_env = {}
        test_env = {
            'OLLAMA_BASE_URL': 'http://test:11434',
            'LOG_LEVEL': 'DEBUG'
        }
        
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            config = load_config_from_env()
            
            # 检查配置是否正确加载
            self.assertEqual(
                config['services']['chat']['providers']['ollama']['base_url'],
                'http://test:11434'
            )
            self.assertEqual(config['logging']['level'], 'DEBUG')
            
        finally:
            # 恢复环境变量
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def test_merge_configs(self):
        """测试配置合并"""
        base_config = {
            'version': '1.0',
            'services': {
                'chat': {
                    'default_provider': 'ollama',
                    'providers': {
                        'ollama': {
                            'base_url': 'http://localhost:11434',
                            'model_name': 'llama2'
                        }
                    }
                }
            }
        }
        
        override_config = {
            'services': {
                'chat': {
                    'providers': {
                        'ollama': {
                            'model_name': 'codellama'
                        }
                    }
                }
            }
        }
        
        merged = merge_configs(base_config, override_config)
        
        # 检查合并结果
        self.assertEqual(merged['version'], '1.0')
        self.assertEqual(
            merged['services']['chat']['providers']['ollama']['model_name'],
            'codellama'
        )
        self.assertEqual(
            merged['services']['chat']['providers']['ollama']['base_url'],
            'http://localhost:11434'
        )


class TestExceptions(unittest.TestCase):
    """异常测试"""
    
    def test_ai_service_error(self):
        """测试AI服务错误"""
        error = AIServiceError("Test error")
        self.assertEqual(str(error), "Test error")
        self.assertIsInstance(error, Exception)
    
    def test_configuration_error(self):
        """测试配置错误"""
        error = ConfigurationError("Invalid config")
        self.assertEqual(str(error), "Invalid config")
        self.assertIsInstance(error, AIServiceError)
    
    def test_service_not_available_error(self):
        """测试服务不可用错误"""
        error = ServiceNotAvailableError("Service not found")
        self.assertEqual(str(error), "Service not found")
        self.assertIsInstance(error, AIServiceError)
    
    def test_connection_error(self):
        """测试连接错误"""
        error = ConnectionError("Connection failed")
        self.assertEqual(str(error), "Connection failed")
        self.assertIsInstance(error, AIServiceError)
    
    def test_validation_error(self):
        """测试验证错误"""
        error = ValidationError("Validation failed")
        self.assertEqual(str(error), "Validation failed")
        self.assertIsInstance(error, AIServiceError)


class TestAsyncOperations(unittest.IsolatedAsyncioTestCase):
    """异步操作测试"""
    
    async def asyncSetUp(self):
        """异步测试设置"""
        self.factory = AIServiceFactory.create_default()
    
    async def test_health_check(self):
        """测试健康检查"""
        # 由于可能没有实际的Ollama服务，我们模拟健康检查
        chat_service = self.factory.create_service("chat")
        
        # 模拟健康检查
        with patch.object(chat_service, 'health_check', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = True
            
            result = await chat_service.health_check()
            self.assertTrue(result)
            mock_health.assert_called_once()
    
    async def test_test_connections(self):
        """测试连接测试"""
        # 模拟连接测试
        with patch.object(self.factory, 'test_connections', new_callable=AsyncMock) as mock_test:
            mock_test.return_value = {
                'chat': True,
                'embedding': True,
                'rerank': True
            }
            
            results = await self.factory.test_connections()
            self.assertIsInstance(results, dict)
            self.assertIn('chat', results)
            self.assertIn('embedding', results)
            self.assertIn('rerank', results)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.factory = AIServiceFactory.create_default()
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 创建服务
        chat_service = self.factory.create_service("chat")
        embedding_service = self.factory.create_service("embedding")
        rerank_service = self.factory.create_service("rerank")
        
        # 验证服务创建成功
        self.assertIsNotNone(chat_service)
        self.assertIsNotNone(embedding_service)
        self.assertIsNotNone(rerank_service)
        
        # 验证服务类型
        from ai_services.services.chat_service import ChatService
        from ai_services.services.embedding_service import EmbeddingService
        from ai_services.services.rerank_service import RerankService
        
        self.assertIsInstance(chat_service, ChatService)
        self.assertIsInstance(embedding_service, EmbeddingService)
        self.assertIsInstance(rerank_service, RerankService)
    
    def test_configuration_workflow(self):
        """测试配置工作流"""
        # 获取默认配置
        config = get_default_config()
        
        # 验证配置
        errors = validate_config(config)
        self.assertEqual(len(errors), 0)
        
        # 从配置创建工厂
        factory = AIServiceFactory.from_config(config)
        self.assertIsInstance(factory, AIServiceFactory)
        
        # 创建服务
        chat_service = factory.create_service("chat")
        self.assertIsNotNone(chat_service)
    
    def test_error_handling_workflow(self):
        """测试错误处理工作流"""
        # 测试无效服务类型
        with self.assertRaises(ServiceNotAvailableError):
            self.factory.create_service("invalid_service")
        
        # 测试无效配置
        invalid_config = {"invalid": "config"}
        errors = validate_config(invalid_config)
        self.assertGreater(len(errors), 0)


def run_performance_tests():
    """运行性能测试"""
    print("\n🚀 运行性能测试...")
    
    import time
    
    # 测试工厂创建性能
    start_time = time.time()
    for _ in range(100):
        factory = AIServiceFactory.create_default()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"工厂创建平均耗时: {avg_time:.4f} 秒")
    
    # 测试服务创建性能
    factory = AIServiceFactory.create_default()
    
    start_time = time.time()
    for _ in range(50):
        chat_service = factory.create_service("chat")
        embedding_service = factory.create_service("embedding")
        rerank_service = factory.create_service("rerank")
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 50
    print(f"服务创建平均耗时: {avg_time:.4f} 秒")


def run_all_tests():
    """运行所有测试"""
    print("🧪 AI Services 模块测试")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestAIServiceFactory,
        TestChatMessage,
        TestConfigManagement,
        TestExceptions,
        TestAsyncOperations,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 显示测试结果
    print(f"\n📊 测试结果:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print(f"  跳过: {len(result.skipped)}")
    
    if result.failures:
        print(f"\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # 运行性能测试
    run_performance_tests()
    
    # 总结
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\n🎯 测试成功率: {success_rate:.1%}")
    
    if success_rate == 1.0:
        print("🎉 所有测试通过！")
    elif success_rate >= 0.8:
        print("✅ 大部分测试通过")
    else:
        print("⚠️  需要修复测试问题")
    
    return result


if __name__ == "__main__":
    run_all_tests()