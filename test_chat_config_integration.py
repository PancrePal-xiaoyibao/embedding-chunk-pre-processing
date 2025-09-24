#!/usr/bin/env python3
"""
Chat配置文件集成测试脚本

测试从配置文件读取Chat配置并创建聊天客户端的功能。

Author: Assistant
Date: 2025-01-24
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.config_manager import ConfigManager
from src.core.chat_factory import (
    create_chat_factory_from_config,
    create_ollama_chat_client,
    test_chat_connection
)
from src.core.ollama_chat import create_ollama_chat_client_from_config
from src.core.chat_interface import ChatMessage, MessageRole

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_config_manager_chat_config():
    """测试配置管理器读取Chat配置"""
    logger.info("=== 测试配置管理器读取Chat配置 ===")
    
    try:
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 读取Chat配置
        chat_config = config_manager.get_chat_config()
        logger.info(f"Chat配置读取成功: {chat_config}")
        
        # 检查Ollama配置
        ollama_config = chat_config.ollama or {}
        logger.info(f"Ollama Chat配置: {ollama_config}")
        
        # 验证必要的配置项
        required_keys = ["base_url", "model_name", "timeout"]
        for key in required_keys:
            if key in ollama_config:
                logger.info(f"✓ {key}: {ollama_config[key]}")
            else:
                logger.warning(f"✗ 缺少配置项: {key}")
        
        return True
        
    except Exception as e:
        logger.error(f"配置管理器测试失败: {str(e)}")
        return False


def test_ollama_chat_client_from_config():
    """测试从配置文件创建Ollama聊天客户端"""
    logger.info("=== 测试从配置文件创建Ollama聊天客户端 ===")
    
    try:
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 从配置创建Ollama聊天客户端
        client = create_ollama_chat_client_from_config(config_manager)
        logger.info(f"Ollama聊天客户端创建成功: {type(client).__name__}")
        
        # 获取模型信息
        model_name = client.get_model_name()
        model_id = client.get_model_id()
        logger.info(f"模型名称: {model_name}")
        logger.info(f"模型ID: {model_id}")
        
        # 测试连接
        logger.info("测试Ollama连接...")
        connection_ok = client.test_connection()
        if connection_ok:
            logger.info("✓ Ollama连接测试成功")
        else:
            logger.warning("✗ Ollama连接测试失败")
        
        return connection_ok
        
    except Exception as e:
        logger.error(f"从配置创建Ollama客户端失败: {str(e)}")
        return False


def test_chat_factory_with_config():
    """测试使用配置管理器的聊天工厂"""
    logger.info("=== 测试使用配置管理器的聊天工厂 ===")
    
    try:
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 创建聊天工厂
        factory = create_chat_factory_from_config(config_manager)
        logger.info(f"聊天工厂创建成功: {type(factory).__name__}")
        
        # 获取可用提供商
        providers = factory.get_available_providers()
        logger.info(f"可用提供商: {[p.value for p in providers]}")
        
        # 创建Ollama聊天客户端
        client = factory.create_chat_client("ollama")
        logger.info(f"通过工厂创建Ollama客户端成功: {type(client).__name__}")
        
        # 测试连接
        connection_ok = factory.test_provider_connection("ollama")
        if connection_ok:
            logger.info("✓ 通过工厂测试Ollama连接成功")
        else:
            logger.warning("✗ 通过工厂测试Ollama连接失败")
        
        return connection_ok
        
    except Exception as e:
        import traceback
        logger.error(f"聊天工厂测试失败: {str(e)}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def test_convenience_functions():
    """测试便捷函数"""
    logger.info("=== 测试便捷函数 ===")
    
    try:
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 测试create_ollama_chat_client便捷函数
        logger.info("测试create_ollama_chat_client便捷函数...")
        client = create_ollama_chat_client(config_manager=config_manager)
        logger.info(f"便捷函数创建Ollama客户端成功: {type(client).__name__}")
        
        # 测试test_chat_connection便捷函数
        logger.info("测试test_chat_connection便捷函数...")
        connection_ok = test_chat_connection("ollama", config_manager=config_manager)
        if connection_ok:
            logger.info("✓ 便捷函数测试连接成功")
        else:
            logger.warning("✗ 便捷函数测试连接失败")
        
        return connection_ok
        
    except Exception as e:
        logger.error(f"便捷函数测试失败: {str(e)}")
        return False


def test_chat_functionality():
    """测试聊天功能"""
    logger.info("=== 测试聊天功能 ===")
    
    try:
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 创建聊天客户端
        client = create_ollama_chat_client_from_config(config_manager)
        
        # 测试连接
        if not client.test_connection():
            logger.warning("Ollama连接失败，跳过聊天功能测试")
            return False
        
        # 测试模型可用性
        model_name = client.get_model_name()
        logger.info(f"检查模型可用性: {model_name}")
        
        if not client.is_model_available(model_name):
            logger.warning(f"模型 {model_name} 不可用，跳过聊天功能测试")
            return False
        
        # 测试简单聊天
        logger.info("测试简单聊天...")
        messages = [ChatMessage(role=MessageRole.USER, content="你好，请简单介绍一下你自己。")]
        
        response = client.chat(messages)
        if response and response.message and response.message.content:
            logger.info(f"✓ 聊天响应成功: {response.message.content[:100]}...")
            return True
        else:
            logger.warning("✗ 聊天响应为空")
            return False
        
    except Exception as e:
        logger.error(f"聊天功能测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    logger.info("开始Chat配置文件集成测试")
    
    # 检查配置文件是否存在
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return False
    
    logger.info(f"使用配置文件: {config_path}")
    
    # 运行测试
    tests = [
        ("配置管理器读取Chat配置", test_config_manager_chat_config),
        ("从配置文件创建Ollama聊天客户端", test_ollama_chat_client_from_config),
        ("使用配置管理器的聊天工厂", test_chat_factory_with_config),
        ("便捷函数", test_convenience_functions),
        ("聊天功能", test_chat_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} - 通过")
            else:
                logger.warning(f"✗ {test_name} - 失败")
        except Exception as e:
            logger.error(f"✗ {test_name} - 异常: {str(e)}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    logger.info(f"\n{'='*50}")
    logger.info("测试结果摘要")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "通过" if result else "失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！配置文件集成成功！")
        return True
    else:
        logger.warning(f"⚠️  有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)