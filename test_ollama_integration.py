#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Embedding集成测试脚本

功能：
- 测试Ollama embedding模型的连接和功能
- 验证配置管理器对Ollama配置的支持
- 测试文档处理器中embedding生成功能
- 提供详细的测试报告

作者: Embedding增强项目团队
版本: 1.0
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def setup_test_logging():
    """
    设置测试日志
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("OllamaIntegrationTest")

def test_ollama_connection():
    """
    测试Ollama服务器连接
    
    Returns:
        bool: 连接是否成功
    """
    logger = logging.getLogger("OllamaIntegrationTest")
    
    try:
        from src.core.ollama_embedding import OllamaEmbeddingClient, OllamaEmbeddingConfig
        
        # 创建配置
        config = OllamaEmbeddingConfig(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text",
            timeout=30
        )
        
        # 创建客户端
        client = OllamaEmbeddingClient(config)
        
        # 测试连接
        logger.info("🔗 测试Ollama服务器连接...")
        is_connected = client.test_connection()
        
        if is_connected:
            logger.info("✅ Ollama服务器连接成功")
            return True
        else:
            logger.warning("❌ Ollama服务器连接失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ollama连接测试异常: {e}")
        return False

def test_ollama_model_availability():
    """
    测试Ollama模型可用性
    
    Returns:
        bool: 模型是否可用
    """
    logger = logging.getLogger("OllamaIntegrationTest")
    
    try:
        from src.core.ollama_embedding import OllamaEmbeddingClient, OllamaEmbeddingConfig
        
        # 创建配置
        config = OllamaEmbeddingConfig(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text"
        )
        
        # 创建客户端
        client = OllamaEmbeddingClient(config)
        
        # 测试模型可用性
        logger.info("🔍 检查模型可用性...")
        is_available = client.is_model_available()
        
        if is_available:
            logger.info("✅ 模型可用")
            return True
        else:
            logger.warning("❌ 模型不可用，尝试拉取模型...")
            # 尝试拉取模型
            success = client.pull_model()
            if success:
                logger.info("✅ 模型拉取成功")
                return True
            else:
                logger.error("❌ 模型拉取失败")
                return False
                
    except Exception as e:
        logger.error(f"❌ 模型可用性测试异常: {e}")
        return False

def test_embedding_generation():
    """
    测试embedding生成功能
    
    Returns:
        bool: embedding生成是否成功
    """
    logger = logging.getLogger("OllamaIntegrationTest")
    
    try:
        from src.core.ollama_embedding import create_ollama_embedding_client
        
        # 创建embedding管理器
        logger.info("🧠 创建Ollama embedding管理器...")
        config_dict = {
            "base_url": "http://localhost:11434",
            "model_name": "nomic-embed-text",
            "timeout": 30,
            "max_retries": 3,
            "batch_size": 32,
            "normalize_embeddings": True,
            "enable_cache": True
        }
        manager = create_ollama_embedding_client(config_dict)
        
        # 测试单个文本embedding
        test_text = "这是一个测试文档，用于验证embedding生成功能。"
        logger.info(f"📝 测试文本: {test_text}")
        
        embedding = manager.get_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            logger.info(f"✅ 单个embedding生成成功，维度: {len(embedding)}")
            
            # 测试批量embedding
            test_texts = [
                "第一个测试文档",
                "第二个测试文档",
                "第三个测试文档"
            ]
            
            logger.info("📚 测试批量embedding生成...")
            embeddings = manager.get_embeddings(test_texts)
            
            if embeddings and len(embeddings) == len(test_texts):
                logger.info(f"✅ 批量embedding生成成功，数量: {len(embeddings)}")
                return True
            else:
                logger.error("❌ 批量embedding生成失败")
                return False
        else:
            logger.error("❌ 单个embedding生成失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embedding生成测试异常: {e}")
        return False

def test_config_manager_integration():
    """
    测试配置管理器对Ollama的支持
    
    Returns:
        bool: 配置管理器集成是否成功
    """
    logger = logging.getLogger("OllamaIntegrationTest")
    
    try:
        from src.config.config_manager import ConfigManager
        
        # 创建配置管理器
        logger.info("⚙️ 测试配置管理器...")
        config_manager = ConfigManager()
        
        # 获取embedding配置
        embedding_config = config_manager.get_embedding_config()
        
        logger.info(f"📋 当前embedding提供商: {embedding_config.provider}")
        logger.info(f"📋 Sentence Transformers配置: {embedding_config.sentence_transformers}")
        logger.info(f"📋 Ollama配置: {embedding_config.ollama}")
        
        # 测试设置Ollama为提供商
        config_manager.set_value("models.embedding.provider", "ollama")
        updated_config = config_manager.get_embedding_config()
        
        if updated_config.provider == "ollama":
            logger.info("✅ 配置管理器Ollama集成成功")
            return True
        else:
            logger.error("❌ 配置管理器Ollama集成失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 配置管理器测试异常: {e}")
        return False

def test_embedding_factory():
    """
    测试embedding工厂
    
    Returns:
        bool: embedding工厂是否正常工作
    """
    logger = logging.getLogger("OllamaIntegrationTest")
    
    try:
        from src.core.embedding_factory import EmbeddingFactory
        from src.config.config_manager import ConfigManager
        
        # 创建配置管理器并直接设置为Ollama
        config_manager = ConfigManager()
        
        # 测试ollama提供商
        logger.info("🏭 测试Embedding工厂 - Ollama...")
        config_manager.set_value("models.embedding.provider", "ollama")
        
        factory = EmbeddingFactory(config_manager)
        ollama_model = factory.create_embedding()
        
        if ollama_model:
            logger.info("✅ Ollama模型创建成功")
            
            # 测试embedding生成
            test_text = "测试embedding工厂功能"
            embedding = ollama_model.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info(f"✅ Embedding工厂测试成功，维度: {len(embedding)}")
                return True
            else:
                logger.error("❌ Embedding工厂生成embedding失败")
                return False
        else:
            logger.error("❌ Ollama模型创建失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embedding工厂测试异常: {e}")
        return False

def test_document_processor_integration():
    """
    测试文档处理器中的embedding集成
    
    Returns:
        bool: 文档处理器embedding集成是否成功
    """
    logger = logging.getLogger("OllamaIntegrationTest")
    
    try:
        # 添加src路径到sys.path
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # 导入模块
        from core.document_processor import DocumentProcessor
        from config.config_manager import ConfigManager
        
        # 创建配置管理器并设置为Ollama
        config_manager = ConfigManager()
        config_manager.set_value("models.embedding.provider", "ollama")
        
        # 创建文档处理器
        logger.info("📄 测试文档处理器embedding集成...")
        processor = DocumentProcessor(config_manager)
        
        # 测试文档处理
        test_content = """
        # 测试文档
        
        这是一个用于测试embedding生成的示例文档。
        
        ## 第一章节
        
        这里是第一章节的内容，包含一些医学相关的术语。
        
        ## 第二章节
        
        这里是第二章节的内容，用于测试分块和embedding功能。
        """
        
        result = processor.process_document(
            content=test_content,
            strategy="semantic",
            extract_keywords=True,
            evaluate_quality=True,
            generate_embeddings=True
        )
        
        if result.success and result.chunks:
            # 检查是否生成了embedding
            embedding_count = sum(1 for chunk in result.chunks if chunk.embedding is not None)
            
            logger.info(f"✅ 文档处理成功")
            logger.info(f"📊 总分块数: {len(result.chunks)}")
            logger.info(f"🧠 生成embedding数: {embedding_count}")
            
            if embedding_count > 0:
                logger.info("✅ 文档处理器embedding集成成功")
                return True
            else:
                logger.error("❌ 未生成任何embedding")
                return False
        else:
            logger.error(f"❌ 文档处理失败: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 文档处理器测试异常: {e}")
        return False

async def run_comprehensive_test():
    """
    运行综合测试
    
    Returns:
        Dict[str, bool]: 各项测试结果
    """
    logger = setup_test_logging()
    
    logger.info("🚀 开始Ollama Embedding集成测试...")
    logger.info("=" * 60)
    
    test_results = {}
    
    # 1. 测试Ollama连接
    logger.info("\n1️⃣ 测试Ollama服务器连接")
    test_results["connection"] = test_ollama_connection()
    
    # 2. 测试模型可用性
    logger.info("\n2️⃣ 测试模型可用性")
    test_results["model_availability"] = test_ollama_model_availability()
    
    # 3. 测试embedding生成
    logger.info("\n3️⃣ 测试embedding生成")
    test_results["embedding_generation"] = test_embedding_generation()
    
    # 4. 测试配置管理器
    logger.info("\n4️⃣ 测试配置管理器集成")
    test_results["config_manager"] = test_config_manager_integration()
    
    # 5. 测试embedding工厂
    logger.info("\n5️⃣ 测试embedding工厂")
    test_results["embedding_factory"] = test_embedding_factory()
    
    # 6. 测试文档处理器集成
    logger.info("\n6️⃣ 测试文档处理器集成")
    test_results["document_processor"] = test_document_processor_integration()
    
    # 输出测试结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试结果汇总:")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"\n🎯 总体结果: {passed_tests}/{total_tests} 项测试通过")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有测试通过！Ollama embedding集成成功！")
    else:
        logger.warning("⚠️ 部分测试失败，请检查相关配置和服务状态")
    
    return test_results

def main():
    """
    主函数
    """
    try:
        # 运行测试
        results = asyncio.run(run_comprehensive_test())
        
        # 根据结果设置退出码
        if all(results.values()):
            sys.exit(0)  # 所有测试通过
        else:
            sys.exit(1)  # 有测试失败
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"测试运行异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()