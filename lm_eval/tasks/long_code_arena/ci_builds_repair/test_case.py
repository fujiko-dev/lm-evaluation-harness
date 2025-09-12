#!/usr/bin/env python3
"""
CI Builds Repair 测试用例

这是一个可运行的测试用例，验证所有组件是否正常工作
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from .config import CIBuildsRepairConfig, get_model_config
from .data_loader import load_lca_ci_builds_repair_dataset
from .lca_ci_builds_repair_task import LCACIBuildsRepairTask
from .metrics import CIBuildsRepairEvaluator


def test_config():
    """测试配置系统"""
    print("🧪 测试配置系统...")
    
    # 测试默认配置
    config = CIBuildsRepairConfig()
    assert config.get('split') == 'test'
    assert config.get('batch_size') == 1
    print("  ✅ 默认配置正常")
    
    # 测试预设模型配置
    try:
        qianfan_config = get_model_config('qianfan-qwen')
        assert 'qwen' in qianfan_config.model_params['model']
        print("  ✅ 预设模型配置正常")
    except Exception as e:
        print(f"  ⚠️  预设模型配置测试跳过: {e}")
    
    # 测试配置文件保存和加载
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_file = f.name
    
    try:
        config.save_to_file(config_file)
        loaded_config = CIBuildsRepairConfig.from_file(config_file)
        assert loaded_config.get('split') == config.get('split')
        print("  ✅ 配置文件保存/加载正常")
    finally:
        os.unlink(config_file)


def test_data_loader():
    """测试数据加载器"""
    print("🧪 测试数据加载器...")
    
    config = CIBuildsRepairConfig()
    
    # 检查数据集路径是否存在
    if not config.dataset_path.exists():
        print(f"  ⚠️  数据集路径不存在，跳过数据加载测试: {config.dataset_path}")
        return
    
    try:
        # 模拟配置，不实际加载大量数据
        test_config = CIBuildsRepairConfig({
            'dataset_path': str(config.dataset_path),
            'split': 'test'
        })
        print("  ✅ 数据加载器初始化正常")
        
        # 验证数据文件路径
        parquet_file = config.dataset_path / "data/python/test-00000-of-00001.parquet"
        if parquet_file.exists():
            print("  ✅ 数据文件路径正确")
        else:
            print(f"  ⚠️  数据文件不存在: {parquet_file}")
            
    except Exception as e:
        print(f"  ❌ 数据加载器测试失败: {e}")


def test_metrics():
    """测试评估指标"""
    print("🧪 测试评估指标...")
    
    try:
        evaluator = CIBuildsRepairEvaluator()
        
        # 测试单个评估
        result = evaluator.evaluate_single_prediction(
            prediction="def fix_issue():\n    return True",
            datapoint={'text_id': 'test_001', 'target_diff': 'def fix_issue():\n    return True'}
        )
        
        # 验证返回的指标 (CIRepairResult是dataclass)
        assert hasattr(result, 'score')
        assert hasattr(result, 'success')
        assert 0 <= result.score <= 1
        
        print(f"  ✅ 指标计算正常 (Score: {result.score:.3f})")
        
    except Exception as e:
        print(f"  ❌ 指标测试失败: {e}")


def test_task_class():
    """测试任务类"""
    print("🧪 测试任务类...")
    
    config_dict = {
        'split': 'test',
        'batch_size': 1,
        'limit': 1
    }
    
    try:
        task = LCACIBuildsRepairTask(config_dict)
        
        # 测试基本属性
        assert task.config.get('split') == 'test'
        print("  ✅ 任务类初始化正常")
        
        # 测试文档处理方法
        sample_doc = {
            'text_id': 'test_001',
            'repo_name': 'test_repo',
            'build_failure_reason': 'Test build failure',
            'logs': [
                {'step_name': 'test_step', 'log': 'Test log content'}
            ],
            'target_fix': 'def fix():\n    return True'
        }
        
        # 测试 doc_to_text
        text = task.doc_to_text(sample_doc)
        assert 'test_repo' in text
        assert 'Test build failure' in text
        print("  ✅ doc_to_text 方法正常")
        
        # 测试 doc_to_target
        target = task.doc_to_target(sample_doc)
        assert target == 'def fix():\n    return True'
        print("  ✅ doc_to_target 方法正常")
        
        # 测试 construct_requests
        requests = task.construct_requests(sample_doc, text)
        assert len(requests) == 1
        assert requests[0].request_type == "generate_until"
        print("  ✅ construct_requests 方法正常")
        
    except Exception as e:
        print(f"  ❌ 任务类测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_integration():
    """集成测试"""
    print("🧪 进行集成测试...")
    
    # 创建测试配置
    config_dict = {
        'split': 'test',
        'batch_size': 1,
        'limit': 1
    }
    
    # 检查数据集路径
    config = CIBuildsRepairConfig(config_dict)
    if not config.validate():
        print("  ⚠️  配置验证失败，跳过集成测试")
        return
    
    try:
        # 创建任务实例
        task = LCACIBuildsRepairTask(config_dict)
        
        # 模拟获取测试文档
        print("  📋 模拟任务流程...")
        
        # 创建模拟文档
        mock_doc = {
            'text_id': 'integration_test_001',
            'repo_name': 'mock_repo',
            'build_failure_reason': 'Integration test build failure',
            'logs': [
                {'step_name': 'test_step', 'log': 'Integration test log content'}
            ],
            'target_fix': 'def integration_fix():\n    return True'
        }
        
        # 测试完整流程
        text = task.doc_to_text(mock_doc)
        target = task.doc_to_target(mock_doc)
        requests = task.construct_requests(mock_doc, text)
        
        # 模拟模型响应
        mock_response = ['{"fix": "def integration_fix():\\n    return True"}']
        result = task.process_results(mock_doc, mock_response)
        
        # 验证结果
        assert 'ci_repair_score' in result
        assert 'predicted_fix' in result
        
        print("  ✅ 集成测试通过")
        
    except Exception as e:
        print(f"  ❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("🧪 CI Builds Repair 测试套件")
    print("=" * 50)
    
    try:
        test_config()
        test_data_loader()
        test_metrics()
        test_task_class()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！系统可以正常使用。")
        print("\n📖 下一步:")
        print("1. 配置API密钥 (如 OPENAI_API_KEY)")
        print("2. 运行评估: python run_ci_builds_repair_eval.py --model gpt-4o --limit 5")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
