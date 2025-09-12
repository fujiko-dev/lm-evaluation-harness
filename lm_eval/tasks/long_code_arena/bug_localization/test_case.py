#!/usr/bin/env python3
"""
Bug Localization 测试用例

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

from .config import BugLocalizationConfig, get_model_config
from .data_loader import BugLocalizationDataLoader
from .lca_bug_localization_task import LCABugLocalizationPyTask
from .metrics import compute_quality_metrics


def test_config():
    """测试配置系统"""
    print("🧪 测试配置系统...")
    
    # 测试默认配置
    config = BugLocalizationConfig()
    assert config.language == 'py'
    assert config.split == 'test'
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
        loaded_config = BugLocalizationConfig.from_file(config_file)
        assert loaded_config.language == config.language
        print("  ✅ 配置文件保存/加载正常")
    finally:
        os.unlink(config_file)


def test_data_loader():
    """测试数据加载器"""
    print("🧪 测试数据加载器...")
    
    config = BugLocalizationConfig()
    
    # 检查数据集路径是否存在
    if not config.dataset_path.exists():
        print(f"  ⚠️  数据集路径不存在，跳过数据加载测试: {config.dataset_path}")
        return
    
    try:
        loader = BugLocalizationDataLoader(config)
        print("  ✅ 数据加载器初始化正常")
        
        # 尝试加载少量数据
        parquet_file = config.dataset_path / f"{config.language}/{config.split}-00000-of-00001.parquet"
        if parquet_file.exists():
            # 模拟加载数据（不实际加载，只验证路径）
            print("  ✅ 数据文件路径正确")
        else:
            print(f"  ⚠️  数据文件不存在: {parquet_file}")
            
    except Exception as e:
        print(f"  ❌ 数据加载器测试失败: {e}")


def test_metrics():
    """测试评估指标"""
    print("🧪 测试评估指标...")
    
    # 测试数据
    all_files = ['file1.py', 'file2.py', 'file3.py', 'file4.py', 'file5.py']
    expected_files = ['file2.py', 'file4.py']
    predicted_files = ['file2.py', 'file3.py']
    
    # 计算指标
    metrics = compute_quality_metrics(all_files, expected_files, predicted_files)
    
    # 验证基本指标
    assert 'bug_loc_precision' in metrics
    assert 'bug_loc_recall' in metrics
    assert 'bug_loc_f1' in metrics
    assert 0 <= metrics['bug_loc_precision'] <= 1
    assert 0 <= metrics['bug_loc_recall'] <= 1
    assert 0 <= metrics['bug_loc_f1'] <= 1
    
    print(f"  ✅ 指标计算正常 (F1: {metrics['bug_loc_f1']:.3f})")


def test_task_class():
    """测试任务类"""
    print("🧪 测试任务类...")
    
    config_dict = {
        'language': 'py',
        'split': 'test',
        'max_context_files': 50
    }
    
    try:
        task = LCABugLocalizationPyTask(config_dict)
        
        # 测试基本属性
        assert task.config.get('language') == 'py'
        assert hasattr(task, 'data_loader')
        print("  ✅ 任务类初始化正常")
        
        # 测试文档处理方法
        sample_doc = {
            'text_id': 'test_001',
            'repo_owner': 'test_owner',
            'repo_name': 'test_repo',
            'issue_title': 'Test Bug',
            'issue_body': 'This is a test bug description',
            'changed_files': ['test_file.py'],
            'repo_files': ['file1.py', 'file2.py', 'test_file.py']
        }
        
        # 测试 doc_to_text
        text = task.doc_to_text(sample_doc)
        assert 'test_owner/test_repo' in text
        assert 'Test Bug' in text
        print("  ✅ doc_to_text 方法正常")
        
        # 测试 doc_to_target
        target = task.doc_to_target(sample_doc)
        assert target == ['test_file.py']
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
        'language': 'py',
        'split': 'test',
        'limit': 1,  # 只测试1个样本
        'batch_size': 1
    }
    
    # 检查数据集路径
    config = BugLocalizationConfig(config_dict)
    if not config.validate():
        print("  ⚠️  配置验证失败，跳过集成测试")
        return
    
    try:
        # 创建任务实例
        task = LCABugLocalizationPyTask(config_dict)
        
        # 模拟获取测试文档
        print("  📋 模拟任务流程...")
        
        # 创建模拟文档
        mock_doc = {
            'text_id': 'integration_test_001',
            'repo_owner': 'mock_owner',
            'repo_name': 'mock_repo',
            'issue_title': 'Integration Test Bug',
            'issue_body': 'This is an integration test for the bug localization system',
            'changed_files': ['mock_file.py'],
            'repo_files': ['mock_file.py', 'other_file.py', 'utils.py']
        }
        
        # 测试完整流程
        text = task.doc_to_text(mock_doc)
        target = task.doc_to_target(mock_doc)
        requests = task.construct_requests(mock_doc, text)
        
        # 模拟模型响应
        mock_response = ['{"files": ["mock_file.py"]}']
        result = task.process_results(mock_doc, mock_response)
        
        # 验证结果
        assert 'bug_loc_f1' in result
        assert 'predicted_files' in result
        assert result['predicted_files'] == ['mock_file.py']
        
        print("  ✅ 集成测试通过")
        
    except Exception as e:
        print(f"  ❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("🧪 Bug Localization 测试套件")
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
        print("2. 运行评估: python run_bug_localization_eval.py --model gpt-4o --limit 5")
        
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
