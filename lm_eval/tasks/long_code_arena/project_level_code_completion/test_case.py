#!/usr/bin/env python3
"""
Project Level Code Completion 任务测试用例

对重构后的代码进行全面测试，确保功能正常
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 导入被测试的模块
from .config import ProjectLevelCodeCompletionConfig
from .data_loader import load_project_level_code_completion_dataset
from .lca_project_level_code_completion_task import LCAProjectLevelCodeCompletionTask


class TestProjectLevelCodeCompletionConfig(unittest.TestCase):
    """测试配置管理类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ProjectLevelCodeCompletionConfig()
        
        # 检查基本属性
        self.assertIsNotNone(config.get('dataset_path'))
        self.assertIsNotNone(config.get('output_dir'))
        self.assertEqual(config.get('context_size'), 'small_context')
        self.assertEqual(config.get('split'), 'test')
        self.assertEqual(config.get('batch_size'), 1)
    
    def test_custom_config(self):
        """测试自定义配置"""
        custom_config = {
            'context_size': 'large_context',
            'batch_size': 4,
            'limit': 10
        }
        config = ProjectLevelCodeCompletionConfig(custom_config)
        
        self.assertEqual(config.get('context_size'), 'large_context')
        self.assertEqual(config.get('batch_size'), 4)
        self.assertEqual(config.get('limit'), 10)
    
    def test_model_args_string(self):
        """测试模型参数字符串生成"""
        config = ProjectLevelCodeCompletionConfig()
        
        # 测试默认参数
        args_str = config.get_model_args_string()
        self.assertIn('model=gpt-4o', args_str)
        self.assertIn('temperature=0.1', args_str)
        
        # 测试千帆参数
        qianfan_str = config.get_model_args_string(use_qianfan=True)
        self.assertIn('model=qwen3-235b-a22b-instruct-2507', qianfan_str)
        self.assertIn('base_url=https://qianfan.baidubce.com', qianfan_str)
    
    def test_config_file_operations(self):
        """测试配置文件操作"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            # 保存配置
            config = ProjectLevelCodeCompletionConfig({'test_key': 'test_value'})
            config.save_to_file(config_file)
            
            # 加载配置
            loaded_config = ProjectLevelCodeCompletionConfig.from_file(config_file)
            self.assertEqual(loaded_config.get('test_key'), 'test_value')
            
        finally:
            os.unlink(config_file)


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时测试数据
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.test_parquet_file = self.test_data_dir / "data" / "small_context" / "test.parquet"
        self.test_parquet_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建测试DataFrame
        import pandas as pd
        test_df = pd.DataFrame([
            {
                'repo': 'test/repo1',
                'file_path': 'src/main.py',
                'completion_lines': 'def hello():\n    print("Hello World")',
                'context': 'import os\nimport sys\n',
                'target_lines': 'def hello():\n    print("Hello World")'
            },
            {
                'repo': 'test/repo2',
                'file_path': 'src/utils.py',
                'completion_lines': 'def add(a, b):\n    return a + b',
                'context': 'def multiply(a, b):\n    return a * b\n',
                'target_lines': 'def add(a, b):\n    return a + b'
            }
        ])
        
        test_df.to_parquet(self.test_parquet_file)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
    
    def test_load_dataset_success(self):
        """测试成功加载数据集"""
        data = load_project_level_code_completion_dataset(
            str(self.test_data_dir),
            'small_context'
        )
        
        self.assertEqual(len(data), 2)
        self.assertIn('repo', data[0])
        self.assertIn('completion_lines', data[0])
        self.assertIn('context', data[0])
        self.assertEqual(data[0]['repo'], 'test/repo1')
    
    def test_load_dataset_missing_path(self):
        """测试数据集路径不存在的情况"""
        with self.assertRaises(FileNotFoundError):
            load_project_level_code_completion_dataset(
                '/nonexistent/path',
                'small_context'
            )


class TestProjectLevelCodeCompletionTask(unittest.TestCase):
    """测试Project Level Code Completion任务实现"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试配置
        self.config = ProjectLevelCodeCompletionConfig({
            'dataset_path': '/tmp/test_dataset',
            'context_size': 'small_context',
            'split': 'test',
            'batch_size': 1
        })
        
        # 创建测试数据
        self.test_doc = {
            'idx': 0,
            'repo': 'test/repo',
            'file_path': 'src/main.py',
            'completion_lines': 'def hello():\n    print("Hello World")',
            'context': 'import os\nimport sys\n',
            'target_lines': 'def hello():\n    print("Hello World")'
        }
    
    def test_task_initialization(self):
        """测试任务初始化"""
        task = LCAProjectLevelCodeCompletionTask(self.config.to_dict())
        
        self.assertIsNotNone(task.config)
        self.assertEqual(task.config.get('context_size'), 'small_context')
        self.assertTrue(task.has_test_docs())
        self.assertFalse(task.has_training_docs())
    
    def test_doc_to_text(self):
        """测试文档转文本"""
        task = LCAProjectLevelCodeCompletionTask(self.config.to_dict())
        
        text = task.doc_to_text(self.test_doc)
        
        self.assertIn('code completion', text.lower())
        self.assertIn('test/repo', text)
        self.assertIn('src/main.py', text)
        self.assertIn('import os', text)
    
    def test_doc_to_target(self):
        """测试目标提取"""
        task = LCAProjectLevelCodeCompletionTask(self.config.to_dict())
        
        target = task.doc_to_target(self.test_doc)
        
        self.assertEqual(target, 'def hello():\n    print("Hello World")')
    
    def test_construct_requests(self):
        """测试请求构建"""
        task = LCAProjectLevelCodeCompletionTask(self.config.to_dict())
        
        text = task.doc_to_text(self.test_doc)
        requests = task.construct_requests(self.test_doc, text)
        
        self.assertEqual(len(requests), 1)
        request = requests[0]
        
        self.assertEqual(request.request_type, "generate_until")
        self.assertEqual(request.doc, self.test_doc)
        self.assertEqual(request.idx, 0)
        self.assertEqual(request.metadata[0], "lca_project_level_code_completion")
    
    def test_process_results(self):
        """测试结果处理"""
        task = LCAProjectLevelCodeCompletionTask(self.config.to_dict())
        
        # 模拟模型输出
        model_outputs = ["def hello():\n    print('Hello World')"]
        
        result = task.process_results(self.test_doc, model_outputs)
        
        self.assertIn('predicted_completion', result)
        self.assertIn('target_completion', result)
        self.assertIn('exact_match', result)
        self.assertIn('edit_distance', result)
        self.assertIn('bleu_score', result)
        
        self.assertEqual(result['predicted_completion'], "def hello():\n    print('Hello World')")
        self.assertEqual(result['target_completion'], 'def hello():\n    print("Hello World")')
        
        # 检查指标是否为合理值
        self.assertTrue(0 <= result['exact_match'] <= 1)
        self.assertTrue(result['edit_distance'] >= 0)
        self.assertTrue(0 <= result['bleu_score'] <= 1)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    @patch('lm_eval.tasks.long_code_arena.project_level_code_completion.data_loader.load_project_level_code_completion_dataset')
    def test_end_to_end_evaluation(self, mock_load_dataset):
        """测试端到端评估流程"""
        # Mock数据
        mock_data = [
            {
                'idx': 0,
                'repo': 'test/repo',
                'file_path': 'src/main.py',
                'completion_lines': 'def hello():\n    print("Hello World")',
                'context': 'import os\nimport sys\n',
                'target_lines': 'def hello():\n    print("Hello World")'
            }
        ]
        mock_load_dataset.return_value = mock_data
        
        # 创建任务
        config = ProjectLevelCodeCompletionConfig({
            'context_size': 'small_context',
            'split': 'test',
            'limit': 1
        })
        task = LCAProjectLevelCodeCompletionTask(config.to_dict())
        
        # 检查文档加载
        docs = list(task.test_docs())
        self.assertEqual(len(docs), 1)
        
        # 检查提示生成
        doc = docs[0]
        text = task.doc_to_text(doc)
        self.assertIn('import os', text)
        
        # 检查请求构建
        requests = task.construct_requests(doc, text)
        self.assertEqual(len(requests), 1)
        
        # 模拟模型响应和结果处理
        mock_response = ["def hello():\n    print('Hello World')"]
        result = task.process_results(doc, mock_response)
        
        self.assertIn('exact_match', result)
        self.assertEqual(result['predicted_completion'], "def hello():\n    print('Hello World')")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_classes = [
        TestProjectLevelCodeCompletionConfig,
        TestDataLoader,
        TestProjectLevelCodeCompletionTask,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # 设置路径以便导入模块
    import sys
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parents[4]
    sys.path.insert(0, str(project_root))
    
    success = run_tests()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败。")
        sys.exit(1)
