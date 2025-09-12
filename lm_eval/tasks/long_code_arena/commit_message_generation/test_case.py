#!/usr/bin/env python3
"""
Commit Message Generation 任务测试用例

对重构后的代码进行全面测试，确保功能正常
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 导入被测试的模块
from .config import CommitMessageGenerationConfig
from .data_loader import load_lca_commit_message_generation_dataset, preprocess_commit_data
from .lca_commit_message_generation_task import LCACommitMessageGenerationTask


class TestCommitMessageGenerationConfig(unittest.TestCase):
    """测试配置管理类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = CommitMessageGenerationConfig()
        
        # 检查基本属性
        self.assertIsNotNone(config.get('dataset_path'))
        self.assertIsNotNone(config.get('output_dir'))
        self.assertEqual(config.get('split'), 'test')
        self.assertEqual(config.get('batch_size'), 1)
    
    def test_custom_config(self):
        """测试自定义配置"""
        custom_config = {
            'split': 'train',
            'batch_size': 4,
            'limit': 10
        }
        config = CommitMessageGenerationConfig(custom_config)
        
        self.assertEqual(config.get('split'), 'train')
        self.assertEqual(config.get('batch_size'), 4)
        self.assertEqual(config.get('limit'), 10)
    
    def test_model_args_string(self):
        """测试模型参数字符串生成"""
        config = CommitMessageGenerationConfig()
        
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
            config = CommitMessageGenerationConfig({'test_key': 'test_value'})
            config.save_to_file(config_file)
            
            # 加载配置
            loaded_config = CommitMessageGenerationConfig.from_file(config_file)
            self.assertEqual(loaded_config.get('test_key'), 'test_value')
            
        finally:
            os.unlink(config_file)


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时测试数据
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.test_parquet_file = self.test_data_dir / "test.parquet"
        
        # 创建测试DataFrame
        import pandas as pd
        test_df = pd.DataFrame([
            {
                'repo': 'test/repo1',
                'commit_sha': 'abc123',
                'message': 'Add new feature',
                'diff': '+def new_function():\n+    return True',
                'mods': ['file1.py'],
                'adds': [],
                'dels': []
            },
            {
                'repo': 'test/repo2',
                'commit_sha': 'def456',
                'message': 'Fix bug in module',
                'diff': '-old_code()\n+new_code()',
                'mods': ['file2.py'],
                'adds': [],
                'dels': []
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
        config = CommitMessageGenerationConfig({
            'dataset_path': str(self.test_data_dir)
        })
        
        data = load_lca_commit_message_generation_dataset(config)
        
        self.assertEqual(len(data), 2)
        self.assertIn('repo', data[0])
        self.assertIn('message', data[0])
        self.assertIn('diff', data[0])
        self.assertEqual(data[0]['repo'], 'test/repo1')
    
    def test_load_dataset_missing_path(self):
        """测试数据集路径不存在的情况"""
        config = CommitMessageGenerationConfig({
            'dataset_path': '/nonexistent/path'
        })
        
        with self.assertRaises(FileNotFoundError):
            load_lca_commit_message_generation_dataset(config)
    
    def test_preprocess_commit_data(self):
        """测试数据预处理"""
        raw_data = [
            {
                'repo': 'test/repo',
                'message': 'Test commit',
                'diff': 'x' * 10000,  # 长diff
                'mods': ['file1.py', 'file2.py']
            }
        ]
        
        config = CommitMessageGenerationConfig({
            'max_context_length': 1000
        })
        
        processed_data = preprocess_commit_data(raw_data, config)
        
        self.assertEqual(len(processed_data), 1)
        self.assertTrue(processed_data[0]['diff_truncated'])
        self.assertEqual(processed_data[0]['num_files_modified'], 2)


class TestCommitMessageGenerationTask(unittest.TestCase):
    """测试Commit Message Generation任务实现"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试配置
        self.config = CommitMessageGenerationConfig({
            'dataset_path': '/tmp/test_dataset',
            'split': 'test',
            'batch_size': 1
        })
        
        # 创建测试数据
        self.test_doc = {
            'idx': 0,
            'repo': 'test/repo',
            'commit_sha': 'abc123',
            'message': 'Add new feature for authentication',
            'diff': '+def authenticate(user):\n+    return validate_user(user)',
            'mods': ['auth.py'],
            'adds': [],
            'dels': [],
            'num_files_modified': 1,
            'num_files_added': 0,
            'num_files_deleted': 0
        }
    
    def test_task_initialization(self):
        """测试任务初始化"""
        task = LCACommitMessageGenerationTask(self.config)
        
        self.assertIsNotNone(task.config)
        self.assertEqual(task.config.get('split'), 'test')
        self.assertTrue(task.has_test_docs())
        self.assertFalse(task.has_training_docs())
    
    def test_doc_to_text(self):
        """测试文档转文本"""
        task = LCACommitMessageGenerationTask(self.config)
        
        text = task.doc_to_text(self.test_doc)
        
        self.assertIn('expert software developer', text)
        self.assertIn('Repository: test/repo', text)
        self.assertIn('1 modified', text)
        self.assertIn('authenticate(user)', text)
        self.assertIn('Commit message:', text)
    
    def test_doc_to_target(self):
        """测试目标提取"""
        task = LCACommitMessageGenerationTask(self.config)
        
        target = task.doc_to_target(self.test_doc)
        
        self.assertEqual(target, 'Add new feature for authentication')
    
    def test_construct_requests(self):
        """测试请求构建"""
        task = LCACommitMessageGenerationTask(self.config)
        
        text = task.doc_to_text(self.test_doc)
        requests = task.construct_requests(self.test_doc, text)
        
        self.assertEqual(len(requests), 1)
        request = requests[0]
        
        self.assertEqual(request.request_type, "generate_until")
        self.assertEqual(request.doc, self.test_doc)
        self.assertEqual(request.idx, 0)
        self.assertEqual(request.metadata[0], "lca_commit_message_generation")
    
    def test_process_results(self):
        """测试结果处理"""
        task = LCACommitMessageGenerationTask(self.config)
        
        # 模拟模型输出
        model_outputs = ["Add user authentication system"]
        
        result = task.process_results(self.test_doc, model_outputs)
        
        self.assertIn('predicted_message', result)
        self.assertIn('target_message', result)
        self.assertIn('commit_bleu', result)
        self.assertIn('commit_rouge1', result)
        self.assertIn('commit_rougeL', result)
        
        self.assertEqual(result['predicted_message'], 'Add user authentication system')
        self.assertEqual(result['target_message'], 'Add new feature for authentication')
        
        # 检查指标是否为合理值
        self.assertTrue(0 <= result['commit_bleu'] <= 1)
        self.assertTrue(0 <= result['commit_rouge1'] <= 1)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    @patch('lm_eval.tasks.long_code_arena.commit_message_generation.data_loader.load_lca_commit_message_generation_dataset')
    def test_end_to_end_evaluation(self, mock_load_dataset):
        """测试端到端评估流程"""
        # Mock数据
        mock_data = [
            {
                'idx': 0,
                'repo': 'test/repo',
                'commit_sha': 'abc123',
                'message': 'Fix parsing issue',
                'diff': '+parser.fix_issue()\n+return result',
                'mods': ['parser.py'],
                'adds': [],
                'dels': []
            }
        ]
        mock_load_dataset.return_value = mock_data
        
        # 创建任务
        config = CommitMessageGenerationConfig({
            'split': 'test',
            'limit': 1
        })
        task = LCACommitMessageGenerationTask(config)
        
        # 检查文档加载
        docs = list(task.test_docs())
        self.assertEqual(len(docs), 1)
        
        # 检查提示生成
        doc = docs[0]
        text = task.doc_to_text(doc)
        self.assertIn('parser.fix_issue', text)
        
        # 检查请求构建
        requests = task.construct_requests(doc, text)
        self.assertEqual(len(requests), 1)
        
        # 模拟模型响应和结果处理
        mock_response = ["Fix parser implementation"]
        result = task.process_results(doc, mock_response)
        
        self.assertIn('commit_bleu', result)
        self.assertEqual(result['predicted_message'], 'Fix parser implementation')


class TestMetrics(unittest.TestCase):
    """测试评估指标"""
    
    def test_metrics_calculation(self):
        """测试指标计算"""
        from .metrics import compute_bleu_score, compute_rouge_scores
        
        predictions = ["Add new feature"]
        references = ["Add feature for users"]
        
        # 测试BLEU
        bleu_score = compute_bleu_score(predictions, references)
        self.assertTrue(0 <= bleu_score <= 1)
        
        # 测试ROUGE
        rouge_scores = compute_rouge_scores(predictions, references)
        self.assertIn('rouge1', rouge_scores)
        self.assertIn('rouge2', rouge_scores)
        self.assertIn('rougeL', rouge_scores)
        
        for score in rouge_scores.values():
            self.assertTrue(0 <= score <= 1)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_classes = [
        TestCommitMessageGenerationConfig,
        TestDataLoader,
        TestCommitMessageGenerationTask,
        TestIntegration,
        TestMetrics
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
