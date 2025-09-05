"""
自定义任务类，实现实时进度显示
"""

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
from typing import List, Dict, Any
import time


class VerboseLongCodeArenaTask(ConfigurableTask):
    """
    带详细日志的Long Code Arena任务类
    """
    
    def __init__(self, config=None):
        super().__init__(config=config)
        self.sample_counter = 0
        self.total_samples = 0
    
    def build_all_requests(self, limit=None, rank=None, world_size=None):
        """重写请求构建过程，添加样本计数"""
        requests = super().build_all_requests(limit=limit, rank=rank, world_size=world_size)
        self.total_samples = len(requests)
        print(f"\n🚀 开始Long Code Arena测试: {self.total_samples} 个样本")
        print("=" * 60)
        return requests
    
    def apply_filters(self, requests, docs):
        """在过滤阶段添加详细日志"""
        print(f"\n🔧 开始处理 {len(requests)} 个请求...")
        return super().apply_filters(requests, docs)
    
    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> List[Instance]:
        """构建请求时添加进度显示"""
        self.sample_counter += 1
        
        # 显示当前样本信息
        instruction = doc.get('instruction', '未知任务')
        print(f"\n📝 样本 {self.sample_counter}/{self.total_samples}")
        print(f"任务: {instruction[:60]}...")
        print(f"开始生成代码... ⏱️  {time.strftime('%H:%M:%S')}")
        
        requests = super().construct_requests(doc, ctx, **kwargs)
        return requests
    
    def process_results(self, doc: dict, results: List[str]) -> Dict[str, Any]:
        """处理结果时显示生成的代码"""
        if results and len(results) > 0:
            generated_code = results[0]
            code_lines = len(generated_code.split('\n'))
            code_chars = len(generated_code)
            
            print(f"✅ 代码生成完成!")
            print(f"   长度: {code_chars} 字符, {code_lines} 行")
            
            # 显示代码预览
            preview_lines = generated_code.split('\n')[:3]
            print(f"   预览:")
            for line in preview_lines:
                print(f"     {line[:80]}{'...' if len(line) > 80 else ''}")
            if code_lines > 3:
                print(f"     ... (还有 {code_lines-3} 行)")
            
            print(f"⏰ 完成时间: {time.strftime('%H:%M:%S')}")
            print("-" * 40)
        else:
            print(f"❌ 代码生成失败!")
            print("-" * 40)
        
        return super().process_results(doc, results)


def create_verbose_task_config():
    """创建详细日志任务的配置"""
    return {
        'task': 'lca_library_based_code_generation_realtime',
        'class': VerboseLongCodeArenaTask,
        'dataset_path': 'json',
        'dataset_kwargs': {
            'data_files': 'lm_eval/tasks/long_code_arena/library_based_code_generation/real_sample_data.json'
        },
        'unsafe_code': True,
        'output_type': 'generate_until',
        'test_split': 'train',
        'doc_to_text': 'Instruction: {{instruction}}\n\nGenerate Python code:',
        'doc_to_target': '{{reference}}',
        'metric_list': [
            {
                'metric': '!function utils.pass_at_1',
                'aggregation': 'mean',
                'higher_is_better': True
            }
        ],
        'generation_kwargs': {
            'until': ['\n\n\n\n', '# End of code'],
            'max_gen_toks': 2048,
            'do_sample': True,
            'temperature': 0.1
        },
        'repeats': 1,
        'num_fewshot': 0,
        'filter_list': [
            {
                'name': 'create_test',
                'filter': [
                    {
                        'function': 'custom',
                        'filter_fn': '!function utils.build_predictions_enhanced_verbose'
                    }
                ]
            }
        ]
    }
