"""
Commit Message Generation 任务配置文件

管理所有可配置的参数，包括路径、模型参数等
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class CommitMessageGenerationConfig:
    """Commit Message Generation 任务配置类"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化配置
        
        Args:
            config_dict: 配置字典，如果为None则使用默认配置
        """
        self._config = config_dict or {}
        
        # 设置默认值
        self._set_defaults()
    
    def _set_defaults(self):
        """设置默认配置值"""
        # 数据集路径配置
        default_dataset_base = os.getenv(
            'LCA_CMG_DATASET_PATH',
            '/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/lca-commit-message-generation'
        )
        
        # 结果输出路径配置
        default_output_base = os.getenv(
            'LCA_CMG_OUTPUT_PATH',
            '/Users/xiaoyunting/Documents/ModelDev/working_path/projects/lm-evaluation-harness/results/long_code_arena/commit_message_generation'
        )
        
        # 默认配置
        defaults = {
            # 路径配置
            'dataset_path': default_dataset_base,
            'output_dir': default_output_base,
            
            # 任务配置
            'split': 'test',
            'batch_size': 1,
            'limit': None,
            'max_context_length': 8000,
            'include_diff': True,
            'max_commit_length': 200,
            
            # 模型配置
            'model_name': 'openai-chat-completions',
            'model_params': {
                'model': 'gpt-4o',
                'base_url': None,  # 使用默认URL
                'api_key': None,   # 从环境变量获取
                'max_tokens': 200,
                'temperature': 0.1,
                'num_concurrent': 5,
                'max_retries': 3,
                'timeout': 120,
                'stop': None,
                'apply_chat_template': True
            },
            
            # 千帆模型专用配置
            'qianfan_params': {
                'model': 'qwen3-235b-a22b-instruct-2507',
                'base_url': 'https://qianfan.baidubce.com/v2/chat/completions',
                'api_key': None,  # 从环境变量OPENAI_API_KEY获取（千帆兼容格式）
                'max_tokens': 200,
                'temperature': 0.1,
                'num_concurrent': 5,
                'max_retries': 10,
                'timeout': 120,
                'stop': '<|endoftext|>',
                'apply_chat_template': True
            }
        }
        
        # 将默认值应用到配置中（如果不存在的话）
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """更新配置"""
        self._config.update(config_dict)
    
    @property
    def dataset_path(self) -> Path:
        """数据集路径"""
        return Path(self._config['dataset_path'])
    
    @property
    def output_dir(self) -> Path:
        """输出目录"""
        return Path(self._config['output_dir'])
    
    @property
    def split(self) -> str:
        """数据拆分"""
        return self._config['split']
    
    @property
    def batch_size(self) -> int:
        """批处理大小"""
        return self._config['batch_size']
    
    @property
    def limit(self) -> Optional[int]:
        """样本限制数量"""
        return self._config['limit']
    
    @property
    def max_context_length(self) -> int:
        """最大上下文长度"""
        return self._config['max_context_length']
    
    @property
    def include_diff(self) -> bool:
        """是否包含代码差异"""
        return self._config['include_diff']
    
    @property
    def max_commit_length(self) -> int:
        """最大提交消息长度"""
        return self._config['max_commit_length']
    
    @property
    def model_name(self) -> str:
        """模型名称"""
        return self._config['model_name']
    
    @property
    def model_params(self) -> Dict[str, Any]:
        """模型参数"""
        return self._config['model_params'].copy()
    
    @property
    def qianfan_params(self) -> Dict[str, Any]:
        """千帆模型参数"""
        return self._config['qianfan_params'].copy()
    
    def get_model_args_string(self, use_qianfan: bool = False) -> str:
        """
        获取模型参数字符串（用于lm_eval）
        
        Args:
            use_qianfan: 是否使用千帆模型参数
            
        Returns:
            格式化的参数字符串
        """
        params = self.qianfan_params if use_qianfan else self.model_params
        
        # 过滤掉None值
        filtered_params = {k: v for k, v in params.items() if v is not None}
        
        # 转换为字符串格式
        arg_strings = []
        for key, value in filtered_params.items():
            if isinstance(value, bool):
                arg_strings.append(f"{key}={str(value).lower()}")
            else:
                arg_strings.append(f"{key}={value}")
        
        return ",".join(arg_strings)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CommitMessageGenerationConfig':
        """从字典创建配置对象"""
        return cls(config_dict)
    
    @classmethod
    def from_file(cls, config_file: str) -> 'CommitMessageGenerationConfig':
        """从配置文件创建配置对象"""
        import json
        
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        import json
        
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必需路径
            if not self.dataset_path.exists():
                print(f"警告: 数据集路径不存在: {self.dataset_path}")
                return False
            
            # 检查数据拆分
            if self.split not in ['train', 'dev', 'test']:
                print(f"错误: 不支持的数据拆分: {self.split}")
                return False
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False


def get_default_config() -> CommitMessageGenerationConfig:
    """获取默认配置"""
    return CommitMessageGenerationConfig()


def get_qianfan_config() -> CommitMessageGenerationConfig:
    """获取千帆模型配置"""
    config = CommitMessageGenerationConfig()
    config.set('model_name', 'openai-chat-completions')
    return config


# 预定义的模型配置
MODEL_CONFIGS = {
    'gpt-4o': {
        'model_name': 'openai-chat-completions',
        'model_params': {
            'model': 'gpt-4o',
            'max_tokens': 200,
            'temperature': 0.1
        }
    },
    'gpt-4o-mini': {
        'model_name': 'openai-chat-completions',
        'model_params': {
            'model': 'gpt-4o-mini',
            'max_tokens': 200,
            'temperature': 0.1
        }
    },
    'claude-3.5-sonnet': {
        'model_name': 'anthropic-chat',
        'model_params': {
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 200,
            'temperature': 0.1
        }
    },
    'qianfan-qwen': {
        'model_name': 'openai-chat-completions',
        'model_params': {
            'model': 'qwen3-235b-a22b-instruct-2507',
            'base_url': 'https://qianfan.baidubce.com/v2/chat/completions',
            'max_tokens': 200,
            'temperature': 0.1,
            'num_concurrent': 5,
            'max_retries': 10,
            'timeout': 120,
            'stop': '<|endoftext|>',
            'apply_chat_template': True
        }
    }
}


def get_model_config(model_key: str) -> CommitMessageGenerationConfig:
    """
    获取预定义的模型配置
    
    Args:
        model_key: 模型配置键
        
    Returns:
        配置对象
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"未知的模型配置: {model_key}. 可用配置: {list(MODEL_CONFIGS.keys())}")
    
    config = CommitMessageGenerationConfig()
    config.update(MODEL_CONFIGS[model_key])
    return config
