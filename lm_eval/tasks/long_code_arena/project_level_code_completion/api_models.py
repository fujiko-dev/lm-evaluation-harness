"""
大模型API接口
支持各种大模型API调用
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import openai
import requests
import json

logger = logging.getLogger(__name__)


class BaseAPIModel(ABC):
    """基础API模型类"""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.generation_config = kwargs
        
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """生成文本"""
        pass
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 100) -> List[str]:
        """批量生成文本"""
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, max_tokens)
                results.append(result)
            except Exception as e:
                logger.error(f"生成失败: {str(e)}")
                results.append("")
        return results


class OpenAIModel(BaseAPIModel):
    """OpenAI API模型"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        
        # 设置OpenAI客户端
        openai.api_key = api_key
        if base_url:
            openai.base_url = base_url
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 默认生成配置
        self.default_config = {
            'temperature': 0.0,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }
        self.default_config.update(kwargs)
        
        # 支持stop参数
        if 'stop' in kwargs:
            self.default_config['stop'] = kwargs['stop']
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """使用OpenAI API生成文本"""
        try:
            # 构建请求配置，避免参数重复
            config = self.default_config.copy()
            config['max_tokens'] = max_tokens  # 使用传入的max_tokens
            
            # 修正：确保stop参数是列表格式（千帆API要求）
            if 'stop' in config and isinstance(config['stop'], str):
                config['stop'] = [config['stop']]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **config
            )
            
            generated_text = response.choices[0].message.content
            return generated_text.strip() if generated_text else ""
            
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise


class QianfanModel(BaseAPIModel):
    """百度千帆API模型"""
    
    def __init__(self, model_name: str, api_key: str, secret_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.secret_key = secret_key
        self.access_token = None
        self.token_expires_at = 0
        
        # 默认生成配置
        self.default_config = {
            'temperature': 0.01,
            'top_p': 1.0,
            'penalty_score': 1.0,
        }
        self.default_config.update(kwargs)
        
        # 模型端点映射
        self.model_endpoints = {
            'ernie-4.0-8k': 'completions_pro',
            'ernie-3.5-8k': 'completions',
            'qianfan-bloomz-7b': 'qianfan_bloomz_7b_compressed',
            'llama2-7b': 'llama_2_7b',
            'llama2-13b': 'llama_2_13b',
            'llama2-70b': 'llama_2_70b',
            'codellama-7b': 'codellama_7b_instruct',
        }
    
    def _get_access_token(self) -> str:
        """获取访问令牌"""
        current_time = time.time()
        
        # 如果token还没过期，直接返回
        if self.access_token and current_time < self.token_expires_at:
            return self.access_token
        
        # 获取新的access token
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        try:
            response = requests.post(url, params=params)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # 提前5分钟过期以防止边界情况
            self.token_expires_at = current_time + token_data.get("expires_in", 3600) - 300
            
            logger.info("成功获取千帆API访问令牌")
            return self.access_token
            
        except Exception as e:
            logger.error(f"获取千帆API访问令牌失败: {str(e)}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """使用千帆API生成文本"""
        try:
            access_token = self._get_access_token()
            
            # 获取模型端点
            endpoint = self.model_endpoints.get(self.model_name.lower(), 'completions')
            url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{endpoint}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # 构建请求数据
            data = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_output_tokens": max_tokens,
                **self.default_config
            }
            
            # 发送请求
            response = requests.post(
                url,
                headers=headers,
                params={"access_token": access_token},
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 检查是否有错误
            if "error_code" in result:
                error_msg = result.get("error_msg", "未知错误")
                raise Exception(f"千帆API错误: {error_msg}")
            
            generated_text = result.get("result", "")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"千帆API调用失败: {str(e)}")
            raise


class ClaudeModel(BaseAPIModel):
    """Anthropic Claude API模型"""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        
        # 默认生成配置
        self.default_config = {
            'temperature': 0.0,
            'top_p': 1.0,
        }
        self.default_config.update(kwargs)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """使用Claude API生成文本"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **self.default_config
            )
            
            generated_text = response.content[0].text
            return generated_text.strip() if generated_text else ""
            
        except Exception as e:
            logger.error(f"Claude API调用失败: {str(e)}")
            raise


class GenericAPIModel(BaseAPIModel):
    """通用API模型（支持自定义端点）"""
    
    def __init__(self, model_name: str, api_key: str, api_url: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_url = api_url
        
        # 默认生成配置
        self.default_config = {
            'temperature': 0.0,
            'top_p': 1.0,
        }
        self.default_config.update(kwargs)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """使用通用API生成文本"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                **self.default_config
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("choices", [{}])[0].get("text", "")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"通用API调用失败: {str(e)}")
            raise


def create_api_model(
    model_type: str,
    model_name: str,
    api_key: str,
    **kwargs
) -> BaseAPIModel:
    """
    创建API模型实例
    
    Args:
        model_type: 模型类型 ('openai', 'qianfan', 'claude', 'generic')
        model_name: 模型名称
        api_key: API密钥
        **kwargs: 其他配置参数
        
    Returns:
        API模型实例
    """
    model_type = model_type.lower()
    
    if model_type == 'openai':
        return OpenAIModel(model_name, api_key, **kwargs)
    elif model_type == 'qianfan':
        secret_key = kwargs.pop('secret_key', None)
        if not secret_key:
            raise ValueError("千帆API需要提供secret_key")
        return QianfanModel(model_name, api_key, secret_key, **kwargs)
    elif model_type == 'claude':
        return ClaudeModel(model_name, api_key, **kwargs)
    elif model_type == 'generic':
        api_url = kwargs.pop('api_url', None)
        if not api_url:
            raise ValueError("通用API需要提供api_url")
        return GenericAPIModel(model_name, api_key, api_url, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 示例配置
EXAMPLE_CONFIGS = {
    'gpt-4': {
        'model_type': 'openai',
        'model_name': 'gpt-4',
        'temperature': 0.0,
        'max_tokens': 150
    },
    'gpt-3.5-turbo': {
        'model_type': 'openai',
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.0,
        'max_tokens': 150
    },
    'ernie-4.0': {
        'model_type': 'qianfan',
        'model_name': 'ernie-4.0-8k',
        'temperature': 0.01,
        'max_tokens': 150
    },
    'claude-3': {
        'model_type': 'claude',
        'model_name': 'claude-3-sonnet-20240229',
        'temperature': 0.0,
        'max_tokens': 150
    }
}
