"""
实时打印模型响应的包装器
"""

from lm_eval.models.openai_completions import LocalChatCompletion
import time
from typing import List, Dict, Any

class RealtimePrintModel(LocalChatCompletion):
    """
    继承自API模型，添加实时打印功能
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.request_counter = 0
        self.total_requests = 0
        
    def generate_until(self, requests) -> List[str]:
        """
        重写generate_until方法，添加实时打印
        """
        if not hasattr(self, 'total_requests_set'):
            self.total_requests = len(requests)
            self.total_requests_set = True
            print(f"\n🚀 开始生成 {self.total_requests} 个样本的代码...")
            print("=" * 60)
        
        results = []
        
        for i, request in enumerate(requests):
            self.request_counter += 1
            
            # 显示当前样本信息
            print(f"\n🔄 样本 {self.request_counter}/{self.total_requests}")
            print("-" * 40)
            
            # 提取指令信息（如果可能）
            context = getattr(request, 'args', [])
            if context:
                context_text = context[0] if isinstance(context, list) else str(context)
                # 提取Instruction部分
                if "Instruction:" in context_text:
                    instruction_start = context_text.find("Instruction:") + 12
                    instruction_end = context_text.find("\\n\\nGenerate Python code:")
                    if instruction_end == -1:
                        instruction_end = instruction_start + 100
                    instruction = context_text[instruction_start:instruction_end].strip()
                    print(f"📝 任务: {instruction[:80]}...")
                
            print(f"⏱️  开始时间: {time.strftime('%H:%M:%S')}")
            print("🤖 正在生成代码...")
            
            # 调用原始的generate_until方法处理单个请求
            single_result = super().generate_until([request])
            
            if single_result and len(single_result) > 0:
                generated_text = single_result[0]
                
                # 实时打印生成的内容
                print(f"✅ 生成完成! 长度: {len(generated_text)} 字符")
                print(f"⏱️  完成时间: {time.strftime('%H:%M:%S')}")
                
                # 显示生成内容的预览
                print("📄 生成的代码预览:")
                lines = generated_text.split('\\n')
                preview_lines = min(8, len(lines))
                
                for j in range(preview_lines):
                    line = lines[j]
                    print(f"   {j+1:2d}| {line[:80]}{'...' if len(line) > 80 else ''}")
                
                if len(lines) > preview_lines:
                    print(f"   ...| (还有 {len(lines) - preview_lines} 行)")
                
                # 简单的代码质量检查
                print("🔍 快速分析:")
                print(f"   总行数: {len(lines)}")
                print(f"   包含import: {'import' in generated_text}")
                print(f"   包含def: {'def' in generated_text}")
                print(f"   包含class: {'class' in generated_text}")
                
                results.extend(single_result)
            else:
                print("❌ 生成失败或为空")
                print(f"⏱️  完成时间: {time.strftime('%H:%M:%S')}")
                results.append("")
            
            print("-" * 40)
            
            # 显示整体进度
            progress = (self.request_counter / self.total_requests) * 100
            print(f"📊 总体进度: {self.request_counter}/{self.total_requests} ({progress:.1f}%)")
            
            if self.request_counter < self.total_requests:
                print("⏳ 准备下一个样本...")
            else:
                print("🎉 所有样本生成完成，开始评估阶段...")
            print()
        
        return results
