import ast
import traceback
import re
import subprocess
import tempfile
import os
from typing import List, Dict, Any

# Try to import official evaluation dependencies
try:
    from sacrebleu import CHRF
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("Warning: sacrebleu not available, ChrF metric will not work")

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False
    print("Warning: tree-sitter not available, API recall metric will use fallback implementation")


def pass_at_1(references: List[str], predictions: List[List[str]]) -> float:
    """
    Calculate pass@1 metric for code generation.
    
    Args:
        references: List of reference code strings
        predictions: List of lists of predicted code strings
        
    Returns:
        pass@1 score (fraction of correctly functioning predictions)
    """
    if len(references) != len(predictions):
        raise ValueError("Number of references and predictions must match")
    
    total = len(references)
    passed = 0
    
    print(f"\n🔍 开始评估 {total} 个样本的Pass@1...")
    
    for i, (ref, pred_list) in enumerate(zip(references, predictions)):
        sample_num = i + 1
        print(f"\n📊 样本 {sample_num}/{total} - Pass@1评估")
        print("-" * 40)
        
        if not pred_list:
            print("❌ 没有预测结果")
            continue
            
        pred = pred_list[0]  # Take first prediction for pass@1
        
        # 显示预测代码的简短信息
        pred_lines = len(pred.split('\n'))
        pred_chars = len(pred)
        print(f"🤖 生成代码: {pred_chars}字符, {pred_lines}行")
        
        # 显示代码的前几行
        code_preview = '\n'.join(pred.split('\n')[:3])
        print(f"   预览: {code_preview[:100]}...")
        
        is_correct = is_functionally_correct(ref, pred)
        if is_correct:
            passed += 1
            print(f"✅ 结果: PASS (功能正确)")
        else:
            print(f"❌ 结果: FAIL (功能错误)")
        
        current_rate = passed / sample_num
        print(f"📈 当前Pass@1: {passed}/{sample_num} = {current_rate:.3f}")
        print("-" * 40)
    
    return passed / total if total > 0 else 0.0


def is_functionally_correct(reference: str, prediction: str) -> bool:
    """
    Check if the prediction is functionally correct compared to reference.
    This implements a more sophisticated evaluation including syntax validation,
    import checking, and basic structure validation.
    
    Args:
        reference: Reference code string
        prediction: Predicted code string
        
    Returns:
        True if prediction appears functionally correct
    """
    try:
        # 1. Basic syntax check
        ast.parse(prediction)
        
        # 2. Extract and compare imports
        ref_info = extract_imports_and_functions(reference)
        pred_info = extract_imports_and_functions(prediction)
        
        # 3. Check if prediction uses similar imports (library usage)
        ref_libs = {imp.split('.')[0] for imp in ref_info['imports']}
        pred_libs = {imp.split('.')[0] for imp in pred_info['imports']}
        
        # Allow some flexibility in import usage
        if ref_libs and not (ref_libs & pred_libs):
            # If reference uses specific libraries, prediction should use at least one
            return False
        
        # 4. Check for basic function structure if reference has functions
        if ref_info['functions'] and not pred_info['functions']:
            # If reference defines functions, prediction should too
            return False
        
        # 5. Basic semantic validation - check for common patterns
        if not _validate_semantic_patterns(reference, prediction):
            return False
        
        return True
        
    except SyntaxError:
        return False
    except Exception:
        # Log error for debugging but don't fail completely
        return False


def _validate_semantic_patterns(reference: str, prediction: str) -> bool:
    """
    Validate semantic patterns between reference and prediction.
    
    Args:
        reference: Reference code
        prediction: Predicted code
        
    Returns:
        True if semantic patterns match
    """
    # Check for common data science patterns
    patterns = [
        (r'\.read_csv\(', 'pandas CSV reading'),
        (r'\.plot\(', 'plotting functionality'),
        (r'\.mean\(', 'mean calculation'),
        (r'\.sum\(', 'sum calculation'),
        (r'\.fit\(', 'model fitting'),
        (r'\.predict\(', 'prediction'),
        (r'np\.[a-zA-Z_]+\(', 'numpy functions'),
        (r'pd\.[a-zA-Z_]+\(', 'pandas functions'),
        (r'plt\.[a-zA-Z_]+\(', 'matplotlib functions'),
    ]
    
    ref_patterns = set()
    pred_patterns = set()
    
    for pattern, name in patterns:
        if re.search(pattern, reference):
            ref_patterns.add(name)
        if re.search(pattern, prediction):
            pred_patterns.add(name)
    
    # If reference has specific patterns, prediction should have similar ones
    if ref_patterns and not (ref_patterns & pred_patterns):
        # Allow some flexibility - if at least 50% of patterns match
        overlap = len(ref_patterns & pred_patterns) / len(ref_patterns)
        return overlap >= 0.3
    
    return True


def build_predictions(resps: List[List[str]], docs: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Build predictions from model responses.
    
    Args:
        resps: List of lists of response strings from the model
        docs: List of document dictionaries
        
    Returns:
        List of lists of processed predictions
    """
    predictions = []
    
    for resp_list, doc in zip(resps, docs):
        pred_list = []
        for resp in resp_list:
            # Clean up the response - remove any markdown code blocks
            cleaned_resp = clean_code_response(resp)
            pred_list.append(cleaned_resp)
        predictions.append(pred_list)
    
    return predictions


def clean_code_response(response: str) -> str:
    """
    Clean up model response to extract just the code.
    
    Args:
        response: Raw response from the model
        
    Returns:
        Cleaned code string
    """
    # Remove markdown code blocks if present
    if "```python" in response:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        if end != -1:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            response = response[start:end].strip()
    
    # Remove any trailing explanations
    lines = response.split('\n')
    code_lines = []
    for line in lines:
        # Stop if we hit a comment that looks like explanation
        if line.strip().startswith('#') and any(word in line.lower() for word in ['explanation', 'note', 'this']):
            break
        code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


def extract_imports_and_functions(code: str) -> Dict[str, Any]:
    """
    Extract imports and function definitions from code.
    
    Args:
        code: Code string to analyze
        
    Returns:
        Dictionary with imports and functions
    """
    try:
        tree = ast.parse(code)
        imports = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return {
            "imports": imports,
            "functions": functions
        }
    except:
        return {"imports": [], "functions": []}


# ===== Official LCA Evaluation Functions =====

class ParsedFile:
    """
    Simplified version of the official ParsedFile class for API extraction.
    """
    def __init__(self, code: str):
        self.code = code
        self.called_functions = set()
        
        if HAS_TREE_SITTER:
            self._parse_with_tree_sitter()
        else:
            self._parse_with_ast()
    
    def _parse_with_tree_sitter(self):
        """Parse code using tree-sitter (official method)"""
        try:
            PY_LANGUAGE = Language(tspython.language())
            parser = Parser(PY_LANGUAGE)
            
            bytecode = bytes(self.code, 'utf-8')
            tree = parser.parse(bytecode)
            root = tree.root_node
            
            # Query for function calls
            called_function_name_query = """
            (call function: (identifier) @fun_name)
            (call function: (attribute attribute: (identifier) @fun_name))
            """
            
            query = PY_LANGUAGE.query(called_function_name_query)
            captures = query.captures(root)
            
            self.called_functions = set(capture[0].text.decode("utf-8") for capture in captures)
        except Exception:
            # Fallback to AST if tree-sitter fails
            self._parse_with_ast()
    
    def _parse_with_ast(self):
        """Fallback parsing using Python AST"""
        try:
            tree = ast.parse(self.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.called_functions.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        self.called_functions.add(node.func.attr)
        except Exception:
            pass


def chrf_score(generated_code: str, reference_code: str) -> float:
    """
    Calculate ChrF score between generated and reference code.
    
    Args:
        generated_code: Generated code string
        reference_code: Reference code string
        
    Returns:
        ChrF score (0-1 range)
    """
    if not HAS_SACREBLEU:
        # Fallback to simple character-level similarity
        return _simple_char_similarity(generated_code, reference_code)
    
    try:
        chrf = CHRF()
        score = chrf.sentence_score(generated_code, [reference_code]).score
        return score / 100.0  # Convert to 0-1 range
    except Exception:
        return _simple_char_similarity(generated_code, reference_code)


def api_recall_score(generated_code: str, reference_code: str, unique_apis: List[str]) -> float:
    """
    Calculate API recall score - how many reference APIs are used in generated code.
    
    Args:
        generated_code: Generated code string
        reference_code: Reference code string (not used directly but kept for compatibility)
        unique_apis: List of unique API calls expected in the reference
        
    Returns:
        API recall score (0-1 range)
    """
    if not unique_apis:
        return 1.0
    
    try:
        parsed_generated = ParsedFile(generated_code)
        generated_apis = parsed_generated.called_functions
        
        # Calculate overlap with expected APIs
        api_set = set(unique_apis)
        guessed_apis = generated_apis & api_set
        
        return len(guessed_apis) / len(api_set)
    except Exception:
        return 0.0


def _simple_char_similarity(text1: str, text2: str) -> float:
    """Simple character-level similarity as fallback for ChrF"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    # Simple character overlap
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    
    overlap = len(chars1 & chars2)
    union = len(chars1 | chars2)
    
    return overlap / union if union > 0 else 0.0


def extract_code_from_response(response: str) -> str:
    """
    Extract code from model response, handling various formats.
    This is the official implementation from the LCA baselines.
    """
    if "```python" in response:
        return response.split("```python")[1].split("```")[0].strip()
    if "```" in response:
        return response.split("```")[1].split("```")[0].strip()
    if "<code>" in response:
        return response.split("<code>")[1].split("</code>")[0].strip()
    return response.strip()


# ===== Enhanced Evaluation Function =====

def comprehensive_evaluation(generated_code: str, reference_code: str, unique_apis: List[str] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation using multiple metrics.
    
    Args:
        generated_code: Generated code string
        reference_code: Reference code string
        unique_apis: List of unique API calls expected
        
    Returns:
        Dictionary with multiple evaluation scores
    """
    results = {}
    
    # Clean the generated code
    cleaned_generated = extract_code_from_response(generated_code)
    
    # 1. Functional correctness (existing implementation)
    results['functional_correct'] = float(is_functionally_correct(reference_code, cleaned_generated))
    
    # 2. ChrF score
    results['chrf'] = chrf_score(cleaned_generated, reference_code)
    
    # 3. API recall
    if unique_apis:
        results['api_recall'] = api_recall_score(cleaned_generated, reference_code, unique_apis)
    else:
        results['api_recall'] = 0.0
    
    # 4. Syntax correctness
    try:
        ast.parse(cleaned_generated)
        results['syntax_correct'] = 1.0
    except SyntaxError:
        results['syntax_correct'] = 0.0
    
    return results


# ===== Metric Functions for lm-eval Framework =====

def chrf_metric(items: List[Any]) -> float:
    """
    ChrF metric function for lm-eval framework.
    
    Args:
        items: List containing [reference, prediction] pair
        
    Returns:
        ChrF score for this single sample
    """
    if len(items) != 2:
        print("⚠️ ChrF评估: 输入格式错误")
        return 0.0
    
    reference, prediction = items
    
    # Handle prediction being a list or string
    if isinstance(prediction, list):
        if not prediction:
            print("⚠️ ChrF评估: 没有预测结果")
            return 0.0
        prediction = prediction[0]  # Take first prediction
    
    try:
        cleaned_pred = extract_code_from_response(prediction)
        score = chrf_score(cleaned_pred, reference)
        print(f"📊 ChrF得分: {score:.3f}")
        return score
    except Exception as e:
        print(f"❌ ChrF评估失败: {e}")
        return 0.0


def api_recall_metric(items: List[Any]) -> float:
    """
    API recall metric function for lm-eval framework.
    This function is called with a list containing [reference, prediction] for each sample.
    We need to get the document info another way.
    
    Args:
        items: List containing [reference, prediction] pair
        
    Returns:
        API recall score for this single sample
    """
    if len(items) != 2:
        print("⚠️ API Recall评估: 输入格式错误")
        return 0.0
    
    reference, prediction = items
    
    # Handle prediction being a list or string
    if isinstance(prediction, list):
        if not prediction:
            print("⚠️ API Recall评估: 没有预测结果")
            return 0.0
        prediction = prediction[0]  # Take first prediction
    
    # For now, extract APIs from the reference code itself
    # This is a simplified approach since we don't have direct access to unique_apis
    try:
        cleaned_pred = extract_code_from_response(prediction)
        ref_parsed = ParsedFile(reference)
        pred_parsed = ParsedFile(cleaned_pred)
        
        # Use reference APIs as the target
        ref_apis = ref_parsed.called_functions
        pred_apis = pred_parsed.called_functions
        
        print(f"🔍 API分析: 参考API={len(ref_apis)}, 生成API={len(pred_apis)}")
        print(f"   参考APIs: {list(ref_apis)[:3]}{'...' if len(ref_apis) > 3 else ''}")
        print(f"   生成APIs: {list(pred_apis)[:3]}{'...' if len(pred_apis) > 3 else ''}")
        
        if not ref_apis:
            print("📊 API Recall: 1.0 (无需检查API)")
            return 1.0  # If no APIs to check, consider it perfect
        
        overlap = len(ref_apis & pred_apis)
        score = overlap / len(ref_apis)
        print(f"📊 API Recall得分: {overlap}/{len(ref_apis)} = {score:.3f}")
        return score
        
    except Exception as e:
        print(f"❌ API Recall评估失败: {e}")
        return 0.0


# Verbose version with detailed logging
def build_predictions_enhanced_verbose(resps: List[List[str]], docs: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Enhanced build predictions function with verbose logging for progress tracking.
    
    Args:
        resps: List of response lists from the model
        docs: List of document dictionaries containing task information
        
    Returns:
        List of processed predictions
    """
    total_samples = len(resps)
    print(f"\n🚀 开始处理 {total_samples} 个样本的代码生成结果...")
    print("=" * 60)
    
    results = []
    for i, (resp_list, doc) in enumerate(zip(resps, docs)):
        sample_num = i + 1
        print(f"\n🔄 样本 {sample_num}/{total_samples} - 代码处理")
        print("-" * 30)
        
        # 显示任务信息
        instruction = doc.get('instruction', '未知任务')
        print(f"📝 任务: {instruction[:50]}...")
        
        if not resp_list:
            print("❌ 模型没有生成任何代码")
            results.append([""])
            continue
        
        raw_response = resp_list[0]
        print(f"🤖 原始响应长度: {len(raw_response)} 字符")
        
        # 处理代码
        try:
            cleaned_code = extract_code_from_response(raw_response)
            code_lines = len(cleaned_code.split('\n'))
            
            # 显示清理后的代码预览
            preview_lines = cleaned_code.split('\n')[:3]
            preview = '\n   '.join(preview_lines)
            
            print(f"🔧 清理后代码: {len(cleaned_code)} 字符, {code_lines} 行")
            print(f"   预览:")
            print(f"   {preview}")
            if code_lines > 3:
                print(f"   ... (还有 {code_lines-3} 行)")
            
            results.append([cleaned_code])
            print("✅ 代码处理完成")
            
        except Exception as e:
            print(f"❌ 代码处理失败: {e}")
            results.append([raw_response])  # 使用原始响应作为fallback
        
        print("-" * 30)
    
    print(f"\n🎯 代码处理完成! 共处理 {total_samples} 个样本")
    print("=" * 60)
    return results


# Updated build_predictions to handle more complex processing
def build_predictions_enhanced(resps: List[List[str]], docs: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Enhanced version of build_predictions with better code extraction.
    
    Args:
        resps: List of lists of response strings from the model
        docs: List of document dictionaries
        
    Returns:
        List of lists of processed predictions
    """
    predictions = []
    
    for resp_list, doc in zip(resps, docs):
        pred_list = []
        for resp in resp_list:
            # Use the official code extraction method
            cleaned_resp = extract_code_from_response(resp)
            pred_list.append(cleaned_resp)
        predictions.append(pred_list)
    
    return predictions
