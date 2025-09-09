"""
API 性能测试工具

用于测试 LLM API 的并发性能，支持 SSE 流式请求，
统计 TTFT（首 token 时间）、完成时间、tokens/s 等指标。
支持 OpenAI、Anthropic 和 Gemini API 格式。

作者: Claude
版本: 1.0.1
"""

import os
import sys
import argparse
import requests
import time
import concurrent.futures
from datetime import datetime
import statistics
import json
import random
import re

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# 默认配置值（仅作为 argparse 的默认值使用）
DEFAULT_API_URL = "https://open.bigmodel.cn/api/anthropic/v1/messages"
DEFAULT_CHAT_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.5"
DEFAULT_TEST_MESSAGE = "What opportunities and challenges will the Chinese large model industry face in 2025?"
DEFAULT_MIN_CONCURRENCY = 5
DEFAULT_MAX_CONCURRENCY = 100
DEFAULT_STEP = 5
DEFAULT_TEST_ROUNDS = 1
DEFAULT_TIMEOUT = 120
DEFAULT_PRINT_SAMPLE_ERRORS = 5
DEFAULT_CHARS_PER_TOKEN = 4.0
DEFAULT_PROMPT_TOKENS = 500


class APIPerformanceTester:
    """API 性能测试工具 - 支持 SSE 流式请求测试"""
    
    def __init__(self, api_url=None, api_key=None, model=None, test_message=None, 
                 min_concurrency=None, max_concurrency=None, step=None, test_rounds=None,
                 timeout=None, print_sample_errors=None, estimate_tokens_by_chars=None,
                 chars_per_token=None, use_chat_api=None, use_stream=None, use_gemini_api=None,
                 prompt_tokens=None):
        """初始化测试配置
        
        Args:
            api_url: API 地址
            api_key: API 密钥
            model: 使用的模型
            test_message: 测试消息
            min_concurrency: 最小并发级别
            max_concurrency: 最大并发级别
            step: 并发步长
            test_rounds: 测试轮数
            timeout: 超时时间
            print_sample_errors: 打印错误数量
            estimate_tokens_by_chars: 是否估算 tokens
            chars_per_token: 字符/token 比率
            use_chat_api: 是否使用 Chat API 接口
            use_stream: 是否使用流式请求
            use_gemini_api: 是否使用 Gemini API 接口
            prompt_tokens: 提示词的 token 数量（默认：500）
        """
        # API 配置
        self.use_chat_api = use_chat_api or False
        self.use_gemini_api = use_gemini_api or False
        if self.use_gemini_api and api_url is None:
            self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
        elif self.use_chat_api and api_url is None:
            self.api_url = DEFAULT_CHAT_API_URL
        else:
            self.api_url = api_url or DEFAULT_API_URL
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.test_message = test_message or DEFAULT_TEST_MESSAGE
        self.use_stream = use_stream if use_stream is not None else True  # 默认启用流式
        
        # 测试参数
        self.min_concurrency = min_concurrency or DEFAULT_MIN_CONCURRENCY
        self.max_concurrency = max_concurrency or DEFAULT_MAX_CONCURRENCY
        self.step = step or DEFAULT_STEP
        self.test_rounds = test_rounds or DEFAULT_TEST_ROUNDS
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.print_sample_errors = print_sample_errors or DEFAULT_PRINT_SAMPLE_ERRORS
        self.estimate_tokens_by_chars = estimate_tokens_by_chars or False
        self.chars_per_token = chars_per_token or DEFAULT_CHARS_PER_TOKEN
        self.prompt_tokens = prompt_tokens or DEFAULT_PROMPT_TOKENS
        
        # 初始化分词器
        self.tokenizer = None
        self._init_tokenizer()

    def run_test(self):
        """运行完整的性能测试"""
        # 检查 API Key
        if not self.api_key:
            print("❌ 错误：请先设置 API_KEY")
            print("提示：创建测试器时传入 api_key 参数")
            return None
            
        print("🚀 开始 API 并发性能测试（SSE + TTFT + tokens/s）")
        if not self.use_stream:
            print("⚠️  流式请求已禁用，使用非流式模式")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API 地址: {self.api_url}")
        print(f"模型: {self.model}")
        print(f"测试范围: {self.min_concurrency}-{self.max_concurrency} 并发 (步长: {self.step})")
        print(f"每个并发级别测试轮数: {self.test_rounds}")
        print(f"单请求超时: {self.timeout}秒")
        print(f"提示词长度: {self.prompt_tokens} tokens")

        results = {}

        for concurrency in range(self.min_concurrency, self.max_concurrency + 5, self.step):
            result = test_concurrency(concurrency, self)
            results[concurrency] = result

            total_req = result.success_count + result.failure_count
            succ_rate = (result.success_count / total_req) if total_req else 0
            if succ_rate < 0.8:
                print(f"\n⚠️  成功率低于 80%，停止继续提升并发")
                break

            time.sleep(2)  # 防止过度压测

        # 打印汇总报告
        self._print_summary(results)
        
        # 打印最大并发上限
        if results:
            max_concurrency_achieved = max(results.keys())
            max_result = results[max_concurrency_achieved]
            total_req = max_result.success_count + max_result.failure_count
            success_rate = max_result.success_count / total_req if total_req else 0
            max_concurrency_count = int(max_concurrency_achieved * success_rate)
            
            print(f"\n最大并发上限: {max_concurrency_count}")
        
        return results
    
    def _init_tokenizer(self):
        """初始化分词器"""
        # 优先使用本地分词器
        if TOKENIZERS_AVAILABLE:
            try:
                # 根据模型选择合适的分词器文件
                tokenizer_file = self._get_tokenizer_file()
                if tokenizer_file and os.path.exists(tokenizer_file):
                    self.tokenizer = Tokenizer.from_file(tokenizer_file)
                    print(f"✅ 使用本地分词器: {tokenizer_file}")
                    return
            except Exception:
                pass
        
        # 如果本地分词器不可用，尝试使用tiktoken
        if TIKTOKEN_AVAILABLE:
            try:
                # 优先尝试使用o200k_base
                import tiktoken
                try:
                    tiktoken.get_encoding("o200k_base")
                    print(f"✅ 使用 tiktoken 分词器")
                    return
                except:
                    # 如果o200k_base不可用，回退到模型特定的编码器
                    encoding_name = self._get_tiktoken_encoding()
                    tiktoken.get_encoding(encoding_name)
                    print(f"✅ 使用 tiktoken 分词器")
                    return
            except Exception:
                pass
        
        # 如果都不可用，使用字符估算
        print(f"📝 使用字符估算模式（{self.chars_per_token} 字符/token）")
    
    def _get_tokenizer_file(self):
        """根据模型名称获取对应的分词器文件路径"""
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 模型到分词器文件的映射
        MODEL_TOKENIZER_MAP = {
            "glm": "tokenizer_glm.json",
            "glm-4": "tokenizer_glm.json",
            "glm-4.5": "tokenizer_glm.json",
            "deepseek": "tokenizer_ds.json",
            "deepseek-chat": "tokenizer_ds.json",
            "deepseek-coder": "tokenizer_ds.json",
            "llama": "tokenizer_llama.json",
            "llama-2": "tokenizer_llama.json",
            "llama-3": "tokenizer_llama.json",
            "llama-3.1": "tokenizer_llama.json",
            "grok": "tokenizer_grok2.json",
            "grok-2": "tokenizer_grok2.json",
            "grok-beta": "tokenizer_grok2.json"
        }
        
        # 检查模型名称（转为小写进行匹配）
        model_lower = self.model.lower()
        
        # 遍历映射表，找到匹配的分词器
        for model_key, tokenizer_filename in MODEL_TOKENIZER_MAP.items():
            if model_key in model_lower:
                tokenizer_path = os.path.join(current_dir, tokenizer_filename)
                return tokenizer_path
        
        return None
    
    def _get_tiktoken_encoding(self):
        """根据模型名称获取对应的tiktoken编码器名称"""
        # 模型到tiktoken编码器的映射（简化版）
        MODEL_TIKTOKEN_MAP = {
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-4": "cl100k_base",
            "gpt-3.5": "cl100k_base",
            "claude": "cl100k_base",
            "deepseek": "cl100k_base",
            "mistral": "cl100k_base",
            "llama": "cl100k_base",
            "grok": "cl100k_base",
            "qwen": "cl100k_base",
            "default": "o200k_base"
        }
        
        # 检查模型名称（转为小写进行匹配）
        model_lower = self.model.lower()
        
        # 遍历映射表，找到匹配的编码器
        for model_key, encoding_name in MODEL_TIKTOKEN_MAP.items():
            if model_key in model_lower:
                return encoding_name
        
        # 使用默认编码器
        return MODEL_TIKTOKEN_MAP.get("default", "o200k_base")
    
    def _count_tokens(self, text):
        """计算文本的 token 数量"""
        # 优先使用本地分词器
        if self.tokenizer:
            try:
                encoding = self.tokenizer.encode(text)
                return len(encoding.ids)
            except Exception:
                pass
        
        # 如果本地分词器不可用，尝试使用tiktoken
        if TIKTOKEN_AVAILABLE:
            try:
                import tiktoken
                # 优先尝试使用o200k_base
                try:
                    encoding = tiktoken.get_encoding("o200k_base")
                    return len(encoding.encode(text))
                except:
                    # 如果o200k_base不可用，回退到模型特定的编码器
                    encoding_name = self._get_tiktoken_encoding()
                    encoding = tiktoken.get_encoding(encoding_name)
                    return len(encoding.encode(text))
            except Exception:
                pass
        
        # 最后回退到字符估算
        return int(len(text) / self.chars_per_token)
    
    def _generate_prompt_content(self, target_tokens):
        """生成指定 token 数量的提示内容"""
        # 基础问题模板
        base_question = "What opportunities and challenges will the Chinese large model industry face in 2025? Please analyze from the following aspects:"
        
        # 分析方面列表
        aspects = [
            "1. Technical development trends and breakthrough points",
            "2. Policy environment and regulatory challenges",
            "3. Commercial application scenarios and market prospects",
            "4. Data security and privacy protection issues",
            "5. International competition and cooperation landscape",
            "6. Talent cultivation and ecosystem construction",
            "7. Computing power and infrastructure constraints",
            "8. Ethical considerations and social responsibility",
            "9. Industry-specific applications and customization needs",
            "10. Future development predictions and strategic recommendations"
        ]
        
        # 生成内容
        content = base_question + "\n\n" + "\n".join(aspects)
        
        # 计算当前token数
        current_tokens = self._count_tokens(content)
        
        # 如果当前token数不足，添加更多内容
        if current_tokens < target_tokens:
            # 生成额外的详细内容
            additional_content = []
            paragraph_num = 0
            
            while current_tokens < target_tokens and paragraph_num < 100:
                # 生成段落内容
                paragraph = self._generate_detailed_paragraph(paragraph_num)
                paragraph_tokens = self._count_tokens(paragraph)
                
                if current_tokens + paragraph_tokens <= target_tokens:
                    additional_content.append(paragraph)
                    current_tokens += paragraph_tokens
                    paragraph_num += 1
                else:
                    # 需要精确控制长度
                    remaining_tokens = target_tokens - current_tokens
                    if remaining_tokens > 50:  # 只有当剩余token足够时才添加
                        # 截取段落的一部分
                        partial_text = self._extract_partial_text(paragraph, remaining_tokens)
                        if partial_text:
                            additional_content.append(partial_text)
                            break
                    break
            
            if additional_content:
                content += "\n\n" + "\n\n".join(additional_content)
        
        # 最终检查和调整
        final_tokens = self._count_tokens(content)
        if final_tokens != target_tokens:
            # 简单的字符级调整
            content = self._adjust_content_length(content, target_tokens)
        
        return content
    
    def _generate_detailed_paragraph(self, num):
        """生成详细的分析段落"""
        # 预定义的详细分析内容模板
        templates = [
            "The Chinese large model industry is experiencing rapid development, with significant investments in research and development. Key players are focusing on improving model capabilities while reducing computational costs.",
            "Market analysis shows that domestic large models have made substantial progress in various applications, including natural language processing, computer vision, and multimodal systems.",
            "Technological innovation remains a critical factor, with Chinese companies developing unique approaches to model architecture, training methodologies, and deployment strategies.",
            "The regulatory landscape continues to evolve, with authorities seeking to balance innovation promotion with necessary oversight and risk management.",
            "International cooperation and competition shape the industry's trajectory, influencing technology transfer, market access, and standard development.",
            "Talent development has become a strategic priority, with educational institutions and companies working to build a skilled workforce in AI and machine learning.",
            "Infrastructure development, particularly computing resources, presents both challenges and opportunities for industry growth and scalability.",
            "Ethical considerations and responsible AI practices are increasingly important, influencing corporate strategies and public perception.",
            "Industry-specific applications demonstrate the practical value of large models across healthcare, finance, education, and manufacturing sectors.",
            "Future prospects depend on sustained innovation, supportive policies, and the ability to address technical and commercial challenges effectively."
        ]
        
        # 选择模板并添加编号
        template = templates[num % len(templates)]
        return f"Detailed Analysis Point {num + 1}: {template} This analysis considers multiple factors including technological maturity, market readiness, regulatory compliance, and competitive positioning. The impact on industry development could be substantial, requiring careful strategic planning and resource allocation."
    
    def _extract_partial_text(self, text, target_tokens):
        """从文本中提取指定token数量的部分"""
        if self.tokenizer:
            try:
                encoding = self.tokenizer.encode(text)
                if len(encoding.ids) <= target_tokens:
                    return text
                # 截断到目标token数
                truncated_encoding = encoding.truncate(target_tokens)
                if truncated_encoding and hasattr(truncated_encoding, 'ids') and truncated_encoding.ids:
                    return self.tokenizer.decode(truncated_encoding.ids)
            except Exception:
                pass
        
        # 使用字符估算
        estimated_chars = int(target_tokens * self.chars_per_token)
        return text[:estimated_chars]
    
    def _adjust_content_length(self, content, target_tokens):
        """调整内容长度以匹配目标tokens"""
        current_tokens = self._count_tokens(content)
        
        if current_tokens <= target_tokens:
            return content
        
        # 计算需要保留的字符数
        estimated_chars = int(target_tokens * self.chars_per_token * 0.95)
        
        # 在句子边界截断
        truncated = content[:estimated_chars]
        sentence_ends = ['. ', '! ', '? ', '。\n', '！\n', '？\n', '\n\n']
        best_pos = 0
        for ending in sentence_ends:
            pos = truncated.rfind(ending)
            if pos > best_pos:
                best_pos = pos + len(ending)
        
        if best_pos > 0:
            return content[:best_pos]
        else:
            return truncated
        
    def _print_summary(self, results):
        """打印测试汇总报告"""
        print("\n" + "=" * 60)
        print(f"📋 测试汇总报告 {self.api_url}")
        print("=" * 60)
        if self.use_stream:
            print("\n并发级别 | 成功率 | 平均完成时间 | 平均TTFT | 平均tokens/s | tokens")
        else:
            print("\n并发级别 | 成功率 | 平均完成时间 | 平均响应时间 | 平均tokens/s | tokens")
        print("-" * 85)

        for concurrency, result in results.items():
            total_req = result.success_count + result.failure_count
            succ_rate = (result.success_count / total_req) * 100 if total_req else 0.0
            avg_time = statistics.mean(result.response_times) if result.response_times else float("nan")
            avg_ttft = statistics.mean(result.first_token_times) if result.first_token_times else float("nan")
            avg_tps = statistics.mean(result.tokens_per_sec) if result.tokens_per_sec else float("nan")
            print(f"{concurrency:8d} | {succ_rate:6.1f}% | {avg_time:10.2f}s | {avg_ttft:8.3f}s | {avg_tps:12.2f} | {self.prompt_tokens:6d}")


class TestResult:
    """测试结果数据类"""
    
    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.response_times = []      # 整体完成时间（秒）
        self.first_token_times = []   # TTFT（秒）
        self.tokens_generated = []    # 每次请求的输出 token 数
        self.tokens_per_sec = []      # 每次请求 tokens/s
        self.status_codes = []
        self.errors = []


def make_request(tester=None):
    """
    发送单个 SSE 流式请求；统计：
      - TTFT（首 token 时间）
      - 完成时间（秒）
      - 输出 token 数（优先取 message_delta.usage.output_tokens；否则可选估算）
      - tokens/s = 输出 token 数 / 完成时间
    
    Args:
        tester: APIPerformanceTester 实例，如果为 None 则使用全局变量
    """
    if tester is None:
        # 不支持无 tester 的调用方式，必须传入 tester
        raise ValueError("make_request 必须传入 tester 参数")
    else:
        # 使用 tester 实例的配置
        api_url = tester.api_url
        api_key = tester.api_key
        model = tester.model
        test_message = tester.test_message
        
        # 如果指定了prompt_tokens且不使用默认消息，则生成指定token数量的内容
        if hasattr(tester, 'prompt_tokens') and tester.prompt_tokens != DEFAULT_PROMPT_TOKENS:
            test_message = tester._generate_prompt_content(tester.prompt_tokens)
            # actual_tokens = tester._count_tokens(test_message)
            # print(f"📝 生成了 {actual_tokens} tokens 的提示内容（目标：{tester.prompt_tokens}）")
        timeout = tester.timeout
        estimate_tokens_by_chars = tester.estimate_tokens_by_chars
        chars_per_token = tester.chars_per_token
        
    start_time = time.time()
    first_token_time = None
    output_tokens = None  # 来自 message_delta 的 usage.output_tokens（累计）
    approx_chars = 0      # 如果需要估算时使用

    # 检测是否为 Cerebras API
    is_cerebras_api = "api.cerebras.ai" in api_url.lower()
    
    # 检测是否为 Gemini API
    is_gemini_api = "generativelanguage.googleapis.com" in api_url.lower()
    
    # 处理 Gemini API URL 中的模型占位符
    if is_gemini_api and "{model}" in api_url:
        api_url = api_url.replace("{model}", model)
    
    # 根据接口类型设置不同的 headers 和 payload
    if is_gemini_api:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": test_message}]
            }],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature": 0.2,
            }
        }
        # Gemini API 总是使用流式
        payload["stream"] = True
    elif tester.use_chat_api or is_cerebras_api:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "你是一个严谨的助手，只返回最终答案。"},
                {"role": "user", "content": test_message}
            ]
        }
        
        # Cerebras API 使用 max_completion_tokens，其他使用 max_tokens
        if is_cerebras_api:
            payload["max_completion_tokens"] = 1024
        else:
            payload["max_tokens"] = 1024
            
        if tester.use_stream:
            payload["stream"] = True
    else:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": test_message}
            ]
        }
        if tester.use_stream:
            payload["stream"] = True

    try:
        with requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload),
            stream=tester.use_stream,  # 根据 use_stream 决定是否使用流式
            timeout=timeout,
        ) as r:
            status = r.status_code
            if status != 200:
                total_time = time.time() - start_time
                text = r.text[:200] if r.text else ""
                return (False, total_time, status, f"HTTP {status}: {text}", None, None, None)

            # 处理非流式响应
            if not tester.use_stream:
                total_time = time.time() - start_time
                response_data = r.json()
                
                # 根据接口类型解析不同的响应格式
                if is_gemini_api:
                    # Gemini API 非流式响应
                    candidates = response_data.get("candidates", [])
                    if candidates:
                        candidate = candidates[0]
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        text = ""
                        for part in parts:
                            if "text" in part:
                                text += part["text"]
                        
                        # Gemini 不提供 usage 信息，需要估算
                        if estimate_tokens_by_chars:
                            output_tokens = max(1, int(len(text) / chars_per_token))
                elif tester.use_chat_api or is_cerebras_api:
                    # Chat API 格式 (包括 Cerebras)
                    usage = response_data.get("usage", {})
                    output_tokens = usage.get("completion_tokens")
                    content = ""
                    choices = response_data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                else:
                    # Anthropic API 格式
                    usage = response_data.get("usage", {})
                    output_tokens = usage.get("output_tokens")
                    content = ""
                    content_blocks = response_data.get("content", [])
                    for block in content_blocks:
                        if block.get("type") == "text":
                            content += block.get("text", "")
                
                # 如果没有获取到 token 数，按需估算
                if output_tokens is None and estimate_tokens_by_chars:
                    output_tokens = max(1, int(len(content) / chars_per_token))
                
                # 计算 tokens/s
                tokens_per_sec = None
                if output_tokens is not None and total_time > 0:
                    tokens_per_sec = output_tokens / total_time
                
                return (True, total_time, status, None, total_time, output_tokens, tokens_per_sec)

            # 处理流式响应
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue

                chunk = raw_line[len("data:"):].strip()
                if not chunk:
                    continue
                
                # 检查是否为流结束标记
                if chunk == "[DONE]":
                    total_time = time.time() - start_time
                    if first_token_time is None:
                        first_token_time = total_time
                    
                    # 若未拿到 tokens，按需估算
                    if output_tokens is None and estimate_tokens_by_chars:
                        output_tokens = max(1, int(approx_chars / chars_per_token))
                    
                    # 计算 tokens/s
                    tokens_per_sec = None
                    if output_tokens is not None and total_time > 0:
                        tokens_per_sec = output_tokens / total_time
                    
                    return (True, total_time, status, None, first_token_time, output_tokens, tokens_per_sec)

                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                # 根据接口类型解析不同的响应格式
                if is_gemini_api:
                    # Gemini API 流式响应
                    candidates = event.get("candidates", [])
                    if candidates:
                        candidate = candidates[0]
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        
                        for part in parts:
                            if "text" in part and part.get("text"):
                                # 记录首 token 时间
                                if first_token_time is None:
                                    first_token_time = time.time() - start_time
                                if estimate_tokens_by_chars:
                                    approx_chars += len(part.get("text", ""))
                        
                        # 检查是否结束
                        finish_reason = candidate.get("finishReason")
                        if finish_reason in ["STOP", "MAX_TOKENS", "SAFETY", "RECITATION"]:
                            total_time = time.time() - start_time
                            if first_token_time is None:
                                first_token_time = total_time
                            
                            # Gemini 不提供 usage 信息，需要估算
                            if output_tokens is None and estimate_tokens_by_chars:
                                output_tokens = max(1, int(approx_chars / chars_per_token))
                            
                            # 计算 tokens/s
                            tokens_per_sec = None
                            if output_tokens is not None and total_time > 0:
                                tokens_per_sec = output_tokens / total_time
                            
                            return (True, total_time, status, None, first_token_time, output_tokens, tokens_per_sec)
                elif tester.use_chat_api or is_cerebras_api:
                    # Chat API 格式 (包括 Cerebras)
                    choices = event.get("choices", [])
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        
                        # 记录首 token 时间（第一个 content 出现）
                        if "content" in delta and delta.get("content"):
                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                            if estimate_tokens_by_chars:
                                approx_chars += len(delta.get("content", ""))
                        
                        # 获取 usage 信息
                        usage = event.get("usage")
                        if usage and "completion_tokens" in usage:
                            output_tokens = usage.get("completion_tokens")
                    
                    # 检查是否结束 - 支持多种结束条件
                    finish_reason = None
                    if choices:
                        finish_reason = choice.get("finish_reason")
                    
                    if finish_reason in ["stop", "length", "content_filter", "function_call"]:
                        total_time = time.time() - start_time
                        if first_token_time is None:
                            first_token_time = total_time
                        
                        # 若未拿到 usage.completion_tokens，按需估算
                        if output_tokens is None and estimate_tokens_by_chars:
                            output_tokens = max(1, int(approx_chars / chars_per_token))
                        
                        # 计算 tokens/s
                        tokens_per_sec = None
                        if output_tokens is not None and total_time > 0:
                            tokens_per_sec = output_tokens / total_time
                        
                        return (True, total_time, status, None, first_token_time, output_tokens, tokens_per_sec)
                else:
                    # Anthropic API 格式
                    etype = event.get("type")

                    # 记录首 token 时间（第一段 text_delta 出现）
                    if etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                            if estimate_tokens_by_chars:
                                approx_chars += len(delta.get("text", ""))

                    # usage 累加通常在 message_delta 事件里
                    if etype == "message_delta":
                        usage = event.get("usage") or {}
                        # 一般是累计值（到当前为止的输出 token 数）
                        if "output_tokens" in usage:
                            output_tokens = usage.get("output_tokens")

                    if etype == "message_stop":
                        total_time = time.time() - start_time
                        if first_token_time is None:
                            first_token_time = total_time  # 极端情况：几乎无输出

                        # 若未拿到 usage.output_tokens，按需估算
                        if output_tokens is None and estimate_tokens_by_chars:
                            output_tokens = max(1, int(approx_chars / chars_per_token))

                        # 计算 tokens/s
                        tokens_per_sec = None
                        if output_tokens is not None and total_time > 0:
                            tokens_per_sec = output_tokens / total_time

                        return (True, total_time, status, None, first_token_time, output_tokens, tokens_per_sec)

            # 未收到结束标志
            total_time = time.time() - start_time
            if is_gemini_api:
                return (False, total_time, status, "Stream ended without finishReason", first_token_time, output_tokens, None)
            elif tester.use_chat_api:
                return (False, total_time, status, "Stream ended without finish_reason=stop", first_token_time, output_tokens, None)
            else:
                return (False, total_time, status, "Stream ended without message_stop", first_token_time, output_tokens, None)

    except Exception as e:
        total_time = time.time() - start_time
        return (False, total_time, None, str(e), first_token_time, output_tokens, None)


def test_concurrency(concurrency_level, tester=None):
    """测试指定并发级别（SSE + TTFT + tokens/s）
    
    Args:
        concurrency_level: 并发级别
        tester: APIPerformanceTester 实例，如果为 None 则使用全局配置
    """
    print(f"\n🔄 测试并发级别: {concurrency_level}")
    print("=" * 50)
    
    # 显示提示词token信息
    if hasattr(tester, 'prompt_tokens') and tester.prompt_tokens != DEFAULT_PROMPT_TOKENS:
        print(f"📝 提示词长度: {tester.prompt_tokens} tokens")
    elif hasattr(tester, 'prompt_tokens'):
        print(f"📝 提示词长度: {tester.prompt_tokens} tokens (默认)")

    result = TestResult()

    test_rounds = tester.test_rounds
    
    for round_num in range(test_rounds):
        print(f"   第 {round_num + 1}/{test_rounds} 轮测试...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
            futures = [executor.submit(make_request, tester) for _ in range(concurrency_level)]

            for future in concurrent.futures.as_completed(futures):
                success, total_time, status_code, error, ttft, out_tokens, tps = future.result()

                if success:
                    result.success_count += 1
                else:
                    result.failure_count += 1
                    if error:
                        result.errors.append(error)

                result.response_times.append(total_time)
                if ttft is not None:
                    result.first_token_times.append(ttft)
                if out_tokens is not None:
                    result.tokens_generated.append(out_tokens)
                if tps is not None:
                    result.tokens_per_sec.append(tps)
                if status_code:
                    result.status_codes.append(status_code)

    # 统计
    total_requests = result.success_count + result.failure_count
    success_rate = (result.success_count / total_requests) * 100 if total_requests else 0.0

    def safe_mean(xs): return statistics.mean(xs) if xs else float("nan")
    def safe_min(xs):  return min(xs) if xs else float("nan")
    def safe_max(xs):  return max(xs) if xs else float("nan")

    def percentile(xs, p):
        if not xs:
            return float("nan")
        xs_sorted = sorted(xs)
        k = (len(xs_sorted) - 1) * p
        f = int(k)
        c = min(f + 1, len(xs_sorted) - 1)
        if f == c:
            return xs_sorted[f]
        return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)

    avg_response_time = safe_mean(result.response_times)
    min_response_time = safe_min(result.response_times)
    max_response_time = safe_max(result.response_times)

    avg_ttft = safe_mean(result.first_token_times)
    p50_ttft = percentile(result.first_token_times, 0.5)
    p95_ttft = percentile(result.first_token_times, 0.95)

    avg_tokens = safe_mean(result.tokens_generated)
    sum_tokens = sum(result.tokens_generated) if result.tokens_generated else 0
    avg_tps = safe_mean(result.tokens_per_sec)
    p50_tps = percentile(result.tokens_per_sec, 0.5)
    p95_tps = percentile(result.tokens_per_sec, 0.95)
    max_tps = safe_max(result.tokens_per_sec)

    # 打印结果
    print(f"📊 测试结果:")
    print(f"   总请求数: {total_requests}")
    print(f"   成功: {result.success_count} | 失败: {result.failure_count}")
    print(f"   成功率: {success_rate:.1f}%")
    print(f"   平均完成时间: {avg_response_time:.2f}s  (最快 {min_response_time:.2f}s | 最慢 {max_response_time:.2f}s)")
    if tester.use_stream:
        print(f"   TTFT(首字响应): 平均 {avg_ttft:.3f}s | P50 {p50_ttft:.3f}s | P95 {p95_ttft:.3f}s")
    else:
        print(f"   响应时间(TTFB): 平均 {avg_ttft:.3f}s | P50 {p50_ttft:.3f}s | P95 {p95_ttft:.3f}s")
    
    if result.tokens_per_sec:
        print(f"   输出Token: 总计 {sum_tokens} | 单次平均 {avg_tokens:.1f}")
        print(f"   输出速率(tokens/s): 平均 {avg_tps:.2f} | P50 {p50_tps:.2f} | P95 {p95_tps:.2f} | 最高 {max_tps:.2f}")
    else:
        if tester.use_gemini_api:
            print("   ⚠️ Gemini API 不提供 usage 信息；如需估算 tokens，请使用 --estimate-tokens 参数。")
        elif tester.use_chat_api:
            print("   ⚠️ 未获取到 usage.completion_tokens；如需估算，请使用 --estimate-tokens 参数。")
        else:
            print("   ⚠️ 未获取到 usage.output_tokens；如需估算，请使用 --estimate-tokens 参数。")

    if result.errors:
        print_limit = tester.print_sample_errors
        print(f"\n❌ 错误汇总 (前{print_limit}个):")
        for i, error in enumerate(result.errors[:print_limit], 1):
            print(f"   {i}. {error}")

    return result


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="API 性能测试工具 - 测试 LLM API 的并发性能（支持 OpenAI、Anthropic 和 Gemini 格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python api_performance_tester.py --key your_api_key_here
  python api_performance_tester.py --key your_api_key_here --model glm-4-0528
  python api_performance_tester.py --key your_api_key_here --min 5 --max 50 --step 5
  python api_performance_tester.py --key your_api_key_here --rounds 3 --timeout 60
  python api_performance_tester.py --key your_api_key_here --chat-api  # 使用 Chat API 接口
  python api_performance_tester.py --key your_api_key_here --no-stream  # 禁用流式请求
  python api_performance_tester.py --key your_api_key_here --gemini-api  # 使用 Gemini API 接口
  python api_performance_tester.py --key your_api_key_here --prompt-tokens 1000  # 使用1000 tokens的提示词
        """
    )
    
    # API 配置参数
    parser.add_argument(
        "--key", 
        required=True,
        help="API 密钥（必需）"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_API_URL,
        help="API 接口地址（默认：%(default)s）"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="使用的模型（默认：%(default)s）"
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_TEST_MESSAGE,
        help="测试消息内容"
    )
    
    # 测试参数
    parser.add_argument(
        "--min",
        type=int,
        default=DEFAULT_MIN_CONCURRENCY,
        help="最小并发级别（默认：%(default)d）"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="最大并发级别（默认：%(default)d）"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help="并发级别步长（默认：%(default)d）"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_TEST_ROUNDS,
        help="每个并发级别测试轮数（默认：%(default)d）"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="单请求超时时间（秒，默认：%(default)d）"
    )
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        help="使用字符数估算 tokens（默认：不启用）"
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help="每个 token 的字符数（默认：%(default).1f）"
    )
    parser.add_argument(
        "--chat-api",
        action="store_true",
        help="使用 Chat API 接口（默认：使用 Anthropic 接口）"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式请求（默认：启用流式）"
    )
    parser.add_argument(
        "--gemini-api",
        action="store_true",
        help="使用 Gemini API 接口（默认：不使用）"
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_PROMPT_TOKENS,
        help="提示词的 token 数量（默认：%(default)d）"
    )
    
    return parser.parse_args()


def main():
    """主函数 - 运行性能测试"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建测试器实例
    # 如果使用 Gemini API 且未指定 URL，则使用默认的 Gemini API URL
    api_url = args.url
    if args.gemini_api and api_url == DEFAULT_API_URL:
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    elif args.chat_api and api_url == DEFAULT_API_URL:
        api_url = DEFAULT_CHAT_API_URL
    
    tester = APIPerformanceTester(
        api_url=api_url,
        api_key=args.key,
        model=args.model,
        test_message=args.message,
        min_concurrency=args.min,
        max_concurrency=args.max,
        step=args.step,
        test_rounds=args.rounds,
        timeout=args.timeout,
        estimate_tokens_by_chars=args.estimate_tokens,
        chars_per_token=args.chars_per_token,
        use_chat_api=args.chat_api,
        use_stream=not args.no_stream,
        use_gemini_api=args.gemini_api,
        prompt_tokens=args.prompt_tokens
    )
    
    # 运行测试
    results = tester.run_test()
    
    return results


if __name__ == "__main__":
    main()
