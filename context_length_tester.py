"""
上下文窗口长度测试工具

用于测试 LLM 模型的最大上下文窗口长度，支持多种预设大小（32k, 64k, 128k）和自定义长度。
通过递增测试法确定模型实际支持的最大上下文大小。

作者: Claude
版本: 1.0.0
"""

import argparse
import requests
import time
from datetime import datetime
import statistics
import json
import random
import os
from typing import List, Dict, Optional, Tuple, Any
import dataclasses

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("⚠️ 警告：tokenizers 未安装，将使用字符数估算。建议运行: pip install tokenizers")


@dataclasses.dataclass
class ContextTestResult:
    """上下文测试结果"""
    success: bool
    success_count: int
    failure_count: int
    avg_response_time: float
    errors: List[str]
    actual_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    is_compressed: bool = False
    compression_ratio: Optional[float] = None


# 默认配置值
DEFAULT_API_URL = "https://open.bigmodel.cn/api/anthropic/v1/messages"
DEFAULT_CHAT_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
DEFAULT_MODEL = "glm-4.5"
DEFAULT_TEST_ROUNDS = 1
DEFAULT_TIMEOUT = 300
DEFAULT_CHARS_PER_TOKEN_EN = 4.0  # 英文默认4字符/token
DEFAULT_CHARS_PER_TOKEN_CN = 2.0  # 中文默认2字符/token
DEFAULT_MAX_PARAGRAPHS = 100000

# 模型到分词器文件的映射
MODEL_TOKENIZER_MAP = {
    "glm": "tokenizer_glm.json",          # GLM系列模型
    "glm-4": "tokenizer_glm.json",        # GLM-4系列
    "glm-4.5": "tokenizer_glm.json",      # GLM-4.5系列  
    "deepseek": "tokenizer_ds.json",      # DeepSeek系列
    "deepseek-chat": "tokenizer_ds.json", # DeepSeek Chat
    "deepseek-coder": "tokenizer_ds.json",# DeepSeek Coder
    "gemini": "tokenizer_glm.json",       # Gemini系列模型（暂用GLM分词器）
    "gemini-pro": "tokenizer_glm.json",   # Gemini Pro
    "gemini-1.5-pro": "tokenizer_glm.json", # Gemini 1.5 Pro
    "gemini-1.5-flash": "tokenizer_glm.json" # Gemini 1.5 Flash
}

# 预设的上下文大小（tokens）
PRESET_SIZES = {
    "1k": 1024*1,
    "2k": 1024*2,
    "4k": 1024*4,
    "8k": 1024*8,
    "16k": 1024*16,
    "32k": 1024*32,
    "64k": 1024*64,
    "128k": 1024*128,
    "256k": 1024*256,
    "512k": 1024*512
}


class ContextLengthTester:
    """上下文窗口长度测试工具"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, 
                 model: Optional[str] = None, test_sizes: Optional[List[str]] = None,
                 test_rounds: Optional[int] = None, timeout: Optional[int] = None, 
                 use_chat_api: Optional[bool] = None, chars_per_token: Optional[float] = None, 
                 output_file: Optional[str] = None, max_paragraphs: Optional[int] = None,
                 use_english: Optional[bool] = None, 
                 disable_thinking: Optional[bool] = None,
                 show_detail: Optional[bool] = None,
                 query_num: Optional[int] = None,
                 use_gemini_api: Optional[bool] = None):
        """初始化测试配置
        
        Args:
            api_url: API 地址
            api_key: API 密钥
            model: 使用的模型
            test_sizes: 测试大小列表（如 ["32k", "64k", "192k"]）
            test_rounds: 测试轮数
            timeout: 超时时间
            use_chat_api: 是否使用 Chat API 接口
            chars_per_token: 字符/token 比率
            output_file: 输出文件路径
            max_paragraphs: 最大段落数量限制
            use_english: 是否使用英文生成prompt（默认使用中文）
            disable_thinking: 是否禁用 GLM 模型的思考模式（默认自动检测）
            show_detail: 是否显示详细的响应内容和payload（默认不显示）
            query_num: 插入的随机数数量（默认为1）
            use_gemini_api: 是否使用 Gemini API 接口
        """
        # API 配置
        self.use_chat_api = use_chat_api or False
        self.use_gemini_api = use_gemini_api or False
        if self.use_gemini_api:
            self.api_url = DEFAULT_GEMINI_API_URL.format(model=model or DEFAULT_MODEL)
        elif self.use_chat_api and api_url is None:
            self.api_url = DEFAULT_CHAT_API_URL
        else:
            self.api_url = api_url or DEFAULT_API_URL
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        
        # 测试参数
        self.test_sizes = test_sizes or []
        self.test_rounds = test_rounds or DEFAULT_TEST_ROUNDS
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.use_english = use_english or False
        self.disable_thinking = disable_thinking
        self.show_detail = show_detail or False
        
        # 根据语言选择合适的字符/token比率
        if chars_per_token is not None:
            self.chars_per_token = chars_per_token
        else:
            self.chars_per_token = DEFAULT_CHARS_PER_TOKEN_EN if self.use_english else DEFAULT_CHARS_PER_TOKEN_CN
            
        self.max_paragraphs = max_paragraphs or DEFAULT_MAX_PARAGRAPHS
        self.query_num = query_num or 1
        
        # 分词器配置 - 默认尝试使用本地分词器
        self.use_tokenizer = TOKENIZERS_AVAILABLE
        self.tokenizer = None
        if self.use_tokenizer:
            try:
                # 根据模型选择合适的分词器文件
                tokenizer_file = self._get_tokenizer_file()
                if tokenizer_file and os.path.exists(tokenizer_file):
                    self.tokenizer = Tokenizer.from_file(tokenizer_file)
                    print(f"✅ 使用本地分词器: {tokenizer_file}")
                else:
                    print(f"⚠️ 分词器文件不存在，回退到字符估算: {tokenizer_file}")
                    self.use_tokenizer = False
            except Exception as e:
                print(f"⚠️ 分词器初始化失败，回退到字符估算: {e}")
                self.use_tokenizer = False
        else:
            print(f"📝 使用字符估算模式（{self.chars_per_token} 字符/token）")
        
        # 生成测试大小列表
        self.test_tokens_list: List[int] = self._generate_test_sizes()
        
        # 输出文件配置
        self.output_file = output_file
        
    def _generate_test_sizes(self) -> List[int]:
        """生成要测试的 tokens 大小列表"""
        sizes = []
        
        # 处理所有指定的大小
        for size_name in self.test_sizes:
            # 首先检查是否是预设大小
            if size_name in PRESET_SIZES:
                sizes.append(PRESET_SIZES[size_name])
            else:
                # 尝试解析为自定义大小
                try:
                    custom_size = parse_custom_size(size_name)
                    sizes.append(custom_size)
                except ValueError as e:
                    print(f"❌ 警告：忽略无效的大小 '{size_name}': {e}")
        
        # 如果没有指定任何大小，使用默认的渐进式测试
        if not sizes:
            # 从小到大，逐步增加直到找到最大值
            sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
        
        return sorted(sizes)
    
    def _get_size_name(self, target_tokens: int) -> str:
        """根据token数量获取用户输入的大小名称"""
        # 首先检查是否是预设大小
        for size_name, tokens in PRESET_SIZES.items():
            if tokens == target_tokens:
                return size_name
        
        # 检查是否是用户输入的自定义大小
        for size_str in self.test_sizes:
            try:
                custom_tokens = parse_custom_size(size_str)
                if custom_tokens == target_tokens:
                    return size_str
            except ValueError:
                continue
        
        # 如果都没有找到，返回最接近的k表示
        if target_tokens >= 1024:
            k_value = target_tokens / 1024
            if k_value.is_integer():
                return f"{int(k_value)}k"
            else:
                return f"{target_tokens}"
        else:
            return f"{target_tokens}"
    
    def _get_tokenizer_file(self) -> Optional[str]:
        """根据模型名称获取对应的分词器文件路径"""
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 检查模型名称（转为小写进行匹配）
        model_lower = self.model.lower()
        
        # 遍历映射表，找到匹配的分词器
        for model_key, tokenizer_filename in MODEL_TOKENIZER_MAP.items():
            if model_key in model_lower:
                tokenizer_path = os.path.join(current_dir, tokenizer_filename)
                return tokenizer_path
        
        # 默认使用GLM分词器（如果没有找到匹配的）
        default_tokenizer = os.path.join(current_dir, "tokenizer_glm.json")
        print(f"⚠️ 模型 {self.model} 未找到对应分词器，使用默认GLM分词器")
        return default_tokenizer
    
    def _count_tokens_all_tokenizers(self, text: str) -> Dict[str, int]:
        """使用所有可用分词器统计token数量"""
        results = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 测试所有分词器
        for tokenizer_name, tokenizer_file in [("GLM", "tokenizer_glm.json"), ("DeepSeek", "tokenizer_ds.json")]:
            tokenizer_path = os.path.join(current_dir, tokenizer_file)
            
            if os.path.exists(tokenizer_path) and TOKENIZERS_AVAILABLE:
                try:
                    tokenizer = Tokenizer.from_file(tokenizer_path)
                    encoding = tokenizer.encode(text)
                    results[tokenizer_name] = len(encoding.ids)
                except Exception:
                    results[tokenizer_name] = None
            else:
                results[tokenizer_name] = None
        
        # 添加字符估算结果
        results["字符估算(EN)"] = int(len(text) / DEFAULT_CHARS_PER_TOKEN_EN)
        results["字符估算(CN)"] = int(len(text) / DEFAULT_CHARS_PER_TOKEN_CN)
        
        return results
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        if self.use_tokenizer and self.tokenizer:
            try:
                encoding = self.tokenizer.encode(text)
                return len(encoding.ids)
            except Exception as e:
                print(f"⚠️ 分词器编码失败，回退到字符估算: {e}")
                # 回退到字符估算
                return int(len(text) / self.chars_per_token)
        else:
            # 回退到字符估算
            return int(len(text) / self.chars_per_token)
    
    def _safe_truncate_text(self, text: str, target_tokens: int) -> Optional[str]:
        """安全地截断文本到目标token数"""
        if not self.use_tokenizer or not self.tokenizer:
            return None
            
        try:
            encoding = self.tokenizer.encode(text)
            if len(encoding.ids) <= target_tokens:
                return text
                
            # 截断到目标token数
            truncated_encoding = encoding.truncate(target_tokens)
            if truncated_encoding and hasattr(truncated_encoding, 'ids') and truncated_encoding.ids:
                return self.tokenizer.decode(truncated_encoding.ids)
            return None
        except Exception:
            return None
    
    def _fine_tune_content_length(self, content: str, target_tokens: int) -> str:
        """精细调整内容长度以精确匹配目标tokens"""
        if not self.use_tokenizer or not self.tokenizer:
            return content
        
        # 找到内容部分的开始和结束
        start_marker_end = content.find("\n\n") + 2
        if start_marker_end < 10:
            start_marker_end = content.find("[START]") + 50
        
        end_marker_start = content.rfind("[END]")
        if end_marker_start == -1:
            return content
        
        # 保护开始和结束标记
        protected_start = content[:start_marker_end]
        protected_end = content[end_marker_start:]
        content_part = content[start_marker_end:end_marker_start]
        
        try:
            # 使用二分查找找到最佳长度
            content_encoding = self.tokenizer.encode(content_part)
            target_content_tokens = target_tokens - self._count_tokens(protected_start + protected_end)
            
            if target_content_tokens <= 0:
                return content
            
            # 二分查找
            left, right = 0, len(content_encoding.ids)
            best_encoding = content_encoding
            
            while left < right:
                mid = (left + right + 1) // 2
                test_encoding = content_encoding.truncate(mid)
                if test_encoding and hasattr(test_encoding, 'ids') and len(test_encoding.ids) <= target_content_tokens:
                    best_encoding = test_encoding
                    left = mid
                else:
                    right = mid - 1
            
            # 重建内容
            fine_tuned_content = protected_start + self.tokenizer.decode(best_encoding.ids) + protected_end
            return fine_tuned_content
        except Exception:
            return content
    
    def _generate_test_content(self, target_tokens: int) -> Tuple[str, List[int]]:
        """生成指定 token 数量的测试内容
        
        使用多样化的文本模式，避免重复内容触发过滤
        在prompt中随机分布指定数量的随机数，测试模型是否真正处理完整上下文
        
        Returns:
            Tuple[str, List[int]]: (测试内容, 随机数列表)
        """
        try:
            # 生成指定数量的随机数
            random_numbers = [random.randint(100, 999) for _ in range(self.query_num)]
            
            # 获取标记（包含随机数提示）
            start_marker = f"[START] Context length test begins. The target token count is: {target_tokens}.\n\n"
            end_marker = f"\n\n[END] Context length test ends. Please find all random numbers hidden in the text and list them. Tell me how many random numbers you found and what each number is."
            
            # 计算标记占用的 tokens
            marker_tokens = self._count_tokens(start_marker + end_marker)
            
            # 计算可用于基础文本的 tokens
            available_tokens = target_tokens - marker_tokens
            
            # 使用变化的内容而不是重复相同内容
            content_parts = []
            current_tokens = 0
            paragraph_num = 0
            
            # 计算每个内容部分的平均大小
            random_number_text_tokens = self._count_tokens("\n[random num: 123]\n\n")
            # 更准确地计算可用内容tokens
            if self.query_num > 0:
                total_content_tokens = available_tokens - (self.query_num * random_number_text_tokens)
            else:
                total_content_tokens = available_tokens
            
            # 更精确的内容生成策略
            if self.query_num == 1:
                # 单个随机数的情况：精确计算每部分大小
                # 计算标记的实际token数
                actual_start_tokens = self._count_tokens(start_marker)
                actual_end_tokens = self._count_tokens(end_marker)
                actual_random_tokens = self._count_tokens("\n[random num: 123]\n\n")
                
                # 重新计算可用tokens
                total_available = target_tokens - actual_start_tokens - actual_end_tokens - actual_random_tokens
                
                # 第一部分占60%，第二部分占40%
                first_part_target = total_available * 0.6
                second_part_target = total_available * 0.4
                
                # 生成第一部分
                part_tokens = 0
                while part_tokens < first_part_target * 0.99 and paragraph_num < self.max_paragraphs:
                    # 计算还需要多少tokens
                    needed = first_part_target - part_tokens
                    if needed < 50:
                        break
                        
                    paragraph_text = self._get_alternative_base_text(paragraph_num)
                    paragraph_tokens = self._count_tokens(paragraph_text)
                    
                    if paragraph_tokens <= needed * 1.1:  # 允许稍微超出，后面会调整
                        content_parts.append(paragraph_text)
                        part_tokens += paragraph_tokens
                        current_tokens += paragraph_tokens
                        paragraph_num += 1
                    else:
                        # 如果段落太大，尝试分割
                        if self.use_tokenizer and self.tokenizer and needed > 100:
                            # 使用安全的方法分割文本
                            partial_text = self._safe_truncate_text(paragraph_text, int(needed))
                            if partial_text:
                                partial_tokens = self._count_tokens(partial_text)
                                if partial_tokens > 0:
                                    content_parts.append(partial_text)
                                    part_tokens += partial_tokens
                                    current_tokens += partial_tokens
                        break
                
                # 插入随机数
                hidden_number_text = f"\n[random num: {random_numbers[0]}]\n\n"
                content_parts.append(hidden_number_text)
                current_tokens += self._count_tokens(hidden_number_text)
                
                # 第二部分内容
                part_tokens = 0
                while part_tokens < second_part_target * 0.99 and paragraph_num < self.max_paragraphs:
                    needed = second_part_target - part_tokens
                    if needed < 50:
                        break
                        
                    paragraph_text = self._get_alternative_base_text(paragraph_num)
                    paragraph_tokens = self._count_tokens(paragraph_text)
                    
                    if paragraph_tokens <= needed * 1.1:
                        content_parts.append(paragraph_text)
                        part_tokens += paragraph_tokens
                        current_tokens += paragraph_tokens
                        paragraph_num += 1
                    else:
                        # 如果段落太大，尝试分割
                        if self.use_tokenizer and self.tokenizer and needed > 100:
                            # 使用安全的方法分割文本
                            partial_text = self._safe_truncate_text(paragraph_text, int(needed))
                            if partial_text:
                                partial_tokens = self._count_tokens(partial_text)
                                if partial_tokens > 0:
                                    content_parts.append(partial_text)
                                    part_tokens += partial_tokens
                                    current_tokens += partial_tokens
                        break
            else:
                # 多个随机数的情况：精确计算每部分
                # 计算实际的标记token数
                actual_start_tokens = self._count_tokens(start_marker)
                actual_end_tokens = self._count_tokens(end_marker)
                actual_random_tokens = self._count_tokens("\n[random num: 123]\n\n")
                
                # 重新计算可用tokens
                total_available = target_tokens - actual_start_tokens - actual_end_tokens - (self.query_num * actual_random_tokens)
                content_sections = self.query_num + 1  # 比随机数多一个部分
                tokens_per_section = total_available / content_sections
                
                for i in range(self.query_num):
                    # 当前部分的内容（除了最后一个随机数后不需要内容）
                    if i < self.query_num:
                        section_target = tokens_per_section
                        section_tokens = 0
                        
                        while section_tokens < section_target * 0.99 and paragraph_num < self.max_paragraphs:
                            needed = section_target - section_tokens
                            if needed < 30:  # 对于多个随机数，可以接受更小的剩余
                                break
                                
                            paragraph_text = self._get_alternative_base_text(paragraph_num)
                            paragraph_tokens = self._count_tokens(paragraph_text)
                            
                            if paragraph_tokens <= needed * 1.1:
                                content_parts.append(paragraph_text)
                                section_tokens += paragraph_tokens
                                current_tokens += paragraph_tokens
                                paragraph_num += 1
                            else:
                                # 如果段落太大，尝试分割
                                if self.use_tokenizer and self.tokenizer and needed > 50:
                                    # 使用安全的方法分割文本
                                    partial_text = self._safe_truncate_text(paragraph_text, int(needed))
                                    if partial_text:
                                        partial_tokens = self._count_tokens(partial_text)
                                        if partial_tokens > 0:
                                            content_parts.append(partial_text)
                                            section_tokens += partial_tokens
                                            current_tokens += partial_tokens
                                break
                    
                    # 插入随机数
                    hidden_number_text = f"\n[random num: {random_numbers[i]}]\n\n"
                    content_parts.append(hidden_number_text)
                    current_tokens += self._count_tokens(hidden_number_text)
            
            # 组合完整内容
            full_content = start_marker + "".join(content_parts) + end_marker
            
            # 调试信息
            if self.show_detail:
                actual_content_tokens = current_tokens - (self.query_num * self._count_tokens("\n[random num: 123]\n\n"))
                print(f"   [DEBUG] 生成内容tokens: {actual_content_tokens:,}, 总计tokens: {current_tokens:,}")
            
            # 如果使用的tokens太少，添加更多内容
            if current_tokens < available_tokens * 0.7:
                # 计算还需要多少tokens
                remaining_tokens = available_tokens - current_tokens
                # 在最后一个随机数前添加内容
                if content_parts:
                    # 找到最后一个随机数的位置
                    for i in range(len(content_parts) - 1, -1, -1):
                        if "[random num:" in content_parts[i]:
                            # 在这个位置之前插入更多内容
                            additional_content = []
                            add_tokens = 0
                            while add_tokens < remaining_tokens * 0.8 and paragraph_num < self.max_paragraphs:
                                paragraph_text = self._get_alternative_base_text(paragraph_num)
                                paragraph_tokens = self._count_tokens(paragraph_text)
                                
                                if add_tokens + paragraph_tokens <= remaining_tokens:
                                    additional_content.append(paragraph_text)
                                    add_tokens += paragraph_tokens
                                    paragraph_num += 1
                                else:
                                    break
                            
                            # 插入额外内容
                            if additional_content:
                                content_parts[i:i] = additional_content
                                current_tokens += add_tokens
                                if self.show_detail:
                                    print(f"   [DEBUG] 添加额外内容tokens: {add_tokens:,}")
                            break
            
            # 调整内容长度以匹配目标 tokens
            adjusted_content = self._adjust_content_length(full_content, target_tokens)
            
            # 最终检查和微调
            final_tokens = self._count_tokens(adjusted_content)
            error_rate = abs(final_tokens - target_tokens) / target_tokens * 100
            
            if self.show_detail:
                print(f"   [DEBUG] 调整前tokens: {current_tokens:,}, 调整后tokens: {final_tokens:,}, 目标: {target_tokens:,}")
                print(f"   [DEBUG] 误差率: {error_rate:.2f}%")
            
            # 如果误差仍然太大（>5%），尝试进一步调整
            if error_rate > 5 and self.use_tokenizer and self.tokenizer:
                if self.show_detail:
                    print(f"   [DEBUG] 误差率过大，进行精细调整...")
                
                # 计算需要添加或删除的tokens
                token_diff = target_tokens - final_tokens
                
                if token_diff > 0:
                    # 需要添加更多内容
                    additional_content = []
                    additional_tokens = 0
                    
                    while additional_tokens < token_diff and paragraph_num < self.max_paragraphs:
                        # 生成一个小段落
                        paragraph_text = self._get_alternative_base_text(paragraph_num)
                        paragraph_tokens = self._count_tokens(paragraph_text)
                        
                        if additional_tokens + paragraph_tokens <= token_diff:
                            additional_content.append(paragraph_text)
                            additional_tokens += paragraph_tokens
                            paragraph_num += 1
                        else:
                            # 使用分词器精确分割
                            remaining = token_diff - additional_tokens
                            if remaining > 10:
                                partial_text = self._safe_truncate_text(paragraph_text, int(remaining))
                                if partial_text:
                                    partial_tokens = self._count_tokens(partial_text)
                                    if partial_tokens > 0:
                                        additional_content.append(partial_text)
                                        additional_tokens += partial_tokens
                            break
                    
                    # 在结束标记前插入额外内容
                    if additional_content:
                        end_pos = adjusted_content.rfind("[END]")
                        if end_pos != -1:
                            adjusted_content = adjusted_content[:end_pos] + "\n\n" + "\n\n".join(additional_content) + adjusted_content[end_pos:]
                            final_tokens = self._count_tokens(adjusted_content)
                            if self.show_detail:
                                print(f"   [DEBUG] 添加额外内容后: {final_tokens:,} tokens")
                else:
                    # 需要删除内容 - 使用更精确的二分查找
                    adjusted_content = self._fine_tune_content_length(adjusted_content, target_tokens)
                    final_tokens = self._count_tokens(adjusted_content)
                    if self.show_detail:
                        print(f"   [DEBUG] 精细调整后: {final_tokens:,} tokens")
            
            return adjusted_content, random_numbers
            
        except Exception as e:
            raise ContentGenerationError(f"生成测试内容失败: {str(e)}")
    
    def _replace_random_number_in_content(self, content: str, old_randoms: List[int], new_randoms: List[int]) -> str:
        """替换测试内容中的随机数，保持其他内容不变"""
        for i in range(len(old_randoms)):
            old_pattern = f"[random num: {old_randoms[i]}]"
            new_pattern = f"[random num: {new_randoms[i]}]"
            content = content.replace(old_pattern, new_pattern)
        
        return content
      
    def run_test(self) -> Optional[Dict[str, Any]]:
        """运行完整的上下文长度测试"""
        # 检查 API Key
        if not self.api_key:
            print("❌ 错误：请先设置 API_KEY")
            print("提示：创建测试器时传入 api_key 参数")
            return None
            
        print("🚀 开始上下文窗口长度测试")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API 地址: {self.api_url}")
        print(f"模型: {self.model}")
        print(f"测试轮数: {self.test_rounds}")
        print(f"单请求超时: {self.timeout}秒")
        print(f"测试大小: {self.test_tokens_list}")
        
        results = {}
        max_successful_tokens = 0
        
        for target_tokens in self.test_tokens_list:
            # 生成测试内容
            test_content, random_numbers = self._generate_test_content(target_tokens)
            actual_tokens = self._count_tokens(test_content)
            
            # 保存测试内容到log目录
            # 获取用户输入的大小名称
            size_name = self._get_size_name(target_tokens)
            self._save_test_content(size_name, test_content, random_numbers)
            
            print(f"\n🔄 测试上下文大小: {actual_tokens:,} tokens (目标: {target_tokens:,})")
            print("=" * 60)
            
            # 显示所有分词器的统计结果
            all_tokenizer_results = self._count_tokens_all_tokenizers(test_content)
            print(f"   生成内容长度统计:")
            
            # 显示当前使用的分词器结果（加粗显示）
            current_tokenizer_name = "GLM" if "glm" in self.model.lower() else "DeepSeek" if "deepseek" in self.model.lower() else "GLM"
            for tokenizer_name, token_count in all_tokenizer_results.items():
                if token_count is not None:
                    if tokenizer_name == current_tokenizer_name and self.use_tokenizer:
                        print(f"   ├─ {tokenizer_name}分词器: {token_count:,} tokens ✅ (当前使用)")
                    else:
                        print(f"   ├─ {tokenizer_name}分词器: {token_count:,} tokens")
                else:
                    print(f"   ├─ {tokenizer_name}分词器: 不可用")
            
            print(f"   ├─ 目标token数: {target_tokens:,}")
            print(f"   └─ 误差: {abs(actual_tokens - target_tokens):,} tokens ({abs(actual_tokens - target_tokens)/target_tokens*100:.1f}%)")
            print(f"   随机数: {random_numbers} (共{self.query_num}个)")
            
            # 显示分词器信息
            if self.use_tokenizer:
                tokenizer_file = os.path.basename(self._get_tokenizer_file()) if self._get_tokenizer_file() else "未知"
                tokenizer_info = f"本地分词器({tokenizer_file})"
            else:
                tokenizer_info = f"字符估算({self.chars_per_token}字符/token)"
            print(f"   分词器: {tokenizer_info}")
            
            # 显示测试内容的首尾部分（用于调试）
            if target_tokens >= 32000:  # 只在较大的测试时显示
                start_preview = test_content[:200]
                end_preview = test_content[-200:]
                print(f"   内容预览 (开始): {repr(start_preview)}")
                print(f"   内容预览 (结束): {repr(end_preview)}")
                # import ipdb; ipdb.set_trace()  # 调试断点
            
            # 运行测试
            result = self._test_single_size(target_tokens, test_content, random_numbers, actual_tokens)
            results[target_tokens] = result
            
            # 记录成功的最大 tokens
            if result.success:
                max_successful_tokens = target_tokens
                print(f"   ✅ 测试成功！")
            else:
                print(f"   ❌ 测试失败: {result.errors[0] if result.errors else '未知错误'}")
                print(f"\n⚠️  在 {target_tokens:,} tokens 处测试失败")
                break
            
            # 成功后短暂休息
            time.sleep(2)
        
        # 打印汇总报告
        self._print_summary(results, max_successful_tokens)
        
        # 导出结果到 JSON 文件
        if self.output_file:
            self._export_results(results, max_successful_tokens)
        
        return {
            "results": results,
            "max_successful_tokens": max_successful_tokens
        }
    
    def _test_single_size(self, target_tokens: int, test_content: str, random_numbers: List[int], actual_tokens: int) -> ContextTestResult:
        """测试单个上下文大小"""
        success_count = 0
        failure_count = 0
        errors = []
        response_times = []
        input_tokens_list = []
        
        for round_num in range(self.test_rounds):
            # 为每轮测试生成新的随机数，但保持内容结构不变
            round_random_numbers = [random.randint(100, 999) for _ in range(self.query_num)]
            
            # 替换测试内容中的随机数
            round_test_content = self._replace_random_number_in_content(test_content, random_numbers, round_random_numbers)
            
            # 显示正确答案（使用新的随机数）
            correct_answers = round_random_numbers
            print(f"   第 {round_num + 1}/{self.test_rounds} 轮测试... 随机数: {correct_answers}")
            
            # 保存每轮的测试内容
            if self.test_rounds > 1:  # 只在多轮测试时保存
                # 获取用户输入的大小名称
                size_name = self._get_size_name(target_tokens)
                self._save_round_content(size_name, round_num, round_test_content, round_random_numbers)
            
            success, response_time, error, model_answer, detail_data, input_tokens = self._make_single_request(round_test_content)
            
            # 记录input_tokens
            if input_tokens:
                input_tokens_list.append(input_tokens)
            
            # 显示模型回答
            if model_answer:
                # 限制模型回答的显示长度
                if len(model_answer) > 200:
                    print(f"      模型回答: {model_answer[:200]}...")
                else:
                    print(f"      模型回答: {model_answer}")
                
                # 验证回答是否正确（检查随机数）
                is_correct = False
                try:
                    # 从回答中提取所有数字
                    import re
                    numbers_in_response = re.findall(r'random num:\s*(\d+)', model_answer.lower())
                    if not numbers_in_response:
                        # 尝试其他格式 - 查找所有3位数字
                        numbers_in_response = re.findall(r'\b\d{3}\b', model_answer)
                    
                    # 转换为整数
                    model_numbers = [int(n) for n in numbers_in_response]
                    
                    # 不去重，保留所有找到的数字（包括重复的）
                    
                    # 检查是否找到了所有随机数（考虑重复）
                    # 创建正确答案的副本用于匹配
                    remaining_answers = correct_answers.copy()
                    found_count = 0
                    found_numbers = []
                    
                    # 按顺序匹配模型回答中的数字
                    for num in model_numbers:
                        if num in remaining_answers:
                            found_count += 1
                            found_numbers.append(num)
                            remaining_answers.remove(num)
                    
                    missing_numbers = remaining_answers
                    extra_numbers = [n for n in model_numbers if n not in correct_answers]
                    
                    # 计算实际需要的匹配数（考虑重复）
                    required_count = len(correct_answers)
                    
                    if found_count >= required_count:
                        is_correct = True
                        print(f"      回答✅正确 (找到{found_count}/{required_count}个随机数)")
                        if extra_numbers:
                            print(f"      额外找到的数字: {extra_numbers}")
                    else:
                        is_correct = False
                        print(f"      回答❌错误 (找到{found_count}/{required_count}个随机数)")
                        if self.show_detail:  # 只在detail模式下显示详细信息
                            if found_numbers:
                                print(f"      正确找到的数字: {found_numbers}")
                            if missing_numbers:
                                print(f"      遗漏的数字: {missing_numbers}")
                            if extra_numbers:
                                print(f"      额外的数字: {extra_numbers}")
                            print(f"      正确的数字: {correct_answers}")
                            print(f"      模型找到的数字: {model_numbers}")
                except Exception as e:
                    is_correct = False
                    print(f"      回答❌错误 (解析失败: {str(e)})")
                
                # 显示详细信息（如果启用）
                if self.show_detail and detail_data:
                    print(f"      === 详细信息 ===")
                    print(f"      响应内容: {json.dumps(detail_data, ensure_ascii=False, indent=2)}")
                    print(f"      ===============")
                
                # 更新成功状态：需要API请求成功且回答正确
                if success and is_correct:
                    success_count += 1
                else:
                    failure_count += 1
                    if not is_correct:
                        errors.append(f"模型回答错误：只找到{found_count}/{self.query_num}个随机数")
                    else:
                        errors.append(error if error else "请求失败")
            else:
                print(f"      模型回答: [空] ❌错误")
                if success:
                    failure_count += 1
                    errors.append("模型回答为空")
                else:
                    failure_count += 1
                    errors.append(error if error else "请求失败")
            
            response_times.append(response_time)
        
        # 计算压缩检测
        avg_input_tokens = None
        is_compressed = False
        compression_ratio = None
        
        if input_tokens_list and actual_tokens:
            avg_input_tokens = int(sum(input_tokens_list) / len(input_tokens_list))
            # 检查是否压缩（误差超过15%）
            compression_diff = abs(avg_input_tokens - actual_tokens)
            compression_ratio = compression_diff / actual_tokens * 100
            is_compressed = compression_ratio > 15
            
            if self.show_detail or is_compressed:
                print(f"   压缩检测: 发送tokens={actual_tokens:,}, 接收tokens={avg_input_tokens:,}, 差异={compression_diff:,} ({compression_ratio:.1f}%)")
                if is_compressed:
                    print(f"   ⚠️  检测到压缩！差异超过15%阈值")
        
        # 计算成功率
        success_rate = success_count / self.test_rounds if self.test_rounds > 0 else 0
        
        # 只有当成功率超过50%时才认为测试成功
        test_success = success_rate >= 0.5
        
        return ContextTestResult(
            success=test_success,
            success_count=success_count,
            failure_count=failure_count,
            avg_response_time=statistics.mean(response_times),
            errors=errors,
            actual_tokens=actual_tokens,
            input_tokens=avg_input_tokens,
            is_compressed=is_compressed,
            compression_ratio=compression_ratio
        )
    
    def _make_single_request(self, test_content: str) -> Tuple[bool, float, Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[int]]:
        """发送单个测试请求"""
        start_time = time.time()
        
        # 设置请求头
        if self.use_gemini_api:
            headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "text": "You are a context testing assistant. Your only task is to read the entire text and report all random numbers hidden in it. Respond with the numbers you found, using the format: random num: xxx\n\n" + test_content
                    }]
                }],
                "generationConfig": {
                    "maxOutputTokens": 2048,
                    "temperature": 0.1,
                }
            }
            print(f"      使用 Gemini API 格式发送请求")
        elif self.use_chat_api:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are a context testing assistant. Your only task is to read the entire text and report all random numbers hidden in it. Respond with the numbers you found, using the format: random num: xxx"},
                    {"role": "user", "content": test_content}
                ]
            }
            
            # 对于 GLM 模型，关闭思考模式以获得更快响应
            should_disable_thinking = (
                self.disable_thinking is True or 
                (self.disable_thinking is None and "glm" in self.model.lower())
            )
            
            if should_disable_thinking:
                payload["thinking"] = {"type": "disabled"}
                print(f"      已设置 thinking.type = disabled（GLM 思考模式关闭）")
                
        else:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": 0.1,
                "messages": [
                    {"role": "user", "content": test_content}
                ]
            }
        
        model_answer = None
        input_tokens = None
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # 检查响应内容
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    return (False, response_time, "响应JSON解析失败", model_answer, None, None)
                
                # 提取input_tokens
                if self.use_gemini_api:
                    # Gemini API格式
                    content = ""
                    
                    # 调试：输出原始响应数据（仅在detail模式下）
                    if self.show_detail:
                        print(f"      [DEBUG] Gemini 原始响应类型: {type(response_data)}")
                        if isinstance(response_data, list) and len(response_data) > 0:
                            print(f"      [DEBUG] 第一项数据: {json.dumps(response_data[0], ensure_ascii=False, indent=2)[:500]}...")
                    
                    # 处理流式响应格式（Gemini 返回的是数组）
                    if isinstance(response_data, list):
                        # 合并所有流式响应的内容
                        for item in response_data:
                            if isinstance(item, dict) and "candidates" in item:
                                candidates = item.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts and len(parts) > 0:
                                        content += parts[0].get("text", "")
                        
                        # 获取第一个有效响应的 usageMetadata
                        for item in response_data:
                            if isinstance(item, dict) and "usageMetadata" in item:
                                usage_metadata = item.get("usageMetadata", {})
                                if isinstance(usage_metadata, dict):
                                    input_tokens = usage_metadata.get("promptTokenCount")
                                    break
                        else:
                            input_tokens = None
                    else:
                        # 标准响应格式（备用）
                        candidates = response_data.get("candidates", [])
                        if candidates and len(candidates) > 0:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and len(parts) > 0:
                                content = parts[0].get("text", "")
                        
                        usage_metadata = response_data.get("usageMetadata", {})
                        if isinstance(usage_metadata, dict):
                            input_tokens = usage_metadata.get("promptTokenCount")
                        else:
                            input_tokens = None
                    
                elif self.use_chat_api:
                    # Chat API格式
                    usage = response_data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens")
                    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    # Anthropic API格式
                    usage = response_data.get("usage", {})
                    input_tokens = usage.get("input_tokens")
                    content = ""
                    content_blocks = response_data.get("content", [])
                    for block in content_blocks:
                        if block.get("type") == "text":
                            content += block.get("text", "")
                
                # 提取完整模型回答
                model_answer = content
                
                # 验证响应是否包含有效内容
                if content and len(content.strip()) > 0:
                    # 有响应内容，但具体是否正确需要在上级方法中验证
                    detail_data = response_data if self.show_detail else None
                    return (True, response_time, None, model_answer, detail_data, input_tokens)
                else:
                    return (False, response_time, "响应内容为空", model_answer, None, input_tokens)
            elif response.status_code == 429:
                return (False, response_time, "API 请求频率限制", model_answer, None, None)
            elif response.status_code == 401:
                return (False, response_time, "API 密钥无效", model_answer, None, None)
            elif response.status_code == 400:
                error_text = response.text[:200] if response.text else ""
                return (False, response_time, f"请求参数错误: {error_text}", model_answer, None, None)
            elif response.status_code >= 500:
                return (False, response_time, f"服务器错误: HTTP {response.status_code}", model_answer, None, None)
            else:
                error_text = response.text[:200] if response.text else ""
                return (False, response_time, f"HTTP {response.status_code}: {error_text}", model_answer, None, None)
                
        except requests.Timeout:
            response_time = time.time() - start_time
            return (False, response_time, f"请求超时（{self.timeout}秒）", model_answer, None, None)
        except requests.ConnectionError:
            response_time = time.time() - start_time
            return (False, response_time, "网络连接错误", model_answer, None, None)
        except requests.RequestException as e:
            response_time = time.time() - start_time
            return (False, response_time, f"请求异常: {str(e)}", model_answer, None, None)
        except Exception as e:
            response_time = time.time() - start_time
            return (False, response_time, f"未知错误: {str(e)}", model_answer, None, None)
    
    def _print_summary(self, results: Dict[int, ContextTestResult], max_successful_tokens: int) -> None:
        """打印测试汇总报告"""
        print("\n" + "=" * 80)
        print(f"📋 上下文窗口测试汇总")
        print("=" * 80)
        print("\n上下文大小 | 成功率 | 平均响应时间 | 压缩状态 | 成功/失败次数")
        print("-" * 85)
        
        for tokens, result in results.items():
            # 计算成功率
            total_requests = result.success_count + result.failure_count
            success_rate = (result.success_count / total_requests * 100) if total_requests > 0 else 0.0
            
            # 平均响应时间
            avg_time = f"{result.avg_response_time:.2f}s" if result.avg_response_time else "N/A"
            
            # 压缩状态
            if result.is_compressed:
                compression_status = f"⚠️ 压缩({result.compression_ratio:.1f}%)"
            elif result.input_tokens:
                compression_status = f"✅ 正常({abs(result.input_tokens - result.actual_tokens)/result.actual_tokens*100:.1f}%)"
            else:
                compression_status = "未知"
            
            # 成功/失败次数
            attempts = f"{result.success_count}/{result.failure_count}"
            
            # 使用实际tokens而不是目标tokens
            display_tokens = result.actual_tokens if result.actual_tokens is not None else tokens
            print(f"{display_tokens:9,} | {success_rate:6.1f}% | {avg_time:13s} | {compression_status:15s} | {attempts}")
        
        print(f"\n🎯 测试结果:")
        print(f"   最大成功上下文: {max_successful_tokens:,} tokens")
        
        # 额外统计信息
        total_tests = sum(result.success_count + result.failure_count for result in results.values())
        total_success = sum(result.success_count for result in results.values())
        total_failure = sum(result.failure_count for result in results.values())
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0.0
        
        print(f"\n📊 整体统计:")
        print(f"   总测试次数: {total_tests}")
        print(f"   总成功次数: {total_success}")
        print(f"   总失败次数: {total_failure}")
        print(f"   整体成功率: {overall_success_rate:.1f}%")
        
        # 压缩统计
        compressed_tests = [result for result in results.values() if result.is_compressed]
        if compressed_tests:
            print(f"\n🗜️  压缩检测:")
            print(f"   检测到压缩的测试: {len(compressed_tests)}")
            print(f"   平均压缩率: {sum(result.compression_ratio for result in compressed_tests) / len(compressed_tests):.1f}%")
            print(f"   最大压缩率: {max(result.compression_ratio for result in compressed_tests):.1f}%")
        
        # 计算平均响应时间（仅成功测试）
        successful_times = [result.avg_response_time for result in results.values() if result.success and result.avg_response_time]
        if successful_times:
            avg_successful_time = sum(successful_times) / len(successful_times)
            print(f"   平均响应时间（成功）: {avg_successful_time:.2f}s")
        
        # 如果有失败的测试，显示错误信息
        failed_results = [(tokens, result) for tokens, result in results.items() if not result.success]
        if failed_results:
            print(f"\n❌ 失败详情:")
            for tokens, result in failed_results:
                if result.errors:
                    print(f"   {tokens:,} tokens: {result.errors[0]}")

    def _export_results(self, results: Dict[int, ContextTestResult], max_successful_tokens: int) -> None:
        """导出测试结果到 JSON 文件"""
        export_data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "api_url": self.api_url,
                "model": self.model,
                "test_rounds": self.test_rounds,
                "timeout": self.timeout,
                "chars_per_token": self.chars_per_token
            },
            "results": {},
            "summary": {
                "max_successful_tokens": max_successful_tokens,
                "max_successful_chars": int(max_successful_tokens * self.chars_per_token),
                "max_successful_chinese_chars": int(max_successful_tokens * self.chars_per_token / 2),
                "max_successful_english_words": int(max_successful_tokens * self.chars_per_token / 5)
            }
        }
        
        # 转换结果数据
        for tokens, result in results.items():
            export_data["results"][str(tokens)] = {
                "success": result.success,
                "success_count": result.success_count,
                "failure_count": result.failure_count,
                "avg_response_time": result.avg_response_time,
                "errors": result.errors,
                "actual_tokens": result.actual_tokens,
                "input_tokens": result.input_tokens,
                "is_compressed": result.is_compressed,
                "compression_ratio": result.compression_ratio
            }
        
        # 写入文件
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"\n📁 测试结果已导出到: {self.output_file}")
        except Exception as e:
            print(f"\n❌ 导出结果失败: {str(e)}")
    
    def _validate_response_content(self, content: str, random_number: int) -> bool:
        """验证响应内容是否包含正确的随机数"""
        import re
        
        # 查找所有数字
        numbers = re.findall(r'\b(\d{3})\b', content)  # 查找3位数字
        
        # 检查是否包含目标随机数
        return str(random_number) in numbers
    
    def _get_base_text(self) -> str:
        """获取基础测试文本"""
        return """这是一段用于测试上下文窗口长度的示例文本。每个段落都包含相同的内容，以便于验证模型是否能够处理完整的上下文。
        
上下文窗口是指大型语言模型在一次交互中能够处理的最大文本量。它包括了输入提示、系统提示以及模型生成的响应。更大的上下文窗口允许模型考虑更多的信息，从而提供更准确和连贯的回答。

测试上下文窗口的重要性在于：
1. 确定模型能够处理的最大文本量
2. 评估模型在长文本下的表现
3. 验证模型的实际能力与宣传是否一致
4. 为应用开发提供参考依据

在实际应用中，上下文窗口的大小直接影响：
- 文档总结的能力
- 多轮对话的连贯性
- 代码分析的范围
- 数据处理的效率

"""
    
    def _get_alternative_base_text(self, paragraph_num: int) -> str:
        """生成随机测试文本，避免重复内容触发过滤"""
        # 如果超过最大段落数限制，返回空字符串
        if paragraph_num >= self.max_paragraphs:
            return ""
            
        # 使用全局随机数生成器，确保每次调用都不同
        # 不重新设置种子，让random模块使用系统时间
        
        if self.use_english:
            # 扩展英文词汇库
            subjects = [
                "Artificial intelligence", "Machine learning", "Deep learning", "Neural networks", 
                "Data science", "Cloud computing", "Blockchain", "Quantum computing",
                "Cybersecurity", "Internet of Things", "Computer vision", "Robotics",
                "Natural language processing", "Edge computing", "Big data analytics",
                "Augmented reality", "Virtual reality", "5G technology", "Autonomous vehicles",
                "Smart cities", "Digital transformation", "Fintech", "Biotechnology",
                "Renewable energy", "Nanotechnology", "3D printing", "Drones",
                "Wearable technology", "Voice assistants", "Predictive analytics", "DevOps"
            ]
            
            verbs = [
                "transforms", "enhances", "revolutionizes", "optimizes", "improves", "automates",
                "streamlines", "innovates", "accelerates", "modernizes", "digitizes", "integrates",
                "facilitates", "enables", "empowers", "simplifies", "standardizes", "customizes",
                "democratizes", "disrupts", "catalyzes", "amplifies", "orchestrates", "synchronizes"
            ]
            
            objects = [
                "business processes", "healthcare systems", "financial services", "manufacturing",
                "education platforms", "retail operations", "transportation networks", "energy grids",
                "communication systems", "supply chains", "customer experiences", "workflows",
                "decision making", "risk management", "quality control", "compliance monitoring",
                "resource allocation", "talent acquisition", "knowledge management", "innovation pipelines",
                "market research", "product development", "service delivery", "stakeholder engagement"
            ]
            
            adjectives = [
                "advanced", "sophisticated", "cutting-edge", "innovative", "revolutionary",
                "state-of-the-art", "modern", "efficient", "scalable", "robust", "flexible", "intelligent",
                "autonomous", "adaptive", "responsive", "proactive", "dynamic", "seamless",
                "comprehensive", "holistic", "integrated", "unified", "centralized", "distributed"
            ]
            
            # 技术特定词汇
            tech_terms = [
                "algorithm", "framework", "architecture", "protocol", "ecosystem", "paradigm",
                "methodology", "infrastructure", "platform", "solution", "approach", "strategy",
                "implementation", "deployment", "integration", "migration", "optimization", "scalability"
            ]
            
            # 为每个段落创建唯一标识符
            paragraph_id = f"P{paragraph_num:04d}"
            
            # 生成随机数量的句子
            num_sentences = random.randint(4, 8)
            sentences = []
            used_combinations = set()  # 跟踪已使用的组合
            
            for i in range(num_sentences):
                # 尝试找到未使用的组合
                max_attempts = 10
                for attempt in range(max_attempts):
                    subject = random.choice(subjects)
                    verb = random.choice(verbs)
                    obj = random.choice(objects)
                    adj = random.choice(adjectives)
                    
                    # 创建组合键
                    combo_key = f"{subject[:8]}_{verb[:5]}_{obj[:8]}_{adj[:5]}"
                    
                    if combo_key not in used_combinations or attempt == max_attempts - 1:
                        used_combinations.add(combo_key)
                        break
                
                if i == 0:
                    sentence = f"{paragraph_id}: {subject} {verb} {obj} through {adj} technologies and methodologies."
                else:
                    # 更多的句子类型变化
                    sentence_templates = [
                        f"The implementation of {subject.lower()} requires careful consideration of {random.choice(['scalability', 'performance', 'security', 'reliability'])} and {random.choice(['compliance', 'governance', 'standards', 'best practices'])}.",
                        f"Recent advances in {subject.lower()} have opened new possibilities for {obj} optimization through {random.choice(tech_terms)} integration.",
                        f"Organizations adopting {subject.lower()} report significant improvements in {random.choice(['efficiency', 'productivity', 'ROI', 'TCO'])} and {random.choice(['agility', 'resilience', 'innovation', 'growth'])}.",
                        f"The future of {subject.lower()} depends on continued research and development in {random.choice(tech_terms)} and {random.choice(['emerging technologies', 'industry standards', 'regulatory frameworks', 'market demands'])}.",
                        f"Integration of {subject.lower()} with existing systems presents both {random.choice(['challenges', 'opportunities', 'risks', 'benefits'])} and {random.choice(['advantages', 'disadvantages', 'trade-offs', 'synergies'])}.",
                        f"Industry experts predict that {subject.lower()} will {random.choice(['disrupt', 'transform', 'redefine', 'reshape'])} the way we approach {obj} in the coming years.",
                        f"Case studies show that successful {subject.lower()} implementations can achieve up to {random.randint(20, 95)}% improvement in key performance indicators.",
                        f"The {random.choice(['ROI', 'TCO', 'NPV', 'IRR'])} of {subject.lower()} projects typically ranges from {random.randint(100, 999)}% depending on the scope and scale."
                    ]
                    sentence = random.choice(sentence_templates)
                
                sentences.append(sentence)
            
            # 添加段落特有的额外内容
            extra_content_types = [
                f"Industry analysts project the global market for these technologies will reach ${random.randint(1, 999)} billion by {2025 + random.randint(1, 10)}.",
                f"A recent survey of {random.randint(100, 999)} organizations revealed that {random.randint(20, 95)}% are planning to increase investment in this area.",
                f"The adoption rate has increased by {random.randint(20, 300)}% year-over-year, indicating strong market momentum.",
                f"Leading vendors in this space include both established technology giants and innovative startups funded with over ${random.randint(10, 500)} million in venture capital."
            ]
            
            if random.random() > 0.6:
                sentences.append(random.choice(extra_content_types))
            
            # 组合成段落
            paragraph = " ".join(sentences)
            
            return f"""{paragraph}

This unique content block {paragraph_id} demonstrates the model's ability to process entirely original textual information without repetition patterns. Each paragraph is algorithmically generated to ensure maximum uniqueness.

The sophisticated randomization algorithm combines vocabulary permutation with structural variation to create content that cannot be predicted or memorized by language models.

Content uniqueness verification: {hash(paragraph) % 1000000:06d}

"""
        else:
            # 扩展中文词汇库
            subjects = [
                "人工智能", "机器学习", "深度学习", "神经网络", "数据科学", "云计算",
                "区块链", "量子计算", "网络安全", "物联网", "计算机视觉", "机器人技术",
                "自然语言处理", "边缘计算", "大数据分析", "增强现实", "虚拟现实",
                "5G技术", "自动驾驶", "智慧城市", "数字化转型", "金融科技", "生物技术",
                "可再生能源", "纳米技术", "3D打印", "无人机", "可穿戴设备", "语音助手",
                "预测分析", "DevOps", "微服务", "容器化", "无服务器计算", "数字化转型"
            ]
            
            verbs = [
                "改变了", "提升了", "革新了", "优化了", "改进了", "自动化了",
                "简化了", "创新了", "加速了", "现代化了", "数字化了", "集成了",
                "促进了", "实现了", "赋能了", "标准化了", "定制化了", "普及了",
                "颠覆了", "催化了", "放大了", "协调了", "同步了", "重构了"
            ]
            
            objects = [
                "业务流程", "医疗系统", "金融服务", "制造业", "教育平台", "零售运营",
                "交通网络", "能源网络", "通信系统", "供应链", "客户体验", "工作流程",
                "决策制定", "风险管理", "质量控制", "合规监控", "资源分配", "人才获取",
                "知识管理", "创新管道", "市场研究", "产品开发", "服务交付", "利益相关者参与"
            ]
            
            adjectives = [
                "先进的", "复杂的", "尖端的", "创新的", "革命性的", "最先进的",
                "现代的", "高效的", "可扩展的", "稳健的", "灵活的", "智能的",
                "自主的", "自适应的", "响应式的", "主动的", "动态的", "无缝的",
                "全面的", "整体的", "集成的", "统一的", "集中的", "分布式的"
            ]
            
            # 技术特定词汇
            tech_terms = [
                "算法", "框架", "架构", "协议", "生态系统", "范式",
                "方法论", "基础设施", "平台", "解决方案", "方法", "策略",
                "实施", "部署", "集成", "迁移", "优化", "可扩展性"
            ]
            
            # 为每个段落创建唯一标识符
            paragraph_id = f"第{paragraph_num + 1:04d}段"
            
            # 生成随机数量的句子
            num_sentences = random.randint(4, 8)
            sentences = []
            used_combinations = set()  # 跟踪已使用的组合
            
            for i in range(num_sentences):
                # 尝试找到未使用的组合
                max_attempts = 10
                for attempt in range(max_attempts):
                    subject = random.choice(subjects)
                    verb = random.choice(verbs)
                    obj = random.choice(objects)
                    adj = random.choice(adjectives)
                    
                    # 创建组合键
                    combo_key = f"{subject}_{verb}_{obj}_{adj}"
                    
                    if combo_key not in used_combinations or attempt == max_attempts - 1:
                        used_combinations.add(combo_key)
                        break
                
                if i == 0:
                    sentence = f"{paragraph_id}：{subject}{verb}{obj}，通过{adj}技术实现突破性进展。"
                else:
                    # 更多的句子类型变化
                    sentence_templates = [
                        f"{subject}的实施需要仔细考虑{random.choice(['可扩展性', '性能', '安全性', '可靠性'])}和{random.choice(['合规性', '治理', '标准', '最佳实践'])}。",
                        f"{subject}的最新进展为{obj}的优化开辟了新的可能性，特别是在{random.choice(tech_terms)}集成方面。",
                        f"采用{subject}的组织报告称在{random.choice(['效率', '生产力', '投资回报率', '总拥有成本'])}和{random.choice(['敏捷性', '韧性', '创新能力', '增长'])}方面有显著改善。",
                        f"{subject}的未来取决于该领域在{random.choice(tech_terms)}和{random.choice(['新兴技术', '行业标准', '监管框架', '市场需求'])}方面的持续研究和开发。",
                        f"将{subject}与现有系统集成既带来了{random.choice(['挑战', '机遇', '风险', '收益'])}，也创造了{random.choice(['优势', '劣势', '权衡', '协同效应'])}。",
                        f"行业专家预测{subject}将在未来几年内{random.choice(['颠覆', '改变', '重新定义', '重塑'])}我们处理{obj}的方式。",
                        f"案例研究表明，成功的{subject}实施可以实现关键绩效指标高达{random.randint(20, 95)}%的提升。",
                        f"{subject}项目的{random.choice(['投资回报率', '总拥有成本', '净现值', '内部收益率'])}通常根据范围和规模在{random.randint(100, 999)}%之间。"
                    ]
                    sentence = random.choice(sentence_templates)
                
                sentences.append(sentence)
            
            # 添加段落特有的额外内容
            extra_content_types = [
                f"行业分析师预测，这些技术的全球市场将在{2025 + random.randint(1, 10)}年达到{random.randint(1, 999)}0亿美元。",
                f"最近一项对{random.randint(100, 999)}家组织的调查显示，{random.randint(20, 95)}%计划增加该领域的投资。",
                f"采用率同比增长了{random.randint(20, 300)}%，表明市场势头强劲。",
                f"该领域的领先供应商既包括技术巨头，也包括获得超过{random.randint(10, 500)}亿美元风险投资支持的初创企业。"
            ]
            
            if random.random() > 0.6:
                sentences.append(random.choice(extra_content_types))
            
            # 组合成段落
            paragraph = "".join(sentences)
            
            return f"""{paragraph}

这段唯一内容块{paragraph_id}展示了模型处理完全原创文本信息而无重复模式的能力。每个段落都通过算法生成，确保最大程度的独特性。

复杂的随机化算法结合词汇排列和结构变化，创造出语言模型无法预测或记忆的内容。

内容唯一性验证：{hash(paragraph) % 1000000:06d}

"""
    
    def _adjust_content_length(self, content: str, target_tokens: int) -> str:
        """调整内容长度以匹配目标 tokens"""
        current_tokens = self._count_tokens(content)
        
        if current_tokens <= target_tokens:
            return content
        
        # 更智能的截断策略
        # 1. 保留完整的开始标记
        start_marker_end = content.find("\n\n") + 2
        if start_marker_end < 10:
            start_marker_end = content.find("[START]") + 50  # fallback
        
        # 2. 保留完整的结束标记
        end_marker_start = content.rfind("[END]")
        if end_marker_start == -1:
            end_marker_start = len(content) - 100  # fallback
        
        # 3. 计算需要保留的内容长度
        if self.use_tokenizer and self.tokenizer:
            # 使用分词器进行精确调整
            left, right = start_marker_end, end_marker_start
            best_content = content
            
            while left < right:
                mid = (left + right + 1) // 2
                test_content = content[:mid] + content[end_marker_start:]
                test_tokens = self._count_tokens(test_content)
                
                if test_tokens <= target_tokens:
                    best_content = test_content
                    left = mid
                else:
                    right = mid - 1
            
            return best_content
        else:
            # 字符估算方式，但更智能
            estimated_chars = int(target_tokens * self.chars_per_token)
            
            # 计算内容部分的大致位置
            content_start = start_marker_end
            content_end = end_marker_start
            content_length = content_end - content_start
            
            if content_length > 0:
                # 按比例缩放内容
                scale = min(1.0, estimated_chars / len(content))
                new_content_end = int(content_start + content_length * scale)
                
                # 寻找合适的截断点（句子边界）
                truncated_content = content[content_start:new_content_end]
                
                # 尝试在句子边界截断
                sentence_endings = ['. ', '! ', '? ', '。\n', '！\n', '？\n', '\n\n']
                best_pos = 0
                for ending in sentence_endings:
                    pos = truncated_content.rfind(ending)
                    if pos > best_pos:
                        best_pos = pos + len(ending)
                
                if best_pos > 0:
                    final_content = content[:content_start + best_pos] + content[end_marker_start:]
                else:
                    # 如果找不到句子边界，在单词边界截断
                    word_boundary = truncated_content.rfind(' ')
                    if word_boundary > 0:
                        final_content = content[:content_start + word_boundary] + content[end_marker_start:]
                    else:
                        final_content = content[:new_content_end] + content[end_marker_start:]
                
                return final_content
            
            # fallback
            return content[:estimated_chars] + content[end_marker_start:]
    
    def _save_test_content(self, size_name: str, test_content: str, random_numbers: List[int]) -> None:
        """保存测试内容到log目录"""
        try:
            # 创建log目录（如果不存在）
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成文件名（包含时间戳和用户输入的大小）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 将多个随机数转换为字符串用于文件名
            random_str = "_".join(map(str, random_numbers))
            filename = f"context_test_{timestamp}_{size_name}_random_{random_str}.txt"
            filepath = os.path.join(log_dir, filename)
            
            # 写入文件（只保存测试内容）
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            print(f"   📁 测试内容已保存到: {filename}")
            
        except Exception as e:
            print(f"   ⚠️ 保存测试内容失败: {str(e)}")
    
    def _save_round_content(self, size_name: str, round_num: int, test_content: str, random_numbers: List[int]) -> None:
        """保存每轮测试内容到log目录"""
        try:
            # 创建log目录（如果不存在）
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成文件名（包含时间戳、用户输入的大小和轮数）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 将多个随机数转换为字符串用于文件名
            random_str = "_".join(map(str, random_numbers))
            filename = f"context_test_{timestamp}_{size_name}_round{round_num + 1}_random_{random_str}.txt"
            filepath = os.path.join(log_dir, filename)
            
            # 写入文件（只保存测试内容）
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            print(f"      📁 第{round_num + 1}轮内容已保存到: {filename}")
            
        except Exception as e:
            print(f"      ⚠️ 保存第{round_num + 1}轮内容失败: {str(e)}")


class APIError(Exception):
    """API 相关错误"""
    pass


class NetworkError(Exception):
    """网络相关错误"""
    pass


class ContentGenerationError(Exception):
    """内容生成错误"""
    pass




def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="上下文窗口长度测试工具 - 测试 LLM 模型的最大上下文大小",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python context_length_tester.py --key your_api_key_here
  python context_length_tester.py --key your_api_key_here --sizes 32k 64k 128k
  python context_length_tester.py --key your_api_key_here --sizes 256k 512k
  python context_length_tester.py --key your_api_key_here --sizes 64k 128k 192k
  python context_length_tester.py --key your_api_key_here --sizes 90k 150k 300k
  python context_length_tester.py --key your_api_key_here --sizes 50000 100000 150000
  python context_length_tester.py --key your_api_key_here --rounds 3 --timeout 600
  python context_length_tester.py --key your_api_key_here --chat-api
  python context_length_tester.py --key your_api_key_here --output-file results.json
  python context_length_tester.py --key your_api_key_here --max-paragraphs 50000
  python context_length_tester.py --key your_api_key_here --detail
  python context_length_tester.py --key your_api_key_here --query-num 10
  python context_length_tester.py --key your_api_key_here --query-num 10 --detail  # 启用调试信息
  python context_length_tester.py --key your_gemini_api_key_here --gemini-api --model gemini-1.5-pro
  python context_length_tester.py --key your_gemini_api_key_here --gemini-api --sizes 32k 64k 128k
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
    
    # 测试大小参数
    parser.add_argument(
        "--sizes",
        nargs="+",
        help="测试大小列表，支持预设大小（1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k）或自定义大小（如 90k, 192k, 50000），k后缀表示乘以1024，数字范围1-65536"
    )
    
    # 测试参数
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_TEST_ROUNDS,
        help="每个大小测试轮数（默认：%(default)d）"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="单请求超时时间（秒，默认：%(default)d）"
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=None,
        help="每个 token 的字符数（默认：英文4.0，中文2.0）"
    )
    parser.add_argument(
        "--chat-api",
        action="store_true",
        help="使用 Chat API 接口（默认：使用 Anthropic 接口）"
    )
    parser.add_argument(
        "--gemini-api",
        action="store_true",
        help="使用 Gemini API 接口（默认：使用 Anthropic 接口）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="导出结果到 JSON 文件"
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=DEFAULT_MAX_PARAGRAPHS,
        help="最大段落数量限制（默认：%(default)d）"
    )
    parser.add_argument(
        "--use-english",
        action="store_true",
        help="使用英文生成prompt内容（默认使用中文）"
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="禁用 GLM 模型的思考模式（默认会自动为 GLM 模型禁用）"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="显示详细的响应内容（默认不显示）"
    )
    parser.add_argument(
        "--query-num",
        type=int,
        default=1,
        help="插入的随机数数量（默认：%(default)d）"
    )
    
    return parser.parse_args()


def parse_custom_size(custom_size_str: str) -> int:
    """解析自定义大小表达式，支持数字+k的形式
    
    Args:
        custom_size_str: 自定义大小字符串（如 "192k", "90k", "50000"）
    
    Returns:
        解析后的token数量
    
    Raises:
        ValueError: 当格式无效或数字超出范围时
    """
    import re
    
    # 匹配数字 + 可选的 k 后缀
    pattern = r'^(\d+)(k?)$'
    match = re.match(pattern, custom_size_str.lower())
    
    if not match:
        raise ValueError(f"无效的自定义大小格式: {custom_size_str}。应为数字或数字+k（如 90k、192k、50000）")
    
    number = int(match.group(1))
    suffix = match.group(2)
    
    # 验证数字范围
    if number < 1 or number > 65536:
        raise ValueError(f"数字必须在1-65536范围内，当前值: {number}")
    
    # 根据后缀计算最终值
    if suffix == 'k':
        return number * 1024  # 使用1024而不是1000，与PRESET_SIZES保持一致
    else:
        return number


def main():
    """主函数 - 运行上下文长度测试"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建测试器实例
    api_url = args.url
    
    tester = ContextLengthTester(
        api_url=api_url,
        api_key=args.key,
        model=args.model,
        test_sizes=args.sizes,
        test_rounds=args.rounds,
        timeout=args.timeout,
        chars_per_token=args.chars_per_token,
        use_chat_api=args.chat_api,
        use_gemini_api=args.gemini_api,
        output_file=args.output_file,
        max_paragraphs=args.max_paragraphs,
        use_english=args.use_english,
        disable_thinking=args.disable_thinking,
        show_detail=args.detail,
        query_num=args.query_num
    )
    
    # 运行测试
    results = tester.run_test()
    
    return results


if __name__ == "__main__":
    main()
