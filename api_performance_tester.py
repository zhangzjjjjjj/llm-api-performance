"""
API 性能测试工具

用于测试 BigModel API 的并发性能，支持 SSE 流式请求，
统计 TTFT（首 token 时间）、完成时间、tokens/s 等指标。

作者: Claude
版本: 1.0.0
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


# 默认配置值（仅作为 argparse 的默认值使用）
DEFAULT_API_URL = "https://open.bigmodel.cn/api/anthropic/v1/messages"
DEFAULT_MODEL = "glm-4.5"
DEFAULT_TEST_MESSAGE = "What opportunities and challenges will the Chinese large model industry face in 2025?"
DEFAULT_MIN_CONCURRENCY = 5
DEFAULT_MAX_CONCURRENCY = 100
DEFAULT_STEP = 5
DEFAULT_TEST_ROUNDS = 1
DEFAULT_TIMEOUT = 120
DEFAULT_PRINT_SAMPLE_ERRORS = 5
DEFAULT_CHARS_PER_TOKEN = 4.0


class APIPerformanceTester:
    """API 性能测试工具 - 支持 SSE 流式请求测试"""
    
    def __init__(self, api_url=None, api_key=None, model=None, test_message=None, 
                 min_concurrency=None, max_concurrency=None, step=None, test_rounds=None,
                 timeout=None, print_sample_errors=None, estimate_tokens_by_chars=None,
                 chars_per_token=None):
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
        """
        # API 配置
        self.api_url = api_url or DEFAULT_API_URL
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.test_message = test_message or DEFAULT_TEST_MESSAGE
        
        # 测试参数
        self.min_concurrency = min_concurrency or DEFAULT_MIN_CONCURRENCY
        self.max_concurrency = max_concurrency or DEFAULT_MAX_CONCURRENCY
        self.step = step or DEFAULT_STEP
        self.test_rounds = test_rounds or DEFAULT_TEST_ROUNDS
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.print_sample_errors = print_sample_errors or DEFAULT_PRINT_SAMPLE_ERRORS
        self.estimate_tokens_by_chars = estimate_tokens_by_chars or False
        self.chars_per_token = chars_per_token or DEFAULT_CHARS_PER_TOKEN

    def run_test(self):
        """运行完整的性能测试"""
        # 检查 API Key
        if not self.api_key:
            print("❌ 错误：请先设置 API_KEY")
            print("提示：创建测试器时传入 api_key 参数")
            return None
            
        print("🚀 开始 API 并发性能测试（SSE + TTFT + tokens/s）")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API 地址: {self.api_url}")
        print(f"模型: {self.model}")
        print(f"测试范围: {self.min_concurrency}-{self.max_concurrency} 并发 (步长: {self.step})")
        print(f"每个并发级别测试轮数: {self.test_rounds}")
        print(f"单请求超时: {self.timeout}秒")

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
        
        return results
        
    def _print_summary(self, results):
        """打印测试汇总报告"""
        print("\n" + "=" * 60)
        print(f"📋 测试汇总报告 {self.api_url}")
        print("=" * 60)
        print("\n并发级别 | 成功率 | 平均完成时间 | 平均TTFT | 平均tokens/s")
        print("-" * 70)

        for concurrency, result in results.items():
            total_req = result.success_count + result.failure_count
            succ_rate = (result.success_count / total_req) * 100 if total_req else 0.0
            avg_time = statistics.mean(result.response_times) if result.response_times else float("nan")
            avg_ttft = statistics.mean(result.first_token_times) if result.first_token_times else float("nan")
            avg_tps = statistics.mean(result.tokens_per_sec) if result.tokens_per_sec else float("nan")
            print(f"{concurrency:8d} | {succ_rate:6.1f}% | {avg_time:10.2f}s | {avg_ttft:8.3f}s | {avg_tps:12.2f}")


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
        timeout = tester.timeout
        estimate_tokens_by_chars = tester.estimate_tokens_by_chars
        chars_per_token = tester.chars_per_token
        
    start_time = time.time()
    first_token_time = None
    output_tokens = None  # 来自 message_delta 的 usage.output_tokens（累计）
    approx_chars = 0      # 如果需要估算时使用

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
        ],
        "stream": True
    }

    try:
        with requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload),
            stream=True,
            timeout=timeout,
        ) as r:
            status = r.status_code
            if status != 200:
                total_time = time.time() - start_time
                text = r.text[:200] if r.text else ""
                return (False, total_time, status, f"HTTP {status}: {text}", None, None, None)

            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue

                chunk = raw_line[len("data:"):].strip()
                if not chunk:
                    continue

                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

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

            # 未收到 message_stop
            total_time = time.time() - start_time
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
    print(f"   TTFT(首字响应): 平均 {avg_ttft:.3f}s | P50 {p50_ttft:.3f}s | P95 {p95_ttft:.3f}s")
    if result.tokens_per_sec:
        print(f"   输出Token: 总计 {sum_tokens} | 单次平均 {avg_tokens:.1f}")
        print(f"   输出速率(tokens/s): 平均 {avg_tps:.2f} | P50 {p50_tps:.2f} | P95 {p95_tps:.2f} | 最高 {max_tps:.2f}")
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
        description="API 性能测试工具 - 测试 BigModel API 的并发性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python api_performance_tester.py --key your_api_key_here
  python api_performance_tester.py --key your_api_key_here --model glm-4-0528
  python api_performance_tester.py --key your_api_key_here --min 5 --max 50 --step 5
  python api_performance_tester.py --key your_api_key_here --rounds 3 --timeout 60
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
    
    return parser.parse_args()


def main():
    """主函数 - 运行性能测试"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建测试器实例
    tester = APIPerformanceTester(
        api_url=args.url,
        api_key=args.key,
        model=args.model,
        test_message=args.message,
        min_concurrency=args.min,
        max_concurrency=args.max,
        step=args.step,
        test_rounds=args.rounds,
        timeout=args.timeout,
        estimate_tokens_by_chars=args.estimate_tokens,
        chars_per_token=args.chars_per_token
    )
    
    # 运行测试
    results = tester.run_test()
    
    return results


if __name__ == "__main__":
    main()