# API 性能测试工具

一个用于测试 BigModel API 并发性能的 Python 工具，支持 SSE（Server-Sent Events）流式请求，可以统计 TTFT（首 token 时间）、完成时间、tokens/s 等关键性能指标。

**支持两种接口模式：**
- Anthropic 兼容接口（默认）
- Chat API 接口

## 功能特性

- 🚀 支持高并发 API 性能测试
- 📊 实时统计 TTFT（Time To First Token）
- ⚡ 测量 tokens/s 输出速率
- 🔍 支持多种模型（如 GLM-4.5）
- 📈 详细的性能报告和错误分析
- 🔧 灵活的配置选项
- 🔄 支持多种 API 接口格式

## 安装要求

- Python 3.7+
- requests 库

```bash
pip install requests
```

## 快速开始

### 1. 命令行使用（推荐）

通过命令行参数直接运行测试：

```bash
# 基本使用
python api_performance_tester.py --key your_api_key_here

# 指定模型
python api_performance_tester.py --key your_api_key_here --model glm-4-0528

# 自定义并发范围
python api_performance_tester.py --key your_api_key_here --min 5 --max 50 --step 5

# 多轮测试和自定义超时
python api_performance_tester.py --key your_api_key_here --rounds 3 --timeout 60

# 使用字符数估算 tokens
python api_performance_tester.py --key your_api_key_here --estimate-tokens

# 使用 Chat API 接口
python api_performance_tester.py --key your_api_key_here --chat-api

# 自定义 Chat API URL
python api_performance_tester.py --key your_api_key_here --url https://open.bigmodel.cn/api/paas/v4/chat/completions

# 查看帮助
python api_performance_tester.py --help
```

### 2. 编程方式使用

```python
from api_performance_tester import APIPerformanceTester

# 使用 Anthropic 接口（默认）
tester = APIPerformanceTester(
    api_url="https://open.bigmodel.cn/api/anthropic/v1/messages",
    api_key="your_api_key",
    model="glm-4.5",
    test_message="测试消息内容"
)

# 使用 Chat API 接口
chat_tester = APIPerformanceTester(
    api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
    api_key="your_api_key",
    model="glm-4.5",
    test_message="测试消息内容",
    use_chat_api=True  # 启用 Chat API 模式
)

# 运行测试
results = tester.run_test()
chat_results = chat_tester.run_test()
```

### 3. 高级配置

```python
# 自定义测试参数
tester = APIPerformanceTester()
tester.min_concurrency = 10      # 最小并发数
tester.max_concurrency = 50      # 最大并发数
tester.step = 5                  # 并发步长
tester.test_rounds = 3           # 每个并发级别测试轮数
tester.timeout = 60              # 请求超时时间（秒）
tester.estimate_tokens_by_chars = True  # 使用字符数估算 tokens
```

## 配置参数

### API 配置
- `API_URL`: API 接口地址
  - Anthropic 接口：`https://open.bigmodel.cn/api/anthropic/v1/messages`（默认）
  - Chat API 接口：`https://open.bigmodel.cn/api/paas/v4/chat/completions`
- `API_KEY`: 您的 API 密钥
- `MODEL`: 使用的模型（如 glm-4.5）
- `TEST_MESSAGE`: 测试消息内容
- `USE_CHAT_API`: 是否使用 Chat API 接口格式（默认：False）

### 测试参数
- `MIN_CONCURRENCY`: 最小并发级别（默认：15）
- `MAX_CONCURRENCY`: 最大并发级别（默认：100）
- `STEP`: 并发级别步长（默认：5）
- `TEST_ROUNDS`: 每个并发级别测试轮数（默认：1）
- `TIMEOUT`: 单请求超时时间（默认：120秒）
- `ESTIMATE_TOKENS_BY_CHARS`: 是否使用字符数估算 tokens（默认：False）

## 测试结果说明

工具会输出以下性能指标：

1. **成功率**: 成功请求占总请求的百分比
2. **平均完成时间**: 从发送请求到收到完整响应的平均时间
3. **TTFT（首字响应时间）**: 
   - 平均值
   - P50（中位数）
   - P95（95分位数）
4. **输出速率（tokens/s）**:
   - 平均值
   - P50、P95
   - 最高值

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--key` | API 密钥（必需） | - |
| `--url` | API 接口地址 | Anthropic API 地址 |
| `--model` | 使用的模型 | glm-4.5 |
| `--message` | 测试消息内容 | 英文测试问题 |
| `--min` | 最小并发级别 | 5 |
| `--max` | 最大并发级别 | 100 |
| `--step` | 并发级别步长 | 5 |
| `--rounds` | 每个并发级别测试轮数 | 1 |
| `--timeout` | 单请求超时时间（秒） | 120 |
| `--estimate-tokens` | 使用字符数估算 tokens | False |
| `--chars-per-token` | 每个 token 的字符数 | 4.0 |
| `--chat-api` | 使用 Chat API 接口格式 | False |

## 使用示例

### 1. Anthropic 接口测试（默认）

```bash
# 运行测试
$ python api_performance_tester.py --key your_api_key_here

🚀 开始 API 并发性能测试（SSE + TTFT + tokens/s）
测试时间: 2025-01-01 10:00:00
API 地址: https://open.bigmodel.cn/api/anthropic/v1/messages
模型: glm-4.5
测试范围: 5-100 并发 (步长: 5)
每个并发级别测试轮数: 1
单请求超时: 120秒

🔄 测试并发级别: 5
==================================================
   第 1/1 轮测试...
📊 测试结果:
   总请求数: 5
   成功: 5 | 失败: 0
   成功率: 100.0%
   平均完成时间: 2.34s  (最快 1.89s | 最慢 3.12s)
   TTFT(首字响应): 平均 0.456s | P50 0.432s | P95 0.678s
   输出Token: 总计 5120 | 单次平均 341.3
   输出速率(tokens/s): 平均 145.89 | P50 148.32 | P95 162.45 | 最高 189.76

============================================================
📋 测试汇总报告 https://open.bigmodel.cn/api/anthropic/v1/messages
============================================================

并发级别 | 成功率 | 平均完成时间 | 平均TTFT | 平均tokens/s
----------------------------------------------------------------------
       5 |  100.0% |       2.34s |    0.456s |       145.89
      10 |  100.0% |       2.89s |    0.523s |       138.45
      15 |   98.0% |       3.45s |    0.612s |       129.87
```

### 2. Chat API 接口测试

```bash
# 使用 Chat API 接口
$ python api_performance_tester.py --key your_api_key_here --chat-api

🚀 开始 API 并发性能测试（SSE + TTFT + tokens/s）
测试时间: 2025-01-01 10:00:00
API 地址: https://open.bigmodel.cn/api/paas/v4/chat/completions
模型: glm-4.5
测试范围: 5-100 并发 (步长: 5)
每个并发级别测试轮数: 1
单请求超时: 120秒

🔄 测试并发级别: 5
==================================================
   第 1/1 轮测试...
📊 测试结果:
   总请求数: 5
   成功: 5 | 失败: 0
   成功率: 100.0%
   平均完成时间: 2.15s  (最快 1.75s | 最慢 2.89s)
   TTFT(首字响应): 平均 0.423s | P50 0.401s | P95 0.612s
   输出Token: 总计 4985 | 单次平均 332.5
   输出速率(tokens/s): 平均 152.34 | P50 155.67 | P95 168.92 | 最高 195.43

============================================================
📋 测试汇总报告 https://open.bigmodel.cn/api/paas/v4/chat/completions
============================================================

并发级别 | 成功率 | 平均完成时间 | 平均TTFT | 平均tokens/s
----------------------------------------------------------------------
       5 |  100.0% |       2.15s |    0.423s |       152.34
      10 |  100.0% |       2.67s |    0.489s |       145.78
      15 |   99.0% |       3.21s |    0.567s |       138.92
```

### 3. 接口对比测试

```bash
# 分别测试两种接口性能
# Anthropic 接口
python api_performance_tester.py --key your_api_key_here --min 10 --max 30 --step 10 --rounds 2 > anthropic_results.txt

# Chat API 接口
python api_performance_tester.py --key your_api_key_here --min 10 --max 30 --step 10 --rounds 2 --chat-api > chat_results.txt

# 对比结果
echo "=== Anthropic 接口 ==="
cat anthropic_results.txt | grep -A 10 "测试汇总报告"

echo -e "\n=== Chat API 接口 ==="
cat chat_results.txt | grep -A 10 "测试汇总报告"
```

## 注意事项

1. **API Key 安全**: 
   - 推荐使用命令行参数 `--key` 传入 API Key
   - 请勿将包含真实 API Key 的代码提交到版本控制系统
   - 可以通过环境变量等方式安全地传递 API Key

2. **配额限制**: 注意您的 API 配额，避免过度消耗

3. **网络环境**: 确保网络连接稳定，以免影响测试结果

4. **超时设置**: 流式请求建议设置较长的超时时间（≥60秒）

5. **性能考虑**: 
   - 高并发测试可能会消耗大量系统资源
   - 建议根据机器性能调整并发级别

6. **接口差异**: 
   - Anthropic 接口使用 `x-api-key` 认证，Chat API 使用 `Authorization: Bearer` 认证
   - 两种接口的响应格式不同，工具会自动识别并解析
   - Chat API 的 token 统计字段为 `completion_tokens`，Anthropic 接口为 `output_tokens`

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.1.0
- 新增 Chat API 接口支持
- 添加 `--chat-api` 命令行参数
- 自动识别并解析不同接口格式的响应
- 支持两种认证方式的自动切换

### v1.0.0
- 初始版本发布
- 支持 SSE 流式请求测试
- 实现基本的性能指标统计
- 添加灵活的配置选项