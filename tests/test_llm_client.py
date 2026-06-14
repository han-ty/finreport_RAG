"""
LLMClient 单元测试
覆盖所有方法的正常路径、异常路径和边界条件
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from test.llm_client import LLMClient
from backend.core.config import AppConfig, LLMConfig


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_llm_configs():
    """构造测试用 LLM 配置列表"""
    return [
        LLMConfig(
            name="model-a",
            base_url="https://api.a.com/v1/",
            api_key="sk-test-a",
            model="model-a-v1",
            temperature=0.7,
            top_p=0.9,
            max_tokens=4096,
            weight=1.0,
            enabled=True,
        ),
        LLMConfig(
            name="model-b",
            base_url="https://api.b.com/v1/",
            api_key="sk-test-b",
            model="model-b-v1",
            temperature=0.5,
            top_p=0.8,
            max_tokens=2048,
            weight=2.0,
            enabled=True,
        ),
        LLMConfig(
            name="model-c-disabled",
            base_url="https://api.c.com/v1/",
            api_key="sk-test-c",
            model="model-c-v1",
            enabled=False,
        ),
    ]


@pytest.fixture
def app_config(sample_llm_configs):
    """构造测试用 AppConfig"""
    config = AppConfig()
    config.llm_configs = sample_llm_configs
    config.agent_llm_indices = [0, 1]
    config.other_llm_indices = [0]
    config.max_concurrent_requests = 10
    return config


@pytest.fixture
def llm_client(app_config):
    """构造 LLMClient 实例（AsyncOpenAI 创建时不做网络请求，安全）"""
    return LLMClient(app_config)


@pytest.fixture
def mock_async_client():
    """创建一个 AsyncMock 模拟的 AsyncOpenAI 客户端"""
    client = AsyncMock()
    return client


# ═══════════════════════════════════════════════════════════════════════
# _repair_json 静态方法测试
# ═══════════════════════════════════════════════════════════════════════

class TestRepairJson:
    """_repair_json 纯逻辑测试 — 无需任何 mock"""

    def test_valid_json_passes_through(self):
        """有效 JSON 直接解析"""
        result = LLMClient._repair_json('{"a": 1}')
        assert result == {"a": 1}

    def test_surrounding_text_extracts_json(self):
        """从周围文本中提取 JSON 对象"""
        result = LLMClient._repair_json('some prefix {"a": 1} some suffix')
        assert result == {"a": 1}

    def test_missing_closing_brace_auto_closes(self):
        """截断的 JSON — 自动补全缺失的 }"""
        result = LLMClient._repair_json('{"a": 1')
        assert result == {"a": 1}

    def test_missing_closing_bracket_auto_closes(self):
        """截断的 JSON — 自动补全缺失的 ]"""
        result = LLMClient._repair_json('{"a": [1, 2')
        assert result == {"a": [1, 2]}

    def test_missing_both_braces_and_brackets(self):
        """同时缺失 } 和 ] — 按正确顺序补全"""
        result = LLMClient._repair_json('{"a": [1, 2')
        assert result == {"a": [1, 2]}

    def test_trailing_comma_in_object(self):
        """对象末尾多余逗号需要移除"""
        result = LLMClient._repair_json('{"a": 1,}')
        assert result == {"a": 1}

    def test_trailing_comma_in_array(self):
        """数组末尾多余逗号需要移除"""
        result = LLMClient._repair_json('{"a": [1, 2,]}')
        assert result == {"a": [1, 2]}

    def test_no_braces_returns_none(self):
        """没有花括号返回 None"""
        result = LLMClient._repair_json('no json here')
        assert result is None

    def test_empty_string_returns_none(self):
        """空字符串返回 None"""
        result = LLMClient._repair_json('')
        assert result is None

    def test_nested_object_missing_outer_brace(self):
        """嵌套对象缺失外层 } — 正确补全"""
        result = LLMClient._repair_json('{"a": {"b": 1}')
        assert result == {"a": {"b": 1}}

    def test_string_containing_braces(self):
        """字符串值中包含花括号字符"""
        result = LLMClient._repair_json('{"a": "text {inside} braces"}')
        assert result == {"a": "text {inside} braces"}

    def test_truncated_key_value_pair(self):
        """截断的 key-value 对被移除后再补全"""
        result = LLMClient._repair_json('{"a": 1, "b')
        assert result == {"a": 1}

    def test_truncated_key_with_colon(self):
        """截断的 key: 被移除"""
        result = LLMClient._repair_json('{"a": 1, "b":')
        assert result == {"a": 1}

    def test_array_with_multiple_objects_finds_first_object(self):
        """数组中含多个对象时，提取第一个匹配的 {...} 对象"""
        # _repair_json 从第一个 { 开始找匹配的 }，只提取第一个 JSON 对象
        result = LLMClient._repair_json('[{"x": 1}, {"y": 2}')
        assert result == {"x": 1}

    def test_complex_nested_structure(self):
        """嵌套对象缺失一个 } — 自动补全"""
        result = LLMClient._repair_json(
            '{"outer": {"inner": [1, 2, 3]}'
        )
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_escaped_quote_in_string(self):
        """字符串中的转义引号正确处理"""
        # \" 在 JSON 中表示转义引号，不应触发字符串结束
        result = LLMClient._repair_json('{"msg": "hello \\"world\\""}')
        assert result == {"msg": 'hello "world"'}

    def test_single_brace_object(self):
        """只有一对花括号的最简对象"""
        result = LLMClient._repair_json('{}')
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════
# _select_llm 逻辑测试
# ═══════════════════════════════════════════════════════════════════════

class TestSelectLLM:
    """_select_llm 测试 — 不依赖 API"""

    def test_selects_from_agent_indices(self, llm_client, sample_llm_configs):
        """prefer_agent=True 时从 agent_llm_indices 中选择"""
        with patch("random.choices") as mock_choices:
            mock_choices.return_value = [sample_llm_configs[0]]
            config, client = llm_client._select_llm(prefer_agent=True)

        assert config.name == "model-a"
        assert len(mock_choices.call_args[0][0]) == 2  # 两个候选
        # 权重应为 [1.0, 2.0]
        assert mock_choices.call_args[1]["weights"] == [1.0, 2.0]

    def test_selects_from_other_indices(self, llm_client, sample_llm_configs):
        """prefer_agent=False 时从 other_llm_indices 中选择"""
        with patch("random.choices") as mock_choices:
            mock_choices.return_value = [sample_llm_configs[0]]
            config, client = llm_client._select_llm(prefer_agent=False)

        assert config.name == "model-a"
        # other_llm_indices=[0]，只有一个候选
        assert len(mock_choices.call_args[0][0]) == 1

    def test_fallback_when_agent_index_is_disabled(self, app_config):
        """agent_llm_indices 指向禁用模型时回退到所有启用的"""
        # model-c-disabled 在 index 2，设为 agent 首选
        app_config.agent_llm_indices = [2]
        llm = LLMClient(app_config)

        with patch("random.choices") as mock_choices:
            mock_choices.return_value = [app_config.llm_configs[0]]
            config, client = llm._select_llm(prefer_agent=True)

        # 应该回退到所有启用的（model-a, model-b）
        candidates = mock_choices.call_args[0][0]
        assert len(candidates) == 2
        assert all(c.enabled for c in candidates)

    def test_raises_when_all_disabled(self, app_config):
        """所有 LLM 被禁用时抛出 ValueError"""
        for c in app_config.llm_configs:
            c.enabled = False
        llm = LLMClient(app_config)

        with pytest.raises(ValueError, match="没有可用的LLM API配置"):
            llm._select_llm()

    def test_fallback_when_index_out_of_range(self, app_config):
        """索引超出范围时回退"""
        app_config.agent_llm_indices = [99]  # 不存在的索引
        llm = LLMClient(app_config)

        with patch("random.choices") as mock_choices:
            mock_choices.return_value = [app_config.llm_configs[0]]
            config, client = llm._select_llm()

        # 应回退到所有启用的
        candidates = mock_choices.call_args[0][0]
        assert len(candidates) == 2

    def test_returns_client_from_clients_dict(self, llm_client):
        """返回的 client 来自 self.clients 字典"""
        with patch("random.choices") as mock_choices:
            mock_choices.return_value = [llm_client.config.llm_configs[0]]
            config, client = llm_client._select_llm()

        # client 应该与 self.clients 中的一致
        assert client is llm_client.clients["model-a"]


# ═══════════════════════════════════════════════════════════════════════
# chat 方法测试
# ═══════════════════════════════════════════════════════════════════════

def _make_chat_response(content: str):
    """构造模拟的 OpenAI chat completion 响应"""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestChat:
    """chat 异步方法测试 — mock AsyncOpenAI"""

    @pytest.mark.asyncio
    async def test_successful_call_returns_content(self, llm_client, mock_async_client):
        """正常调用返回响应文本"""
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=_make_chat_response("测试回复")
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            result = await llm_client.chat(
                messages=[{"role": "user", "content": "你好"}],
            )

        assert result == "测试回复"

    @pytest.mark.asyncio
    async def test_passes_temperature_and_max_tokens(self, llm_client, mock_async_client):
        """验证自定义 temperature 和 max_tokens 被传入 API"""
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=_make_chat_response("ok")
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            await llm_client.chat(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.3,
                max_tokens=100,
            )

        call_kwargs = mock_async_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_json_mode_adds_response_format(self, llm_client, mock_async_client):
        """json_mode=True 时传入 response_format 参数"""
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=_make_chat_response('{"key": "val"}')
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            await llm_client.chat(
                messages=[{"role": "user", "content": "输出JSON"}],
                json_mode=True,
            )

        call_kwargs = mock_async_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_retry_on_timeout_then_succeed(self, llm_client, mock_async_client):
        """超时一次后重试成功"""
        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=[
                asyncio.TimeoutError(),
                _make_chat_response("重试成功"),
            ]
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            result = await llm_client.chat(
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == "重试成功"
        assert mock_async_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_error_then_succeed(self, llm_client, mock_async_client):
        """普通异常一次后重试成功"""
        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("网络错误"),
                _make_chat_response("重试成功"),
            ]
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            result = await llm_client.chat(
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == "重试成功"
        assert mock_async_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises(self, llm_client, mock_async_client):
        """3 次全部失败后抛出异常"""
        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            with pytest.raises(asyncio.TimeoutError):
                await llm_client.chat(
                    messages=[{"role": "user", "content": "hi"}],
                )

        assert mock_async_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_extra_kwargs_passed_through(self, llm_client, mock_async_client):
        """额外 kwargs 透传到 API 调用"""
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=_make_chat_response("ok")
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            await llm_client.chat(
                messages=[{"role": "user", "content": "hi"}],
                extra_param="extra_value",
            )

        call_kwargs = mock_async_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_param"] == "extra_value"


# ═══════════════════════════════════════════════════════════════════════
# chat_stream 方法测试
# ═══════════════════════════════════════════════════════════════════════

class TestChatStream:
    """chat_stream 异步生成器测试"""

    @pytest.mark.asyncio
    async def test_yields_content_chunks(self, llm_client, mock_async_client):
        """验证流式输出逐块 yield 内容"""
        # 构造流式 chunk（使用 delta.content 而非 message.content）
        def _make_stream_chunk(content):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
            return chunk

        chunks = [
            _make_stream_chunk("Hello"),
            _make_stream_chunk(" World"),
            _make_stream_chunk("!"),
        ]

        class MockStream:
            def __aiter__(self):
                self._iter = iter(chunks)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        mock_async_client.chat.completions.create = AsyncMock(
            return_value=MockStream()
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            results = []
            async for chunk in llm_client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
            ):
                results.append(chunk)

        assert results == ["Hello", " World", "!"]

    @pytest.mark.asyncio
    async def test_stream_with_custom_temperature(self, llm_client, mock_async_client):
        """流式调用传入自定义参数"""
        # 构造流式 chunk（使用 delta.content）
        def _make_stream_chunk(content):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
            return chunk
        chunks = [_make_stream_chunk("ok")]

        # 包装成 async iterable（普通 list 不支持 async for）
        class MockStream:
            def __init__(self, items):
                self._items = items
            def __aiter__(self):
                self._iter = iter(self._items)
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        mock_async_client.chat.completions.create = AsyncMock(
            return_value=MockStream(chunks)
        )

        with patch.object(
            llm_client, "_select_llm",
            return_value=(llm_client.config.llm_configs[0], mock_async_client),
        ):
            async for _ in llm_client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
            ):
                pass

        call_kwargs = mock_async_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["stream"] is True


# ═══════════════════════════════════════════════════════════════════════
# query 方法测试
# ═══════════════════════════════════════════════════════════════════════

class TestQuery:
    """query 方法测试 — 验证消息构造"""

    @pytest.mark.asyncio
    async def test_basic_query_wraps_prompt(self, llm_client, mock_async_client):
        """query 将 prompt 包装为 user 消息"""
        expected_response = "回答内容"

        with patch.object(
            llm_client, "chat", AsyncMock(return_value=expected_response),
        ) as mock_chat:
            result = await llm_client.query("今天天气怎么样？")

        assert result == expected_response
        # query 内部调用 self.chat(messages, **kwargs)，messages 是位置参数
        messages = mock_chat.call_args.args[0]
        assert messages == [{"role": "user", "content": "今天天气怎么样？"}]

    @pytest.mark.asyncio
    async def test_query_with_system_message(self, llm_client, mock_async_client):
        """带 system 消息的 query"""
        with patch.object(llm_client, "chat", AsyncMock(return_value="ok")) as mock_chat:
            await llm_client.query(
                prompt="查询",
                system="你是一个有用的助手",
            )

        messages = mock_chat.call_args.args[0]
        assert messages == [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "查询"},
        ]

    @pytest.mark.asyncio
    async def test_query_passes_kwargs_to_chat(self, llm_client, mock_async_client):
        """query 将额外参数透传到 chat"""
        with patch.object(llm_client, "chat", AsyncMock(return_value="ok")) as mock_chat:
            await llm_client.query(
                prompt="test",
                prefer_agent=False,
                temperature=0.1,
                max_tokens=50,
            )

        assert mock_chat.call_args.kwargs["prefer_agent"] is False
        assert mock_chat.call_args.kwargs["temperature"] == 0.1
        assert mock_chat.call_args.kwargs["max_tokens"] == 50


# ═══════════════════════════════════════════════════════════════════════
# query_json 方法测试
# ═══════════════════════════════════════════════════════════════════════

class TestQueryJson:
    """query_json 测试 — JSON 解析和修复"""

    @pytest.mark.asyncio
    async def test_parses_valid_json(self, llm_client):
        """有效 JSON 直接解析"""
        with patch.object(
            llm_client, "query", AsyncMock(return_value='{"result": 42}'),
        ):
            result = await llm_client.query_json("返回JSON")

        assert result == {"result": 42}

    @pytest.mark.asyncio
    async def test_extracts_json_from_markdown_block(self, llm_client):
        """从 markdown 代码块中提取 JSON"""
        markdown_response = '以下是结果：\n```json\n{"name": "test"}\n```\n完毕'
        with patch.object(
            llm_client, "query", AsyncMock(return_value=markdown_response),
        ):
            result = await llm_client.query_json("返回JSON")

        assert result == {"name": "test"}

    @pytest.mark.asyncio
    async def test_extracts_json_from_code_block_no_lang(self, llm_client):
        """从无语言标记的代码块中提取 JSON"""
        response = '```\n{"items": [1,2,3]}\n```'
        with patch.object(
            llm_client, "query", AsyncMock(return_value=response),
        ):
            result = await llm_client.query_json("返回JSON")

        assert result == {"items": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_repairs_broken_json(self, llm_client):
        """通过 _repair_json 修复损坏的 JSON"""
        with patch.object(
            llm_client, "query", AsyncMock(return_value='{"a": 1'),
        ):
            result = await llm_client.query_json("返回JSON")

        assert result == {"a": 1}

    @pytest.mark.asyncio
    async def test_raises_on_unrepairable(self, llm_client):
        """无法修复时抛出 JSONDecodeError"""
        with patch.object(
            llm_client, "query", AsyncMock(return_value="这不是JSON"),
        ):
            with pytest.raises(json.JSONDecodeError):
                await llm_client.query_json("返回JSON")

    @pytest.mark.asyncio
    async def test_repairs_truncated_in_code_block(self, llm_client):
        """markdown 代码块中截断的 JSON 也会被修复"""
        response = '```json\n{"items": [1, 2\n```'
        with patch.object(
            llm_client, "query", AsyncMock(return_value=response),
        ):
            result = await llm_client.query_json("返回JSON")

        assert result == {"items": [1, 2]}


# ═══════════════════════════════════════════════════════════════════════
# batch_query 方法测试
# ═══════════════════════════════════════════════════════════════════════

class TestBatchQuery:
    """batch_query 批量并发测试"""

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, llm_client):
        """批量查询并发执行"""
        async def mock_query(prompt, **kwargs):
            await asyncio.sleep(0.001)
            return f"response: {prompt}"

        with patch.object(llm_client, "query", side_effect=mock_query):
            results = await llm_client.batch_query(
                prompts=["p1", "p2", "p3"],
            )

        assert len(results) == 3
        assert results == [
            "response: p1",
            "response: p2",
            "response: p3",
        ]

    @pytest.mark.asyncio
    async def test_respects_semaphore_limit(self, llm_client):
        """验证信号量限制并发数"""
        concurrent_count = 0
        max_concurrent = 0

        async def mock_query(prompt, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.001)
            concurrent_count -= 1
            return prompt

        with patch.object(llm_client, "query", side_effect=mock_query):
            await llm_client.batch_query(
                prompts=[f"p{i}" for i in range(10)],
                max_concurrent=2,  # 限制为 2
            )

        # 最大并发数不应超过 2（允许少量误差）
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_exceptions_returned_not_raised(self, llm_client):
        """单个查询异常时，异常作为结果返回而不是中断整个批次"""
        async def mock_query(prompt, **kwargs):
            if prompt == "fail":
                raise ValueError("查询失败")
            return f"ok: {prompt}"

        with patch.object(llm_client, "query", side_effect=mock_query):
            results = await llm_client.batch_query(
                prompts=["ok1", "fail", "ok2"],
            )

        assert results[0] == "ok: ok1"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok: ok2"

    @pytest.mark.asyncio
    async def test_passes_system_message(self, llm_client):
        """system 参数被传递给每个子查询"""
        with patch.object(llm_client, "query", AsyncMock(return_value="ok")) as mock_query:
            await llm_client.batch_query(
                prompts=["p1", "p2"],
                system="系统提示词",
            )

        # 验证 system 参数传递
        for call in mock_query.call_args_list:
            assert call.kwargs["system"] == "系统提示词"
