"""
大模型客户端模块
支持多API提供商的异步调用，负载均衡，重试机制
"""
import asyncio
import logging
import random
import json
import re
from typing import Optional, Dict, Any, List, Union
from openai import AsyncOpenAI

from .config import AppConfig, LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的大模型客户端"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.clients: Dict[str, AsyncOpenAI] = {}
        self._init_clients()

    def _init_clients(self):
        """初始化所有启用的LLM客户端"""
        import httpx
        for llm_config in self.config.get_enabled_llms():
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=30.0),
                limits=httpx.Limits(max_keepalive_connections=2, max_connections=10),
                trust_env=False,
            )
            self.clients[llm_config.name] = AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                http_client=http_client,
            )

    def _select_llm(self, prefer_agent: bool = True) -> tuple:
        """选择LLM配置（加权随机）"""
        if prefer_agent:
            indices = self.config.agent_llm_indices
        else:
            indices = self.config.other_llm_indices

        candidates = []
        for idx in indices:
            if idx < len(self.config.llm_configs):
                c = self.config.llm_configs[idx]
                if c.enabled and c.name in self.clients:
                    candidates.append(c)

        if not candidates:
            candidates = [c for c in self.config.llm_configs
                          if c.enabled and c.name in self.clients]

        if not candidates:
            raise ValueError("没有可用的LLM API配置")

        # 加权随机选择
        weights = [c.weight for c in candidates]
        selected = random.choices(candidates, weights=weights, k=1)[0]
        return selected, self.clients[selected.name]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        prefer_agent: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """
        发送对话请求
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            prefer_agent: 是否优先使用Agent LLM
            temperature: 温度参数
            max_tokens: 最大token数
            json_mode: 是否要求JSON格式输出
        Returns:
            模型回复文本
        """
        llm_config, client = self._select_llm(prefer_agent)

        params = {
            "model": llm_config.model,
            "messages": messages,
            "temperature": temperature or llm_config.temperature,
            "top_p": llm_config.top_p,
            "max_tokens": max_tokens or llm_config.max_tokens,
        }

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        params.update(kwargs)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**params),
                    timeout=90,
                )
                result = response.choices[0].message.content
                logger.debug(f"LLM [{llm_config.name}] 回复成功, 长度: {len(result)}")
                return result
            except asyncio.TimeoutError:
                logger.warning(f"LLM [{llm_config.name}] 第{attempt+1}次调用超时(90s)")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
            except Exception as e:
                err_name = type(e).__name__
                logger.warning(f"LLM [{llm_config.name}] 第{attempt+1}次调用失败: {err_name}: {e}")
                # Recreate HTTP client on connection errors (fixes Windows httpx connection issues)
                if 'Connection' in err_name or 'ConnectError' in err_name:
                    import httpx
                    logger.info(f"重建 HTTP 客户端 [{llm_config.name}]")
                    new_http = httpx.AsyncClient(
                        timeout=httpx.Timeout(120.0, connect=30.0),
                        limits=httpx.Limits(max_keepalive_connections=2, max_connections=10),
                        trust_env=False,
                    )
                    self.clients[llm_config.name] = AsyncOpenAI(
                        api_key=llm_config.api_key,
                        base_url=llm_config.base_url,
                        http_client=new_http,
                    )
                    client = self.clients[llm_config.name]
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        prefer_agent: bool = True,
        **kwargs,
    ):
        """流式对话"""
        llm_config, client = self._select_llm(prefer_agent)

        params = {
            "model": llm_config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", llm_config.temperature),
            "top_p": llm_config.top_p,
            "max_tokens": kwargs.get("max_tokens", llm_config.max_tokens),
            "stream": True,
        }

        response = await client.chat.completions.create(**params)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def query(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """简化的单轮查询接口"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, **kwargs)

    async def query_json(self, prompt: str, system: Optional[str] = None, **kwargs) -> Any:
        """查询并解析JSON回复（含自动修复）"""
        result = await self.query(prompt, system=system, json_mode=True, **kwargs)
        # 尝试提取JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pass
        # 尝试从markdown代码块提取
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', result)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                result = match.group(1).strip()
        # 尝试修复JSON
        repaired = self._repair_json(result)
        if repaired is not None:
            return repaired
        raise json.JSONDecodeError("无法解析或修复JSON", result, 0)

    @staticmethod
    def _repair_json(text: str) -> Optional[Any]:
        """尝试修复损坏的JSON"""
        # 找到最外层 { 
        start = text.find('{')
        if start == -1:
            return None
        # 找匹配的 }
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            json_text = text[start:end + 1]
        else:
            # 截断输出 - 尝试关闭所有开放结构
            json_text = text[start:]
            # 移除最后一个不完整的key-value
            json_text = re.sub(r',\s*"[^"]*"?\s*:?\s*[^,}\]]*$', '', json_text)
            # 关闭所有开放括号
            open_braces = json_text.count('{') - json_text.count('}')
            open_brackets = json_text.count('[') - json_text.count(']')
            if open_brackets > 0:
                json_text += ']' * open_brackets
            if open_braces > 0:
                json_text += '}' * open_braces
        # 修复常见问题
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None

    async def batch_query(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """批量并发查询"""
        semaphore = asyncio.Semaphore(
            max_concurrent or self.config.max_concurrent_requests
        )

        async def _query_one(prompt):
            async with semaphore:
                return await self.query(prompt, system=system, **kwargs)

        tasks = [_query_one(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)



