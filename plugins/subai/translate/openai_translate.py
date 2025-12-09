import time
from typing import List, Union

from openai import OpenAI
from cacheout import Cache

OpenAISessionCache = Cache(maxsize=100, ttl=3600, timer=time.time, default=None)


class OpenAi:
    _api_key: str = None
    _api_url: str = None
    _model: str = "gpt-3.5-turbo"

    def __init__(self, api_key: str = None, api_url: str = None, proxy: dict = None, model: str = None):
        self._api_key = api_key
        self._api_url = api_url
        
        # 处理代理设置
        http_client = None
        if proxy and proxy.get("https"):
            import httpx
            # 兼容不同版本的httpx
            try:
                # 新版本httpx使用proxies参数
                http_client = httpx.Client(proxies=proxy)
            except TypeError:
                try:
                    # 旧版本httpx使用不同的参数格式
                    http_client = httpx.Client(proxy=proxy.get("https"))
                except TypeError:
                    # 如果两种方式都不支持，则不使用代理
                    http_client = httpx.Client()
        
        # 处理base_url，避免重复添加/v1
        base_url = None
        if self._api_url:
            if self._api_url.endswith("/v1"):
                base_url = self._api_url
            else:
                base_url = self._api_url + "/v1"
        
        self._client = OpenAI(
            base_url=base_url,
            api_key=self._api_key,
            http_client=http_client
        )
        if model:
            self._model = model

    @staticmethod
    def __save_session(session_id: str, message: str):
        """
        保存会话
        :param session_id: 会话ID
        :param message: 消息
        :return:
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "assistant",
                "content": message
            })
            OpenAISessionCache.set(session_id, seasion)

    @staticmethod
    def __get_session(session_id: str, message: str) -> List[dict]:
        """
        获取会话
        :param session_id: 会话ID
        :return: 会话上下文
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "user",
                "content": message
            })
        else:
            seasion = [
                {
                    "role": "system",
                    "content": "请在接下来的对话中请使用中文回复，并且内容尽可能详细。"
                },
                {
                    "role": "user",
                    "content": message
                }]
            OpenAISessionCache.set(session_id, seasion)
        return seasion

    def __get_model(self, message: Union[str, List[dict]],
                    prompt: str = None,
                    user: str = "MoviePilot",
                    **kwargs):
        """
        获取模型
        """
        if not isinstance(message, list):
            if prompt:
                message = [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            else:
                message = [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
        return self._client.chat.completions.create(
            model=self._model,
            messages=message,
            **kwargs
        )

    @staticmethod
    def __clear_session(session_id: str):
        """
        清除会话
        :param session_id: 会话ID
        :return:
        """
        if OpenAISessionCache.get(session_id):
            OpenAISessionCache.delete(session_id)

    def translate_to_zh(self, text: str, context: str = None):
        """
        翻译为中文
        :param text: 输入文本
        :param context: 翻译上下文
        """
        system_prompt = """您是一位专业字幕翻译专家，请严格遵循以下规则：
1. 语义精准：必须准确传达原文含义，不得增删、曲解或主观发挥；
2. 语言风格：使用自然、地道的简体中文口语表达，符合影视字幕的节奏与观影习惯；
3. 上下文协同：允许参考上下文（包括前后行）以确保人物称谓、专有名词、情感语气、时态逻辑等在本行译文中保持一致与合理，但仅输出当前行的译文，不得引入其他行内容；
4. 逐行对应：输入多少行，必须输出完全相同数量的译文行，每行译文严格对应输入的同一行；
5. 格式纯净：输出内容仅包含译文文本，禁止任何额外文字（如序号、说明、注释、空行、开场白、总结等）；
6. 不可合并或拆分：无论原文长短、标点或语义是否完整，均不得将多行合并为一行，也不得将一行拆分为多行；
7. 无法翻译处理：若某行原文因缺失、乱码、非语言内容等原因无法翻译，必须在对应位置原样输出“[未翻译]”，不得留空、跳过或替换为其他占位符；
8. 术语统一：同一术语、角色名、地点名等在全文中须保持一致，首次出现后不得随意更改；
9. 文化适配：在不改变原意前提下，可对文化特定表达进行必要本地化，但不得过度意译或添加原文没有的信息；
10.标点规范：使用中文全角标点，符合中文书写习惯，但需保留原文的语气强度（如感叹、疑问、停顿等）。"""
        user_prompt = f"翻译上下文：\n{context}\n\n需要翻译的内容：\n{text}" if context else f"请翻译：\n{text}"
        result = ""
        try:
            completion = self.__get_model(prompt=system_prompt,
                                          message=user_prompt,
                                          temperature=0.2,
                                          top_p=0.9)
            result = completion.choices[0].message.content.strip()
            return True, result
        except Exception as e:
            print(f"{str(e)}：{result}")
            return False, f"{str(e)}：{result}"