"""
点击验证码 LLM 解算器

使用硅基流动 Qwen 多模态大模型识别验证码中的图标位置，按参考图标顺序返回点击坐标。

策略:
1. 下载参考图标条，按三等分裁剪为3个独立图标
2. 下载主图，将所有图片以 data URI 发送（确保 LLM 能访问）
3. 单次 API 调用，一次找到所有3个图标（避免重复匹配同一位置）
"""

import base64
import io
import json
import logging
import os
import re
from typing import List, Optional, Tuple

import requests
from PIL import Image
from openai import OpenAI

import const

logger = logging.getLogger(__name__)


class LLMFatalError(Exception):
    """大模型侧不可恢复的错误（余额不足 / 鉴权失败 / 模型下线）。

    这类错误重试多少次都不会变好：继续刷验证码只会白白耗掉本次登录机会，
    还会让国网看到更多次失败登录。上层收到后必须立刻中止本轮任务。
    """


def _is_fatal_llm_error(exc: Exception) -> bool:
    """判断异常是否属于“重试也没用”的大模型错误。"""
    status = getattr(exc, "status_code", None)
    if status in (401, 402, 403):
        return True
    text = str(exc)
    for pat in ("402", "401 ", "403", "insufficient", "balance",
                "Invalid token", "invalid_api_key", "Unauthorized",
                "Model disabled", "model_not_found", "does not exist"):
        if pat.lower() in text.lower():
            return True
    return False


class ClickCaptchaSolver:
    """基于大模型的点击验证码解算器。"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv('LLM_API_KEY') or const.LLM_API_KEY
        self.model = model or os.getenv('LLM_MODEL') or const.LLM_MODEL
        self.base_url = base_url or os.getenv('LLM_BASE_URL') or const.LLM_BASE_URL
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise LLMFatalError("LLM_API_KEY 未设置，无法识别验证码")
            # 超时保护：模型响应偶发 3~6 分钟（含思考链 / 上游 503 重试），
            # 不设上限会把整次登录的时间预算耗光。默认 90 秒，超时即放弃本张图，
            # 由上层刷新验证码重试。
            llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key,
                timeout=llm_timeout, max_retries=0)
        return self._client

    def solve(self, ref_url: str, main_url: str,
              main_width: int, main_height: int) -> List[Tuple[int, int]]:
        """下载图片 → 拆分参考条 → 单次LLM调用找到所有3个图标。"""
        # 1. 下载参考图标条并拆分为3个独立图标
        ref_raw = self._download(ref_url)
        if not ref_raw:
            return []
        icon_uris = self._split_strip(ref_raw)
        if len(icon_uris) < 3:
            return []

        # 2. 下载主图并转为 data URI（确保LLM能访问）
        main_raw = self._download(main_url)
        if not main_raw:
            return []
        main_uri = "data:image/png;base64," + base64.b64encode(main_raw).decode("ascii")

        # 3. 单次调用找到所有图标
        coords = self._find_all_icons(icon_uris, main_uri, main_width, main_height)
        if len(coords) < 2:
            return []

        # 钳制到主图范围内
        return [
            (max(0, min(x, main_width - 1)), max(0, min(y, main_height - 1)))
            for x, y in coords
        ]

    def _download(self, url: str) -> Optional[bytes]:
        """下载图片，支持 http 和 data URI。"""
        try:
            if url.startswith("data:"):
                _, encoded = url.split(",", 1)
                try:
                    return base64.b64decode(encoded)
                except Exception:
                    return base64.b64decode(__import__('urllib.parse').unquote(encoded))
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.content
            logger.error(f"下载失败: HTTP {resp.status_code}")
            return None
        except Exception as e:
            logger.error(f"下载错误: {e}")
            return None

    def _split_strip(self, raw: bytes) -> List[str]:
        """将参考图标条三等分为独立图标的 data URI。"""
        try:
            img = Image.open(io.BytesIO(raw))
            w, h = img.size
            logger.info(f"参考图标条: {w}x{h}")

            part_w = w // 3
            uris = []
            for i in range(3):
                left = i * part_w
                right = (i + 1) * part_w if i < 2 else w
                icon = img.crop((left, 0, right, h))
                # 放大图标以便LLM看清细节
                icon = icon.resize((icon.width * 3, icon.height * 3), Image.LANCZOS)
                buf = io.BytesIO()
                icon.save(buf, format="PNG")
                uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                uris.append(uri)
                logger.info(f"图标 #{i + 1}: {icon.width}x{icon.height}")
            return uris
        except Exception as e:
            logger.error(f"拆分错误: {e}")
            return []

    def _find_all_icons(self, icon_uris: List[str], main_uri: str,
                        main_width: int, main_height: int) -> List[Tuple[int, int]]:
        """单次API调用，让LLM找到所有3个图标。"""
        prompt = (
            f"大图（{main_width}×{main_height}像素）是一个图标网格。\n"
            "找到3个参考图标(A, B, C)各自在大图网格中的位置。\n"
            "匹配规则：形状和颜色必须一致，空心/实心、线条粗细是关键区分点，允许旋转。\n\n"
            '输出JSON：{"coords":[[xA,yA],[xB,yB],[xC,yC]]}\n'
            "其中x、y为图标中心的比例坐标（0~1）。"
        )

        content = []
        labels = ["A", "B", "C"]
        for i, uri in enumerate(icon_uris[:3]):
            content.append({"type": "image_url", "image_url": {"url": uri}})
            content.append({"type": "text", "text": f"参考图标{labels[i]}"})

        content.append({"type": "image_url", "image_url": {"url": main_uri}})
        content.append({"type": "text", "text": "Use only the visible captcha image above. Return center coordinates for A, B, C as JSON. Do not use page or hidden DOM coordinates."})
        content.append({"type": "text", "text": prompt})

        params = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": "Output valid JSON only. No markdown, no explanation."},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        try:
            # Qwen3 系列默认会输出大段“思考链”，单张图能拖到 3~6 分钟；
            # 关闭思考后通常 3~10 秒返回。部分模型不认这个参数，则退回普通调用。
            try:
                response = self.client.chat.completions.create(
                    **params, extra_body={"enable_thinking": False})
            except Exception as inner:
                if "enable_thinking" in str(inner) or "400" in str(inner):
                    logger.warning("模型不接受 enable_thinking 参数，改用普通调用")
                    response = self.client.chat.completions.create(**params)
                else:
                    raise
            output = response.choices[0].message.content or ""
            logger.info(f"大模型响应: {output[:400]}")
            return self._parse_coordinates(output, main_width, main_height)
        except Exception as e:
            if _is_fatal_llm_error(e):
                raise LLMFatalError(
                    f"验证码大模型不可用（{self.base_url} / {self.model}）：{e}"
                ) from e
            # 超时/503/网络抖动都属于“这张图没解出来”，返回空让上层刷新验证码重试；
            # 不要抛 RuntimeError（上层会当成致命错误直接退出进程）。
            logger.error(f"大模型错误（本张验证码放弃，将刷新重试）: {e}")
            return []

    def _parse_coordinates(self, text: str,
                           main_width: int, main_height: int) -> List[Tuple[int, int]]:
        """从LLM返回文本中提取JSON坐标并转为像素。"""
        # 优先尝试JSON解析
        match = re.search(r'\{.*"coords"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                result = []
                for x, y in data["coords"]:
                    x, y = float(x), float(y)
                    if max(x, y) <= 1.5:
                        result.append((round(x * main_width), round(y * main_height)))
                    else:
                        result.append((round(x), round(y)))
                return result
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        # 回退：正则解析
        coords = []
        paren_pairs = re.findall(r'\(\s*(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*)\s*\)', text)
        for x_str, y_str in paren_pairs:
            x, y = float(x_str), float(y_str)
            coords.append((x, y))

        if not coords:
            # 宽松匹配
            nums = re.findall(r'(\d+\.?\d+)', text)
            for i in range(0, len(nums) - 1, 2):
                coords.append((float(nums[i]), float(nums[i + 1])))

        result = []
        for x, y in coords[:3]:
            max_val = max(x, y)
            if max_val <= 1.5:
                result.append((round(x * main_width), round(y * main_height)))
            else:
                result.append((round(x), round(y)))
        return result


def probe_llm() -> None:
    """登录前的轻量探活（1 token 请求）。

    只拦“重试也不会好”的错误（余额不足 / 密钥失效 / 模型下线），直接抛
    LLMFatalError，让上层在**打开浏览器之前**就放弃本轮，避免白白登录一次、
    平白增加风控暴露面。网络抖动/超时等一律放行，交给正常流程重试。
    """
    solver = ClickCaptchaSolver()
    try:
        solver.client.chat.completions.create(
            model=solver.model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
        )
        logger.info(f"验证码大模型探活成功（{solver.model} @ {solver.base_url}）")
    except Exception as e:  # noqa: BLE001
        if _is_fatal_llm_error(e):
            raise LLMFatalError(
                f"验证码大模型不可用（{solver.base_url} / {solver.model}）：{e}"
            ) from e
        logger.warning(f"验证码大模型探活未成功（按可恢复处理，继续登录）：{e}")
