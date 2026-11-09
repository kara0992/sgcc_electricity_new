"""
点击验证码 LLM 解算器

使用多模态大模型识别验证码中的图标位置，按参考图标顺序返回点击坐标。

策略:
1. 下载参考图标条，按三等分裁剪为3个独立图标
2. 下载主图，将所有图片以 data URI 发送（确保 LLM 能访问）
3. 单次 API 调用，一次找到所有3个图标（避免重复匹配同一位置）

LLM_MODEL 支持逗号分隔的模型列表（如 "glm-4.1v-thinking-flash,glm-4.6v-flash"），
按顺序尝试，前一个不可用（免费额度用完/模型拥堵）自动换下一个。
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
    """大模型侧不可恢复的错误（余额不足 / 鉴权失败 / 密钥失效）。

    这类错误重试多少次都不会变好：继续刷验证码只会白白耗掉本次登录机会，
    还会让国网看到更多次失败登录。上层收到后必须立刻中止本轮任务。
    """


# 密钥/账户级错误：换模型也没用
_FATAL_PATTERNS = (
    "401", "402", "403", "insufficient", "balance", "余额不足", "无可用资源包",
    "invalid token", "invalid_api_key", "unauthorized", "authentication",
    "api key", "apikey",
)

# 模型级错误：换下一个模型即可（免费模型经常拥堵/下线/无免费额度）
_MODEL_PATTERNS = (
    "429", "1305", "1302", "1303", "访问量过大", "rate limit", "too many requests",
    "model disabled", "model_not_found", "does not exist", "模型不存在", "1113",
    "无可用资源包",
)


def _msg(exc: Exception) -> str:
    return str(exc).lower()


def _is_fatal_llm_error(exc: Exception) -> bool:
    """判断异常是否属于“账户/密钥级、重试也没用”的错误。"""
    status = getattr(exc, "status_code", None)
    if status in (401, 402, 403):
        return True
    body = getattr(exc, "body", None)
    text = _msg(exc)
    if isinstance(body, dict):
        text += " " + str(body.get("code", "")) + " " + str(body.get("message", "")).lower()
    for pat in _FATAL_PATTERNS:
        if pat in text:
            return True
    return False


def _is_model_unavailable(exc: Exception) -> bool:
    """模型本身不可用（拥堵 / 下线 / 无免费额度）→ 换下一个模型。"""
    body = getattr(exc, "body", None)
    text = _msg(exc)
    if isinstance(body, dict):
        text += " " + str(body.get("code", "")) + " " + str(body.get("message", "")).lower()
    for pat in _MODEL_PATTERNS:
        if pat in text:
            return True
    return False


def _split_models(raw: Optional[str]) -> List[str]:
    models = [m.strip() for m in (raw or "").split(",") if m.strip()]
    return models or [raw or ""]


class ClickCaptchaSolver:
    """基于大模型的点击验证码解算器。"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv('LLM_API_KEY') or const.LLM_API_KEY
        self.model = model or os.getenv('LLM_MODEL') or const.LLM_MODEL
        self.base_url = base_url or os.getenv('LLM_BASE_URL') or const.LLM_BASE_URL
        self.models = _split_models(self.model)
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

    # ------------------------------------------------------------------
    # 验证码识别主流程
    # ------------------------------------------------------------------
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
        self._dump(ref_raw, main_raw)
        coords = self._find_all_icons(icon_uris, main_uri, main_width, main_height)
        if len(coords) < 2:
            return []

        # 钳制到主图范围内
        return [
            (max(0, min(x, main_width - 1)), max(0, min(y, main_height - 1)))
            for x, y in coords
        ]

    def _dump(self, ref_raw: bytes, main_raw: bytes) -> None:
        """把最近一次验证码图片留档到 /config/sgcc_captcha/，便于事后离线复盘。

        只保留最近 8 组（ref_*/main_*），不影响主流程；目录不可写时静默跳过。
        """
        import glob
        import time as _time

        dump_dir = os.getenv("CAPTCHA_DUMP_DIR", "/config/sgcc_captcha").strip()
        if not dump_dir:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            stamp = _time.strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(dump_dir, f"ref_{stamp}.png"), "wb") as f:
                f.write(ref_raw)
            with open(os.path.join(dump_dir, f"main_{stamp}.png"), "wb") as f:
                f.write(main_raw)
            olds = sorted(glob.glob(os.path.join(dump_dir, "main_*.png")))
            for p in olds[:-8]:
                os.remove(p)
                ref = os.path.join(dump_dir, "ref_" + os.path.basename(p)[5:])
                if os.path.exists(ref):
                    os.remove(ref)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"验证码留档失败（忽略）: {e}")

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

    def _build_messages(self, icon_uris: List[str], main_uri: str,
                        main_width: int, main_height: int) -> List[dict]:
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
        return [
            {"role": "system", "content": "Output valid JSON only. No markdown, no explanation."},
            {"role": "user", "content": content},
        ]

    def _find_all_icons(self, icon_uris: List[str], main_uri: str,
                        main_width: int, main_height: int) -> List[Tuple[int, int]]:
        """调用大模型找到所有3个图标；模型列表逐个降级，参数逐个降级。"""
        messages = self._build_messages(icon_uris, main_uri, main_width, main_height)
        params = dict(messages=messages, max_tokens=2048, temperature=0.1)
        # 不同供应商对可选参数的支持不一样（智谱认 response_format 且思考链单独返回、
        # 不认 enable_thinking；硅基流动认 enable_thinking）。按「特性从多到少」降级。
        variants = [
            dict(response_format={"type": "json_object"}),
            dict(response_format={"type": "json_object"}, extra_body={"enable_thinking": False}),
            dict(extra_body={"enable_thinking": False}),
            dict(),
        ]

        errors: List[Exception] = []
        for model in self.models:
            for idx, extra in enumerate(variants):
                try:
                    response = self.client.chat.completions.create(model=model, **params, **extra)
                except Exception as inner:  # noqa: BLE001
                    errors.append(inner)
                    if _is_model_unavailable(inner):
                        logger.warning(f"模型 {model} 暂不可用（{str(inner)[:120]}），换下一个模型")
                        break
                    if _is_fatal_llm_error(inner):
                        logger.error(f"模型 {model} 账户/密钥错误（{str(inner)[:120]}），换下一个模型")
                        break
                    logger.warning(f"[{model}] 第 {idx + 1} 种调用方式失败，降级重试：{inner}")
                    continue

                choice = response.choices[0]
                output = choice.message.content or ""
                logger.info(f"[{model}] 大模型响应({choice.finish_reason}): {output[:400]}")
                coords = self._parse_coordinates(output, main_width, main_height)
                if coords or choice.finish_reason != "length":
                    return coords
                logger.warning(f"[{model}] 输出被截断且未解析出坐标，降级重试")
            # 当前模型所有参数组合都失败 → 换下一个模型

        # 所有模型都失败：如果每条错误都是账户级（余额/密钥），本轮登录直接放弃
        if errors and all(_is_fatal_llm_error(e) for e in errors):
            raise LLMFatalError(
                f"验证码大模型不可用（{self.base_url} / {self.model}）：{errors[-1]}"
            ) from errors[-1]
        # 否则（拥堵/超时/网络抖动）当作“这张图没解出来”，返回空让上层刷新重试；
        # 不要抛 RuntimeError（上层会当成致命错误直接退出进程）。
        logger.error(f"所有模型均未成功（{self.model}）：{errors[-1] if errors else '无响应'}")
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

    只拦“重试也不会好”的错误（余额不足 / 密钥失效），直接抛 LLMFatalError，
    让上层在**打开浏览器之前**就放弃本轮，避免白白登录一次、平白增加风控暴露面。
    模型拥堵/网络抖动一律放行，交给正常流程重试。
    """
    solver = ClickCaptchaSolver()
    errors: List[Exception] = []
    for model in solver.models:
        try:
            solver.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
            )
            logger.info(f"验证码大模型探活成功（{model} @ {solver.base_url}）")
            return
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            logger.warning(f"探活 {model} 未成功：{str(e)[:150]}")

    if errors and all(_is_fatal_llm_error(e) for e in errors):
        raise LLMFatalError(
            f"验证码大模型不可用（{solver.base_url} / {solver.model}）：{errors[-1]}"
        ) from errors[-1]
    logger.warning(f"验证码大模型探活未成功（按可恢复处理，继续登录）：{errors[-1] if errors else '无响应'}")
