import base64
import io
import os
import json
import copy
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from PIL import Image


class _HostResolveAdapter(HTTPAdapter):
    """
    将指定 host 的连接目标改为固定 IP，但保持：
    - Host header 为原域名
    - TLS SNI(server_hostname) 与证书校验(assert_hostname) 仍为原域名
    """

    def __init__(self, host: str, dest_ip: str, is_https: bool, **kwargs):
        self.host = host
        self.dest_ip = dest_ip
        self.is_https = is_https
        # 注意：HTTPAdapter.__init__ 会调用 init_poolmanager，必须先设置 host/dest_ip
        super().__init__(**kwargs)

    def init_poolmanager(
        self, connections: int, maxsize: int, block: bool = False, **pool_kwargs
    ) -> None:
        # 仅 HTTPS 场景设置主机名验证；HTTP 连接不接受 assert_hostname/server_hostname
        if self.is_https:
            pool_kwargs.setdefault("assert_hostname", self.host)
            pool_kwargs.setdefault("server_hostname", self.host)
        else:
            pool_kwargs.pop("assert_hostname", None)
            # pool_kwargs.pop("server_hostname", None)
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def send(self, request, **kwargs):
        split = urlsplit(request.url)
        if split.hostname != self.host:
            return super().send(request, **kwargs)

        # 连接到指定 IP，但保留原 path/query
        port = split.port
        if port is None:
            port = 443 if split.scheme == "https" else 80

        new_netloc = f"{self.dest_ip}:{port}"
        request.url = urlunsplit(
            (split.scheme, new_netloc, split.path, split.query, split.fragment)
        )

        # Host header 仍然保持原域名（必要时带端口）
        host_header = self.host
        if (split.scheme == "https" and port != 443) or (
            split.scheme == "http" and port != 80
        ):
            host_header = f"{self.host}:{port}"
        request.headers["Host"] = host_header

        return super().send(request, **kwargs)


def mount_session(session: requests.Session, url: str, server_ip: str):
    split = urlsplit(url)
    if split.hostname:
        adapter = _HostResolveAdapter(
            host=split.hostname,
            dest_ip=server_ip,
            is_https=split.scheme == "https",
        )
        session.mount(f"{split.scheme}://{split.hostname}", adapter)


class ZnzmoClient:
    """
    Znzmo 自定义图像生成客户端。
    接口参数约定：
        model: 模型名称
        prompt: 提示词
        aspectRatio: 宽高比
        imageSize: 输出尺寸
        imageList: 参考图列表（base64 字符串）
        batchSize: 生成数量
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        server_ip: str = "",
        timeout: int = 120,
        proxies: Optional[Dict[str, str]] = None,
    ):
        # 明确要求调用方传入 api_url，不再强制依赖环境变量
        if not base_url:
            raise ValueError("api_url 不能为空，请在节点参数中填写服务地址。")

        self.base_url = base_url
        self.api_key = api_key or os.getenv("ZNZMO_IMAGE_API_KEY", "")
        # 0 表示不超时（requests 需要 None）
        self.timeout = None if timeout == 0 else timeout

        # 代理读取环境变量，允许显式传入覆盖
        if proxies is not None:
            self.proxies = proxies
        else:
            proxy = (
                os.getenv("HTTP_PROXY")
                or os.getenv("HTTPS_PROXY")
                or os.getenv("ALL_PROXY")
            )
            self.proxies = {"http": proxy, "https": proxy} if proxy else None

        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # 统一使用 session，便于做 host->ip 覆盖
        self.session = requests.Session()
        self.server_ip = server_ip.strip() if server_ip else ""
        if self.server_ip:
            mount_session(self.session, self.base_url, self.server_ip)

    @staticmethod
    def _encode_image_png_b64(img: Image.Image) -> str:
        """
        将 PIL Image 转为 PNG base64，使用高压缩。
        """
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=9)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _normalize_image_list(self, image_list: Any) -> List[str]:
        """
        支持传入：
        - None / 空
        - JSON 字符串（数组或单个字符串）
        - 单个 base64 字符串
        - 字符串列表
        - PIL Image 或 PIL 列表（自动压缩为 PNG base64）
        """
        if not image_list:
            return []
        parsed = image_list
        if isinstance(image_list, str):
            try:
                parsed = json.loads(image_list)
            except json.JSONDecodeError:
                parsed = image_list

        # PIL Image -> b64
        if isinstance(parsed, Image.Image):
            return [self._encode_image_png_b64(parsed)]

        if isinstance(parsed, str):
            return [parsed]

        out: List[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and item:
                    out.append(item)
                elif isinstance(item, Image.Image):
                    out.append(self._encode_image_png_b64(item))
        return out

    def generate_image(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str = "auto",
        image_size: str = "auto",
        batch_size: int = 1,
        image_list: Optional[Any] = None,
        task_id: Optional[str] = None,
        source: str = "agent",
    ) -> tuple[List[Image.Image], str]:
        normalized_list = self._normalize_image_list(image_list)

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "batchSize": batch_size,
        }
        if aspect_ratio != "auto":
            payload["aspectRatio"] = aspect_ratio
        if image_size != "auto":
            payload["imageSize"] = image_size
        if normalized_list:
            payload["imageList"] = normalized_list
        if task_id:
            payload["taskId"] = task_id
        if source:
            payload["source"] = source
        # 打印请求参数，但是要先把图片列表base64的图片去掉，不然会占用太多空间
        log_payload = copy.deepcopy(payload)
        log_data = log_payload.get("imageList")
        if isinstance(log_data, list):
            for i in range(len(log_data)):
                image = log_data[i]
                if isinstance(image, str):
                    if len(image) > 50:
                        log_data[i] = image[:30] + "..." + image[-10:]
        elif isinstance(log_data, str):
            if len(log_data) > 50:
                log_data = log_data[:30] + "..." + log_data[-10:]
                log_payload["imageList"] = log_data
        print(f"znzmo generate image payload: \n{log_payload}")
        del log_payload

        try:
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
                proxies=self.proxies,
            )
        except requests.exceptions.ConnectTimeout as e:
            error_msg = f"连接超时: 无法连接到服务器"
            if self.server_ip:
                error_msg += f"\n尝试连接到 IP: {self.server_ip}"
                error_msg += f"\n原始域名: {urlsplit(self.base_url).hostname}"
            error_msg += f"\n超时设置: {self.timeout} 秒"
            error_msg += f"\n请检查:\n  1. IP 地址是否正确: {self.server_ip if self.server_ip else '未设置'}\n  2. 网络是否可达\n  3. 端口是否正确（HTTPS 默认 443）"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接错误: 无法连接到服务器"
            if self.server_ip:
                error_msg += f"\n尝试连接到 IP: {self.server_ip}"
                error_msg += f"\n原始域名: {urlsplit(self.base_url).hostname}"
            error_msg += f"\n请检查网络连接和服务器状态"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"请求超时: 服务器响应时间超过 {self.timeout} 秒") from e
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Znzmo 图像服务调用失败: HTTP {response.status_code} {response.text}"
            )

        try:
            result = response.json()
            # 打印返回结果，但是要先把图片列表中base64的图片去掉，不然会占用太多空间
            log_result = copy.deepcopy(result)
            log_data = log_result.get("data")
            if isinstance(log_data, list):
                for i in range(len(log_data)):
                    image = log_data[i]
                    if isinstance(image, str):
                        if len(image) > 50:
                            log_data[i] = image[:30] + "..." + image[-10:]
            elif isinstance(log_data, str):
                if len(log_data) > 50:
                    log_data = log_data[:30] + "..." + log_data[-10:]
                    log_result["data"] = log_data
            print(f"znzmo generate image response: \n{log_result}")
            del log_result
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Znzmo 图像服务返回非 JSON 响应: {exc}") from exc

        if result.get("code") != 0:
            raise RuntimeError(f"Znzmo 图像服务调用失败: {result.get('message')}")

        image_urls = result.get("data")

        images = []
        for image_url in image_urls:
            # 判断是url还是base64
            if image_url.startswith("data:image/"):
                img = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1])))
            else:
                response = self.session.get(image_url, timeout=self.timeout, proxies=self.proxies)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            img.load()
            images.append(img)

        return images, result.get("message")
