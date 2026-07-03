from typing import List, Optional
from PIL import Image
from .znzmo_client import ZnzmoClient


class ZnzmoGenerateImage:
    """
    Znzmo Generate Image 节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": (
                    "STRING",
                    {
                        "default": "https://api.znzmo.cn/ai-draw/third-api/ai-draw-api/dispatch/getResult",
                        "tooltip": "Znzmo 图像服务地址（必填）",
                    },
                ),
                "server_ip": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "可选：将 api_url 的域名解析到该 IP 用于测试",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "nanoBanana",
                        "tooltip": "模型名称，例如 nanoBanana",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "生成提示词",
                    },
                ),
                "aspect_ratio": (
                    [
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    {
                        "default": "auto",
                        "tooltip": "宽高比 (仅部分模型支持)",
                    },
                ),
                "image_size": (
                    [
                        "auto",
                        "1k",
                        "2k",
                        "4k",
                        "8k",
                    ],
                    {
                        "default": "2k",
                        "tooltip": "生成尺寸",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "一次生成的图像数量",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "请求超时(秒)，0 表示不超时",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "task_id": ("STRING",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")

    FUNCTION = "generate_image"

    _NODE_NAME = "Znzmo Generate Image"
    DESCRIPTION = "按指定模型与参数生成图像 (支持 base64 参考图)"
    CATEGORY = "ZnzmoNodes/LLM"

    def generate_image(
        self,
        api_url: str,
        server_ip: str,
        model: str,
        prompt: str,
        aspect_ratio: str,
        image_size: str,
        batch_size: int,
        timeout: int,
        image: Optional[Image.Image] = None,
        image1: Optional[Image.Image] = None,
        image2: Optional[Image.Image] = None,
        image3: Optional[Image.Image] = None,
        image4: Optional[Image.Image] = None,
        task_id: Optional[str] = None,
    ):
        # 收集参考图（直接使用 PIL Image）
        images: List[Image.Image] = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                images.append(img)

        # 去掉尺寸中的注释部分
        size = image_size
        if size != "auto" and "(" in size:
            size = size[: size.find("(")]

        client = ZnzmoClient(base_url=api_url, server_ip=server_ip, timeout=timeout)
        out_images, text = client.generate_image(
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_size=size,
            batch_size=batch_size,
            image_list=images,
            task_id=task_id,
        )

        # 直接返回 PIL Image 列表
        if len(out_images) == 1:
            return (out_images[0], text)
        elif len(out_images) > 1:
            return (out_images, text)
        else:
            return (None, text)
