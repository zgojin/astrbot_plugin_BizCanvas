import os
import re
import json
import uuid
import base64
import hashlib
import aiohttp
import asyncio
import io
from PIL import Image as PILImage

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

@register("AstrBot_Plugin_Biz_Canvas", "长安某", "各种逆向生图自用", "1.9.4_FINAL_CLEAN")
class GeminiBizCanvas(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 初始化网关地址
        raw_url = self.config.get("gateway_url", "").strip()
        if raw_url:
            raw_url = raw_url.rstrip("/")
            if raw_url.endswith("/v1"):
                raw_url = raw_url[:-3]
            raw_url = raw_url.rstrip("/")
        self.gateway_url = raw_url
        
        self.api_keys = self.config.get("gateway_api_keys")
        self.model_name = self.config.get("model_name")
        self.api_route_type = self.config.get("api_route_type", "chat_completions")
        self.img_ratio = self.config.get("image_config_ratio", "16:9")
        self.img_size = self.config.get("image_config_size", "4K")
        self.prompts_config = self.config.get("prompts", {})
        
        self.current_key_idx = 0
        self.plugin_dir = os.path.dirname(__file__)
        self.temp_dir = os.path.join(self.plugin_dir, "temp_biz_images")
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        if not self.api_keys:
            logger.warning("[BizCanvas] 未配置 Token")
        else:
            logger.info(f"[BizCanvas] 加载成功 | 网关: {self.gateway_url}")

    def _get_api_key(self):
        if not self.api_keys: return None
        return self.api_keys[self.current_key_idx]

    def _rotate_key(self):
        if not self.api_keys: return
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        logger.info(f"[BizCanvas] 切换 Key 索引至: {self.current_key_idx}")

    def _get_prompt_by_key(self, key: str, default_text: str = "") -> str:
        val = self.prompts_config.get(key)
        if not val:
            val = self.config.get(key)
        if val and isinstance(val, str) and val.strip():
            return val
        return default_text

    def _compress_image(self, file_path: str, max_size: int = 1024, quality: int = 85) -> bytes:
        try:
            with PILImage.open(file_path) as img:
                if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                width, height = img.size
                if max(width, height) > max_size:
                    ratio = max_size / max(width, height)
                    img = img.resize((int(width * ratio), int(height * ratio)), PILImage.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                return buffer.getvalue()
        except Exception:
            with open(file_path, "rb") as f: return f.read()

    async def _download_avatar(self, user_id: str) -> str | None:
        if not user_id: return None
        try:
            url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
            save_path = os.path.join(self.temp_dir, f"avatar_{user_id}_{uuid.uuid4()}.png")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        with open(save_path, "wb") as f: f.write(await resp.read())
                        return save_path
        except: pass
        return None

    async def _extract_images_from_event(self, event: AstrMessageEvent) -> list[str]:
        image_paths = []
        try:
            for comp in event.message_obj.message:
                if isinstance(comp, Comp.Image): image_paths.append(await comp.convert_to_file_path())
            if not image_paths:
                for comp in event.message_obj.message:
                    if isinstance(comp, Comp.Reply):
                        for c in comp.chain:
                            if isinstance(c, Comp.Image): image_paths.append(await c.convert_to_file_path())
            if not image_paths and event.get_sender_id():
                path = await self._download_avatar(event.get_sender_id())
                if path: image_paths.append(path)
        except Exception as e: logger.error(f"[BizCanvas] 图片提取失败: {e}")
        return image_paths

    async def _download_generated_image(self, url: str) -> bytes | None:
        try:
            url = url.strip().rstrip(".,;!?'\"").replace("\n", "").replace("\r", "")
            if not url.startswith("http"):
                full_url = f"{self.gateway_url}{url}" if url.startswith("/") else f"{self.gateway_url}/{url}"
            else: full_url = url
            
            key = self._get_api_key()
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.get(full_url, headers=headers) as resp:
                    if resp.status == 200: return await resp.read()
                    else: logger.warning(f"[BizCanvas] 图片下载失败 HTTP {resp.status}")
        except Exception as e: logger.warning(f"[BizCanvas] 图片下载异常: {e}")
        return None

    def _extract_from_json_responses(self, json_data):
        # Responses 模式专用提取器
        extracted_imgs = []
        extracted_text = ""
        target_keys = {"result", "image_url", "image_base64", "base64", "image", "url"}

        def _recursive_search(data):
            nonlocal extracted_text
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in ["text", "content"] and isinstance(v, str):
                        extracted_text += v + "\n"
                        extracted_imgs.extend(re.findall(r'!\[.*?\]\((.*?)\)', v))
                    if k in target_keys:
                        if isinstance(v, dict) and "url" in v:
                            extracted_imgs.append(v["url"])
                        elif isinstance(v, str) and v:
                            if k == "url":
                                if v.startswith("http") or len(v) > 500: extracted_imgs.append(v)
                            else:
                                extracted_imgs.append(v)
                    if isinstance(v, (dict, list)):
                        _recursive_search(v)
            elif isinstance(data, list):
                for item in data:
                    _recursive_search(item)

        _recursive_search(json_data.get("output", []))
        return extracted_imgs, extracted_text.strip()

    async def _call_gateway_brute_force(self, user_prompt: str, image_paths: list[str] = None):
        input_img_hashes = set()
        b64_images = []
        
        # 处理输入图片指纹
        if image_paths:
            for path in image_paths:
                img_bytes = self._compress_image(path)
                input_img_hashes.add(hashlib.md5(img_bytes).hexdigest())
                b64_str = base64.b64encode(img_bytes).decode('utf-8')
                b64_images.append(f"data:image/jpeg;base64,{b64_str}")

        logger.info(f"[BizCanvas] Payload构造, 模式: {self.api_route_type}")
        
        # 构造请求体
        if self.api_route_type == "responses":
            url = f"{self.gateway_url}/v1/responses"
            input_content = [{"type": "input_text", "text": user_prompt}]
            for b64_str in b64_images:
                input_content.append({"type": "input_image", "image_url": b64_str})
            
            payload = {
                "model": self.model_name,
                "input": [{"type": "message", "role": "user", "content": input_content}],
                "stream": False,
                "tools": [{"type": "image_generation"}],
                "image_config": {"aspect_ratio": self.img_ratio, "image_size": self.img_size}
            }
        else:
            url = f"{self.gateway_url}/v1/chat/completions"
            content_payload = [{"type": "text", "text": user_prompt}]
            for b64_str in b64_images:
                content_payload.append({"type": "image_url", "image_url": {"url": b64_str}})
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": content_payload}],
                "image_config": {"aspect_ratio": self.img_ratio, "image_size": self.img_size}
            }

        if not self.api_keys: return [], "未配置 API Key", ""

        last_error_msg = ""
        
        # 重试循环
        for i in range(3): 
            key = self._get_api_key()
            if not key: break
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            
            try:
                logger.info(f"[BizCanvas] 发起请求 (Try {i+1}): {url}")
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        
                        full_text = await resp.text()
                        logger.info(f"[BizCanvas] HTTP: {resp.status} | Len: {len(full_text)}")
                        if len(full_text) > 0:
                            logger.info(f"[BizCanvas] Resp Preview: {full_text[:500]}")

                        if resp.status in [401, 403]:
                            logger.warning(f"[BizCanvas] 鉴权失败")
                            self._rotate_key(); continue
                        
                        if resp.status != 200:
                            logger.warning(f"[BizCanvas] 错误响应: {full_text[:200]}")
                            last_error_msg = f"HTTP {resp.status}"
                            await asyncio.sleep(2); self._rotate_key(); continue
                        
                        # 开始解析
                        raw_candidates = []
                        captured_text = ""
                        
                        try:
                            json_data = json.loads(full_text)
                            
                            if self.api_route_type == "responses":
                                j_imgs, j_text = self._extract_from_json_responses(json_data)
                                raw_candidates.extend(j_imgs)
                                if j_text: captured_text = j_text
                            else:
                                # OpenAI Chat 解析器
                                for item in json_data.get("choices", []):
                                    message = item.get("message", {})
                                    
                                    # 检查非标准 images 字段
                                    images_field = message.get("images", [])
                                    if images_field:
                                        logger.info(f"[BizCanvas] 发现 message.images: {len(images_field)}")
                                        for img_obj in images_field:
                                            url_val = img_obj.get("image_url", {}).get("url")
                                            if url_val: raw_candidates.append(url_val)

                                    # 检查 Markdown 图片
                                    content = message.get("content", "")
                                    if content:
                                        captured_text += content + "\n"
                                        matches = re.findall(r'!\[.*?\]\((.*?)\)', content)
                                        if matches:
                                            logger.info(f"[BizCanvas] 发现 Markdown 图片: {len(matches)}")
                                            raw_candidates.extend(matches)
                                
                        except Exception as e:
                            logger.error(f"[BizCanvas] JSON 解析异常: {e}")
                        
                        # 兜底提取
                        if not raw_candidates:
                            raw_candidates.extend(re.findall(r'(/image/[a-zA-Z0-9_.-]+\.png)', full_text))
                            if not raw_candidates:
                                raw_candidates.extend(re.findall(r'(https?://[^\s"\'<>)]+)', full_text))

                        # 过滤与去重
                        results = []
                        seen_candidates = set()
                        for c in raw_candidates:
                            c = c.strip().rstrip(".,;!?'\"")
                            if c in seen_candidates: continue
                            
                            is_url = c.startswith(("http", "/"))
                            is_b64 = c.startswith("data:image/")
                            if not (is_url or is_b64): continue
                            
                            seen_candidates.add(c)
                            if is_url: 
                                results.append({'type': 'url', 'data': c})
                            else: 
                                try: 
                                    if "base64," in c: _, b64_part = c.split("base64,", 1)
                                    else: b64_part = c
                                    img_data = base64.b64decode(b64_part)
                                    # 指纹去重
                                    if input_img_hashes:
                                        if hashlib.md5(img_data).hexdigest() in input_img_hashes: continue
                                    results.append({'type': 'base64', 'data': img_data})
                                except: pass
                        
                        logger.info(f"[BizCanvas] 有效图片: {len(results)}")
                        if results: return results, captured_text, ""
                        
                        if not last_error_msg:
                            last_error_msg = f"无图片数据: {captured_text[:50]}"

            except Exception as e:
                last_error_msg = f"异常: {str(e)}"
                logger.error(f"[BizCanvas] {last_error_msg}")
                self._rotate_key()
        
        return [], f"失败: {last_error_msg}", ""

    async def _process_render_chain(self, results, saved_paths):
        chain = []
        seen_hashes = set()

        for idx, item in enumerate(results):
            img_bytes = None
            if item['type'] == 'base64':
                img_bytes = item['data']
            elif item['type'] == 'url':
                img_bytes = await self._download_generated_image(item['data'])
                if not img_bytes:
                    chain.append(Comp.Image.fromURL(item['data'])); continue

            # 输出图片 MD5 去重
            if img_bytes:
                h = hashlib.md5(img_bytes).hexdigest()
                if h in seen_hashes: continue
                seen_hashes.add(h)
                
                save_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}_{idx}.png")
                with open(save_path, "wb") as f: f.write(img_bytes)
                saved_paths.append(save_path)
                chain.append(Comp.Image.fromFileSystem(save_path))
                
        return chain

    @filter.regex(r"(?i)^(手办化|cos化|宝可梦)", priority=3)
    async def handle_style_transform(self, event: AstrMessageEvent):
        msg = event.message_obj.message_str.strip()
        match = re.match(r"(?i)^(手办化|cos化|宝可梦)", msg)
        if not match: return
        
        cmd = match.group(1).lower()
        image_paths = await self._extract_images_from_event(event)
        if not image_paths: yield event.plain_result("未找到图片"); return

        user_input = re.sub(r"(?i)^(手办化|cos化|宝可梦)\s*", "", msg, count=1).strip()
        
        key_map = {"手办化": "figurine", "cos化": "cosplay", "宝可梦": "pokemon"}
        prompt_key = key_map.get(cmd, "pokemon")
        base_prompt = self._get_prompt_by_key(prompt_key, f"{cmd} prompt missing")
        
        yield event.plain_result(f"正在进行 {cmd}，请稍候...")
        final_prompt = f"Style: {base_prompt}\nRequest: {user_input}\nOutput: Image only."
        
        saved_paths = []
        try:
            results, text, _ = await self._call_gateway_brute_force(final_prompt, image_paths)
            if results:
                chain = await self._process_render_chain(results, saved_paths)
                if chain: yield event.chain_result(chain)
                else: yield event.plain_result("图片渲染失败")
            else: yield event.plain_result(f"{text[:200]}") 
        finally:
            for p in saved_paths + image_paths: self._remove_file(p)

    @filter.llm_tool(name="biz_generate_image")
    async def biz_generate_image(self, event: AstrMessageEvent, keywords: str):
        '''调用此工具生成图片。

        Args:
            keywords(string): 图片的详细描述或提示词
        '''
        if not keywords: keywords = event.message_str
        saved_paths = []
        try:
            results, text, _ = await self._call_gateway_brute_force(f"Draw: {keywords}", None)
            if results:
                chain = await self._process_render_chain(results, saved_paths)
                if chain: yield event.chain_result(chain)
                else: yield event.plain_result("绘图渲染失败")
            else: yield event.plain_result(f"{text[:200]}") 
        finally:
            for p in saved_paths: self._remove_file(p)

    @filter.llm_tool(name="biz_edit_image")
    async def biz_edit_image(self, event: AstrMessageEvent, keywords: str):
        '''调用此工具编辑或修改已有的图片。

        Args:
            keywords(string): 修改图片的指令，例如"变成动漫风格"
        '''
        image_paths = await self._extract_images_from_event(event)
        if not image_paths: yield event.plain_result("未找到参考图片"); return
        saved_paths = []
        try:
            results, text, _ = await self._call_gateway_brute_force(f"Edit: {keywords}", image_paths)
            if results:
                chain = await self._process_render_chain(results, saved_paths)
                if chain: yield event.chain_result(chain)
                else: yield event.plain_result("修图渲染失败")
            else: yield event.plain_result(f"{text[:200]}") 
        finally:
            for p in saved_paths + image_paths: self._remove_file(p)

    def _remove_file(self, path: str):
        if path and os.path.exists(path):
            try: os.remove(path)
            except: pass
