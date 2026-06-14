import asyncio
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, StarTools


class ImageCollectorPlugin(Star):
    """图片收藏插件 - LLM 自主判断收藏价值，分类存储，嵌入检索匹配。"""

    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config: AstrBotConfig = config or {}
        self.data_dir: Path = StarTools.get_data_dir()
        self.images_dir: Path = self.data_dir / "images"
        self.index_path: Path = self.data_dir / "index.json"
        self._index_lock = asyncio.Lock()

    # ═══════════════════════════════════════════════
    #  辅助方法
    # ═══════════════════════════════════════════════

    async def _load_index(self) -> list[dict]:
        async with self._index_lock:
            try:
                if self.index_path.exists():
                    data = json.loads(self.index_path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.error(f"[ImageCollector] 加载索引失败: {e}")
            return []

    async def _save_index(self, data: list[dict]):
        async with self._index_lock:
            try:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self.index_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.error(f"[ImageCollector] 保存索引失败: {e}")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', "_", name).strip().strip(".") or "unnamed"

    @staticmethod
    def _extract_image_urls(event: AstrMessageEvent) -> list[dict]:
        urls: list[dict] = []
        raw = event.message_obj.raw_message
        if not isinstance(raw, dict):
            msg = getattr(event.message_obj, "message", None)
            if isinstance(msg, list):
                for comp in msg:
                    if isinstance(comp, Comp.Image):
                        url = getattr(comp, "url", "") or getattr(comp, "file", "")
                        if url:
                            urls.append({"url": url, "file": url})
            return urls

        message = raw.get("message", [])
        if isinstance(message, list):
            for item in message:
                if isinstance(item, dict) and item.get("type") == "image":
                    data = item.get("data", {})
                    url = data.get("url", "")
                    file_id = data.get("file", "")
                    if url:
                        urls.append({"url": url, "file": file_id or url})
        return urls

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _embedding_enabled(self) -> bool:
        return bool(self.config.get("embedding_enabled", False))

    async def _get_embedding(self, event: AstrMessageEvent, text: str) -> list[float] | None:
        """通过 AstrBot 内置 provider 获取文本嵌入向量。"""
        if not self._embedding_enabled() or not text:
            return None
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
            if provider is None:
                return None

            # 尝试多种常见的嵌入方法名
            for method_name in ("embedding", "text_embedding", "get_embedding", "embed"):
                method = getattr(provider, method_name, None)
                if method is not None:
                    result = await method(text)
                    if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                        return result
                    if hasattr(result, "embedding"):
                        return result.embedding
                    if isinstance(result, dict):
                        emb = result.get("embedding") or result.get("data", [{}])[0].get("embedding")
                        if emb:
                            return emb

            logger.debug("[ImageCollector] 当前 provider 不支持嵌入方法")
            return None
        except Exception as e:
            logger.warning(f"[ImageCollector] 获取嵌入向量失败: {e}")
            return None

    # ═══════════════════════════════════════════════
    #  on_llm_request 钩子
    # ═══════════════════════════════════════════════

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        extra_lines: list[str] = []

        # ── 1. 提取当前消息中的图片 URL ──
        image_urls = self._extract_image_urls(event)
        if image_urls:
            lines = ["[重要] 当前消息中包含以下图片，如果你认为这些图片有收藏价值（如表情包、梗图、美图、实用截图等），请务必调用 collect_image 工具收藏:"]
            for i, img in enumerate(image_urls, 1):
                lines.append(f"  图片{i}: {img['url']}")
            lines.append(
                "collect_image 参数: url(图片URL), category(分类,如\"表情包\"/\"梗图\"/\"美图\"/\"实用截图\"), "
                "filename(文件名不含扩展名), description(收藏理由/图片描述), tags(逗号分隔标签)"
            )
            extra_lines.append("\n".join(lines))

        # ── 2. 嵌入检索匹配 ──
        if self._embedding_enabled():
            user_text = event.message_str
            if user_text and len(user_text) > 2:
                embedding = await self._get_embedding(event, user_text)
                if embedding:
                    index = await self._load_index()
                    threshold = float(self.config.get("similarity_threshold", 0.7))
                    max_count = int(self.config.get("max_context_images", 3))
                    scored: list[tuple[float, dict]] = []
                    for entry in index:
                        stored_emb = entry.get("embedding")
                        if stored_emb and len(stored_emb) == len(embedding):
                            sim = self._cosine_similarity(embedding, stored_emb)
                            if sim >= threshold:
                                scored.append((sim, entry))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    if scored:
                        lines = ["以下收藏图片可能与当前话题相关:"]
                        for sim, entry in scored[:max_count]:
                            desc = entry.get("description", "") or entry.get("filename", "")
                            lines.append(
                                f"  - {entry['local_path']} "
                                f"(分类: {entry.get('category', '')}, 描述: {desc})"
                            )
                        lines.append("你可以在回复中提及或使用这些图片。")
                        extra_lines.append("\n".join(lines))

        if extra_lines:
            req.system_prompt += "\n\n[系统提示] " + "\n\n".join(extra_lines)

    # ═══════════════════════════════════════════════
    #  LLM Tools
    # ═══════════════════════════════════════════════

    @filter.llm_tool(name="collect_image")
    async def collect_image(
        self,
        event: AstrMessageEvent,
        url: str,
        category: str,
        filename: str,
        description: str = "",
        tags: str = "",
    ):
        """收藏一张图片到指定分类。当聊天中出现有收藏价值的图片时调用。

        Args:
            url(string): 图片的完整 URL 地址
            category(string): 分类文件夹名，如 \"表情包\"、\"梗图\"、\"美图\"、\"实用截图\"
            filename(string): 文件名（不含扩展名），如 \"猫猫震惊\"
            description(string): 收藏理由/图片描述。默认空。
            tags(string): 逗号分隔的标签，如 \"猫,震惊,可爱\"。默认空。
        """
        if not url:
            return "错误: 缺少图片 URL。"

        category = self._sanitize_name(category) or "未分类"
        filename = self._sanitize_name(filename) or f"image_{uuid.uuid4().hex[:8]}"

        ext = ".jpg"
        url_lower = url.lower()
        if ".png" in url_lower or url_lower.startswith("data:image/png"):
            ext = ".png"
        elif ".gif" in url_lower or url_lower.startswith("data:image/gif"):
            ext = ".gif"
        elif ".webp" in url_lower or url_lower.startswith("data:image/webp"):
            ext = ".webp"

        safe_filename = filename + ext
        category_dir = self.images_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        dest_path = category_dir / safe_filename
        if dest_path.exists():
            stem = filename
            suffix = f"_{uuid.uuid4().hex[:6]}{ext}"
            dest_path = category_dir / (stem + suffix)
            safe_filename = stem + suffix

        try:
            if url.startswith("data:"):
                import base64 as _b64

                header, b64 = url.split(",", 1)
                raw_bytes = _b64.b64decode(b64)
            else:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    raw_bytes = resp.content

            dest_path.write_bytes(raw_bytes)
            file_size = len(raw_bytes)
        except Exception as e:
            logger.error(f"[ImageCollector] 下载图片失败 url={url[:80]}: {e}")
            return f"下载图片失败: {e}"

        embedding: list[float] | None = None
        if self._embedding_enabled() and description:
            embedding = await self._get_embedding(event, description)

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        rel_path = str(dest_path.relative_to(self.data_dir))
        sender_id = str(event.get_sender_id())
        group_id = str(event.get_group_id()) if event.get_group_id() else ""

        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "category": category,
            "filename": safe_filename,
            "local_path": rel_path,
            "original_url": url[:500],
            "description": description,
            "tags": tag_list,
            "embedding": embedding,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "collected_from": f"group:{group_id}" if group_id else f"private:{sender_id}",
            "file_size": file_size,
        }

        index = await self._load_index()
        index.append(entry)
        await self._save_index(index)

        logger.info(
            f"[ImageCollector] 收藏图片: {safe_filename} -> {category}/ "
            f"来源={entry['collected_from']}"
        )
        return (
            f"已收藏图片「{filename}」到分类「{category}」下。"
            + (f" 标签: {', '.join(tag_list)}。" if tag_list else "")
        )

    @filter.llm_tool(name="search_collected_images")
    async def search_collected_images(
        self,
        event: AstrMessageEvent,
        query: str,
        category: str = "",
        limit: int = 5,
    ):
        """搜索已收藏的图片。按关键词匹配标签和描述，或使用嵌入相似度排序。

        Args:
            query(string): 搜索关键词
            category(string): 限定分类文件夹。留空则搜索全部。默认空。
            limit(int): 返回结果数量上限。默认 5。
        """
        index = await self._load_index()
        if not index:
            return "收藏库中暂无图片。"

        results: list[tuple[float, dict]] = []

        if self._embedding_enabled():
            embedding = await self._get_embedding(event, query)
            if embedding:
                for entry in index:
                    if category and entry.get("category") != category:
                        continue
                    stored_emb = entry.get("embedding")
                    if stored_emb and len(stored_emb) == len(embedding):
                        sim = self._cosine_similarity(embedding, stored_emb)
                        results.append((sim, entry))
                results.sort(key=lambda x: x[0], reverse=True)
                results = results[:limit]

        # 文本匹配（嵌入未命中或无嵌入时的降级）
        if not results:
            query_lower = query.lower()
            scored: list[tuple[int, dict]] = []
            for entry in index:
                if category and entry.get("category") != category:
                    continue
                score = 0
                if query_lower in entry.get("description", "").lower():
                    score += 3
                if query_lower in entry.get("filename", "").lower():
                    score += 2
                if query_lower in entry.get("category", "").lower():
                    score += 1
                for tag in entry.get("tags", []):
                    if query_lower in tag.lower():
                        score += 2
                if score > 0:
                    scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            for s, e in scored[:limit]:
                results.append((float(s), e))

        if not results:
            return f"未找到与「{query}」匹配的收藏图片。"

        lines = [f"搜索「{query}」找到 {len(results)} 张图片:"]
        for _, e in results:
            lines.append(
                f"  - {e['local_path']} "
                f"(分类: {e['category']}"
                + (f", 标签: {', '.join(e.get('tags', []))}" if e.get("tags") else "")
                + ")"
            )
        return "\n".join(lines)

    @filter.llm_tool(name="send_collected_image")
    async def send_collected_image(
        self,
        event: AstrMessageEvent,
        local_path: str = "",
        filename: str = "",
        category: str = "",
    ):
        """在当前聊天中发送一张已收藏的图片。可通过 local_path 或 filename+category 指定。

        Args:
            local_path(string): 图片在收藏库中的相对路径，如 \"images/表情包/猫猫震惊.jpg\"。默认空。
            filename(string): 文件名（含扩展名），如 \"猫猫震惊.jpg\"。与 category 配合使用。默认空。
            category(string): 分类文件夹名，与 filename 配合定位图片。默认空。
        """
        index = await self._load_index()

        match: dict | None = None
        if local_path:
            for e in index:
                if e.get("local_path") == local_path:
                    match = e
                    break
        elif filename:
            for e in index:
                cat_match = not category or e.get("category") == category
                if cat_match and e.get("filename") == filename:
                    match = e
                    break

        if match is None:
            hint = local_path or f"{category}/{filename}" if category else filename
            return f"未找到图片: {hint}。请先使用 search_collected_images 确认路径。"

        full_path = self.data_dir / match["local_path"]
        if not full_path.exists():
            return f"图片文件不存在: {match['local_path']}（可能已被移动或删除）。"

        try:
            await event.send(event.image_result(str(full_path)))
            desc = match.get("description", "") or match.get("filename", "")
            logger.info(
                f"[ImageCollector] 发送收藏图片: {match['local_path']} "
                f"来源={event.get_sender_id()}"
            )
            return f"已发送图片「{desc}」。"
        except Exception as e:
            logger.error(f"[ImageCollector] 发送图片失败: {e}")
            return f"发送图片失败: {e}"

    @filter.llm_tool(name="list_collection_categories")
    async def list_collection_categories(self, event: AstrMessageEvent):
        """列出所有收藏分类及图片数量。"""
        index = await self._load_index()
        if not index:
            return "收藏库中暂无图片。"

        cats: dict[str, int] = {}
        for entry in index:
            cat = entry.get("category", "未分类")
            cats[cat] = cats.get(cat, 0) + 1

        lines = [f"收藏库共 {len(index)} 张图片，{len(cats)} 个分类:"]
        for cat, count in sorted(cats.items()):
            lines.append(f"  - {cat}: {count} 张")
        return "\n".join(lines)

    async def terminate(self):
        """插件卸载时调用。"""
        logger.info("ImageCollectorPlugin 已卸载。")
