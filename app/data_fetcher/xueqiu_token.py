# app/data_fetcher/xueqiu_token.py
import datetime as dt
import logging
from typing import Optional, Tuple

from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import Session
from playwright.sync_api import sync_playwright

from app.database import Base, get_db

logger = logging.getLogger(__name__)


class XqToken(Base):
    __tablename__ = "xq_token_cache"
    key = Column(String(32), primary_key=True)   # 固定 'xq_a_token'
    value = Column(String(2048), nullable=False)
    updated_at = Column(DateTime, nullable=False)


class XueqiuTokenManager:
    """服务器用：纯 headless 自动获取 xq_a_token，带数据库缓存（TTL=23h）。"""
    TOKEN_KEY = "xq_a_token"
    TTL = dt.timedelta(hours=23)
    START_URL = "https://xueqiu.com/S/SH000001"

    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/124.0.0.0 Safari/537.36")

    @classmethod
    def get_token(cls, allow_refresh: bool = True) -> str:
        """读取缓存，若过期则刷新"""
        with get_db() as db:
            cached = cls._read_cache(db)
            if cached:
                val, ts = cached
                if not cls.is_expired(ts):
                    return val
                elif not allow_refresh:
                    return val

        if not allow_refresh:
            raise RuntimeError("缓存 token 已过期且禁止刷新。")

        token = cls._fetch_headless_token()
        with get_db() as db:
            cls._write_cache(db, token)
        return token

    @classmethod
    def force_refresh(cls) -> str:
        token = cls._fetch_headless_token()
        with get_db() as db:
            cls._write_cache(db, token)
        return token

    # ------------ 内部方法 ------------
    @classmethod
    def _read_cache(cls, db: Session) -> Optional[Tuple[str, dt.datetime]]:
        row = db.get(XqToken, cls.TOKEN_KEY)
        if not row:
            return None
        return row.value, row.updated_at

    @classmethod
    def _write_cache(cls, db: Session, token: str) -> None:
        obj = XqToken(key=cls.TOKEN_KEY, value=token, updated_at=dt.datetime.now())
        db.merge(obj)
        db.commit()

    @classmethod
    def is_expired(cls, ts: dt.datetime) -> bool:
        return dt.datetime.now() - ts > cls.TTL

    @classmethod
    def _fetch_headless_token(cls, timeout_ms: int = 25000) -> str:
        """在 headless 模式下模拟真实浏览器访问雪球页面并提取 token。"""
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--lang=zh-CN,zh",
                ],
            )
            context = browser.new_context(
                user_agent=cls.UA,
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
                viewport={"width": 1366, "height": 768},
                java_script_enabled=True,
            )

            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(cls.START_URL, wait_until="networkidle")
            page.wait_for_timeout(10000)  # 给二次请求充分时间跑完

            # 取 cookie
            cookies = context.cookies("https://xueqiu.com")
            token = next(
                (c["value"] for c in cookies if c["name"] == "xq_a_token"), None
            )

            # 兜底再查 document.cookie
            if not token:
                try:
                    cookie_str = page.evaluate("() => document.cookie || ''")
                    for kv in cookie_str.split(";"):
                        kv = kv.strip()
                        if kv.startswith("xq_a_token="):
                            token = kv.split("=", 1)[1]
                            break
                except Exception:
                    pass

            context.close()
            browser.close()

            if not token:
                raise RuntimeError("未能在 headless 模式下获取 xq_a_token，可能被风控。")

            logger.info("成功刷新雪球 token。")
            return token
