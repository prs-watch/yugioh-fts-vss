"""カード画像の取得と署名付き URL 生成。"""

import hashlib
import hmac
import os
import time

import requests
import streamlit as st
from requests.adapters import HTTPAdapter

from consts import IMAGE_BASE_URL

_session = requests.Session()
_session.mount("http://", HTTPAdapter(pool_maxsize=128))
_session.mount("https://", HTTPAdapter(pool_maxsize=128))


@st.cache_data(ttl=86400)
def get_image(card_id: str) -> bytes | None:
    """署名付き URL でカード画像を取得する（24時間キャッシュ）。

    Args:
        card_id: 取得するカードの ID 文字列。

    Returns:
        画像のバイト列。取得失敗時は None。
    """
    secret = os.environ.get("IMAGE_SIGNING_SECRET", "dev-secret")
    exp = str(int(time.time()) + 300)
    sig = hmac.new(
        secret.encode(), f"{card_id}:{exp}".encode(), hashlib.sha256
    ).hexdigest()
    url = f"{IMAGE_BASE_URL}/card/{card_id}?exp={exp}&sig={sig}"
    resp = _session.get(url, timeout=5)
    if resp.ok:
        return resp.content
    return None
