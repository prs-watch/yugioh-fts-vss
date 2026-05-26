import hashlib
import hmac
import os
import time

import requests
import streamlit as st

from consts import IMAGE_BASE_URL


@st.cache_data(ttl=86400)
def get_image(card_id: str) -> bytes | None:
    secret = os.environ.get("IMAGE_SIGNING_SECRET", "dev-secret")
    exp = str(int(time.time()) + 300)
    sig = hmac.new(secret.encode(), f"{card_id}:{exp}".encode(), hashlib.sha256).hexdigest()
    url = f"{IMAGE_BASE_URL}/card/{card_id}?exp={exp}&sig={sig}"
    resp = requests.get(url, timeout=5)
    if resp.ok:
        return resp.content
    return None
