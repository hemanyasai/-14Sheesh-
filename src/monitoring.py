"""
monitoring.py - Langfuse wrapper using v4 decorators
"""

import os
from dotenv import load_dotenv

load_dotenv()

_LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

if _LANGFUSE_ENABLED:
    try:
        from langfuse import observe
    except Exception:
        _LANGFUSE_ENABLED = False
        from functools import wraps
        def observe(*args, **kwargs):
            def decorator(func):
                @wraps(func)
                def wrapper(*a, **kw):
                    return func(*a, **kw)
                return wrapper
            return decorator
else:
    # Dummy decorators for graceful fallback
    from functools import wraps
    def observe(*args, **kwargs):
        def decorator(func):
            @wraps(func)
            def wrapper(*a, **kw):
                return func(*a, **kw)
            return wrapper
        return decorator
