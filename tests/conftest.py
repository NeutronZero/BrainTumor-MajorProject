"""Shared test configuration.

The API rate limiter (app/api/main.py) is a module-level singleton keyed by
client IP, and Starlette's TestClient always presents the same client
("testclient"). Without this, every API test in the suite shares one
120-requests/minute budget and can trip spurious 429s as coverage grows.
Tests that exercise the limiter set ``_MAX_REQUESTS`` on the live middleware
directly, so raising only the import-time default here keeps them meaningful.

Must be set before ``main`` is imported; a conftest is loaded before test
collection, so this is the right place.
"""

import os

os.environ.setdefault("BT_RATE_LIMIT", "1000000")
