import pytest

import ratelimit


def test_allows_up_to_the_limit():
    key = "test:allow"
    for _ in range(5):
        ratelimit.check(key, max_hits=5, window_seconds=60)


def test_blocks_over_the_limit():
    key = "test:block"
    for _ in range(5):
        ratelimit.check(key, max_hits=5, window_seconds=60)
    with pytest.raises(ratelimit.RateLimitExceeded):
        ratelimit.check(key, max_hits=5, window_seconds=60)


def test_reset_clears_history():
    key = "test:reset"
    for _ in range(5):
        ratelimit.check(key, max_hits=5, window_seconds=60)
    ratelimit.reset(key)
    ratelimit.check(key, max_hits=5, window_seconds=60)  # should not raise


def test_keys_are_independent():
    ratelimit.reset("a")
    ratelimit.reset("b")
    for _ in range(5):
        ratelimit.check("a", max_hits=5, window_seconds=60)
    # "b" should still have full quota
    ratelimit.check("b", max_hits=5, window_seconds=60)
