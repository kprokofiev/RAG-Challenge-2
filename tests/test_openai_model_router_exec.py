import os
import unittest
from unittest import mock

import src.openai_model_router as router


class _FakePipeline:
    def __init__(self, client):
        self.client = client
        self.ops = []

    def hincrby(self, key, field, amount):
        self.ops.append(("hincrby", key, field, amount))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    def execute(self):
        for op in self.ops:
            if op[0] == "hincrby":
                _, key, field, amount = op
                self.client.hincrby(key, field, amount)
            elif op[0] == "expire":
                _, key, ttl = op
                self.client.expire(key, ttl)
        return True


class _FakeRedis:
    def __init__(self):
        self.data = {}

    def hgetall(self, key):
        return dict(self.data.get(key, {}))

    def hset(self, key, mapping=None, **kwargs):
        current = self.data.setdefault(key, {})
        payload = mapping or kwargs
        for field, value in payload.items():
            current[field] = str(value)
        return True

    def hincrby(self, key, field, amount):
        current = self.data.setdefault(key, {})
        current[field] = str(int(current.get(field, "0")) + int(amount))
        return int(current[field])

    def expire(self, key, ttl):
        return True

    def delete(self, key):
        self.data.pop(key, None)
        return True

    def pipeline(self):
        return _FakePipeline(self)

    def ping(self):
        return True


class OpenAIModelRouterExecTests(unittest.TestCase):
    def setUp(self):
        router._redis_client_cache.clear()
        self.redis = _FakeRedis()
        self.redis_patcher = mock.patch.object(router, "_redis_client", return_value=self.redis)
        self.redis_patcher.start()
        self.env = mock.patch.dict(
            os.environ,
            {
                "OPENAI_MODEL_ROUTER_ENABLED": "1",
                "DDKIT_EXEC_ELITE_RESERVE_MIN": "40000",
                "DDKIT_EXEC_MINI_RESERVE_MIN": "300000",
                "OPENAI_MODEL_ROUTER_ELITE_DAILY_TOKENS": "250000",
                "OPENAI_MODEL_ROUTER_MINI_DAILY_TOKENS": "2500000",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self):
        try:
            self.redis_patcher.stop()
        except RuntimeError:
            pass
        self.env.stop()
        router._redis_client_cache.clear()

    def test_elite_exhausted_falls_back_to_mini(self):
        state_key = router._redis_state_key(router._utc_day_key())
        self.redis.hset(state_key, mapping={"elite_exhausted": 1})
        routed = router.choose_routed_model("gpt-5.4")
        self.assertEqual(routed.tier, "mini")

    def test_reservation_and_commit_flow_updates_state(self):
        routed = router.reserve_routed_model(
            requested_model="gpt-5.4",
            estimated_total_tokens=10000,
            block_class="critical",
            thinking_mode="high",
            allow_nano_final=False,
        )
        snapshot = router.get_budget_snapshot()
        self.assertEqual(snapshot["elite"]["reserved_tokens"], 10000)

        after = router.commit_routed_usage(
            routed,
            {
                "input_tokens": 2000,
                "output_tokens": 3000,
                "total_tokens": 5000,
                "cached_tokens": 100,
                "reasoning_tokens": 1200,
            },
        )
        self.assertEqual(after["elite"]["reserved_tokens"], 0)
        self.assertEqual(after["elite"]["used_tokens"], 5000)

    def test_mini_shortfall_falls_back_to_nano(self):
        state_key = router._redis_state_key(router._utc_day_key())
        self.redis.hset(
            state_key,
            mapping={
                "elite_exhausted": 1,
                "mini_used_tokens": 2250000,
                "mini_reserved_tokens": 0,
            },
        )
        routed = router.reserve_routed_model(
            requested_model="gpt-5.4-mini",
            estimated_total_tokens=100000,
            block_class="secondary",
            thinking_mode="medium",
        )
        self.assertEqual(routed.tier, "nano")

    def test_router_tries_localhost_when_env_uses_container_hostname(self):
        self.redis_patcher.stop()

        attempts = []

        class _FailingRedis:
            def ping(self):
                raise OSError("name resolution failed")

        def _from_url(url, **kwargs):
            attempts.append(url)
            self.assertEqual(kwargs["decode_responses"], True)
            self.assertIn("socket_connect_timeout", kwargs)
            self.assertIn("socket_timeout", kwargs)
            if url == "redis://redis:6379/0":
                return _FailingRedis()
            self.assertEqual(url, "redis://localhost:6379/0")
            return self.redis

        with mock.patch.dict(
            os.environ,
            {
                "REDIS_URL": "redis://redis:6379/0",
            },
            clear=False,
        ), mock.patch.object(router.redis_lib.Redis, "from_url", side_effect=_from_url):
            snapshot = router.get_budget_snapshot()

        self.assertEqual(snapshot["mini"]["remaining_tokens"], 2500000)
        self.assertEqual(attempts, ["redis://redis:6379/0", "redis://localhost:6379/0"])

    def test_router_disabled_does_not_touch_redis_url(self):
        self.redis_patcher.stop()

        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_MODEL_ROUTER_ENABLED": "0",
                "REDIS_URL": "redis://redis:6379/0",
            },
            clear=False,
        ), mock.patch.object(router.redis_lib.Redis, "from_url") as from_url:
            routed = router.reserve_routed_model(
                requested_model="gpt-5.4-mini",
                estimated_total_tokens=10000,
                block_class="critical",
                thinking_mode="high",
            )
            snapshot = router.get_budget_snapshot()

        self.assertEqual(routed.tier, "disabled")
        self.assertEqual(routed.model, "gpt-5.4-mini")
        self.assertEqual(snapshot["mini"]["remaining_tokens"], 2500000)
        from_url.assert_not_called()

    def test_quota_autostop_status_disabled_does_not_block(self):
        state_key = router._redis_state_key(router._utc_day_key())
        self.redis.hset(
            state_key,
            mapping={
                "mini_exhausted": 1,
                "mini_last_error": "429 insufficient_quota",
            },
        )
        with mock.patch.dict(os.environ, {"DDKIT_EXEC_QUOTA_AUTOSTOP": "0"}, clear=False):
            status = router.get_quota_autostop_status()

        self.assertFalse(status["enabled"])
        self.assertFalse(status["blocked"])
        self.assertEqual(status["exhausted_tiers"], ["mini"])
        self.assertEqual(status["last_errors"]["mini"], "429 insufficient_quota")

    def test_quota_autostop_status_blocks_when_enabled_and_tier_exhausted(self):
        state_key = router._redis_state_key(router._utc_day_key())
        self.redis.hset(
            state_key,
            mapping={
                "elite_exhausted": 1,
                "elite_last_error": "billing hard stop",
                "mini_exhausted": 1,
                "mini_last_error": "429 insufficient_quota",
            },
        )
        with mock.patch.dict(os.environ, {"DDKIT_EXEC_QUOTA_AUTOSTOP": "1"}, clear=False):
            status = router.get_quota_autostop_status()

        self.assertTrue(status["enabled"])
        self.assertTrue(status["blocked"])
        self.assertEqual(status["exhausted_tiers"], ["elite", "mini"])
        self.assertIn("reset_at_utc=", status["reason"])
        self.assertEqual(status["last_errors"]["elite"], "billing hard stop")


if __name__ == "__main__":
    unittest.main()
