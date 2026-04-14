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


class OpenAIModelRouterExecTests(unittest.TestCase):
    def setUp(self):
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
        self.redis_patcher.stop()
        self.env.stop()

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


if __name__ == "__main__":
    unittest.main()
