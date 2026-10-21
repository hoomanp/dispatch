"""Tests for DISPATCH Phase 4 enterprise extensions:
1. HybridRateLimiter (In-Memory + Redis Fallback)
2. BanditRouter (Adaptive Contextual Bandit tier routing with Thompson Sampling & Epsilon-Greedy)
3. PrivacyMask (Deterministic PII/PHI token redaction & reversible unmasking)
4. Redis event broadcasting fallback in TrustLedger
"""

import unittest
from dispatcher.rate_limiter import InMemorySlidingWindowRateLimiter, HybridRateLimiter
from dispatcher.bandit_router import BanditRouter
from dispatcher.privacy_mask import PrivacyMask
from dispatcher.trust_ledger import TrustLedger


class TestHybridRateLimiter(unittest.TestCase):
    def test_in_memory_sliding_window(self):
        limiter = InMemorySlidingWindowRateLimiter(window_seconds=60.0)
        key = "agent-alpha"
        # 3 requests allowed
        allowed1, remaining1, _ = limiter.check(key, max_rpm=3)
        allowed2, remaining2, _ = limiter.check(key, max_rpm=3)
        allowed3, remaining3, _ = limiter.check(key, max_rpm=3)
        allowed4, remaining4, retry_after = limiter.check(key, max_rpm=3)

        self.assertTrue(allowed1)
        self.assertTrue(allowed2)
        self.assertTrue(allowed3)
        self.assertFalse(allowed4)
        self.assertEqual(remaining4, 0)
        self.assertGreater(retry_after, 0)

    def test_hybrid_fallback_without_redis(self):
        # Should gracefully use in-memory rate limiter when redis is disabled or unavailable
        limiter = HybridRateLimiter(redis_url=None)
        self.assertFalse(limiter.using_redis)
        allowed1, _, _ = limiter.check("agent-beta", max_rpm=2)
        allowed2, _, _ = limiter.check("agent-beta", max_rpm=2)
        allowed3, _, _ = limiter.check("agent-beta", max_rpm=2)

        self.assertTrue(allowed1)
        self.assertTrue(allowed2)
        self.assertFalse(allowed3)


class TestBanditRouter(unittest.TestCase):
    def setUp(self):
        self.candidates = ["local-llama", "deepseek-cloud", "claude-sonnet-cloud"]
        self.router = BanditRouter(epsilon=0.0)

    def test_arm_selection(self):
        selected, scores = self.router.select_arm(self.candidates, strategy="bandit")
        self.assertIn(selected, self.candidates)
        self.assertEqual(len(scores), len(self.candidates))

    def test_reward_feedback_and_learning(self):
        # Reinforce local-llama with excellent latency and zero cost
        for _ in range(10):
            self.router.record_feedback("local-llama", success=True, latency_ms=120, cost_usd=0.0)

        # Penalize deepseek-cloud with failures and high latency
        for _ in range(10):
            self.router.record_feedback("deepseek-cloud", success=False, latency_ms=4500, cost_usd=0.05)

        stats = self.router.stats()
        self.assertGreater(stats["local-llama"]["expected_value"], stats["deepseek-cloud"]["expected_value"])
        self.assertEqual(stats["local-llama"]["pulls"], 10)
        self.assertEqual(stats["local-llama"]["success_rate"], 1.0)
        self.assertEqual(stats["deepseek-cloud"]["success_rate"], 0.0)

    def test_epsilon_exploration(self):
        exploring_router = BanditRouter(epsilon=1.0)
        # With epsilon 1.0, it explores randomly across candidates
        selections = {c: 0 for c in self.candidates}
        for _ in range(90):
            arm, _ = exploring_router.select_arm(self.candidates, strategy="bandit")
            selections[arm] += 1
        for count in selections.values():
            self.assertGreater(count, 5)

    def test_contextual_task_conditioning_and_user_satisfaction(self):
        # Train local-llama to be great at coding, but poor at math
        for _ in range(8):
            self.router.record_feedback(
                "local-llama", success=True, latency_ms=100, cost_usd=0.0,
                user_satisfaction=0.95, task="coding"
            )
            self.router.record_feedback(
                "local-llama", success=False, latency_ms=2000, cost_usd=0.0,
                user_satisfaction=-0.8, task="math"
            )
            self.router.record_feedback(
                "deepseek-cloud", success=True, latency_ms=400, cost_usd=0.005,
                user_satisfaction=0.9, task="math"
            )

        coding_stats = self.router.stats(task="coding")
        math_stats = self.router.stats(task="math")
        self.assertGreater(coding_stats["local-llama"]["expected_value"],
                           math_stats["local-llama"]["expected_value"])
        self.assertGreater(math_stats["deepseek-cloud"]["expected_value"],
                           math_stats["local-llama"]["expected_value"])


class TestPrivacyMask(unittest.TestCase):
    def setUp(self):
        self.masker = PrivacyMask()

    def test_mask_emails_and_phones(self):
        sample = "Please email john.doe@company.org or call 555-123-4567 regarding the patient."
        masked, token_map, detected = self.masker.mask_text(sample)
        
        self.assertNotIn("john.doe@company.org", masked)
        self.assertNotIn("555-123-4567", masked)
        self.assertIn("<MASK:EMAIL_", masked)
        self.assertIn("<MASK:PHONE_", masked)
        self.assertIn("EMAIL", detected)
        self.assertIn("PHONE", detected)

        # Test deterministic reverse unmasking
        unmasked = self.masker.unmask_text(masked, token_map)
        self.assertEqual(unmasked, sample)

    def test_mask_api_keys_and_jwts(self):
        jwt_sample = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        api_sample = "sk-proj-1234567890abcdef1234567890abcdef"
        text = f"Auth using Bearer {jwt_sample} and API token {api_sample}."

        masked, token_map, detected = self.masker.mask_text(text)
        self.assertNotIn("sk-proj-1234567890abcdef1234567890abcdef", masked)
        self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", masked)
        self.assertIn("<MASK:API_KEY_", masked)
        self.assertIn("<MASK:JWT_", masked)

        unmasked = self.masker.unmask_text(masked, token_map)
        self.assertEqual(unmasked, text)

    def test_mask_messages_pipeline(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My SSN is 000-12-3456 and CC is 4111222233334444."}
        ]
        masked_msgs, token_map, detected = self.masker.mask_messages(messages)
        
        user_content = masked_msgs[1]["content"]
        self.assertNotIn("000-12-3456", user_content)
        self.assertNotIn("4111222233334444", user_content)
        self.assertIn("<MASK:SSN_", user_content)
        self.assertIn("<MASK:CREDIT_CARD_", user_content)

        # Roundtrip unmask on model reply that echoes the token
        synthetic_keys = list(token_map.keys())
        model_reply = f"Acknowledged request with token {synthetic_keys[0]}."
        unmasked_reply = self.masker.unmask_text(model_reply, token_map)
        self.assertIn(token_map[synthetic_keys[0]], unmasked_reply)

    def test_multi_turn_bijection_and_prefix_collision_safety(self):
        messages = [
            {"role": "user", "content": "Contact alice@example.com on host 10.0.0.1 or 10.0.0.123."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Now CC bob@example.com and alice@example.com."},
        ]
        masked_msgs, token_map, _ = self.masker.mask_messages(messages)
        # Ensure alice and bob got distinct tokens and neither was overwritten
        self.assertEqual(token_map.get("<MASK:EMAIL_1>"), "alice@example.com")
        self.assertEqual(token_map.get("<MASK:EMAIL_2>"), "bob@example.com")
        # Ensure 10.0.0.1 did not corrupt 10.0.0.123
        turn0_unmasked = self.masker.unmask_text(masked_msgs[0]["content"], token_map)
        turn2_unmasked = self.masker.unmask_text(masked_msgs[2]["content"], token_map)
        self.assertEqual(turn0_unmasked, messages[0]["content"])
        self.assertEqual(turn2_unmasked, messages[2]["content"])


class TestTrustLedgerRedisBroadcast(unittest.TestCase):
    def test_trust_ledger_recording_with_fallback(self):
        ledger = TrustLedger(db_path=":memory:")
        # Should record and snapshot without error even when redis is not configured
        snap = ledger.record("agent-gamma", "eval_pass", detail={"score": 0.95})
        self.assertGreater(snap.score, 0.5)
        self.assertEqual(snap.tier_name, "standard")


if __name__ == "__main__":
    unittest.main()
