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


class TestTrustLedgerRedisBroadcast(unittest.TestCase):
    def test_trust_ledger_recording_with_fallback(self):
        ledger = TrustLedger(db_path=":memory:")
        # Should record and snapshot without error even when redis is not configured
        snap = ledger.record("agent-gamma", "eval_pass", detail={"score": 0.95})
        self.assertGreater(snap.score, 0.5)
        self.assertEqual(snap.tier_name, "standard")


if __name__ == "__main__":
    unittest.main()
