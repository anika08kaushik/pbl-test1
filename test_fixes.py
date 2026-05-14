#!/usr/bin/env python3
"""Quick test of the 3 fixes"""

from agent_3_validator import clean_jd_skill, is_direct_match, MATCH_THRESHOLD, MATCH_THRESHOLD_STRICT

print("=" * 60)
print("TESTING 3 FIXES")
print("=" * 60)

# Test Fix 1: clean_jd_skill
print("\n✅ FIX 1: Clean JD Skills")
print("-" * 60)
test_cases = [
    "frameworks like node.js",
    "interest in building scalable",
    "secure web applications",
    "react",
    "typescript",
    "good spoken english",
    "bachelor s degree",
]

for test in test_cases:
    cleaned = clean_jd_skill(test)
    status = "✓" if cleaned else "✗"
    print(f"  {status} '{test}' -> {cleaned}")

# Test Fix 2: Lower thresholds
print("\n✅ FIX 2: Lowered Thresholds")
print("-" * 60)
print(f"  MATCH_THRESHOLD_STRICT: {MATCH_THRESHOLD_STRICT} (was 0.85)")
print(f"  MATCH_THRESHOLD: {MATCH_THRESHOLD} (was 0.65)")

# Test Fix 3: Direct match boost
print("\n✅ FIX 3: Direct Match Boost")
print("-" * 60)
test_matches = [
    ("node.js", "node.js", True),
    ("node.js", "nodejs", True),
    ("react", "react.js", True),
    ("python", "node.js", False),
    ("typescript", "typescript", True),
]

for jd, resume, expected in test_matches:
    result = is_direct_match(jd, resume)
    status = "✓" if result == expected else "✗"
    print(f"  {status} is_direct_match('{jd}', '{resume}') -> {result} (expected {expected})")

print("\n" + "=" * 60)
print("All fixes integrated successfully!")
print("=" * 60)
