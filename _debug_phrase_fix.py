from syncom.phrase_fix import (
    parse_phrase_report, _load_comment_texts_from_psv, _load_persona_contexts_from_psv,
    build_rule_ngrams, triage_repeated_phrases, rewrite_comment_for_phrase
)
from syncom.rewriter import RewriterConfig
import json, os

report_path = 'HHS-ONC-2026-0001/synthetic_comments/synthetic_phrase_report.md'
psv_path = 'HHS-ONC-2026-0001/synthetic_comments/synthetic.txt'
world_model_path = 'HHS-ONC-2026-0001/world_model.json'

# Load rule text
with open('HHS-ONC-2026-0001/rule/rule.txt', 'r', encoding='utf-8', errors='replace') as f:
    rule_text = f.read()

repeated = parse_phrase_report(report_path)
rule_ngrams = build_rule_ngrams(rule_text, world_model_path)
expected, suspicious = triage_repeated_phrases(repeated, rule_ngrams)
print(f'Expected: {len(expected)}, Suspicious: {len(suspicious)}')

comment_texts = _load_comment_texts_from_psv(psv_path)
persona_contexts = _load_persona_contexts_from_psv(psv_path)

# Pick the first suspicious phrase and first affected comment
rp = suspicious[0]
match = rp.matches[0]
cid = match.comment_id
print(f'\nTesting rewrite for: "{rp.phrase}" in {cid}')
print(f'Sentence: {match.sentence}')
print(f'Comment text length: {len(comment_texts.get(cid, ""))}')
print(f'Persona context: {persona_contexts.get(cid, {})}')

# Try a single rewrite
cfg = RewriterConfig()
print(f'\nRewriterConfig.is_available(): {cfg.is_available()}')
print(f'rewrite_api_key[:8]: {cfg.rewrite_api_key[:8]}...')

try:
    result = rewrite_comment_for_phrase(
        comment_text=comment_texts[cid],
        phrase=rp.phrase,
        sentence=match.sentence,
        total_occurrences=rp.count,
        persona_context=persona_contexts.get(cid, {}),
        config=cfg,
    )
    print(f'\nRewrite succeeded! New length: {len(result)}')
    print(f'First 200 chars: {result[:200]}')
except Exception as e:
    import traceback
    print(f'\nRewrite FAILED: {e}')
    traceback.p
