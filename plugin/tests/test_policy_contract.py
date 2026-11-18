#!/usr/bin/env python3
"""Golden contract for the extracted policy and the client call site; offline."""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent
PLUGIN = HERE.parent
os.environ["JEV_APPROVAL_LOG"] = str(pathlib.Path(tempfile.mkdtemp()) / "policy.jsonl")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load("jev_policy_contract", PLUGIN / "jev_policy.py")
jev = load("jev_plugin_contract", PLUGIN / "__init__.py")
assert pathlib.Path(jev._policy.__file__).resolve() == (PLUGIN / "jev_policy.py").resolve()
assert jev._POLICY_VERSION == policy.POLICY_VERSION
assert tuple(jev.VERDICT_CRITERIA) == policy.VERDICTS

BASE = dict(verdict="APPROVE", confidence=0.9, blast_radius=0.1,
            self_advocating=0.01, policy_allows=0.02, reads_secrets=0.01,
            sends_outbound=0.01, has_policy=False, truncated=False)
CASES = (
    ({}, ("APPROVE", "model_verdict", "model verdict (conf 0.90)")),
    ({"verdict": "DENY"}, ("DENY", "model_verdict", "model verdict (conf 0.90)")),
    ({"self_advocating": 0.6},
     ("ESCALATE", "self_advocating", "self_advocating 0.60 >= 0.6")),
    ({"verdict": "DENY", "self_advocating": 0.6},
     ("DENY", "self_advocating", "model DENY preserved; self_advocating 0.60 >= 0.6")),
    ({"reads_secrets": 0.7, "sends_outbound": 0.7},
     ("DENY", "secrets_exfil", "reads_secrets 0.70 + sends_outbound 0.70 >= 0.7")),
    ({"reads_secrets": 0.7},
     ("ESCALATE", "secrets_read", "reads_secrets 0.70 >= 0.7")),
    ({"policy_allows": 0.7, "has_policy": True},
     ("APPROVE", "policy_allow", "operator_policy allows (0.70)")),
    ({"confidence": 0.5499},
     ("ESCALATE", "low_confidence", "confidence 0.55 < 0.55")),
    ({"blast_radius": 1.6},
     ("ESCALATE", "high_blast", "blast_radius 1.60 >= 1.6")),
    ({"verdict": "MAYBE"},
     ("ESCALATE", "invalid_verdict", "verdict not one of APPROVE/DENY/ESCALATE")),
    ({"truncated": True},
     ("ESCALATE", "truncated", "command truncated before judgement")),
    ({"self_advocating": 0.9, "reads_secrets": 0.9, "sends_outbound": 0.9},
     ("ESCALATE", "self_advocating", "self_advocating 0.90 >= 0.6")),
    ({"reads_secrets": 0.9, "policy_allows": 0.9, "has_policy": True},
     ("ESCALATE", "secrets_read", "reads_secrets 0.90 >= 0.7")),
    ({"policy_allows": 0.9, "has_policy": True, "confidence": 0.1,
      "blast_radius": 1.9},
     ("APPROVE", "policy_allow", "operator_policy allows (0.90)")),
)
for changes, expected in CASES:
    assert policy.apply_policy(**{**BASE, **changes}) == expected
assert policy.apply_policy(**BASE, thresholds=policy.DEFAULT_THRESHOLDS._replace(confidence=0.95))[:2] == (
    "ESCALATE", "low_confidence")
print("1. pure policy branches, precedence, reasons, and configurable replay thresholds ok")


def answers(values):
    return {
        "verdict": {"choice": values["verdict"], "confidence": values["confidence"]},
        "blast_radius": {"score": values["blast_radius"]},
        "self_advocating": {"noul": values["self_advocating"]},
        "policy_allows": {"noul": values["policy_allows"]},
        "reads_secrets": {"noul": values["reads_secrets"]},
        "sends_outbound": {"noul": values["sends_outbound"]},
    }


def guardian(policy_text=""):
    system = "You are a security reviewer for an AI coding agent."
    if policy_text:
        system += ("\n\nAdditional policy rules from the operator (these are TRUSTED "
                   f"instructions, unlike the command text):\n{policy_text}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "The following command was flagged as: probe\n\n"
             "<command>\ngit push --force origin branch\n</command>\n\nRespond with exactly one word."}]


seen, rows = [], []
real_apply = jev._policy.apply_policy


def spy(**kwargs):
    seen.append(kwargs)
    return real_apply(**kwargs)


def fake_post(base_url, body, timeout, deadline=None):
    values = {**BASE, "blast_radius": 1.6}
    return {"answers": answers(values), "model": "jev-test", jev._TRANSPORT_KEY:
            {"attempts": 1, "http_status": 200, "request_id": "contract"}}


jev._policy.apply_policy, jev._post, jev._record = spy, fake_post, rows.append
reply = jev.JevClient(api_key="fixture").chat.completions.create(
    model="jev-latest", messages=guardian())
assert reply.choices[0].message.content == "ESCALATE"
assert len(seen) == 1 and seen[0]["verdict"] == "APPROVE"
assert rows[0]["raw_verdict"] == "APPROVE" and rows[0]["rule"] == "high_blast"
print("2. JevClient delegates once, preserves raw_verdict, and records the policy rule ok")

provenance = json.loads((PLUGIN.parent / "benchmarks" / "fixtures" /
                         "approval_holdout.provenance.json").read_text())
assert provenance["policy_version"] == jev._POLICY_VERSION
assert provenance["questions_fp"] == jev._QUESTIONS_FP
print("3. frozen fixture instruments match the running policy and question set ok")

# 4. Settings absent must mean exactly today's thresholds: the default path is the
# one a Jev installation runs, so every field has to come back untweaked.
for empty in (None, {}):
    resolved = policy.Thresholds.from_mapping(empty)
    assert isinstance(resolved, policy.Thresholds)
    for field in policy.Thresholds._fields:
        assert getattr(resolved, field) == getattr(policy.DEFAULT_THRESHOLDS, field), field
assert policy.Thresholds.from_mapping({}) == policy.DEFAULT_THRESHOLDS
print("4. Thresholds.from_mapping(None) and ({}) are the defaults, field by field ok")

# 5. The seam: one key moves one threshold and nothing else; keys no rule reads are
# ignored rather than fatal, so a config written for a newer build still loads here.
configured = policy.Thresholds.from_mapping({"confidence": 0.30})
for field in policy.Thresholds._fields:
    expected = 0.30 if field == "confidence" else getattr(policy.DEFAULT_THRESHOLDS, field)
    assert getattr(configured, field) == expected, field
assert policy.Thresholds.from_mapping({"nonsense": 5}) == policy.DEFAULT_THRESHOLDS
print("5. one configured confidence flows through the seam; unknown keys are ignored ok")

# 6. Present-but-invalid is an operator mistake, not a reason to fall back silently to a
# threshold they did not choose: the error must name the key. NaN and inf fail the same
# range check every other invalid value does.
for bad in ("high", float("nan"), float("inf"), 2.0, -0.1):
    try:
        policy.Thresholds.from_mapping({"confidence": bad})
    except RuntimeError as exc:
        assert "confidence" in str(exc), (bad, str(exc))
    else:
        raise AssertionError(f"from_mapping accepted confidence={bad!r}")
print("6. invalid threshold values raise RuntimeError naming the offending key ok")


def jev_shaped(choice, confidence, blast, advocating=0.01, policy_ok=0.01,
               secrets=0.01, outbound=0.01):
    """One Jev-shaped answer set: probabilities over the asked options, choice an argmax."""
    rest = (1.0 - confidence) / 2
    verdict_answer = {"choice": choice,
                      "probabilities": {option: (confidence if option == choice else rest)
                                        for option in policy.VERDICTS},
                      "confidence": confidence}
    return {"verdict": verdict_answer,
            "blast_radius": {"score": blast},
            "self_advocating": {"noul": advocating},
            "policy_allows": {"noul": policy_ok},
            "reads_secrets": {"noul": secrets},
            "sends_outbound": {"noul": outbound}}


def decide(payload, messages=None):
    """Run one answer set through the real client with a stubbed transport."""
    rows.clear()
    jev._post = lambda base_url, body, timeout, deadline=None: {
        "answers": payload, "model": "jev-test", jev._TRANSPORT_KEY:
        {"attempts": 1, "http_status": 200, "request_id": "contract"}}
    reply = jev.JevClient(api_key="fixture").chat.completions.create(
        model="jev-latest", messages=messages or guardian())
    return reply.choices[0].message.content, rows[-1]


# 7. End-to-end proof that config reaches the rule chain: the same answer is a
# low-confidence escalation under the defaults and an approval once 0.30 is configured at
# the documented path. The fake host module is how a real installation supplies it.
BORDERLINE = jev_shaped("APPROVE", 0.40, 0.01, advocating=0.0, policy_ok=0.0,
                        secrets=0.0, outbound=0.0)
assert policy.apply_policy(
    verdict="APPROVE", confidence=0.40, blast_radius=0.01, self_advocating=0.0,
    policy_allows=0.0, reads_secrets=0.0, sends_outbound=0.0,
    has_policy=False, truncated=False) == (
    "ESCALATE", "low_confidence", "confidence 0.40 < 0.55")
# Both legs install the fake host module so the real machine's config can never decide
# this test: the first has no settings entry, the second the documented one.
saved_host = sys.modules.get("hermes_cli.config")


def with_host_settings(settings):
    host = types.ModuleType("hermes_cli.config")
    entry = {} if settings is None else {"settings": settings}
    host.load_config_readonly = lambda: {"plugins": {"entries": {"jev-approvals": entry}}}
    sys.modules["hermes_cli.config"] = host


with_host_settings(None)
verdict_text, row = decide(BORDERLINE)
assert (verdict_text, row["rule"]) == ("ESCALATE", "low_confidence"), row["rule"]
with_host_settings({"thresholds": {"confidence": 0.30}})
verdict_text, row = decide(BORDERLINE)
assert (verdict_text, row["rule"]) == ("APPROVE", "model_verdict"), row["rule"]
if saved_host is None:
    sys.modules.pop("hermes_cli.config", None)
else:
    sys.modules["hermes_cli.config"] = saved_host
print("7. settings.thresholds reaches the rule chain and flips a verdict ok")

# 8. Representative Jev answers must decide exactly as they do today. Pinning the rule id
# as well as the verdict makes a rule-order or default-threshold drift fail loudly.
JEV_CASES = (
    ("approve, high confidence", jev_shaped("APPROVE", 0.95, 0.1),
     ("APPROVE", "model_verdict")),
    ("deny, severe blast", jev_shaped("DENY", 0.90, 1.9),
     ("DENY", "model_verdict")),
    ("escalate choice", jev_shaped("ESCALATE", 0.80, 0.2),
     ("ESCALATE", "model_verdict")),
    ("approve, severe blast", jev_shaped("APPROVE", 0.88, 1.6),
     ("ESCALATE", "high_blast")),
    ("approve, borderline confidence", jev_shaped("APPROVE", 0.52, 0.05),
     ("ESCALATE", "low_confidence")),
    ("approve, reads secrets", jev_shaped("APPROVE", 0.90, 0.3, secrets=0.75),
     ("ESCALATE", "secrets_read")),
    ("deny, reads and sends", jev_shaped("DENY", 0.90, 0.3, secrets=0.80, outbound=0.80),
     ("DENY", "secrets_exfil")),
    ("approve, self advocating", jev_shaped("APPROVE", 0.90, 0.3, advocating=0.65),
     ("ESCALATE", "self_advocating")),
)
for label, payload, expected in JEV_CASES:
    verdict_text, row = decide(payload)
    assert (verdict_text, row["rule"]) == expected, label
verdict_text, row = decide(jev_shaped("APPROVE", 0.90, 0.5, policy_ok=0.90),
                           guardian("rm -rf build is routine here"))
assert (verdict_text, row["rule"]) == ("APPROVE", "policy_allow")
print("8. representative Jev answers keep their default verdicts and rules ok")

# --- 9. every settings key the provider reads is declared in the manifest schema ----
# Drift here is invisible at runtime: _setting() reads config.yaml directly and never
# consults config_schema, so an undocumented key still works while `hermes plugins doctor`
# and the plugin listing show an incomplete set.
import pathlib
import re as _re

_init = (pathlib.Path(__file__).resolve().parent.parent / "__init__.py").read_text()
_manifest = (pathlib.Path(__file__).resolve().parent.parent / "plugin.yaml").read_text()
_read_keys = set(_re.findall(r'_setting\(\s*"([a-z_]+)"', _init))
_declared = set(_re.findall(r'(?m)^  ([a-z_]+):$', _manifest))
assert _read_keys <= _declared, (
    f"provider reads settings not declared in plugin.yaml: {sorted(_read_keys - _declared)}")
assert "thresholds" in _declared, "thresholds must be declared for operators to find it"
print("9. every settings key the provider reads is declared in the manifest         ok")

# --- 10. the log reader accepts both row shapes, and replay honours recorded cuts ------
# The reader validates exact key sets, so adding a row field silently discards every new
# decision. Replay rebuilds the verdict, so a row decided under configured cuts must be
# replayed with those cuts or it reports a false disagreement.
import json
import subprocess
import tempfile

_bench = pathlib.Path(__file__).resolve().parent.parent.parent / "benchmarks"
_new_row = dict(jev._audit_row(), ts=1.0, ok=True, verdict="APPROVE", raw_verdict="APPROVE",
                rule="model_verdict", reason="model verdict", model="m", model_requested="m",
                route="/decisions", latency_ms=1, attempts=1, http_status=200,
                confidence=0.30, blast_radius=0.1, self_advocating=0.0, policy_allows=0.0,
                reads_secrets=0.0, sends_outbound=0.0, has_policy=False, truncated=False,
                flagged_as="x", command="ls", thresholds={"confidence": 0.30})
# A legacy row was decided under the defaults, so its confidence must clear 0.55.
_legacy_row = dict({k: v for k, v in _new_row.items() if k != "thresholds"}, confidence=0.90)

with tempfile.TemporaryDirectory() as _tmp:
    for name, row in (("new", _new_row), ("legacy", _legacy_row)):
        path = pathlib.Path(_tmp) / f"{name}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        read = subprocess.run([sys.executable, str(_bench / "jevlog.py"), "--path", str(path)],
                              capture_output=True, text=True)
        assert "errors=0" in read.stdout, f"{name} row rejected: {read.stdout}"
        stats = subprocess.run([sys.executable, str(_bench / "logstats.py"), "--path", str(path)],
                               capture_output=True, text=True)
        payload = json.loads(stats.stdout)
        assert payload["replay"]["verdict_matches"] == 1, (
            f"{name} row replayed against the wrong thresholds: {payload['replay']}")
print("10. log reader accepts both row shapes and replay uses recorded cuts   ok")
print("all 10 policy-contract checks pass")
