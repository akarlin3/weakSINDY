"""Deterministic unit tests for the shared AveryCorp AuditReport contract in weakSINDY."""
from __future__ import annotations

import json

from click.testing import CliRunner

from weaksindy import AuditReport, __version__, run_audit
from weaksindy.cli import main as cli_main

REQUIRED_AUDIT_KEYS = {
    "package",
    "version",
    "suite",
    "stage_role",
    "status",
    "verdict",
    "metrics",
    "thresholds",
    "seed",
}


def test_run_audit_contract_and_determinism() -> None:
    rep = run_audit(seed=42)
    assert isinstance(rep, AuditReport)
    assert rep.package == "weak-sindy"
    assert rep.version == __version__
    assert rep.suite == "Suite II: Scientific Identifiability & Null-Audit Suite"
    assert "Weak-Form Fractional PDE/ODE Identifiability" in rep.stage_role
    assert rep.status in ("PASS", "WARN", "HALT")
    assert rep.status == "PASS"
    assert "IDENTIFIABLE_SUPPORT_RECOVERED" in rep.verdict
    assert rep.seed == 42

    parsed = json.loads(rep.to_json())
    assert set(parsed.keys()) == REQUIRED_AUDIT_KEYS
    assert parsed["metrics"]["support_f1"] >= 0.90
    assert parsed["metrics"]["order_abs_error"] <= 0.08
    assert parsed["metrics"]["min_active_inclusion_prob"] >= 0.75
    assert rep.summary_line().startswith(f"[{rep.status}] weak-sindy v{__version__}")

    rep2 = run_audit(seed=42)
    assert rep.to_dict() == rep2.to_dict()


def test_cli_audit_json_and_missing_input_error() -> None:
    runner = CliRunner()
    res = runner.invoke(cli_main, ["--audit", "--json", "--seed", "42"])
    assert res.exit_code == 0, res.output
    parsed = json.loads(res.output)
    assert set(parsed.keys()) == REQUIRED_AUDIT_KEYS
    assert parsed["package"] == "weak-sindy"
    assert parsed["seed"] == 42

    res_missing = runner.invoke(cli_main, [])
    assert res_missing.exit_code != 0
    assert "Missing argument 'INPUT_PATH'" in res_missing.output
