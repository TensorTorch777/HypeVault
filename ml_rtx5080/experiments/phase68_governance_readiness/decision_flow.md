# Decision flow — fail-closed acquisition recommendation state machine

**Non-authorization notice:** Every state below leaves `ACQUISITION_AUTHORIZED=false` and all other production/readiness gates closed. No environment variable, Python constant, JSON policy, scope contract, network permission, or CLI flag is updated by advancing this flow.

```
                  ┌─────────┐
                  │  DRAFT  │
                  └────┬────┘
                       │ evidence + reviews assembled
           ┌───────────┼───────────┐
           ▼           ▼           ▼
       ┌───────┐  ┌────────────┐  ┌────────────────────────────────────────┐
       │ NO_GO │  │CONDITIONAL │  │ GO_RECOMMENDED_SEPARATE_AUTHORIZATION  │
       │       │  │   _NO_GO   │  │           _REQUIRED                    │
       └───────┘  └─────┬──────┘  └───────────────────┬────────────────────┘
                        │ remediate + re-review       │
                        └──────────► DRAFT            │
                                                      ▼
                                    Separate future authorization phase
                                    (explicit human/institutional decision)
                                    — not Phase 68 —
```

## States

### 1. `DRAFT`

- Evidence collection and review preparation are incomplete.
- Template default status.
- Record may be edited by the preparer; prior review outcomes must not be silently rewritten—append notes or version the record.

### 2. `NO_GO`

- A mandatory applicable condition fails, or an unresolved critical blocker remains.
- Acquisition consideration stops until a new draft cycle is explicitly opened after remediation evidence exists.

### 3. `CONDITIONAL_NO_GO`

- Remediation or additional evidence is required before reconsideration.
- Each condition lists owner, evidence needed, and target review date.
- Re-entry is only to `DRAFT` after conditions are addressed—not automatic promotion to a GO recommendation.

### 4. `GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED`

- Reviewers recommend that a **separate** authorization review be scheduled based on assembled evidence.
- **Acquisition remains unauthorized.**
- Does not set `ACQUISITION_AUTHORIZED`, scraper `--allow-network`, `DATA_READINESS`, inference, publication, or promotion gates.
- Consuming systems and humans must treat this as a recommendation token only.

## Participants

| Action | Who |
|---|---|
| Prepare / update draft | Record preparer / custodian |
| Rights, privacy, GT, integrity, security, budget reviews | Named reviewers in the decision record |
| Set overall outcome | Lead reviewer + recorded concurrence rules in the record |
| Open a future authorization phase | Institutional / project decision-maker **outside** Phase 68 artifacts |

Vacant required reviewer seats ⇒ outcome cannot leave `DRAFT` (or must be `NO_GO` if a deadline lapses with vacancies).

## Conflicts and withdrawal

- Conflicting reviewer outcomes without adjudication ⇒ `NO_GO` or `CONDITIONAL_NO_GO`.
- Withdrawn approval, expired license evidence, or superseded terms ⇒ reopen as `DRAFT`/`NO_GO`; prior recommendation is void.
- Suspected bypass of final-test/OOD locks or fabricated evidence ⇒ immediate `NO_GO` and security escalation (owner named in security plan).

## Evidence standard

Evidence means durable references (title, URI/path, retrieval date, checksum when retained), not mere checklist ticks. Human judgment is required for legal rights, ethics applicability, and factual accuracy—JSON Schema only checks structure.

## What a future authorization phase would still need

Even after `GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED`, a later phase would still need, at minimum:

1. Explicit written authorization to change `ACQUISITION_AUTHORIZED` (or an equivalent future source of truth), with named approver and date.
2. Confirmation that live gates (`DATA_READINESS`, inference, publication, promotion) remain separately controlled.
3. Operational runbook limits (sources, volume, budget, retention) matching the approved record.
4. No expansion beyond the approved source inventory without a new decision record.

Phase 68 does not perform those steps.
