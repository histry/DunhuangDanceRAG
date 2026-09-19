# Paper 2 manifests

Before a mechanism, development, or sealed run, create an immutable JSON file:

```json
{
  "schema": "paper2_case_manifest_v1",
  "immutable": true,
  "role": "mechanism_train",
  "case_uids": ["transaction_id:case_index"],
  "source_case_uids": ["source_case_uid"],
  "implementation_commit": "full git SHA",
  "protocol_sha256": "frozen protocol SHA256",
  "source_bank_sha256": "teacher bank SHA256",
  "selection_rule": "predeclared deterministic rule",
  "created_before_results": true,
  "content_sha256": "canonical manifest SHA256"
}
```

Allowed roles are `mechanism_train`, `development`, and `sealed_held_out`.
Create manifests with `prepare_case_manifest.py`; the runner rejects edited
content, a different commit/protocol/bank, or missing split provenance.  The
mechanism manifest must contain 12--20 train cases.  Development and sealed
manifests exclude both prior transaction IDs and source-case IDs.  A sealed
launch writes a persistent receipt keyed by manifest content SHA: completion,
failure, or interruption forbids reusing that manifest from any output root.
