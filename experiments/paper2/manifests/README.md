# Paper 2 manifests

Before a mechanism, development, or sealed run, create an immutable JSON file:

```json
{
  "schema": "paper2_case_manifest_v1",
  "role": "mechanism_train",
  "case_uids": ["transaction_id:case_index"],
  "selection_rule": "predeclared deterministic rule",
  "created_before_results": true
}
```

Allowed roles are `mechanism_train`, `development`, and `sealed_held_out`.
The runner records the manifest content SHA256 before launching any job.  The
mechanism manifest must contain 12--20 train cases.  A sealed manifest must not
be edited or reused after its first completed run.
