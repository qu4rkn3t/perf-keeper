{regression}

## Commits

{pr_blocks}

---

Commit keys in this batch: {commit_keys}

## Response Format

Up to 15 entries ordered by triage_score descending.

```json
{
  "rankings": [
    {
      "commit_key": "<string>",
      "triage_score": 85,
      "confidence": "<low | medium | high>",
      "reasoning": "<string>"
    }
  ]
}
```
