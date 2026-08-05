{regression}

## Commits

{pr_blocks}

---

Commit keys in this batch: {commit_keys}

## Response Format

One entry per commit key listed above.

```json
{
  "decisions": [
    {
      "commit_key": "<string>",
      "worth_investigating": <boolean>
    }
  ]
}
```
