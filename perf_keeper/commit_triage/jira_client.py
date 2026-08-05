"""JIRA client for Phase 2 — extends commons JiraClient with attachment support."""

from __future__ import annotations

from io import BytesIO

from commons.jira import JiraClient as _Base
from commons.jira.exceptions import JiraQueryError, JiraUpdateError


class JiraClient(_Base):
    """Adds attachment download/upload on top of the commons base client."""

    def get_markdown_attachment(self, jira_key: str) -> str:
        """Return the text of the first Markdown attachment on the issue."""
        try:
            issue = self.jira.issue(jira_key)
        except Exception as e:
            raise JiraQueryError(f"Failed to fetch {jira_key}: {e}") from e

        for att in getattr(issue.fields, "attachment", []):
            if att.filename.lower().endswith(".md"):
                return att.get().decode("utf-8", errors="replace")

        raise FileNotFoundError(f"No Markdown attachment found on {jira_key}")

    def upload_attachment(self, jira_key: str, filename: str, content: str) -> None:
        """Upload a text file as a new attachment to the issue."""
        buf = BytesIO(content.encode("utf-8"))
        buf.name = filename
        try:
            self.jira.add_attachment(issue=jira_key, attachment=buf, filename=filename)
        except Exception as e:
            raise JiraUpdateError(
                f"Failed to upload {filename!r} to {jira_key}: {e}"
            ) from e
