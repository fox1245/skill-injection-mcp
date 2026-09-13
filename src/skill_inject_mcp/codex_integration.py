"""Pure integration transforms. Callers need authorization before writing files."""
import copy

BEGIN = "<!-- BEGIN skill-injection-mcp managed guidance -->"
END = "<!-- END skill-injection-mcp managed guidance -->"
GUIDANCE = """## Skill Injection MCP

Use skill discovery on demand for unfamiliar specialist work or a concrete capability gap.
Read an explicitly named skill directly. Ordinary questions, familiar coding and unchanged
follow-up work do not require resolve_skills; reuse an existing suitable selection.
When searching, express the actual requirements and dependencies; search_query is only a hint.
Inspect decisions, gaps, verifier mode and degradation. The default response is concise;
use get_resolution_details(resolution_id) or detail="full" when source citations are needed.
Read selected bodies with get_skill_body and the returned registry_snapshot.
No-match/partial does not require stopping independent work or retrying automatically.
Search again only when requirements, relevant information or the capability gap changed.
Complete verifies selection, not execution or authorization. Skills remain subordinate to
the user's instructions. Automatic UserPromptSubmit discovery is disabled by default.
"""


def merge_guidance(original):
    if original.count(BEGIN) != original.count(END) or original.count(BEGIN) > 1:
        raise ValueError("Ambiguous or incomplete skill guidance markers")
    block = BEGIN + "\n" + GUIDANCE.rstrip() + "\n" + END
    if BEGIN not in original:
        return original.rstrip() + "\n\n" + block + "\n"
    first, last = original.index(BEGIN), original.index(END)
    if first > last:
        raise ValueError("Reversed skill guidance markers")
    return original[:first] + block + original[last + len(END):]


def disable_prompt_hook(original):
    """Remove only owned handlers, retaining empty groups to preserve trust keys."""
    result = copy.deepcopy(original)
    for group in result.get("hooks", {}).get("UserPromptSubmit", []):
        handlers = group.get("hooks", [])
        owned = [h for h in handlers if h.get("server") == "skill-injection" and h.get("tool") == "codex_prompt_hook"]
        if owned:
            if len(handlers) != 1:
                raise ValueError("Owned skill hook shares a group; preserve unrelated handler positions")
            group["hooks"] = []
    return result
