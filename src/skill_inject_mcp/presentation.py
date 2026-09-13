"""Bounded process-local detail storage; summaries never change verifier results."""
from collections import OrderedDict
import time
import uuid

from skill_inject_mcp.schemas import SkillInjectResponse


class ResolutionDetails:
    def __init__(self, max_entries=32, max_bytes=4 * 1024 * 1024, ttl=900, clock=time.monotonic):
        self.max_entries, self.max_bytes, self.ttl, self.clock = max_entries, max_bytes, ttl, clock
        self.entries = OrderedDict()
        self.bytes = 0

    def _remove(self, key):
        _, _, size = self.entries.pop(key)
        self.bytes -= size

    def _expire(self):
        for key, (expires, _, _) in list(self.entries.items()):
            if expires <= self.clock():
                self._remove(key)

    def present(self, result: SkillInjectResponse, detail="summary") -> SkillInjectResponse:
        self._expire()
        full = result.model_copy(deep=True)
        full.output_detail = "full"
        full.resolution_id = uuid.uuid4().hex
        size = len(full.model_dump_json().encode())
        if size > self.max_bytes or self.max_entries < 1:
            # Never return a shortened response whose full evidence is unavailable.
            full.resolution_id = None
            return full
        while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
            self._remove(next(iter(self.entries)))
        self.entries[full.resolution_id] = (self.clock() + self.ttl, full, size)
        self.bytes += size
        if detail == "full":
            return full.model_copy(deep=True)
        summary = full.model_copy(deep=True)
        summary.output_detail = "summary"
        summary.execution = []
        summary.evidence = []
        summary.notes = []
        for check in summary.checks:
            check.evidence = []
            check.citations = []
            if check.reason and len(check.reason) > 360:
                check.reason = check.reason[:360] + "… (full reason in get_resolution_details)"
        return summary

    def get(self, resolution_id):
        self._expire()
        if resolution_id not in self.entries:
            raise KeyError("Resolution details expired, were evicted, or belong to another server process")
        return self.entries[resolution_id][1].model_copy(deep=True)
