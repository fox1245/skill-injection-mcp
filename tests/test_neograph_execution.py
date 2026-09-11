from concurrent.futures import ThreadPoolExecutor
import pytest

from skill_inject_mcp.neograph_runtime import run_stages
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest


def test_native_failure_does_not_execute_downstream():
    seen = []
    def fail():
        raise RuntimeError('stage failed')
    with pytest.raises(RuntimeError, match='stage failed'):
        run_stages('fail', [('one', lambda: seen.append(1)), ('fail', fail), ('later', lambda: seen.append(3))])
    assert seen == [1]


def test_native_invocations_do_not_share_callbacks():
    def run(i):
        seen = []
        result = run_stages('parallel_requests', [('one', lambda: seen.append(i)), ('two', lambda: seen.append(i + 1))])
        assert seen == [i, i + 1]
        return result['run_id']
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(run, range(20)))
    assert len(set(ids)) == 20


def test_skill_resolution_records_native_stage_execution(engine):
    result = engine.resolve(SkillInjectRequest(requirements=[Requirement(id='pip', description='Install Python packages')]))
    assert result.execution[0]['executor'] == 'neograph-engine'
    assert result.execution[0]['nodes'] == ['retrieve', 'verify', 'bind']
