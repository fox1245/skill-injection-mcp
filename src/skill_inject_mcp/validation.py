"""Reference validation runs before indexing or any remote request."""
from graphlib import CycleError, TopologicalSorter

from skill_inject_mcp.schemas import SkillInjectRequest, ValidationErrorItem


def validate_request(request: SkillInjectRequest) -> list[ValidationErrorItem]:
    errors = []
    ids = set()
    for req in request.requirements:
        if req.id in ids:
            errors.append(ValidationErrorItem(
                code="duplicate_requirement_id", message=f"Duplicate requirement '{req.id}'", path=req.id,
            ))
        ids.add(req.id)
    graph = {r.id: list(r.depends_on) for r in request.requirements}
    for req in request.requirements:
        for dep in req.depends_on:
            if dep not in ids:
                errors.append(ValidationErrorItem(
                    code="unknown_requirement_dependency",
                    message=f"Requirement '{req.id}' depends on unknown '{dep}'", path=req.id,
                ))
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as exc:
        errors.append(ValidationErrorItem(
            code="requirement_dependency_cycle", message=f"Requirement cycle: {exc.args[1]}",
        ))
    if request.draft_plan is not None:
        steps = set()
        for step in request.draft_plan.steps:
            if step.id in steps:
                errors.append(ValidationErrorItem(
                    code="duplicate_step_id", message=f"Duplicate plan step '{step.id}'", path=step.id,
                ))
            steps.add(step.id)
            for rid in step.requirement_ids:
                if rid not in ids:
                    errors.append(ValidationErrorItem(
                        code="unknown_plan_requirement",
                        message=f"Step '{step.id}' references unknown requirement '{rid}'", path=step.id,
                    ))
    return errors
