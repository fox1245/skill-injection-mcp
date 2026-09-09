from skill_inject_mcp.retrieve.hybrid import reciprocal_rank_fusion


def test_rrf_tie_break_dense_rank_then_skill_id():
    dense = [("b-skill", 0.9), ("a-skill", 0.8)]
    sparse = [("a-skill", 1.0, 1), ("b-skill", 0.5, 2)]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    scores = {r.skill_id: r.ranking_score for r in fused}
    assert abs(scores["a-skill"] - scores["b-skill"]) < 1e-12
    assert fused[0].skill_id == "b-skill"
    assert fused[1].skill_id == "a-skill"
