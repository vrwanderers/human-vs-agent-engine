import asyncio
from random import Random

import pytest

from hva_engine.cognition import AgentIdentity, CognitiveProfile, RuntimeBehaviorPolicy
from hva_engine.fact_graph import AgentFactGraph, FactGraphError, FactStatus
from hva_engine.fact_store import InMemoryFactStore
from hva_engine.llm import LLMDecisionClient, LLMMessage, LLMResponse
from hva_engine.models import AgentTuning


def _graph() -> tuple[AgentFactGraph, InMemoryFactStore]:
    rng = Random(4)
    policy = RuntimeBehaviorPolicy.from_tuning(AgentTuning())
    profile = CognitiveProfile.sample(rng, "opponent", policy)
    identity = AgentIdentity.sample("Astra", profile, "opponent", rng)
    store = InMemoryFactStore()
    return AgentFactGraph.from_identity("agent-a", identity, store), store


def test_fact_graph_protects_identity_and_keeps_explicit_revision_history() -> None:
    graph, store = _graph()
    name_fact = next(
        fact for fact in graph.private_view()["facts"] if fact["predicate"] == "identity.name"
    )
    with pytest.raises(FactGraphError, match="reserved"):
        graph.propose(
            subject="agent-a",
            predicate="identity.name",
            value="Someone Else",
            basis_fact_ids=[name_fact["id"]],
        )
    first = graph.propose(
        subject="agent-a",
        predicate="preference.local",
        value={"drink": "tea"},
        basis_fact_ids=[name_fact["id"]],
    )
    with pytest.raises(FactGraphError, match="explicitly supersede"):
        graph.propose(
            subject="agent-a",
            predicate="preference.local",
            value={"drink": "coffee"},
            basis_fact_ids=[name_fact["id"]],
        )
    revised = graph.propose(
        subject="agent-a",
        predicate="preference.local",
        value={"drink": "coffee"},
        basis_fact_ids=[name_fact["id"]],
        supersedes_fact_id=first.id,
    )
    assert revised.revision == 2
    assert revised.supersedes == first.id
    assert store.facts[f"agent-a:{first.id}"]["status"] == FactStatus.SUPERSEDED
    assert len(store.rejections) == 2


class ProposalProvider:
    name = "proposal"

    async def complete(self, _request):
        return LLMResponse(
            '{"action_index":0,"reason":"test","fact_proposals":['
            '{"subject":"self","predicate":"preference.local",'
            '"object":{"drink":"tea"},"basis_fact_ids":["fact-0001"]}]}',
            "proposal-1",
            {},
            {},
        )


def test_llm_decision_can_return_bounded_fact_proposals() -> None:
    client = LLMDecisionClient(ProposalProvider())
    index, proposals, _response = asyncio.run(
        client.choose_action_and_facts([LLMMessage("user", "choose")], [{"type": "wait"}])
    )
    assert index == 0
    assert proposals[0]["predicate"] == "preference.local"
    assert proposals[0]["basis_fact_ids"] == ["fact-0001"]
