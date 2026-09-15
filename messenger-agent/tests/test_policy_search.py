from types import SimpleNamespace

from app.agent.tools import policy_search


class FakePineconeClient:
    class Inference:
        async def embed(self, **kwargs):
            assert kwargs["model"] == "llama-text-embed-v2"
            assert kwargs["parameters"]["input_type"] == "query"
            return SimpleNamespace(data=[SimpleNamespace(values=[0.1, 0.2])])

    inference = Inference()


class FakePineconeIndex:
    async def query(self, **kwargs):
        assert kwargs["namespace"] == "faq"
        assert kwargs["include_metadata"] is True
        return SimpleNamespace(
            matches=[SimpleNamespace(metadata={"text": "Returns: 7 days, unworn."})]
        )


async def test_policy_search_uses_pinecone_rag():
    result = await policy_search.ainvoke(
        {"question": "How long can I return an item?"},
        config={
            "configurable": {
                "pinecone_client": FakePineconeClient(),
                "pinecone_index": FakePineconeIndex(),
            }
        },
    )

    assert result == "Returns: 7 days, unworn."