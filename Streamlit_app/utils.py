from typing import List, Dict, Any, Tuple


def query(pipeline, query, filters={}, top_k_reader=5, top_k_retriever=5) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}

    response = pipeline.run(query=query, params=params)

    # Format response
    results = []
    answers = response["answers"]
    for answer in answers:
        if answer.to_dict().get("answer", None):
            results.append(
                {
                    "context": "..." + answer.to_dict()["context"] + "...",
                    "answer": answer.to_dict().get("answer", None),
                    "source": answer.to_dict()["meta"]["name"],
                    "relevance": round(answer.to_dict()["score"] * 100, 2),
                    "document": [doc for doc in response["documents"] if doc.to_dict()["id"] == answer.to_dict()["document_id"]][0],
                    "offset_start_in_doc": answer.to_dict()["offsets_in_document"][0]["start"],
                    "_raw": answer
                }
            )
        else:
            results.append(
                {
                    "context": None,
                    "answer": None,
                    "document": None,
                    "relevance": round(answer.to_dict()["score"] * 100, 2),
                    "_raw": answer,
                }
            )
    return results, response


def get_backlink(result) -> Tuple[str, str]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get("title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None