from langchain_community.vectorstores import Chroma
from indexing_chroma import embeddings
import pandas as pd
from tqdm import tqdm
if __name__ == "__main__":
    vector_store = Chroma(
        collection_name="shipping",
        embedding_function=embeddings,
        persist_directory="db/shipping",
    )
    golden = pd.read_csv("golden_dataset.csv")
    queries = golden['query'].tolist()
    query_ids_at_3 = []
    for q in tqdm(queries):
        results = vector_store.similarity_search(
            q,
            k=3
        )
        query_ids_at_3.append(",".join([res.metadata["index"] for res in results]))

    golden["retrieved_ids_at_3"] = query_ids_at_3  
    golden.to_csv("experiment.csv", index=False)