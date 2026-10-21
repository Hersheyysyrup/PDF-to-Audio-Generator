def get_mmr_retriever(vector_store, k=4, fetch_k=20, lambda_mult=0.5):
    
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    return retriever 