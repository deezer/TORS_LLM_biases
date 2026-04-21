#A Study of Biases in LLM-Generated Musical Taste Profiles for Recommendation

This repository provides our Python code to reproduce the experiments from the paper "A Study of Biases in LLM-Generated Musical Taste Profiles for Recommendation". Submited to ACM Transactions on Recommender Systems. 

The data can be found in the [repository of the recsys version of this paper](https://github.com/deezer/recsys25_llm_biases). 

# Recommendation-as-retrieval experiments

This repository is aimed to test the recommendation-as-retrieval when using generated user profiles as queries and track metadata as documents.


## Installation 

```bash
make build
make run
hf auth login 
```

Place the data in `data/` following [the instructions](https://github.deezerdev.com/Research/TORS_LLM_biases/tree/main/data).

## Testing pre-trained encoders

We selected models relying on a variety of architectures and sizes.
The criteria for selecting the models were that they are open-source and achieve top retrieval results within their size category according to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

When running experiments with a model for the first time, embeddings are computed from scratch and saved in the output directory provided as argument (see below). Subsequent runs load the embeddings from the corresponding directory. To force recomputation of the embeddings, the corresponding directory must be manually removed.

### Average model size

`google/embeddinggemma-300m`

- encoder-decoder architecture, decoder uses Gemma 3 as backbone

- 1155 MB memory usage

- 768 embedding dimension

```bash
 python -m src.eval --input_dir data/ --output_dir output --model_name google/embeddinggemma-300m
  ```


### Smaller model size

`Alibaba-NLP/gte-multilingual-base`

- encoder-only

- 582 memory usage

- 768 embedding dimension

```bash
 python -m src.eval --input_dir data/ --output_dir output --model_name Alibaba-NLP/gte-multilingual-base
  ```


### Compact model size

`sentence-transformers/all-MiniLM-L6-v2` 

- encoder-only

- 87 memory usage

- 384 embedding size
 
 ```bash
 python -m src.eval --input_dir data/ --output_dir output --model_name sentence-transformers/all-MiniLM-L6-v2
  ```
