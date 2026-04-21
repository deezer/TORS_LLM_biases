import os
import ast
import json
import random
import logging
import argparse
from collections import defaultdict
from typing import Dict, Iterable, Set, Optional


import torch
import numpy as np
import pandas as pd
from tqdm import trange

from evaluator import InformationRetrievalEvaluator

from sentence_transformers import SentenceTransformer, util, models


# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# ---------------------------
# Constants
# ---------------------------

NB_NEGATIVES = 1000
NB_POSITIVES = 10
EMB_CHUNK_SIZE = 16
SEEDS = [1, 2, 3, 4, 5]

#model
model = None
model_name = None

# ---------------------------
# Helpers
# ---------------------------

def parse_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--input_dir",
		type=str,
		required=True,
		help="Directory where test and train data is stored"
	)

	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
		help="Directory where to save the results"
	)

	model_group = parser.add_mutually_exclusive_group(required=True)
	model_group.add_argument(
		"--model_name",
		type=str,
		help="SentenceTransformer model name (e.g. all-MiniLM-L6-v2)"
	)
	model_group.add_argument(
		"--model_path",
		type=str,
		help="Path to a local SentenceTransformer model (e.g. own fine-tuned model)"
	)

	return parser.parse_args()

def get_model_identifier(args) -> str:
	"""
	Get the model's name
	"""
	model_id = args.model_name or args.model_path
	if not model_id:
		raise ValueError("Either model_name or model_path must be provided")
	
	# Get clean name for filesystem/logging
	model_name = os.path.basename(os.path.normpath(model_id))
	model_name = model_name.replace("/", "_").replace(" ", "_")
	return model_name

def load_model(args, model=None) -> SentenceTransformer:
	"""Load and return the model (or return existing if already loaded)"""
	if model is not None:
		return model
	
	model_identifier = args.model_name or args.model_path
	if not model_identifier:
		raise ValueError("Either model_name or model_path must be provided")
	
	return SentenceTransformer(model_identifier, trust_remote_code=True)


def embed(
	model: SentenceTransformer,
	corpus: list[str], 
	embed_prompt: str = '', 
	chunk_size: int = EMB_CHUNK_SIZE):
	
	corpus = [embed_prompt + t for t in corpus]
	if len(corpus) < chunk_size:
		# Single batch
		return (
			model.encode(
				corpus,
				show_progress_bar=True,
				convert_to_tensor=True,
			).detach().cpu()
		)

	# Chunked processing
	all_embs = []
	for start in trange(0, len(corpus), chunk_size, desc="Corpus Chunks"):
		end = start + chunk_size
		embs = model.encode(
				corpus[start:end],
				show_progress_bar=False,
				convert_to_tensor=True,
				)
		all_embs.append(embs.detach().cpu())
	return torch.cat(all_embs, dim=0)


def build_all_test_tracks(
    test: pd.DataFrame,
    test_cases: list[str],
    n_negatives: int = NB_NEGATIVES,
    seed: Optional[int] = None,
) -> Dict:
    """Build test tracks for all users at once."""
    if seed is not None:
        random.seed(seed)
    
    # Group by user_id and set, get unique products
    grouped = test.groupby(['user_id', 'set'])['product_id'].apply(lambda x: list(x.unique()))
    
    test_tracks_per_user = {}
    for user_id in test.user_id.unique():
        # Sample negatives for this user
        negatives_key = (user_id, 'negative')
        all_negatives = grouped.get(negatives_key, [])
        
        if len(all_negatives) < n_negatives:
            raise ValueError(
                f"User {user_id}: not enough negatives: requested {n_negatives}, got {len(all_negatives)}"
            )
        negatives = random.sample(all_negatives, n_negatives)
        
        # Build tracks for all test cases
        test_tracks_per_user[user_id] = {
            test_case: {
                '+': grouped.get((user_id, test_case), []),
                '-': negatives
            }
            for test_case in test_cases
        }
    
    return test_tracks_per_user

def extract_genre(value):
    """Extract genre from string representation of tuple or actual tuple"""
    if pd.isna(value):
        return None
    
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else None
    
    if isinstance(value, str):
        value = value.strip()
        if value in ['', 'nan', 'None', '()']:
            return None
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return parsed[0]
            return None  # ← Add this! Empty tuple/list case
        except:
            # Plain string genre like 'Blues' - return as-is
            return value
    
    return None

def build_track_embeddings(
    test: pd.DataFrame,
    test_tracks_per_user: Dict,
    model: SentenceTransformer,
    model_name: str
) -> (Dict, Dict):
    """
    Build track embeddings for all unique tracks in tracks_test
    """

    # Collect all unique track IDs
    unique_track_ids = set()
    for user_tracks in test_tracks_per_user.values():
        for test_case_tracks in user_tracks.values():
            unique_track_ids.update(test_case_tracks['+'])
            unique_track_ids.update(test_case_tracks['-'])
    
    # Filter test DataFrame once for all needed tracks
    tracks_data = test[test.product_id.isin(unique_track_ids)].drop_duplicates(subset=['product_id'])
    
    # Set index for lookups
    tracks_data = tracks_data.set_index('product_id')
    
    # Build tracks dict 
    tracks = {}

    for track_id in unique_track_ids:
        if track_id not in tracks_data.index:
            continue

        row = tracks_data.loc[track_id]

        parts = []

        # Title
        if pd.notna(row["song_title"]):
            parts.append(f"title: {row['song_title']}")

        # Artist
        if pd.notna(row["artist_name"]):
            parts.append(f"artist: {row['artist_name']}")

        # Country
        if pd.notna(row["country_name"]):
            parts.append(f"origin country: {row['country_name']}")

        # Year
        if pd.notna(row["year_release"]):
            y = int(row["year_release"])
            if 1972 <= y <= 2025:
                parts.append(f"album release date: {y}")

        # Genres
        main_genre = extract_genre(row["Main"])
        if main_genre:
            parts.append(f"main genre: {main_genre}")

        secondary_genre = extract_genre(row["Secondary"])
        if secondary_genre:
            parts.append(f"secondary genre: {secondary_genre}")

        tracks[track_id] = "\n".join(parts)
        
    if not tracks:
        return {}, {}
    
    corpus_ids = list(tracks.keys())
    corpus_content = list(tracks.values())
    
    if model_name == 'embeddinggemma-300m':
        embeddings = embed(
            model, 
            corpus_content, 
            embed_prompt='title: "none" | text: '
        )
    else:
        embeddings = embed(model, corpus_content)
    
    return dict(zip(corpus_ids, corpus_content)), dict(zip(corpus_ids, embeddings))


def build_query_embeddings(
    users_ids: Iterable,
    train: pd.DataFrame,
    model: SentenceTransformer,
    model_name: str,
    profile_time_windows: Iterable,
    profile_llms: Iterable
) -> (Dict, Dict):
    """
    Build query / profile embeddings per user
    """
    
    # Filter once for all positive samples from relevant users
    relevant_data = train[
        (train["user_id"].isin(users_ids)) &
        (train["time_window"].isin(profile_time_windows)) &
        (train["model"].isin(profile_llms)) &
        (train["y"] == 1)
    ]
    
    # Group by the key components and get first profile for each group
    grouped = relevant_data.groupby(
        ["user_id", "model", "time_window"], 
        as_index=False
    )["profile"].first()
    
    # Build queries dict
    queries = {}
    for _, row in grouped.iterrows():
        key = f"{row['user_id']}/{row['model']}/{row['time_window']}"
        queries[key] = row["profile"]
    
    if not queries:
        return {}, {}
    
    queries_ids = list(queries.keys())
    queries_content = list(queries.values())
    
    # Determine embed_prompt based on model_name
    if model_name == 'embeddinggemma-300m':
        embed_prompt = 'task: search result | query: '
    elif model_name == 'Qwen3-Embedding-4B':
        embed_prompt = 'Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: '
    else:
        embed_prompt = None
    
    # Embed
    if embed_prompt:
        embeddings = embed(model, queries_content, embed_prompt=embed_prompt)
    else:
        embeddings = embed(model, queries_content)
    
    return dict(zip(queries_ids, queries_content)), dict(zip(queries_ids, embeddings))


def evaluate_recsys(
	test_case: str,
	users_ids: Iterable,
	profile_time_windows: Iterable,
	profile_llms: Iterable,
	test_tracks_per_user: Dict,
	track_corpus: Dict,
	track_embeddings: Dict,
	queries: list[str],
	query_embeddings: Dict,
	score_function: Dict,
) -> Dict:
    """
    Run recommendation via retrieval evaluation per user profile
    """

    results: Dict = {}
    for user_id in users_ids:
        for ptw in profile_time_windows:
            for pllm in profile_llms:

                rkey = f"{user_id}/{pllm}/{ptw}"
                if rkey not in queries:
                	continue
                user_queries = {user_id: queries[rkey]}
                user_query_embeddings = torch.stack(
                    [query_embeddings[rkey]]
                )

                relevant_tracks = Dict = {} # mapping between query and tracks
                relevant_tracks[user_id] = test_tracks_per_user[user_id][test_case]['+']

                all_tracks = test_tracks_per_user[user_id][test_case]['+'] + test_tracks_per_user[user_id][test_case]['-']

                user_corpus = {cid: track_corpus[cid] for cid in all_tracks}
                user_corpus_embeddings = torch.stack(
                    [track_embeddings[cid] for cid in all_tracks]
                )

                evaluator = InformationRetrievalEvaluator(
                    user_queries, user_corpus, relevant_tracks, score_functions= score_function
                )

                results[rkey] = evaluator.compute_metrices(
                    user_corpus_embeddings, user_query_embeddings
                )

    return results


def save_results(
	results: Dict,
	test_case: str,
	score_function_name: str,
	output_dir: str = ".",
):
	"""
	Save evaluation results to CSV.
	"""
	output_path = f"{output_dir}/results_{test_case}.csv"

	with open(output_path, "w") as f:
		f.write("user_id,time_window,model,recall@10,ndcg@10\n")

		for rkey, metrics in results.items():
			user_id, model, time_window = rkey.split("/")
			
			recall10 = metrics[score_function_name]["recall@k"][10]
			ndcg10 = metrics[score_function_name]["ndcg@k"][10]
			f.write(
				f"{user_id},{model},{time_window},"
				f"{recall10:.5f},{ndcg10:.5f}\n"
			)


def save_embeddings_npz(
	corpus: Dict,
	embeddings: Dict,
	path: str,
):
	"""
	Save pre-computed embeddings
	"""
	ids = [cid for cid in corpus]
	texts = [corpus[cid] for cid in ids]
	embeddings = torch.stack(
                    [embeddings[cid] for cid in ids]
                )
	np.savez(path, embeddings=embeddings, texts=texts, ids=ids)


def load_embeddings_npz(
	path: str,
	required_keys=("embeddings",),
): 
	"""
	Safely load embeddings from an .npz file.

	Args:
		path: Path to .npz file
		required_keys: Keys that must exist in the file

	Returns:
		dict[str, np.ndarray]
	"""
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Embeddings file not found: {path}")

	try:
		data = np.load(path, allow_pickle=False)
	except Exception as e:
		raise RuntimeError(f"Failed to load npz file: {path}") from e

	missing = [k for k in required_keys if k not in data]
	if missing:
		raise KeyError(f"Missing required keys in {path}: {missing}")

	return {k: data[k] for k in data.files}


# ---------------------------
# Main pipeline
# ---------------------------


def main():
	args = parse_args()
	# read train data
	# from the train we will only read the user generated profiles
	# 	which become queries for recommendation as retrieval algorithm
	train = pd.read_csv(f'{args.input_dir}/train.csv')
	# the various time windows used for user profile generation
	profile_time_windows = train['time_window'].dropna().unique()

	# the various LLMs used for user profile generation
	profile_llms = train['model'].dropna().unique()
	
	# read test data
	test = pd.read_csv(f'{args.input_dir}/test.csv')

	# filter out users who do not have NB_POSITIVES positive tracks
	counts = test[test.set != 'negative'].groupby(["user_id", "set"]).size().reset_index(name="cnt")
	users_ids = counts[counts.cnt == NB_POSITIVES]['user_id'].unique()
	logging.info(f"Filtered out the users who did not have {NB_POSITIVES} positives")
	logging.info(f"{len(users_ids)} users to be used in evaluation")

	# the various test cases specific to the test sets
	test_cases = test[test.set!= "negative"].set.unique()

	# the scoring function
	score_function = {"cosine": util.cos_sim}
	#score_function = {model.similarity_fn_name: model.similarity}

	model_name = get_model_identifier(args)
	model = None
	
	# we have multiple runs for each set-up 
	#	in order to have a different sample of negatives 
	for seed in SEEDS:

		random.seed(seed)

		# path where to save results and embeddings
		output_dir = os.path.join(args.output_dir, model_name, f'seed{seed}')

		# prepare the list of tracks per user
		test_tracks_per_user = build_all_test_tracks(test, test_cases, NB_NEGATIVES, seed)
		# check if pre-computed track embeddings are available
		# if not compute and save them 
		embedding_path = f"{output_dir}/track_embeddings.npz"
		if os.path.isfile(embedding_path):
			logging.info("Loading pre-computed track embeddings")
			data = load_embeddings_npz(
				embedding_path, 
				required_keys=["embeddings", "texts", "ids"]
			)
			track_corpus = dict(
				zip(data["ids"],
					data["texts"])
			)
			track_embeddings = dict(
				zip(data["ids"],
					torch.from_numpy(data["embeddings"]).float())
			)

		else:
			#create the output directory to log the embeddings
			os.makedirs(output_dir, exist_ok=True)
			# get the model
			model = load_model(args)
	
			# compute track embeddings for all users
			logging.info("Computing the track corpus embeddings")
			track_corpus, track_embeddings = build_track_embeddings(
				test, 
				test_tracks_per_user, 
				model,
				model_name
			)
			# and save them
			logging.info(f"Saving track embeddings to {embedding_path}")
			save_embeddings_npz(track_corpus, track_embeddings, embedding_path)
		
		# check if pre-computed user profile embeddings are available
		# if not compute and save them 
		embedding_path = f"{output_dir}/profile_embeddings.npz"
		if os.path.isfile(embedding_path):
			logging.info("Loading pre-computed user profile embeddings")
			data = load_embeddings_npz(
				embedding_path, 
				required_keys=["embeddings", "texts", "ids"]
			)
			queries = dict(
				zip(data["ids"],
					data["texts"])
			)
			query_embeddings = dict(
				zip(data["ids"],
					torch.from_numpy(data["embeddings"]).float())
			)
		else:
			# get the model
			model = load_model(args, model = model)
			# compute query / user profile embeddings for all user
			logging.info("Computing the user profile embeddings")
			queries, query_embeddings = build_query_embeddings(
				users_ids, 
				train, 
				model, 
				model_name,
				profile_time_windows, 
				profile_llms
			)
			logging.info(f"Saving user profile embeddings to {embedding_path}")
			save_embeddings_npz(queries, query_embeddings, embedding_path)

		for test_case in test_cases:

			# compute results
			results = evaluate_recsys(test_case, users_ids, profile_time_windows, profile_llms, test_tracks_per_user, track_corpus, track_embeddings, queries, query_embeddings, score_function)

			score_function_name = next(iter(score_function))
			save_results(results, test_case, score_function_name, output_dir)


if __name__ == "__main__":
    main()
