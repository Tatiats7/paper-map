from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pdf_reader import extract_text_from_pdf
import os
import ast

import pandas as pd
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import numpy as np


embeddings_model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-en",
    trust_remote_code=True)
embeddings_model.max_seq_length = 4096

def make_embeddings_for_new_papers(papers_not_havin_embeds):
  df = pd.DataFrame()
  all_embeddings = []
  all_paragraphs = []
  all_names = []
  all_paths = []
  for i, paper in tqdm(enumerate(papers_not_havin_embeds)):
    pdf_path = paper
    extracted_text = extract_text_from_pdf(pdf_path)
    paragraphs = extracted_text.split(".\n")
    paragraphs = [tx for tx in paragraphs if len(''.join(tx.split())) > 20] # to exclude table things
    embeddings = embeddings_model.encode(paragraphs)
    embeds_row = [list(embeddings[i]) for i in range(0,len(embeddings))]
    all_embeddings.extend(embeds_row)
    all_paragraphs.extend(paragraphs)
    paths = [paper]*len(paragraphs)
    all_paths.extend(paths)
    names = ["".join(paper.split("/")[-1].split(".")[:-1])]*len(paragraphs)
    all_names.extend(names)

  df["name"] = all_names
  df["path"] = all_paths
  df["embeds"] = all_embeddings
  df["paragraphs"] = all_paragraphs
  return df

def check_and_add_embeds(root_path):
  all_papers = os.listdir(root_path)
  all_paper_pdf_names = [root_path +paper for paper in all_papers if ".pdf" in paper]
  read = pd.read_csv(root_path+"embeddings.csv",lineterminator='\n')
  read["embeds"] = read["embeds"].apply(ast.literal_eval)
  papers_not_havin_embeds = [paper for paper in all_paper_pdf_names if paper not in read["path"].tolist()]
  if len(papers_not_havin_embeds):
    print("NEW papers: ", papers_not_havin_embeds)
    additional_table = make_embeddings_for_new_papers(papers_not_havin_embeds)
    df = pd.concat([read, additional_table])
    df.to_csv(root_path+"embeddings.csv", index=False, escapechar="\\")
    return df
  else:
    print("NO new papers")
    return read
  
def compute_cosine_similarities(query_embedding, df_embeddings):
    query_embedding_reshaped = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_reshaped, df_embeddings)[0]
    return similarities

def find_similar_papers(query, data, top_n=5):
  query_embeds = embeddings_model.encode(query)
  df_embeddings = np.stack(data['embeds'].values)
  similarities = compute_cosine_similarities(query_embeds, df_embeddings)
  data_new = data.copy()
  data_new['similarity'] = similarities
  data_new = data_new.sort_values(by='similarity', ascending=False)

  top_n_similar = data_new.head(top_n)
  context = "" 
  source_papers = ""
  for i in range(len(top_n_similar)):
    context += top_n_similar.iloc[i]["paragraphs"] + ". "
    source_papers += top_n_similar.iloc[i]["name"] + ", "
    print(top_n_similar.iloc[i]["similarity"])
  return context, source_papers