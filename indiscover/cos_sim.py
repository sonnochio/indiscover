import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from indiscover.preprocessing import remove_punctuation, tokenize



def return_top_k_images(img_latent,df,k):
    df=df.dropna()
    breakpoint()
    df["cos_sim"]=df["img_latent"].apply(lambda x : cosine_similarity(img_latent,[x]))
    df=df.sort_values("cos_sim", ascending=False)[:k]
    return df

def get_similar_text(query_text,tfidf_mat,k,vectorizer):

    #tokenize and embed query text
    query_tokens=tokenize(remove_punctuation(query_text))
    embed_query=vectorizer.transform(query_tokens)

    #calculate the similarity between each token vs the entire tfidf matrix
    sim_mat=cosine_similarity(embed_query,tfidf_mat)

    #returns the k number of indeces with the highest similarity
    cos_sim=np.mean(sim_mat, axis=0)
    index=np.argsort(cos_sim)[::-1]

    mask=np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index]!=0, mask)
    return index[mask][:k]
