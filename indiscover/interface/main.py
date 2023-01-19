import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import PIL
import os
from indiscover.encode import build_encoder, encode_chunks_save_pickle, d2v
from indiscover.data import load_query_image_nparray, load_all_latent_chunks, load_full_df, get_products_df
from indiscover.cos_sim import return_top_k_images
from indiscover.preprocessing import clean_sentence,remove_punctuation, tokenize, query_clean, preprocessor
from PIL import Image
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#load dataframes
df=get_products_df(read_csv=True)
df_full=load_full_df()
df_interface=df_full.merge(df, how='right',on='product_page').drop_duplicates("product_name")
df_interface=df_interface[['product_name', 'designer_name', 'designer_page',
       'designer_description', 'product_page', 'product_image_url',
       'product_description']]

#preprocess dataframe
df_proc=preprocessor(df)


def image_workflow(encode_all_image=False):

    #build latent space enocder for imgae processing
    encoder=build_encoder(50)
    print("✅built encoder")

    #encode the whole image dataset and load csv
    if encode_all_image:
        encode_chunks_save_pickle(encoder)
        print("✅added clean text column")
    else:
        print("✅not coding anything, opening the text")

    df=load_all_latent_chunks()
    #encode query image
    query_img_data=load_query_image_nparray(3)
    query_latent=encoder.predict(query_img_data)

    df_top_k_images=return_top_k_images(query_latent,df,3)

    return df_top_k_images

def show_response_images(url,df_top_k_images):
    product_list=[]


    for i in df_top_k_images["file_num"].tolist():
        product_name=df_full[df_full["num"]==i]["product_name"].values[0]

        if product_name not in product_list:

            product_list.append(product_name)

            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,5))
            ax1.imshow(Image.open(f'images/image_{i}.jpg'))
            ax2.imshow(Image.open(url))

    pass

def text_query(query, topk):


    #set english stopwords and vectorize texts
    stop_words = get_stop_words('en')
    vectorizer=TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    tfidf_mat=vectorizer.fit_transform(df_proc["clean_text"].values)

    def get_similar_text(query_text,tfidf_mat,k):
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


    return get_similar_text(query, tfidf_mat, topk)

def similar_product_query(query_list):
    response_ls=[]
    for query in query_list:
        most_similar_indexs=d2v.docvecs.most_similar(query,topn=5)
        response_ls.append(most_similar_indexs)

    df_proc=preprocessor(df)

    return df_proc[response_ls]



"""
to run everything,

"""


query_response=text_query("hello bitch dress", 3)




# for i in query_response:
#     print(df_interface[i])
#     plt.imshow(Image.open(df_['image'])


print ("main page running✅")
