import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import PIL
import os
from indiscover.encode import build_encoder, encode_chunks_save_pickle, d2v
from indiscover.data import load_query_image_nparray, load_all_latent_chunks, load_full_df, get_products_df, save_pickle
from indiscover.cos_sim import return_top_k_images, get_similar_text
from indiscover.preprocessing import clean_sentence,remove_punctuation, tokenize, query_clean, preprocessor
from PIL import Image
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests


#load dataframes
df=get_products_df(read_csv=True)
df_full=load_full_df()

#prepare user dataframe for front end
df_interface=df_full.merge(df, how='right',on='product_page').drop_duplicates("product_name")
df_interface=df_interface[['product_name', 'designer_name', 'designer_page',
       'designer_description', 'product_page', 'product_image_url',
       'product_description']]


def image_workflow(num,encode_all_image=False, load_model=False):


    #build latent space enocder for imgae processing
    encoder=build_encoder(50)
    print("✅built encoder")

    #encode the whole image dataset and load csv
    if encode_all_image:
        breakpoint()
        df_full["num"].apply(lambda x: save_pickle(x, df_full,try_except=True))
        encode_chunks_save_pickle(encoder,1)
        print("✅latent encoded all chunks of image numpy arrays")

    else:
        print("✅not encoding, about to load all_latent_chunks")

    df=load_all_latent_chunks()

    #encode query image, num is test image number
    query_img_data=load_query_image_nparray(num)
    query_latent=encoder.predict(query_img_data)
    df_top_k_images=return_top_k_images(query_latent,df,3)

    return df_top_k_images

def image_query(url,df_top_k_images):
    product_list=[]

    for i in df_top_k_images["file_num"].tolist():
        product_name=df_full[df_full["num"]==i]["product_name"].values[0]

        if product_name not in product_list:

            product_list.append(product_name)

            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,5))


            ax1_url=df_full["product_image_url"].loc[i]

            ax2_url=url

            ax1.imshow(Image.open(requests.get(ax1_url, stream=True).raw))
            ax2.imshow(Image.open(ax2_url))

            plt.show()
    pass

def text_query(query, topk):

    #set english stopwords and vectorize texts
    stop_words = get_stop_words('en')
    vectorizer=TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    tfidf_mat=vectorizer.fit_transform(df_proc["clean_text"].values)

    #return query text

    return get_similar_text(query, tfidf_mat, topk, vectorizer)

def similar_product_query(query_list):
    response_ls=[]
    for query in query_list:
        most_similar_indexs=d2v.docvecs.most_similar(query,topn=5)
        response_ls.append(most_similar_indexs)

    df_proc=preprocessor(df)

    return df_proc[response_ls]

# def text_image_query(text, image_url):

#     indexes=text_query(text, 100)





"""
to run everything,

"""


# query_response=text_query("hello bitch dress", 3)
# print ("finished running text query✅")

num=1
test_img_url=f"test_img/test{num}.jpg"
df_top_k_images=image_workflow(num, encode_all_image=True)
image_query(test_img_url,df_top_k_images)

# for i in query_response:
#     print(df_interface[i])
#     plt.imshow(Image.open(df_['image'])


print ("main page running✅")
