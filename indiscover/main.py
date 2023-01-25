import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
from tensorflow.keras.models import load_model
from encode import build_encoder, encode_chunks_save_pickle
from data_util import load_query_image_nparray, load_all_latent_chunks, load_full_df, get_products_df, save_pickle
from cos_sim import return_top_k_images, get_similar_text
from preprocessing import tokenize,  preprocessor
from PIL import Image
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import streamlit as st

#load dataframes
df_products=get_products_df(read_csv=True)
df_full=load_full_df()

#preprocess df
df_proc=preprocessor(df_products)


#prepare user dataframe for front end
df_interface=df_full.merge(df_products.drop("num", axis=1), how='left',on='product_page')


def image_workflow(topk, image=None, url=None, encode_all_image=False, load_tf_model=False, return_images=True):
    if load_tf_model==True:
        encoder=load_model('models/encoder')
        print("✅loaded encoder")

    else:
    #build latent space enocder for imgae processing
        encoder=build_encoder(50)
        print("✅built encoder")

    #encode the whole image dataset and load csv
    if encode_all_image:

        df_full["num"].apply(lambda x: save_pickle(x, df_full,try_except=True))
        encode_chunks_save_pickle(encoder,1)
        print("✅latent encoded all chunks of image numpy arrays")

    else:
        print("✅not encoding, about to load all_latent_chunks")

    if return_images:
        df=load_all_latent_chunks()

        #encode query image, num is test image number
        if image is not None:
            query_img_data=load_query_image_nparray(image)
        else:
            query_img_data=load_query_image_nparray(url)


        query_latent=encoder.predict(query_img_data)
        df_top_k_images=return_top_k_images(query_latent,df,topk)

        return df_top_k_images

def return_image_query(df_top_k_images, query_image=None, url=None, front_end=False, cropping=True):
    """
    query on an image per user input and return similar products

    """
    product_list=[]

    columns=st.columns(2, gap='small')

    new_height=360
    if cropping==False:
        query_width, query_height = Image.open(query_image).size
    else:
        query_width, query_height = query_image.size

    new_query_width=query_width*(new_height/query_height)


    for ind,i in enumerate(df_top_k_images["file_num"].tolist()):
        product_name=df_full[df_full["num"]==i]["product_name"].values[0]

        if front_end==False:
            #opening query image from url and display in terminal
            if product_name not in product_list:

                product_list.append(product_name)

                fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,5))


                ax1_url=df_full["product_image_url"].loc[i]

                ax2_url=url

                ax1.imshow(Image.open(requests.get(ax1_url, stream=True).raw))
                ax2.imshow(Image.open(ax2_url))

                plt.show()
        else:
            #opening query image from user input and display on front end
            if product_name not in product_list:

                product_list.append(product_name)
                df_img_url=df_full["product_image_url"].loc[i]
                img_df=Image.open(requests.get(df_img_url, stream=True).raw)
                width, height = img_df.size
                new_width=width*(new_height/height)
                img_df=img_df.resize((int(new_width),int(new_height)))
                columns[0].image(img_df)



                columns[1].image(query_image, width=int(new_query_width))




    pass

def text_query(query, topk):


    #set english stopwords and vectorize texts
    stop_words = get_stop_words('en')
    vectorizer=TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    tfidf_mat=vectorizer.fit_transform(df_proc["clean_text"].values)

    #return df_proc indexes
    proc_indexes=get_similar_text(query, tfidf_mat, topk, vectorizer)


    return df_proc.loc[proc_indexes]["num"].tolist()

#recommend prodcuts based on shopping cart

# def similar_product_query(query_index_list):
#     response_ls=[]
#     for query in query_index_list:
#         most_similar_indexs=d2v.docvecs.most_similar(query,topn=5)
#         response_ls.append(most_similar_indexs)

#     df_proc=preprocessor(df_products)

#     return df_proc[response_ls]

def text_image_query(text, query_image, load_tf_model=False, df_products=df_products):

    response_indexes=text_query(text, 30)
    df=load_all_latent_chunks()
    #similar text retrieval
    # df_sim_text=df_full.loc[indexes]

    df=df.merge(df_full, how="right", left_on="file_num", right_on="num")
    # breakpoint()

    df=df[df["num"].isin(response_indexes)]
    df=df.merge(df_products.drop('num', axis=1), how="inner", on="product_page")

    # df_products=df_sim_text.merge(df_products, how="right", on="num")

    # df=df.merge(df_products, how="inner", on="product_name" )

    #load model
    if load_tf_model:
        encoder=load_model('models/encoder')
        print("✅loaded encoder")

    else:
    #build latent space enocder for imgae processing
        encoder=build_encoder(50)
        print("✅built encoder")


    query_img_data=load_query_image_nparray(query_image)
    query_latent=encoder.predict(query_img_data)
    df_top_k_images=return_top_k_images(query_latent,df, 5)
    return_image_query(df_top_k_images,query_image=query_image, front_end=True)

    pass


print ("main page running✅")
