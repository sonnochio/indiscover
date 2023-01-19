import pandas as pd
import pickle as pk
import numpy as np
import os
import re
from PIL import Image

"""
===construct product data for Natural Language Processing===

"""
def load_full_df():
    """load full dataset, extracted from ap0cene

    """
    df_full=pd.read_csv('data/df_full_image_url.csv')
    df_full.drop('Unnamed: 0',axis=1,inplace=True)
    df_full=df_full.dropna()

    return df_full

def get_products_df(read_csv=True):
    """
    load products dataframe and do basic cleaning
    """
    if read_csv:
        return pd.read_csv("data/df_products.csv").drop("Unnamed: 0", axis=1)
    else:
        df_full=pd.read_csv("data/df_full_image_url.csv")
        df_full=df_full.dropna()

        #dropping duplicated pages
        df_products=df_full[["num","product_page"]].drop_duplicates("product_page")

        return df_products

def load_image_nparray(num):
    """
    load the dataset images and convert them to numpy array.

    """
    url=(f"images/image_{num}.jpg")

    img=Image.open(url).resize((360,360))
    if img.mode=="RGB" :
        img_data=np.asarray(img,dtype="int32" ).astype('float32')/255
    else :
        img_data=np.asarray(img.convert('RGB'), dtype="int32" ).astype('float32')/255

    return img_data

def load_query_image_nparray(num):
    """
    load the query image and convert it to numpy array.
    """
    url=(f"test_img/test{num}.jpg")
    img=Image.open(url).resize((360,360))
    if img.mode=="RGB" :
        img_data=np.asarray(img,dtype="int32" ).astype('float32')/255
    else :
        img_data=np.asarray(img.convert('RGB'), dtype="int32" ).astype('float32')/255
    img_data=np.expand_dims(img_data, axis=0)
    return img_data

def save_pickle(num):
    """
    saving a numpy image as a pickle file.
    """

    img_data=load_image_nparray(num)

    with open(f'image_data/image_{num}.pickle', 'wb') as handle:
        pk.dump(img_data, handle, protocol=pk.HIGHEST_PROTOCOL)


def load_all_latent_chunks():

    """
    load all latent encoded chunks and save them in a dataframe
    """

    latent_names=os.listdir("latent_chunks")
    df=pd.DataFrame()

    for name in latent_names:
        with open(f"latent_chunks/{name}", "rb") as h:
            df=pd.concat([df,pk.load(h).transpose()])

    df=df.rename({0:"file_name",1:"img_latent"},axis=1)

    def file_num(name):
        return int(re.findall("\d+",name)[0])

    df["file_num"]=df["file_name"].apply(file_num)
    df=df.sort_values("file_num").reset_index(drop=True)
    df.to_csv("latent_pics.csv")

    return df
