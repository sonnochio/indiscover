import pandas as pd
import pickle as pk
import numpy as np
import os
import re
from PIL import Image
import streamlit as st


"""
===construct product data for Natural Language Processing===

"""
def load_full_df():
    """load full dataset, extracted from ap0cene

    """
    df_full=pd.read_csv('../data/df_full_image_url.csv')
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

def return_image(img, cropping=False):

    if  cropping==False:
        img=Image.open(img).resize((360,360))
    else:
        img=img.resize((360,360))


    if img.mode=="RGB" :
        img_data=np.asarray(img,dtype="int32" ).astype('float32')/255
    else :
        img_data=np.asarray(img.convert('RGB'), dtype="int32" ).astype('float32')/255

    return img_data

def load_image_nparray(num, try_except=False):
    """
    load the dataset images and convert them to numpy array.

    """
    url=(f"images/image_{num}.jpg")

    if try_except:
        try:
            return return_image(url)
        except:
            pass
    else:
        return return_image(url)

def load_query_image_nparray(image=None,url=None,use_url=False, try_except=False):
    """
    load the query image and convert it to numpy array.
    """
    if use_url==True:
        url=url
        img_data=return_image(url)

    else:
        img_data=return_image(image, cropping=True)

    img_data=np.expand_dims(img_data, axis=0)
    return img_data


def save_pickle(num, df_full, try_except=False):
    """
    saving a numpy image as a pickle file.
    """
    if try_except:
        try:
            img_data=load_image_nparray(num)

            with open(f'image_data/image_{num}.pickle', 'wb') as handle:
                pk.dump(img_data, handle, protocol=pk.HIGHEST_PROTOCOL)
        except:
            pass
            # df_full[].drop()

    # else:
    #     img_data=load_image_nparray(num, try_except)
    #     with open(f'image_data/image_{num}.pickle', 'wb') as handle:
    #         pk.dump(img_data, handle, protocol=pk.HIGHEST_PROTOCOL)


def load_all_latent_chunks(to_csv=False, from_csv=True):

    """
    load all latent encoded chunks and save them in a dataframe
    """
    if from_csv:
        with open('../data/full_latent_embedding.pickle','rb') as h:
            df=pk.load(h)

    else:
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


    if to_csv:
        df.to_csv("latent_pics.csv")

    return df


def open_streamlit(img=None):
    st.image(img)
    pass
