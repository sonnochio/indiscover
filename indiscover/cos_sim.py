import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image



def return_top_k_images(img_latent,df,k):
    df["cos_sim"]=df["img_latent"].apply(lambda x : cosine_similarity(img_latent,[x]))
    df=df.sort_values("cos_sim", ascending=False)[:k]
    return df

def image_cos_sim(img_latent,df,k):
    top_k_df=return_top_k_images(img_latent,df,10)
    product_list=[]

    for i in top_k_df["file_num"].tolist():
        product_name=df_full[df_full["num"]==i]["product_name"].values[0]

        if product_name not in product_list:

            product_list.append(product_name)

            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,5))
            ax1.imshow(Image.open(f'images/image_{i}.jpg'))
            ax2.imshow(Image.open(url))

    pass
