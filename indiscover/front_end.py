import streamlit as st
from main import text_query, image_workflow, return_image_query,text_image_query
import pandas as pd
import numpy as np
from data_util import load_full_df, get_products_df
from preprocessing import preprocessor
from PIL import Image
import requests
from data_util import open_streamlit

#load dataframes
df_products=get_products_df(read_csv=True)
df_full=load_full_df()

#preprocess df
df_proc=preprocessor(df_products)


#prepare user dataframe for front end
df_interface=df_full.merge(df_products.drop("num", axis=1), how='left',on='product_page')

# user_text=np.nan

st.set_option('deprecation.showfileUploaderEncoding', False)




with st.form('form1'):
    user_text=st.text_input('Describe the item you want to see:')
    image_query= st.file_uploader("Insert an image example:")
    submitted = st.form_submit_button("inDiscover")

    if len(user_text) >1 and image_query is not None:
        st.write("bimbim bab")
        text_image_query(text=user_text, query_image=image_query,load_tf_model=True)


    elif len(user_text) >1:
        query_response=text_query(user_text, 3)

        for i in query_response:
            try:
                st.write(df_interface.loc[i])
                st.image(Image.open(requests.get(df_full['product_image_url'][i], stream=True).raw))
            except:
                pass
    elif image_query is not None:

        df_top_k_images=image_workflow(image=image_query, topk=10,load_tf_model=True)
        return_image_query(df_top_k_images, query_image=image_query, front_end=True)
