import streamlit as st
from main import text_query, image_workflow, return_image_query,text_image_query
from data_util import load_full_df, get_products_df
from preprocessing import preprocessor
from PIL import Image
import requests
from data_util import open_streamlit
from streamlit_cropper import st_cropper
from streamlit_lottie import st_lottie
st.set_page_config(layout="wide")
#load dataframes
df_products=get_products_df(read_csv=True)
df_full=load_full_df()

#preprocess df
df_proc=preprocessor(df_products)

#prepare user dataframe for front end
df_interface=df_full.merge(df_products.drop("num", axis=1), how='left',on='product_page')

# user_text=np.nan

st.set_option('deprecation.showfileUploaderEncoding', False)


columns=st.columns([2,1])

with columns[0]:

    st.markdown("""
    <style>
    .big-font {
        font-size:100px !important;
        style="font-family:Verdana, sans-serif;;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font"><b>inDiscover</b> </p>', unsafe_allow_html=True)

with columns[1]:

    logo_url="https://assets2.lottiefiles.com/packages/lf20_sen1ai7d.json"
    logo_json=requests.get(logo_url).json()
    st_lottie(logo_json, height=200)



st.markdown("""
<style>
.mid-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('''<p class="mid-font">Discover fashion items from independent desingers with a cotent-based ML recommendation system! </p>''', unsafe_allow_html=True)

st.caption("Describe your item or upload an image, or do both for better results. Crop your image to item for even better results! (Hint: Be creative with your image, upload a pattern or a moodboard and see what happens 🤪!) ")


st.markdown("""
<style>
.text-font {
    font-size:12px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown(' ')
st.markdown('''<p class="text-font">Ex. Prompt: A green top with egdy logo, uploded picture on the right </p>''', unsafe_allow_html=True)
st.image(Image.open('demo.png'), width=400)


st.markdown("""---""")


user_text=st.text_area('Prompt', height=0)
st.markdown("""---""")

columns=st.columns([1,2])




with columns[0] :
    realtime_update = st.checkbox(label="Update in Real Time", value=True)
    box_color = st.color_picker(label="Crop box Color", value='#011813')

    # with columns[1] :
    aspect_dict = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "2:3": (2, 3),
            "Free": None
    }
    aspect_choice = st.radio(label="Crop Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])

    aspect_ratio = aspect_dict[aspect_choice]

with columns[1] :
    image_query= st.file_uploader("Image")

if image_query:
    img = Image.open(image_query)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    # st.markdown("""
    #                 <style>iframe {background-color: red;}</style>
    #                 """
    #                 , unsafe_allow_html=True)



    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)

    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((150,150))
    st.image(cropped_img)

st.markdown("""---""")

submitted = st.button("inDiscover")



if submitted and len(user_text) >1 and image_query is not None:

    text_image_query(text=user_text, query_image=cropped_img,load_tf_model=True)

elif submitted and len(user_text) >1:
    query_response=text_query(user_text, 3)

    for i in query_response:

        with st.expander('', expanded=True):
            st.image(Image.open(requests.get(df_full['product_image_url'][i], stream=True).raw))

            df_item=df_interface[df_interface.num==i]
            url =df_item['product_page'].reset_index(drop=True).loc[0]

            txt=f"{df_item['product_name'].reset_index(drop=True).loc[0]} by {df_item['designer_name'].reset_index(drop=True).loc[0]}"
            page=df_item['product_page'].reset_index(drop=True).loc[0]
            link=f"   [{txt}]({page})"
            st.markdown(link, unsafe_allow_html=True)

elif submitted and image_query is not None:

    df_top_k_images=image_workflow(image=cropped_img, topk=10,load_tf_model=True)
    return_image_query(df_top_k_images, query_image=cropped_img, front_end=True)
