import pandas as pd
import numpy as np
import os
import pickle as pk
# from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def build_encoder(latent_dimension):
    encoder=Sequential()

    encoder.add(Conv2D(8,(3,3), input_shape=(360,360,3), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(16, (3, 3), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(32, (3, 3), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation='sigmoid'))

    return encoder



def encode_chunks_save_pickle(encoder, chunk_num):

    data_pickles=np.array(os.listdir('image_data'))
    chunks=np.array_split(data_pickles,chunk_num)

    for i in np.arange(len(chunks)):
        img_data_ls=[]
        df=pd.DataFrame()

        for p in chunks[i]:
            with open(f'image_data/{p}','rb') as handle:
                img_data=pk.load(handle)
            try:
                if img_data == None:
                    pass
            except:
                img_data_ls.append(img_data)

        X=np.array(img_data_ls)

        # breakpoint()
        latent_X=encoder.predict(X)

        df=pd.DataFrame([chunks[i],latent_X])

        with open(f"latent_chunks/latent{i}.pickle","wb") as handle:
            pk.dump(df,handle, protocol=pk.HIGHEST_PROTOCOL)

    pass

#for topic modeling

# def d2v(df_proc,save_model=False):
#     tagged_data=[TaggedDocument(words=text, tags=[i]) for i, text in enumerate(df_proc["text_tokens"])]
#     d2v=Doc2Vec(vector_size=30, epochs=80, min_count=2)
#     d2v.build_vocab(tagged_data)
#     d2v.train(tagged_data, total_examples=d2v.corpus_count, epochs=80)

#     if save_model:
#         d2v.save("models/d2v.model")

#     return d2v
