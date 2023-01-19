import requests
import time
from bs4 import BeautifulSoup as bs
import re
import pandas as pd

"""
This module will perform data extraction tasks through multilayered website ap0cene. 😭
The resulting dataframe is "df_full_image_url.csv", which includes


 #   Column                 Non-Null Count  Dtype
---  ------                 --------------  -----
 0   Unnamed: 0             11240 non-null  int64
 1   product_name           11240 non-null  object
 2   designer_name          11240 non-null  object
 3   designer_page          11240 non-null  object
 4   designer_description   11240 non-null  object
 5   product_suffix         11240 non-null  object
 6   product_page           11240 non-null  object
 7   product_page_response  11240 non-null  bool
 8   product_image_url      11240 non-null  object
 9   num                    11240 non-null  int64


Only run this file if ap0cene announces they are adding new designers/items.

"""



"""
Utility functions used before interface
"""
def extract_designer_name(ind):
    url=df.loc[ind]["designer_page"]

    r=requests.get(url)
    soup=bs(r.content, "html.parser")
    designer_name=soup.find("h1", class_="font-heading text-lg").text

    return designer_name

#extracting designer bio/description from their personal page
#imputing missing names with "A designer"

def extract_designer_description(url):
    r=requests.get(url)
    soup=bs(r.content, "html.parser")
    try :
        description=soup.find("div", class_="rte mt-4").text
        return description
    except:
        return "A designer."


def designer_page_scrap(url):
    r=requests.get(url)
    soup=bs(r.content, "html.parser")
    names=[]
    items=[]
    descriptions=[]
    for i in soup.find_all("img", class_="responsive-image block absolute top-0 left-0 w-full h-full lazyload transition-opacity duration-200 ease-in-out w-full max-w-full h-auto" ):
        names.append(i['alt'])
        items.append('-'.join(i['alt'].lower().split()))

    return names,items,descriptions

def get_product_page(url):
    response=requests.get(url)
    soup=bs(response.content,"html.parser")
    page_objects=soup.find_all("a",class_="increase-target")
    suffix=[]

    for ob in page_objects:
        suffix.append(ob["href"])

    return suffix

def product_url(product_suffix):
    url=f'https://ap0cene.com{product_suffix}'
    print(url)
    return url


def url_response_check(product_page):
    #check if product page is valid
    status=(requests.get(product_page).ok)
    return status

def get_pic(url):
    response=requests.get(url)
    soup=bs(response.content,"html.parser")
    pic_objects=soup.find_all("div",class_="responsive-image-wrapper")
    pic_url=[]
    width=360
    for ob in pic_objects:
        pic_url.append(ob.find_all("img")[0]["data-src"].replace("{width}","360").replace("//"," http://"))
    return pic_url

def save_images(index,url):
    r = requests.get(url, stream=True).content
    with open(f'../images/image_{index}.jpg', 'wb') as handler:
        handler.write(r)
    print('image saved locally')
    pass

def save_all_images():
    """
    Apply to all urls to save all iamges
    🔞 Do not use this if you are not ready to update your image dataset!!

    """

    images=df_full["product_image_url"]
    images.apply(lambda x: save_images(x["num"],x["product_image_url"]), axis=1)
    print('✅all images saved locally')

    pass



def web_scraping():
    """
    executing all above function to achieve main purpose of this module
    
    """
    all_designers_url="https://ap0cene.com/pages/ap0cene-designers-a-z"
    all_designers_page=requests.get(all_designers_url)
    soup=bs(all_designers_page.content, "html.parser")
    all_designers_soup=soup.find_all(href=re.compile("https:\/\/ap0cene.com\/collections.+"))[2:]
    df=pd.DataFrame()
    designer_names=[]
    designer_links=[]

    #constructing designer dataframe
    for i,node in enumerate(all_designers_soup):
        designer_names.append(node.text)
        designer_links.append(node["href"])

    df['designer_name']=designer_names
    df['designer_page']=designer_links

    #dropping invalide page
    df.drop(21,inplace=True)
    #finding missing designer info

    designers_with_name_missing=df[df['designer_name']=='']

    #extract designer name from their personal page
    #imputing missing names

    for ind in designers_with_name_missing.index:
        try :
            df.loc[ind]["designer_name"]=extract_designer_name(ind)
        except:
            print(ind,"problem")


    #assigning columns to designer info
    df["designer_description"]=df["designer_page"].apply(extract_designer_description)
    df['designer_description']=df['designer_description'].apply(lambda x: x.replace('\n',''))
    df_designer=df


    #extracting products, scrap the designer items page and return names,items,descriptions
    scrapped=df_designer['designer_page'].apply(designer_page_scrap)
    df_product=pd.DataFrame()
    df_product=pd.DataFrame(data={'product_name':scrapped.apply(lambda x:x[0])})
    df_full=pd.concat([df_product,df_designer],axis=1)

    #constructing full product urls
    df_full["product_suffix"]=df_full["designer_page"].apply(get_product_page)
    df_full=df_full.explode(['names', 'items','product_suffix']).reset_index(drop=True)

    #find the desingers with no products
    df_missing=df_full[df_full.isnull().any(axis=1)]
    missing_index=df_missing.index

    #dropping designers with no products
    df_full=df_full.drop(missing_index).reset_index(drop=True)

    #get product url
    df_full['product_page']=df_full.apply(lambda x: product_url(x['product_suffix']), axis=1)

    #check responses
    df_full['product_page_response']=df_full['product_page'].apply(url_response_check)

    #saving prouct page validity df
    df_full.to_csv("../data/df_full_response.csv")

    #testing if there's any missing pages
    df_missing=df_full[df_full['product_page_response']==False]


    #applying the scraping fucntion
    df_full["product_image_url"]=df_full["product_page"].apply(get_pic)

    #expand the df on their image url
    df_full=df_full.explode('product_image_url').reset_index(drop=True)

    #writing a new column called num for easy index
    df_full["num"]=df_full.index

    #saving full image url df
    df_full.to_csv("../data/df_full_image_url.csv")

    pass
