# Import useful packages

import gdal
import pandas as pd
import geopandas as gpd
from io import StringIO
import osr
import matplotlib.pyplot as plt
import pandas as pd
import rtree
import pygeos
import os, json
import geopandas as gpd
from ipywidgets import widgets
from ipyleaflet import Map, GeoData, basemaps, LayersControl, ScaleControl, FullScreenControl, WidgetControl
from ipywidgets import widgets, IntSlider, jslink


import seaborn as sns
#bokeh
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")

# Load image from local storage
from IPython.display import Image

from ipyleaflet import *


# Take the mean of the percentage cover for the whole dataframe 
def mean_col(df, col_name):
    return df[col_name].mean()

# Split the data frames based on the years
def split_years(dt):
    dt['year'] = dt['surveydate'].dt.year
    return [dt[dt['year'] == y] for y in dt['year'].unique()]

# Make Geodata-frame
def make_geo_frame(df):
    return gpd.GeoDataFrame(df, geometry = 'geometry')

def export_shape(df, name):
    return df.to_file(name + '.shp', driver='ESRI Shapefile')


# Look if any transectid (square) has several surveys in it 
def trans_surv(df):
    num = df.transectid.duplicated().sum()
    if (num):
        tran_id = df[df.transectid.duplicated()]["transectid"].item()
        
        
        return [num,df[df["transectid"] == tran_id]]
    else :
        return [0,0]
    
# New dataframe with the trasectid as index and year as feature with only "True"
def smart_df(df, year):
    df_ = df["transectid"]
    df_ = pd.DataFrame(df_)
    df_.set_index("transectid", inplace = True) 
    df_[year] = True
    return df_

# function to keep only columns of given list
def get_df_col(df, list_col):
    return df[list_col]

# in this function you get a list of indexes that correspond to condition
def get_index_list(df, cond):
    return df[df["Sum"] == cond].index.tolist()
# you return a dataframe given a list of indeces
def return_df(df, ind_list):
    return df.loc[ind_list]

def mean_region(df_s, df_joined, cond):
    df_s = df_s.set_index("transectid") 
    list_col = ["pr_hard_coral", "pr_soft_coral", "pr_algae", "pr_oth_invert", "pr_other", "geometry", "year"]
    df = get_df_col(df_s, list_col)
    return return_df(df, get_index_list(df_joined,  cond))

def plot_mean_stack(df):

    # I will now group the pr_oth_invert and pr_other
    df["others"] = df["pr_oth_invert"] + df["pr_other"] + df["pr_soft_coral"]
    df = df.drop(columns = ["pr_oth_invert", "pr_other", "pr_soft_coral"])

    # Here I make a specific dataframe to enable a stacked graph 
    #survey_melt = pd.melt(Survey_mean, id_vars = ["year"], value_vars=["pr_hard_coral", "pr_algae", "pr_soft_coral", "others"])
    survey_melt = pd.melt(df, id_vars = ["year"], value_vars=["pr_hard_coral", "pr_algae","others"])
    fig, ax = plt.subplots(figsize=(10,7))  
    import numpy as np
    months = survey_melt['variable'].drop_duplicates()
    margin_bottom = np.zeros(len(survey_melt['year'].drop_duplicates()))
    # colors = ["#004C1E","#006D2C", "#31A354","#74C476"]
    colors = ["#004C1E","#006D2C", "#31A354"]

    for num, month in enumerate(months):
        values = list(survey_melt[survey_melt['variable'] == month].loc[:, 'value'])

        survey_melt[survey_melt['variable'] == month].plot.bar(x='year',y='value', ax=ax, stacked=True, color = colors[num],
                                        bottom = margin_bottom, label=month)
        margin_bottom += values
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Percentage")
    plt.xlabel("Year")
    plt.title("Types of corals distribution")
    plt.show()
    
def hard_algae_plot(df):
    sns.set_style("darkgrid")
    sns.lineplot("year","pr_hard_coral", data = df, label = "Hard coral")
    sns.lineplot("year","pr_algae", data = df, label = "Algae")
    plt.ylabel("Proportion")
    plt.xlabel("Year")
    
def fun(df, list_12_17):
    list_col = ["pr_hard_coral", "pr_soft_coral", "pr_algae", "pr_oth_invert", "pr_other", "geometry", "year"]
    df = get_df_col(df.set_index("transectid"), list_col)
    return return_df(df, list_12_17)


def group_others(df):
    df["others"] = df["pr_soft_coral"] + df["pr_oth_invert"] + df["pr_other"]
    df = df.drop(columns = ["pr_oth_invert", "pr_other", "pr_soft_coral"])
    return df
    
"""    Useless map drawing
#get shapefile name list
shpList = [x[:-4] for x in os.listdir('../Data/GBR_shapes_surveys') if x[-3:]=='shp']

#dropdown widget
selDrop = widgets.Dropdown(
    options=shpList,
    value=shpList[0],
    description='Shapefile:',
    disabled=False,
)

display(selDrop)

#button widget
selBot = widgets.Button(
    description='Select shapefile',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Select shapefile',
    icon='check'
)

display(selBot)

#map object
m = Map(center=(52.3,8.0), zoom = 10, 
        basemap = basemaps.OpenTopoMap)

zoom_slider = IntSlider(description='Zoom level:', min=0, max=15, value=7)
jslink((zoom_slider, 'value'), (m, 'zoom'))
widgetControl = WidgetControl(widget=zoom_slider, position='topright')
m.add_control(widgetControl)

m.add_layer(basemaps.OpenStreetMap.Mapnik)
m.add_control(ScaleControl(position='bottomleft'))
m.add_control(FullScreenControl(position='topright'))
m.add_control(LayersControl())
display(m)

#show map function
def showMap(selShp):

    selDf = gpd.read_file('../Data/GBR_shapes_surveys/'+selShp+'.shp')


    geoDf = selDf.set_crs(epsg=4326)
    lonCent = (geoDf.bounds.maxx + geoDf.bounds.minx).mean()/2
    latCent = (geoDf.bounds.maxy + geoDf.bounds.miny).mean()/2

    m.center = (latCent,lonCent)
    geoData = GeoData(geo_dataframe=geoDf, name=selShp)
    m.add_layer(geoData)

#on click function
def on_button_clicked(b):
    showMap(selDrop.value)

selBot.on_click(on_button_clicked)




# NOT USEFUL
# This plots the triangular shapes on the GBR shape
shx_GBR_regions = gpd.read_file("../Data/GBR_shapes/GBR_1_finale.shp")
ax = shx_GBR_regions.plot(figsize=(15, 5), edgecolor='purple')
shx.plot(ax = ax)
# NOT USEFUL
# This function is used to separate the regions using the triangle wich were cut on QGIS 
def GBR_region(df,ID):
    return df[df["id"] == ID]

GBR_region1 = GBR_region(shx_GBR_regions,1)
GBR_region2 = GBR_region(shx_GBR_regions,2)
GBR_region3 = GBR_region(shx_GBR_regions,3)
GBR_region4 = GBR_region(shx_GBR_regions,4)

# NOT USEFUL
# This plots the means of the different "species" for each region
def inter_mean(df_region,df_survey,year):
    inter = gpd.sjoin(df_region, df_survey.set_crs(epsg=4326))
    plot_means(inter, year)
    

inter_mean(GBR_region1, gSurvey_2012, 2012)
inter_mean(GBR_region1, gSurvey_2014, 2014)
inter_mean(GBR_region1, gSurvey_2016, 2016)
inter_mean(GBR_region1, gSurvey_2017, 2017)

plt.title("Distribution of types for Region 1")
plt.legend()

inter_mean(GBR_region2, gSurvey_2012, 2012)
inter_mean(GBR_region2, gSurvey_2014, 2014)
inter_mean(GBR_region2, gSurvey_2016, 2016)
inter_mean(GBR_region2, gSurvey_2017, 2017)

inter_mean(GBR_region2, gSurvey_2012, 2012)
inter_mean(GBR_region2, gSurvey_2014, 2014)
inter_mean(GBR_region2, gSurvey_2016, 2016)
inter_mean(GBR_region2, gSurvey_2017, 2017)

plt.title("Distribution of types for Region 2")
plt.legend()


inter_mean(GBR_region3, gSurvey_2012, 2012)
inter_mean(GBR_region3, gSurvey_2014, 2014)
inter_mean(GBR_region3, gSurvey_2016, 2016)
inter_mean(GBR_region3, gSurvey_2017, 2017)

plt.title("Distribution of types for Region 3")
plt.legend()

inter_mean(GBR_region4, gSurvey_2012, 2012)
inter_mean(GBR_region4, gSurvey_2014, 2014)
inter_mean(GBR_region4, gSurvey_2016, 2016)
inter_mean(GBR_region4, gSurvey_2017, 2017)

plt.title("Distribution of types for Region 4")
plt.legend()


# 
# Not very insightful
def plot_means(df, year):
    return df[["pr_hard_coral", "pr_algae", "pr_soft_coral", "pr_oth_invert", "pr_other"]].mean().plot(label = year)#kind = "bar")
plot_means(gSurvey_2012, "2012")
plot_means(gSurvey_2014, "2014")
plot_means(gSurvey_2016, "2016")
plot_means(gSurvey_2017, "2017")
plt.legend()

"""