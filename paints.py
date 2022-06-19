import pandas as pd
import numpy as np
from PIL import ImageColor

def closest_color(colors:pd.Series,color):
    colors = colors.values
    colors = [list(col) for col in colors]
    colors = np.array(colors)
    color = np.array(color)

    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    return tuple(smallest_distance[0])

def return_paint_name(color_rgb, paint_df):
    return paint_df.loc[paint_df['RGB'] == color_rgb]['Paint_Name'].values[0]

citadel_paints = pd.read_csv('paints/citadel_paints.csv', header=None, names=['Paint_Name','Type','Hex'])
citadel_paints.dropna(axis=0, subset=['Hex'], inplace=True)
citadel_paints = citadel_paints.loc[citadel_paints['Type'].isin(['Layer','Base']) ]

citadel_paints['RGB'] = citadel_paints['Hex'].apply(lambda x: ImageColor.getcolor(x, 'RGB'))
