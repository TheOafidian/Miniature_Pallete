import cv2
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paints import closest_color, return_paint_name, citadel_paints


test_img = cv2.imread("test_imgs/test_mini.jpeg")
citadel_paint_colors = citadel_paints["RGB"]


def extract_miniature(image):
    """Apply the grabCut method from cv2 to extract the miniature in the foreground."""

    mask = np.zeros(image.shape[:2], np.uint8)

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # set these through selection
    rectangle = (20, 20, image.shape[0], image.shape[1])

    # num of iterations = 5
    cv2.grabCut(
        image,
        mask,
        rectangle,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

    return image * mask2[:, :, np.newaxis]


def cluster_colors(image, K):
    """Cluster the colors in the miniature using KMeans to get K amount of colors."""
    Z = image.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((image.shape))


def count_occurences_color(color, image):
    count = 0
    for col in image:
        if col.tolist() == color:
            count += 1
    return count


image = extract_miniature(test_img)
clustered_img = cluster_colors(image, 19)
res2_rgb = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)

flattened_cluster = res2_rgb.reshape(-1, res2_rgb.shape[2])

uq_cols = np.unique(flattened_cluster, axis=0)

col_frame = pd.DataFrame({"color_rgb": uq_cols.tolist()})
# remove RGB 0,0,0 -> make this more specific! (sort or select 0,0,0)
col_frame.drop(index=0, inplace=True)
col_frame["count"] = col_frame["color_rgb"].apply(
    lambda x: count_occurences_color(x, flattened_cluster)
)
col_frame["percent"] = col_frame["count"] / col_frame["count"].sum()

col_frame["rgb_plt"] = col_frame["color_rgb"].apply(lambda x: [c / 255 for c in x])

# find closest existing paint color
col_frame["color_matched"] = col_frame["color_rgb"].apply(
    lambda x: closest_color(citadel_paint_colors, x)
)
# add color names from a df...
col_frame["Paint"] = col_frame["color_matched"].apply(
    lambda x: return_paint_name(x, citadel_paints)
)
col_frame_shortened = (
    col_frame.groupby(by="Paint")
    .agg({"percent": "sum", "rgb_plt": "first"})
    .sort_values(by=["percent"], ascending=False)
)

# colors = col_frame['rgb_plt']
# stackplot = col_frame[['percent']].T.plot.bar(stacked=True, color=colors, label='Paint')
# pieplot = plt.pie(col_frame['percent'], labels=col_frame['Paint'],autopct='%1.1f%%',
#        shadow=True, startangle=90, colors=col_frame['rgb_plt'])
bottom = 1

for row in col_frame_shortened.iterrows():
    label = row[0]
    height, color = row[1]
    bottom -= height

    plt.bar(0, height, 0.4, bottom=bottom, color=color, label=label)


plt.axis("off")
plt.legend()
plt.tight_layout()

plt.savefig("test.png")


plt.imsave("clustered.jpeg", res2_rgb)
plt.imsave("extracted.jpeg", image)
