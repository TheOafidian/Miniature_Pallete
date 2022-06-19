import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

import matplotlib.pyplot as plt
from paints import closest_color, return_paint_name, citadel_paints

matplotlib.use("Agg")


citadel_paint_colors = citadel_paints["RGB"]


class Palette_Extract:
    def __init__(
        self, image_file, palette_file="palette_extract.png", save_outputs=False
    ) -> None:
        # Read the image
        self.test_img = cv2.imread(image_file)
        self.miniature = self._extract_miniature(self.test_img)
        clustered_img = self._cluster_colors(self.miniature, 19)
        # Convert to RGB
        res2_rgb = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)
        # Flatten data
        flattened_cluster = res2_rgb.reshape(-1, res2_rgb.shape[2])
        self.paint_data = self._dataframe_miniature_paints_used(flattened_cluster)

        # Generate pieplot
        self.plot_palette(palette_file)

        if save_outputs:
            plt.imsave("clustered.jpeg", res2_rgb)
            plt.imsave("extracted.jpeg", self.miniature)

    def _extract_miniature(self, image):
        """Apply the grabCut method from cv2 to extract the miniature in the foreground."""

        mask = np.zeros(image.shape[:2], np.uint8)

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)

        # set these through selection in future
        rectangle = (15, 15, image.shape[0] - 10, image.shape[1] - 10)

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

    def _cluster_colors(self, image, K):
        """Cluster the colors in the miniature using KMeans to get K amount of colors."""
        Z = image.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        return res.reshape((image.shape))

    def _count_occurences_color(self, color, image):
        count = 0
        for col in image:
            if col.tolist() == color:
                count += 1
        return count

    def _dataframe_miniature_paints_used(self, flat_clustered_img) -> pd.DataFrame:
        uq_cols = np.unique(flat_clustered_img, axis=0)

        col_frame = pd.DataFrame({"color_rgb": uq_cols.tolist()})

        # remove RGB 0,0,0 -> make this more specific! (sort or select 0,0,0)
        col_frame.drop(index=0, inplace=True)
        col_frame["count"] = col_frame["color_rgb"].apply(
            lambda x: self._count_occurences_color(x, flat_clustered_img)
        )
        col_frame["percent"] = col_frame["count"] / col_frame["count"].sum()

        col_frame["rgb_plt"] = col_frame["color_rgb"].apply(
            lambda x: [c / 255 for c in x]
        )

        # find closest existing paint color
        col_frame["color_matched"] = col_frame["color_rgb"].apply(
            lambda x: closest_color(citadel_paint_colors, x)
        )
        # add color names from a df...
        col_frame["Paint"] = col_frame["color_matched"].apply(
            lambda x: return_paint_name(x, citadel_paints)
        )
        return col_frame

    def plot_palette(self, filename):
        col_frame_shortened = (
            self.paint_data.groupby(by="Paint")
            .agg({"percent": "sum", "rgb_plt": "first"})
            .sort_values(by=["percent"], ascending=False)
        )

        bottom = 1
        plt.figure(figsize=(4, 4))

        for row in col_frame_shortened.iterrows():
            label = row[0]
            height, color = row[1]
            bottom -= height

            plt.bar(0, height, 0.2, bottom=bottom, color=color, label=label)

        plt.axis("off")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        if filename is None:
            filename = "palette_extract.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    import time

    tic = time.perf_counter()
    Palette_Extract("test_imgs/test_mini.jpeg")
    toc = time.perf_counter()
    print(f"Took {toc-tic} seconds")
