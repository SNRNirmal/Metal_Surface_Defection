import os
import pandas as pd


class XMLExtraction:
    """
    Reads all XML annotation files from a folder and builds a DataFrame
    with columns: label, image_path, annotation_path.

    The XML filenames follow the pattern: {label}_{number}.xml
    The corresponding image files are: IMAGES/{label}_{number}.jpg
    """

    def __init__(self, annotation_folder: str, image_folder: str = '.\\IMAGES'):
        self.annotation_folder = annotation_folder
        self.image_folder = image_folder
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        records = []
        for filename in os.listdir(self.annotation_folder):
            if not filename.endswith('.xml'):
                continue

            annotation_path = os.path.join(self.annotation_folder, filename)
            name = filename[:-4]  # strip .xml

            parts = name.rsplit('_', 1)
            label = parts[0] if len(parts) == 2 else name

            image_path = os.path.join(self.image_folder, name + '.jpg')

            records.append({
                'label': label,
                'image_path': image_path,
                'annotation_path': annotation_path,
            })

        df = pd.DataFrame(records, columns=['label', 'image_path', 'annotation_path'])
        return df