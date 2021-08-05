import pandas as pd

from copy_annotations.anchor import Anchor
from copy_annotations.annotation import Annotation
from copy_annotations.selection import Selection

ANCHOR = 'anchor'

VALID = 'valid'

X = 'x'
Y = 'y'
CONTENT = 'content'


class Sheet:
    transformed_df = None
    transformed_annotations = None
    transformed_row_map = None
    transformed_column_map = None
    annotations = None

    def __init__(self, dataframe: pd.DataFrame, annotations: dict = None):
        if annotations:
            self.annotations = [Annotation(annotation) for annotation in annotations]

        self.df = dataframe
        self.transform()

    def find_anchors(self, target_df):
        anchors = {}
        candidates = self._get_anchors_candidates()

        for y, row in target_df.iterrows():
            for x, value in row.items():
                if isinstance(value, str):
                    for candidate in candidates:
                        if value.lower() == candidate[CONTENT].lower():
                            if anchors.get(value):
                                anchors[value][VALID] = False
                                continue

                            anchors[value] = {
                                ANCHOR: Anchor(value, candidate[X], candidate[Y], x + 1, y + 1),
                                VALID: True
                            }

        return [v[ANCHOR] for k, v in anchors.items() if v[VALID]]

    def _get_anchors_candidates(self):
        candidates = []
        for y, row in self.df.iterrows():
            for x, value in row.items():
                is_annotated = any([a.source_selection.contains(Selection(x + 1, x + 1, y + 1, y + 1)) for a in self.annotations])
                if isinstance(value, str) and not is_annotated:
                    candidates.append({
                        CONTENT: value,
                        X: x + 1,
                        Y: y + 1,
                    })

        return candidates

    def represent_transformed_annotations(self, key):
        default_content = 'UNLABELED' if self.annotations else ''
        annotation_df = pd.DataFrame(default_content, columns=self.transformed_df.columns, index=self.transformed_df.index)

        if not self.transformed_annotations:
            return annotation_df

        for y, row in self.transformed_df.iterrows():
            for x, value in row.items():
                for a in self.transformed_annotations:
                    if a.source_selection.contains(Selection(x + 1, x + 1, y + 1, y + 1)):
                        annotation_df.iloc[y, x] = a.__dict__[key]

        return annotation_df

    def transform(self):
        transformed_df = self.df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        self.transformed_row_map = list(transformed_df.index)
        self.transformed_column_map = list(transformed_df.columns)

        intermediate_target_df = transformed_df.reset_index(drop=True)
        intermediate_target_df.columns = [i for i in range(0, intermediate_target_df.shape[1])]
        self.transformed_df = intermediate_target_df

        if not self.annotations:
            return

        self.transformed_annotations = []
        for a in self.annotations:
            y1, x1 = self.get_transformed_coordinates(a.source_selection.y1, a.source_selection.x1)
            y2, x2 = self.get_transformed_coordinates(a.source_selection.y2, a.source_selection.x2)
            self.transformed_annotations.append(Annotation(a.generate_target_annotations(x1, x2, y1, y2)))

    def get_row_index(self, index):
        return self.transformed_row_map[index - 1] + 1

    def get_column_index(self, index):
        return self.transformed_column_map[index - 1] + 1

    def get_transformed_coordinates(self, original_row, original_column):
        transformed_row = self.transformed_row_map.index(original_row - 1) + 1
        transformed_column = self.transformed_column_map.index(original_column - 1) + 1
        return transformed_row, transformed_column
