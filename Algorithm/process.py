from pathlib import Path

import SimpleITK

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


class Lnq2023(SegmentationAlgorithm):
    def __init__(self):
        output_path=Path('/output/images/mediastinal-lymph-node-segmentation/')
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        super().__init__(
            input_path=Path('/input/images/mediastinal-ct/'),
            output_path = output_path, 
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        # Segment all values greater than 2 in the input image
        print('abc')
        return SimpleITK.BinaryThreshold(
            image1=input_image, lowerThreshold=2, insideValue=1, outsideValue=0
        )


if __name__ == "__main__":
    Lnq2023().process()
