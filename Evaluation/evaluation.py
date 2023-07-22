import SimpleITK

from pathlib import Path
from jsonloader import load_predictions_json

from evalutils import ClassificationEvaluation
from evalutils.io import SimpleITKLoader
from evalutils.validators import (
    NumberOfCasesValidator, UniquePathIndicesValidator, UniqueImagesValidator
)


class Lnq2023(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=SimpleITKLoader(),
            validators=(
                NumberOfCasesValidator(num_cases=2),
                UniquePathIndicesValidator(),
                UniqueImagesValidator(),
            ),
        )
        self.mapping_dict = load_predictions_json(Path("/input/predictions.json"))

    def score_case(self, *, idx, case):
        gt_path = case["path_ground_truth"]
        pred_path = case["path_prediction"]

        # Load the images for this case
        gt = self._file_loader.load_image(gt_path)
        pred = self._file_loader.load_image(pred_path)

        # Check that they're the right images
        if (self._file_loader.hash_image(gt) != case["hash_ground_truth"] or
            self._file_loader.hash_image(pred) != case["hash_prediction"]):
            raise RuntimeError("Images do not match")

        # Cast to the same type
        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkUInt8)
        caster.SetNumberOfThreads(1)
        gt = caster.Execute(gt)
        pred = caster.Execute(pred)

        # Score the case
        overlap_measures = SimpleITK.LabelOverlapMeasuresImageFilter()
        overlap_measures.SetNumberOfThreads(1)
        overlap_measures.Execute(gt, pred)

        return {
            'DiceCoefficient': overlap_measures.GetDiceCoefficient(),
            'pred_fname': pred_path.name,
            'gt_fname': gt_path.name,
        }

    def load(self):
        self._ground_truth_cases = self._load_cases(folder=self._ground_truth_path)
        self._predictions_cases = self._load_cases(folder=self._predictions_path)

        self._predictions_cases["ground_truth_path"] = [
            self._ground_truth_path / self.mapping_dict[Path(path).name]
            for path in self._predictions_cases.path
        ]

        self._ground_truth_cases = self._ground_truth_cases.sort_values(
            "path"
        ).reset_index(drop=True)
        self._predictions_cases = self._predictions_cases.sort_values(
            "ground_truth_path"
        ).reset_index(drop=True)


if __name__ == "__main__":
    Lnq2023().evaluate()
