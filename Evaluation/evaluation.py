import os

import json
from statistics import mean
from pathlib import Path
from pprint import pprint

import numpy as np
import SimpleITK as sitk

# for local test of the evaluation code uncomment the following two lines
# INPUT_DIRECTORY = "test_submission"
# OUTPUT_DIRECTORY = "output"

# for grand-challenge evaluation docker website uncomment the following two lines
INPUT_DIRECTORY = "/input"
OUTPUT_DIRECTORY = "/output"


def surface_mean_distance(reference_segmentation, seg):
    """
    Compute symmetric surface distances and take the mean.
    """
    reference_surface = sitk.LabelContour(reference_segmentation)
    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            reference_surface, squaredDistance=False, useImageSpacing=True
        )
    )

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_surface = sitk.LabelContour(seg)
    segmented_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            segmented_surface, squaredDistance=False, useImageSpacing=True
        )
    )

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(
        segmented_surface, sitk.sitkFloat32
    )
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(
        reference_surface, sitk.sitkFloat32
    )

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(
        np.zeros(num_segmented_surface_pixels - len(seg2ref_distances))
    )

    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(
        np.zeros(num_reference_surface_pixels - len(ref2seg_distances))
    )

    # ref2seg_distances = ref2seg_distances
    all_surface_distances = np.mean(seg2ref_distances) / 2 + np.mean(ref2seg_distances) / 2
    return all_surface_distances


def main():
    print_inputs()

    metrics = {"case": {}}
    predictions = read_predictions()

    for job in predictions:
        ct = get_image_name(values=job["inputs"], slug="mediastinal-ct")

        batch_id = ct.split("-")[2]
        pprint(job)
        pprint(f"Processing batch {batch_id}")
        pprint((ct))

        segmentation_location = get_file_location(job_pk=job["pk"], values=job["outputs"],
                                                  slug="mediastinal-lymph-node")

        segmentation = load_image(location=segmentation_location)
        pprint(segmentation)

        ground_truth = sitk.ReadImage(os.path.join('ground_truth', ct.replace('-ct', '-seg')))
        pprint(ground_truth)

        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkUInt8)
        caster.SetNumberOfThreads(1)
        gt = caster.Execute(ground_truth)
        pred = caster.Execute(segmentation)

        # Score the case
        overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures.SetNumberOfThreads(1)
        overlap_measures.Execute(gt, pred)

        metrics["case"][batch_id] = {'DiceCoefficient': overlap_measures.GetDiceCoefficient(),
                                     'AverageSymmetricSurfaceDistance': surface_mean_distance(gt, pred)}

    # generate an aggregate score
    metrics["aggregates"] = {"DiceCoefficient": mean(batch["DiceCoefficient"] for batch in metrics["case"].values()),
                             "AverageSymmetricSurfaceDistance": mean(batch["AverageSymmetricSurfaceDistance"]
                                                                     for batch in metrics["case"].values())}

    pprint(metrics)

    write_metrics(metrics=metrics)

    return 0


def print_inputs():
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("-"*100)
    print("Input Files:")
    pprint(input_files)
    print("-"*100)


def read_predictions():
    with open(f"{INPUT_DIRECTORY}/predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return f"{INPUT_DIRECTORY}/{job_pk}/output/{relative_path}"


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image(*, location, extension="mha"):
    mha_files = {f for f in Path(location).glob(f"*.{extension}") if f.is_file()}
    print(location, mha_files)
    if len(mha_files) == 1:
        mha_file = mha_files.pop()
        return sitk.ReadImage(mha_file)
    elif len(mha_files) > 1:
        raise RuntimeError(
            f"More than one mha file was found in {location!r}"
        )
    else:
        raise NotImplementedError


def write_metrics(*, metrics):
    # Write a json document that is used for ranking results on the leaderboard
    with open(f"{OUTPUT_DIRECTORY}/metrics.json", "w") as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    raise SystemExit(main())
