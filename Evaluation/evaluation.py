import os

import json
from statistics import mean
from pathlib import Path
from pprint import pprint

import SimpleITK

# for local test of the evaluation code uncomment the following two lines
# INPUT_DIRECTORY = "test_submission"
# OUTPUT_DIRECTORY = "output"

# for building docker for Grand-Challenge website uncomment the following two lines
INPUT_DIRECTORY = "/input"
OUTPUT_DIRECTORY = "/output"


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

        ground_truth = SimpleITK.ReadImage(os.path.join('ground_truth', ct.replace('-ct', '-seg')))
        pprint(ground_truth)

        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkUInt8)
        caster.SetNumberOfThreads(1)
        gt = caster.Execute(ground_truth)
        pred = caster.Execute(segmentation)

        # Score the case
        overlap_measures = SimpleITK.LabelOverlapMeasuresImageFilter()
        overlap_measures.SetNumberOfThreads(1)
        overlap_measures.Execute(gt, pred)

        metrics["case"][batch_id] = {'DiceCoefficient': overlap_measures.GetDiceCoefficient(),}

    # generate an aggregate score
    metrics["aggregates"] = {"DiceCoefficient": mean(batch["DiceCoefficient"] for batch in metrics["case"].values())}

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
        return SimpleITK.ReadImage(mha_file)
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
