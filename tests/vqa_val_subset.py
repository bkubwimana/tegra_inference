
"""VQA v2 loading script - MODIFIED for first 600 validation samples."""

import json
import os
from pathlib import Path
import datasets


_DESCRIPTION = """\
A subset of the VQA v2 dataset containing the first 600 open-ended questions
from the validation split, along with their corresponding images and annotations.
Intended for focused experiments like prompt engineering studies.
"""

_HOMEPAGE = "https://visualqa.org"

_LICENSE = "CC BY 4.0"

# MODIFIED: Only include URLs needed for the validation subset
_VAL_SUBSET_URLS = {
    "questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "images": "http://images.cocodataset.org/zips/val2014.zip",
}

# MODIFIED: Only include file/folder names for validation
_VAL_SUBSET_FILENAMES = {
    "questions": "v2_OpenEnded_mscoco_val2014_questions.json",
    "annotations": "v2_mscoco_val2014_annotations.json",
    "images": "val2014",
}

# MODIFIED Class Name
class VQAv2ValidationSubset(datasets.GeneratorBasedBuilder):
    """VQA v2 dataset - First 600 validation samples."""

    VERSION = datasets.Version("1.0.0")
    # No Builder Configs needed for this simple subset

    # The features (structure of one example) remain the same
    def _info(self):
        features = datasets.Features(
            {
                "question_type": datasets.Value("string"),
                "multiple_choice_answer": datasets.Value("string"),
                "answers": [
                    {
                        "answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                        "answer_id": datasets.Value("int64"),
                    }
                ],
                "image_id": datasets.Value("int64"),
                "answer_type": datasets.Value("string"),
                "question_id": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    # MODIFIED: Only define one split generator for the validation subset
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download only the necessary validation files
        data_dir = dl_manager.download_and_extract(_VAL_SUBSET_URLS)

        # Prepare paths for the generator
        val_kwargs = {
            "questions_path": Path(data_dir["questions"]) / _VAL_SUBSET_FILENAMES["questions"],
            "annotations_path": Path(data_dir["annotations"]) / _VAL_SUBSET_FILENAMES["annotations"],
            "images_path": Path(data_dir["images"]) / _VAL_SUBSET_FILENAMES["images"],
        }

        return [
            datasets.SplitGenerator(
                name="validation_subset",  # Custom split name
                gen_kwargs=val_kwargs,
            ),
        ]

    # MODIFIED: Generate only the first 600 examples
    def _generate_examples(self, questions_path, annotations_path, images_path):
        """Yields examples based on validation data, limited to first 600."""
        questions_data = json.load(open(questions_path, "r"))
        annotations_data = json.load(open(annotations_path, "r"))

        # Create a lookup for annotations by question_id for efficiency
        annotations_lookup = {ann["question_id"]: ann for ann in annotations_data["annotations"]}

        count = 0
        max_examples = 600

        # NOTE: We take the first 600 examples based on the iteration order
        # of questions in the v2_OpenEnded_mscoco_val2014_questions.json file.
        # This order is usually stable but not strictly guaranteed across file versions.
        for question in questions_data["questions"]:
            if count >= max_examples:
                break # Stop after yielding 600 examples

            question_id = question["question_id"]
            if question_id in annotations_lookup:
                annotation = annotations_lookup[question_id]

                # Basic checks (optional but good practice)
                # assert question["image_id"] == annotation["image_id"]

                # Combine question and annotation info
                record = {**question, **annotation} # Merge dicts (Python 3.5+)

                # Construct the image path
                image_filename = f"COCO_{images_path.name}_{record['image_id']:0>12}.jpg"
                record["image"] = str(images_path / image_filename)

                # Yield the combined record with a unique key for this subset
                yield f"val_subset_{question_id}", record
                count += 1
            # else:
            #     # Should not happen for validation set if files are correct
            #     print(f"Warning: Annotation not found for question_id {question_id}")