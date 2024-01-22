import csv
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_images_path = "/home/alex/DATASETS/TODO/TICaM/Train_images/RGB"
    train_anns_path = "/home/alex/DATASETS/TODO/TICaM/Train_labels/Train_Labels"
    test_images_path = "/home/alex/DATASETS/TODO/TICaM/Test_images/RGB"

    batch_size = 10
    group_tag_name = "im id"
    images_ext = "_RGB.png"
    depth_ext = "_DEPTH.png"
    ir_ext = "_IR.png"
    masks_ext = "_DEPTH_classes.png"
    bboxes_file_name = "boxes_2d_depth.csv"
    activities_file_name = "activities.csv"

    ds_name_to_images = {
        "train": (train_images_path, train_anns_path),
        "test": (test_images_path, None),
    }

    def create_ann(image_path):
        labels = []
        tags = []
        group_tag_value = "_".join(get_file_name(image_path).split("_")[:-1])
        group_tag = sly.Tag(group_tag_meta, value=group_tag_value)
        tags.append(group_tag)

        tag_subfolder = sly.Tag(subfolder_meta, value=subfolder)
        tags.append(tag_subfolder)

        img_height = 512
        img_wight = 512

        if anns_path is None:
            if get_file_name(image_path)[-3:] == "RGB":
                img_height = 720
                img_wight = 1280
            else:
                img_height = 512
                img_wight = 512
            return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

        frame = get_file_name(image_path).split("_")[1]
        activity_data = frame_to_activity_data.get(int(frame))
        if activity_data is not None:
            activity = sly.Tag(activity_meta, value=activity_data[0])
            tags.append(activity)
            person_id = sly.Tag(person_id_meta, value=activity_data[1])
            tags.append(person_id)
            status = sly.Tag(status_meta, value=activity_data[2])
            tags.append(status)
            duration = sly.Tag(duration_meta, value=activity_data[3])
            tags.append(duration)

        if (
            get_file_name(image_path)[-3:] == "RGB"
        ):  # another shape, resize not possible(a lower field of view) https://arxiv.org/pdf/2103.11719.pdf page_3
            img_height = 720
            img_wight = 1280
            return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

        mask_path = os.path.join(
            anns_path,
            subfolder,
            image_path.split("/")[-2],
            "masks_depth",
            image_path.split("/")[-1].replace(images_ext, masks_ext),
        )

        if not file_exists(mask_path):
            mask_path = mask_path.replace(depth_ext, masks_ext)
            if not file_exists(mask_path):
                mask_path = mask_path.replace(ir_ext, masks_ext)

        ann_np = sly.imaging.image.read(mask_path)[:, :, 0]
        unique_pixels = np.unique(ann_np)[1:]
        for pixel in unique_pixels:
            class_name = idx_to_class.get(pixel)
            obj_class = meta.get_obj_class(class_name)
            mask = ann_np == pixel
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                curr_bitmap = sly.Bitmap(obj_mask)
                if curr_bitmap.area > 30:
                    curr_label = sly.Label(curr_bitmap, obj_class)
                    labels.append(curr_label)

        bboxes_data = frame_to_bboxes_data[frame]
        for curr_bboxes_data in bboxes_data:
            obj_class = meta.get_obj_class(curr_bboxes_data[0])
            remission = sly.Tag(remission_meta, value=int(curr_bboxes_data[1]))
            left = curr_bboxes_data[2][0]
            right = curr_bboxes_data[2][2]
            top = curr_bboxes_data[2][1]
            bottom = curr_bboxes_data[2][3]
            rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            label = sly.Label(rectangle, obj_class, tags=[remission])
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    activity_meta = sly.TagMeta("activity", sly.TagValueType.ANY_STRING)
    person_id_meta = sly.TagMeta("person id", sly.TagValueType.ANY_NUMBER)
    status_meta = sly.TagMeta("status", sly.TagValueType.ANY_STRING)
    duration_meta = sly.TagMeta("duration", sly.TagValueType.ANY_NUMBER)
    subfolder_meta = sly.TagMeta("sequence", sly.TagValueType.ANY_STRING)
    remission_meta = sly.TagMeta("low remission", sly.TagValueType.ANY_NUMBER)
    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)
    meta = sly.ProjectMeta(
        tag_metas=[
            group_tag_meta,
            subfolder_meta,
            remission_meta,
            activity_meta,
            person_id_meta,
            status_meta,
            duration_meta,
        ]
    )

    idx_to_class = {
        0: "background",
        1: "person",
        2: "backpack",
        3: "winter jacket",
        4: "box",
        5: "water bottle",
        6: "mobile phone",
        7: "blanket",
        8: "accessory",
        9: "book",
        10: "laptop",
        11: "laptop bag",
        12: "infant",
        13: "handbag",
        14: "ff",  # Front-facing childseat
        15: "rf",  # Rearfacing childseat
        16: "child",
    }

    for idx, name in idx_to_class.items():
        obj_class = sly.ObjClass(name, sly.AnyGeometry)
        meta = meta.add_obj_class(obj_class)

    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    for ds_name, ds_data in ds_name_to_images.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path, anns_path = ds_data

        for subfolder in os.listdir(images_path):
            subpath = os.path.join(images_path, subfolder)
            if anns_path is not None:
                subpath_ann = os.path.join(anns_path, subfolder)
            for curr_subdir in os.listdir(subpath):
                curr_images_path = os.path.join(subpath, curr_subdir)
                if anns_path is not None:
                    curr_anns_path = os.path.join(subpath_ann, curr_subdir)
                    frame_to_bboxes_path = os.path.join(curr_anns_path, bboxes_file_name)

                    frame_to_bboxes_data = defaultdict(list)
                    with open(frame_to_bboxes_path, "r") as file:
                        csvreader = csv.reader(file)
                        for idx, row in enumerate(csvreader):
                            if idx == 0:
                                continue
                            frame_to_bboxes_data[row[0]].append(
                                [row[1], row[-1], list(map(int, row[2:6]))]
                            )

                    frame_to_activity_path = os.path.join(curr_anns_path, activities_file_name)
                    frame_to_activity_data = {}
                    if file_exists(frame_to_activity_path):
                        with open(frame_to_activity_path, "r") as file:
                            csvreader = csv.reader(file)
                            for idx, row_str in enumerate(csvreader):
                                if idx == 0:
                                    continue
                                if len(row_str) == 1:
                                    row = row_str[0].split(" ")
                                else:
                                    row = row_str

                                for i in range(int(row[4]), int(row[5])):
                                    frame_to_activity_data[i] = [
                                        row[1],
                                        int(row[2]),
                                        row[3],
                                        int(row[-1]),
                                    ]

                images_names = os.listdir(curr_images_path)

                progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

                for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                    img_pathes_batch = []
                    images_names_batch = []
                    ann_img_pathes_batch = []
                    for im_name in img_names_batch:
                        images_names_batch.append(im_name)
                        im_path = os.path.join(curr_images_path, im_name)
                        img_pathes_batch.append(im_path)

                        depth_name = im_name.replace("_RGB.png", "_DEPTH.png")
                        images_names_batch.append(depth_name)
                        img_pathes_batch.append(
                            os.path.join(curr_images_path.replace("/RGB/", "/Depth/"), depth_name)
                        )

                        ir_name = im_name.replace("_RGB.png", "_IR.png")
                        images_names_batch.append(ir_name)
                        img_pathes_batch.append(
                            os.path.join(curr_images_path.replace("/RGB/", "/IR/"), ir_name)
                        )

                    img_infos = api.image.upload_paths(
                        dataset.id, images_names_batch, img_pathes_batch
                    )
                    img_ids = [im_info.id for im_info in img_infos]

                    anns = []
                    anns = [create_ann(image_path) for image_path in img_pathes_batch]
                    api.annotation.upload_anns(img_ids, anns)

                    progress.iters_done_report(len(img_names_batch))

    return project
