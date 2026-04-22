from src.utils.args import parse_args
from src.utils.process_config import load_config
import os.path as osp
from src.utils.db import DatabaseManager
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
           


def extract_dino(db: DatabaseManager, config: dict, feature_info: list[dict]) -> None:

    # get infos where the brep doesn't already have a dino feature
    print("\nExtracting DINO features for breps...")

    to_extract = []
    for info in feature_info:
        if "feature_dino" not in db.breps.find_one({"_id": info["brep_id"]}) and info["frame_path"] is not None:
            to_extract.append(info)

    if not to_extract:
        print("No breps to extract features for.")
        return
    
    # extract dino features
    extractor = pipeline(
        task="image-feature-extraction",
        model=config["dino_model"],
    )

    input_paths = [e["frame_path"] for e in to_extract]

    features = extractor(
        input_paths,
        batch_size=config["dino_batch_size"],
        return_tensors=True
    )

    # loop over all features and insert them into the database
    for i, info in enumerate(to_extract):
        feature = features[i]
        feature = feature.squeeze().numpy()
        feature = feature.flatten().tolist()  # flatten the feature to a list
        db.breps.update_one(
            {"_id": info["brep_id"]},
            {"$set": {"feature_dino": feature}}
        )


def load_image_from_path(image_path: str):
    if not osp.exists(image_path):
        print(f"Image path {image_path} does not exist.")
        return None
    image = Image.open(image_path)
    return image

def extract_clip_visual(db: DatabaseManager, config: dict, feature_info: list[dict]) -> None:
    
    print("\nExtracting CLIP visual features for breps...")

    to_extract = []
    for info in feature_info:
        if "feature_clip_visual" not in db.breps.find_one({"_id": info["brep_id"]}) and info["frame_path"] is not None:
            to_extract.append(info)

    if not to_extract:
        print("No breps to extract features for.")
        return
    
    device = config["clip_device"]
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.to(device)

    input_paths = [e["frame_path"] for e in to_extract]

    # loop over batches of images. The last batch may be smaller than the batch size
    # and we will handle that automatically
    print(f"Extracting CLIP visual features for {len(input_paths)} breps in batches of {config['clip_batch_size']}...")
    features = []
    for i in range(0, len(input_paths), config["clip_batch_size"]):
        batch_paths = input_paths[i: min(i + config["clip_batch_size"], len(input_paths))]

        print(batch_paths)
        images = [load_image_from_path(p) for p in batch_paths if load_image_from_path(p) is not None]

        # images = [get_brep_path_from_folder(p) for p in batch_paths]
        inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features_np = image_features.cpu().numpy()
            features_list = features_np.tolist()
            features.extend(features_list)

    # loop over all features and insert them into the database
    for i, info in enumerate(to_extract):
        feature = features[i]
        db.breps.update_one(
            {"_id": info["brep_id"]},
            {"$set": {"feature_clip_visual": feature}}
        )

def extract_clip_text(db: DatabaseManager, config: dict, feature_text: list[dict]) -> None:
    print("\nExtracting CLIP text features for requests...")

    to_extract = []
    for info in feature_text:
        if "feature_clip_text" not in db.requests.find_one({"_id": info["id"]}) and "text" in info:
            to_extract.append(info)

    if not to_extract:
        print("No requests to extract features for.")
        return
    
    device = config["clip_device"]
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.to(device)

    texts = [f"A photo of {e['text']}" for e in to_extract]

    inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        # Normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        features_np = text_features.cpu().numpy()
        features_list = features_np.tolist()

    # loop over all features and insert them into the database
    for i, info in enumerate(to_extract):
        feature = features_list[i]
        db.requests.update_one(
            {"_id": info["id"]},
            {"$set": {"feature_clip_text": feature}}
        )

def extract_all_features(db: DatabaseManager, config: dict) -> None:

    # go through all requests and extract features from the breps
    all_requests_iterator = db.requests.find()

    feature_info = []

    for request in all_requests_iterator:
        brep_id = request["brep_start"]
        if not brep_id:
            continue
        frame_path = db.get_brep_images(brep_id)
        frame_path = osp.join(db.root_dir, frame_path[0]) if frame_path else None
        info = {"request_or_edit": "request", "id": request["_id"], "frame_path": frame_path, "brep_id": brep_id}
        feature_info.append(info)

    for edit in db.edits.find():
        brep_id = edit["brep_end"]
        if not brep_id:
            continue
        frame_path = db.get_brep_images(brep_id)
        frame_path = osp.join(db.root_dir, frame_path[0]) if frame_path else None

        info = {"request_or_edit": "edit", "id": edit["_id"], "frame_path": frame_path, "brep_id": brep_id}
        feature_info.append(info)

    # extract dino features
    extract_dino(db, config, feature_info)
    extract_clip_visual(db, config, feature_info)

    all_requests_iterator = db.requests.find({"request_type": "text2brep"})
    feature_text = []
    for request in all_requests_iterator:
        if "text" in request:
            feature_text.append({
                "request_or_edit": "request",
                "id": request["_id"],
                "text": request["text"]
            })
    extract_clip_text(db, config, feature_text)

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    db = DatabaseManager(config)

    extract_all_features(db, config)

    db.print_db_summary()

    db.close_connection()

if __name__ == "__main__":
    main()



