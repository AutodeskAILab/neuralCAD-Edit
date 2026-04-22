from src.utils.args import parse_args
from src.utils.process_config import load_config
import os
import os.path as osp
import json
from src.utils.db import DatabaseManager
import shutil
           


def crawl_and_load(db, annotations_dir):
    # Find all folders containing "brep_end" subdirectory
    for root, dirs, files in os.walk(annotations_dir):
        # Check if this directory contains settings.json (for requests)
        if "settings.json" in files and "brep_start" in dirs:

            print(f"Processing folder: {root}")

            folder_path = root
            
            file_path = osp.join(folder_path, "settings.json")
            
            with open(file_path, "r") as f:
                data = json.load(f)

            # check if @ in userId
            if "@" in data["userId"]:
                data["userId"] = data["userId"].split("@")[0]

            db.insert_user(
                user_id=data["userId"],
                email=data.get("userName", None),
                vlm_config=None,
                is_human=data.get("isHuman", True)
            )

            video_path = osp.join(folder_path, "screen_recording.mov")
            video_path = video_path if osp.exists(video_path) else None

            # brep_start_folder = osp.join(folder_path, "brep_start")
            brep_start_folder = os.listdir(osp.join(folder_path, "brep_start"))[0]
            brep_start_folder = osp.join(folder_path, "brep_start", brep_start_folder)


            if osp.exists(brep_start_folder):

                brep_id = db.insert_brep(
                    orig_path=brep_start_folder,
                    user=data["userId"],
                    end_time=data["end_time"],
                )

                frames_src = osp.join(folder_path, "frames")
                procedure_id = data["edit_request_id"]
                frames_dst = osp.join(db.root_dir, db.frames_dir, procedure_id)

                # copy frames to the frames directory if it doesn't exist
                if osp.exists(frames_src):
                    if not osp.exists(frames_dst):
                        os.makedirs(frames_dst, exist_ok=True)
                        for frame_file in os.listdir(frames_src):
                            src_file = osp.join(frames_src, frame_file)
                            dst_file = osp.join(frames_dst, frame_file)
                            shutil.copy(src_file, dst_file)

                db.insert_request(
                    request_id=data["edit_request_id"],
                    user=data["userId"],
                    difficulty=data.get("edit_difficulty", None),
                    brep_start=brep_id,
                    instructions=video_path,
                    start_time=None, # start time from quicktime is not valid
                    end_time=data["end_time"],
                    events=data.get("events", []),
                    text=data.get("edit_note", None),
                    frames_dir=frames_dst,
                    filename=data.get("fileName", None),
                    prompt=data.get("prompt", None),
                )

            # check if it already contains a corrected transcription (not aligned yet)
            transcript_fn = osp.join(folder_path, "screen_recording_transcript", "screen_recording_transcript_segments.json")
            if osp.exists(transcript_fn):
                with open(transcript_fn, "r") as f:
                    transcript_data = json.load(f)
                db.requests.update_one(
                    {"_id": data["edit_request_id"]},
                    {"$set": {"corrected_transcript_segments": transcript_data}}
                )
        
        # Check if this directory contains a "brep_end" subdirectory for actions
        if "brep_end" in dirs:
            folder_path = root
            
            # Process for edit paths (folders with brep_end)
            brep_end_path = osp.join(folder_path, "brep_end")
            
            # get all folders in the brep_end folder and sort them by name
            brep_end_folders = sorted(
                [f for f in os.listdir(brep_end_path) if osp.isdir(osp.join(brep_end_path, f))]
            )
            brep_final_folder = osp.join(brep_end_path, brep_end_folders[-1]) if brep_end_folders else None
            
            if brep_final_folder is None:
                continue
                
            file_path = osp.join(brep_final_folder, "settings.json")
            if not osp.exists(file_path):
                continue
                
            with open(file_path, "r") as f:
                data = json.load(f)

            # check if @ in userId
            if "@" in data["userId"]:
                data["userId"] = data["userId"].split("@")[0]

            db.insert_user(
                user_id=data["userId"],
                email=data.get("userName", None),
                vlm_config=None,
                is_human=data.get("isHuman", True)
            )

            brep_id = db.insert_brep(
                orig_path=brep_final_folder,
                user=data["userId"],
                end_time=data.get("end_time", None),
            )

            frames_src = osp.join(folder_path, "frames")
            procedure_id = data["edit_id"]
            frames_dst = osp.join(db.root_dir, db.frames_dir, procedure_id)

            # copy frames to the frames directory if it doesn't exist
            if osp.exists(frames_src):
                if not osp.exists(frames_dst):
                    os.makedirs(frames_dst, exist_ok=True)
                    for frame_file in os.listdir(frames_src):
                        src_file = osp.join(frames_src, frame_file)
                        dst_file = osp.join(frames_dst, frame_file)
                        shutil.copy(src_file, dst_file)

            db.insert_edit(
                edit_id=data["edit_id"],
                request_id=data["edit_request_id"],
                brep_end_id=brep_id,
                user_id=data["userId"],
                start_time=data.get("start_time", None),
                end_time=data.get("end_time", None),
                events=data.get("events", []),
                frames_dir=frames_dst,
                filename=data.get("fileName", None),
                token_counts=data.get("token_counts", None),
                completion=data.get("completion", None),
                prompt_completion=data.get("prompt_completion", None),
                failed_run=data.get("failed_run", False)
                )
        

def create_database_from_annotations_dir(config: dict) -> None:
    """
    Creates the instructions database.

    Args:
        config (dict): Configuration dictionary.
    """
    db = DatabaseManager(config)

    for dir in config["annotations_dir"]["paths"]:
        if not osp.exists(dir):
            print(f"Annotations directory {dir} does not exist.")
            continue
        crawl_and_load(db, dir)

    for dir in config["models_dir"]["paths"]:
        if not osp.exists(dir):
            print(f"Models directory {dir} does not exist.")
            continue
        crawl_and_load(db, dir)

    return  db

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    db = create_database_from_annotations_dir(config)

    # db.print_db()
    db.print_db_summary()

    db.close_connection()

if __name__ == "__main__":
    main()



