from src.utils.args import parse_args
from src.utils.process_config import load_config
import os.path as osp
import pandas as pd
from src.utils.db import DatabaseManager
import re
           
def read_tags(db: DatabaseManager, config: dict) -> None:
    for path in config["assets_dir"]["paths"]:
        csv_path = osp.join(path, "metadata", "tags.csv")
        if not osp.exists(csv_path):
            print(f"Tags CSV file not found at {csv_path}. Skipping tag reading.")
            return
        tags_df = pd.read_csv(csv_path)

        print(f"Found {len(tags_df)} tags in the CSV file.")
        for _, row in tags_df.iterrows():

            row = row.to_dict()
            if "notes" in row:
                del row["notes"]

            # set values to True if they are "yes" or "true", False if they are "no" or "false" or blank
            for k, v in row.items():
                if isinstance(v, str):
                    v = v.strip().lower()
                    if v in ["yes", "true", "y"]:
                        row[k] = True
                    elif v in ["no", "false", "n"]:
                        row[k] = False
                    elif v in [""]:
                        del row[k]

            if not db.request_exists(row["request_id"]):
                if (not pd.isna(row["user_id"])) and (not db.user_exists(row["user_id"])):
                    db.insert_user(user_id=row["user_id"])

                db.insert_request(
                    request_id=row["request_id"],
                    user=row["user_id"],
                    request_type=row["request_type"]
                )

            # remove all nan values from the row
            row = {k: v for k, v in row.items() if pd.notna(v)}

            request_id = row.pop("request_id")
            db.requests.update_one(
                {"_id": request_id},
                {"$set": {k: v for k, v in row.items()}}
            )

def clean_filename(filename):
    # orig_filename = filename
    # remove everything in ()
    filename = re.sub(r'\(.*?\)', '', filename)

    # remove trailing whitespace
    filename = filename.strip()

    # remove ' v1', ' v2' at the end of the filename
    filename = re.sub(r' v\d+$', '', filename)

    filename = filename.strip()

    # remove _v1, _v2 at the end of the filename
    filename = re.sub(r'_v\d+$', '', filename)

    # if filename != orig_filename:
    #     print(f"Cleaned filename from '{orig_filename}' to '{filename}'")

    return filename.strip()

def read_model_data(db: DatabaseManager, config: dict) -> None:

    for path in config["assets_dir"]["paths"]:
        model_data_path = osp.join(path, "metadata", "data.csv")
        if not osp.exists(model_data_path):
            print(f"Model data CSV file not found at {model_data_path}. Skipping model data reading.")
            return
        model_data_df = pd.read_csv(model_data_path)

        #build a mapping from filename to a list of request ids in the database that have that filename. This is needed because the CSV files may not have the same extension as the filenames in the database, and Mongita doesn't support regex queries.
        filename_to_request_ids = {}
        for request in db.requests.find({}):
            filename = request.get("filename", None)
            if filename:
                filename_without_ext = osp.splitext(filename)[0]
                filename_without_ext = clean_filename(filename_without_ext)
                if filename_without_ext not in filename_to_request_ids:
                    filename_to_request_ids[filename_without_ext] = []
                filename_to_request_ids[filename_without_ext].append(request["_id"])

        for _, row in model_data_df.iterrows():
            row = row.to_dict()
            if "notes" in row:
                del row["notes"]

            # remove all nan values from the row
            row = {k: v for k, v in row.items() if pd.notna(v)}

            # set values to True if they are "yes" or "true", False if they are "no" or "false" or blank
            for k, v in row.items():
                if isinstance(v, str):
                    v = v.strip().lower()
                    if v in ["yes", "true", "y"]:
                        row[k] = True
                    elif v in ["no", "false", "n", ""]:
                        row[k] = False


            filename_without_ext = osp.splitext(row["filename"])[0]
            filename_without_ext = clean_filename(filename_without_ext)

            row["filename_cleaned"] = filename_without_ext


            for request_id in filename_to_request_ids.get(filename_without_ext, []):
                db.requests.update_one(
                    {"_id": request_id},
                    {"$set": {k: v for k, v in row.items() if k != "filename"}}
                )
                # print(f"Updated request {request_id} with model data from filename '{row['filename']}'")






def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    db = DatabaseManager(config)

    read_tags(db, config)
    read_model_data(db, config)

    # db.print_db_summary()

    db.close_connection()

if __name__ == "__main__":
    main()



