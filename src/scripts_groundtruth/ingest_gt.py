#!/usr/bin/env python3
"""
Ingest Ground Truth Script

Reads JSONL files from groundtruth directories specified in the config,
and updates the database with ground truth annotations.
"""

from datetime import datetime
from src.utils.args import parse_args
from src.utils.process_config import load_config
import os
import os.path as osp
import json
from src.utils.db import DatabaseManager


def load_jsonl(file_path: str) -> list:
    """
    Load a JSONL file and return a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line in the JSONL file
    """
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file_path}: {e}")
    return records


def find_jsonl_files(directory: str) -> list:
    """
    Recursively find all JSONL files in a directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of paths to JSONL files
    """
    jsonl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(osp.join(root, file))
    return jsonl_files


def ingest_groundtruth_record(db: DatabaseManager, record: dict) -> bool:
    """
    Ingest a single ground truth record into the database.
    
    Creates a user entry for the rater (if doesn't exist) and inserts
    a rating with instruction_rating and quality_rating.
    
    Args:
        db: DatabaseManager instance
        record: Dictionary containing ground truth data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get rater worker id
        rater_worker_id = record.get('rater_worker_id')
        if not rater_worker_id:
            print(f"Record missing rater_worker_id: {record}")
            return False
        # update rater_worker_id to user key
        record['user'] = rater_worker_id
        del record['rater_worker_id']
        
        # Get edit_id from result_info
        result_info = record.get('result_info', {})
        edit_id = result_info.get('edit_id')
        if not edit_id:
            print(f"Record missing edit_id in result_info: {record}")
            return False
        record['edit'] = edit_id
        
        # # Get ratings
        # instruction_rating = record.get('instruction_rating')
        # quality_rating = record.get('quality_rating')
        # creation_date = record.get('creation_date')
        
        # if instruction_rating is None and quality_rating is None:
        #     print(f"Record missing both instruction_rating and quality_rating: {edit_id}")
        #     return False
        
        # # Insert rating
        # rating_kwargs = {}
        # if instruction_rating is not None:
        #     rating_kwargs['score_instr'] = instruction_rating
        # if quality_rating is not None:
        #     rating_kwargs['score_quality'] = quality_rating
        # if creation_date is not None:
        #     rating_kwargs['creation_date'] = creation_date


        key_modifications = {'instruction_rating': 'score_instr', 'quality_rating': 'score_quality'}
        necessary_fields = ['score_instr', 'score_quality']
        # update record keys
        for old_key, new_key in key_modifications.items():
            if old_key in record:
                record[new_key] = record[old_key]
                # remove old key
                del record[old_key]

        # check all necessary fields are present
        for field in necessary_fields:
            if field not in record:
                print(f"Record missing necessary field {field}: {record}")
                return False

        # Create user entry for rater if doesn't exist
        db.insert_user(
            user_id=rater_worker_id,
            email=None,
            vlm_config=None,
            is_human=True
        )

        # correct times to handle different timezones if necesssary
        record['submission_time'] = datetime.fromisoformat(record['submission_time'].replace("Z", "+00:00"))
        record['job_info']['creation_date'] = datetime.fromisoformat(record['job_info']['creation_date'].replace("Z", "+00:00"))

        # update existing rating if it doesn't contain the necessary ratings or it is older
        if db.rating_exists(rater_worker_id, edit_id):
            rating = db.ratings.find_one({"edit": edit_id, "user": rater_worker_id})
            rating_id = rating["_id"]
            to_update = False
            if all(field in rating for field in necessary_fields):
                if 'submission_time' in rating:
                    # overwrite only if the current entry is newer
                    if record.get('submission_time', '') >= rating['submission_time']:
                        to_update = True
                else:
                    to_update = True
            else:
                to_update = True

            if to_update:
                print(f"Updating existing rating {rating_id} for edit {edit_id} by rater {rater_worker_id}")
                db.ratings.update_one(
                    {"_id": rating_id},
                    {"$set": record}
                )
                return True
            else:
                print(f"Skipping update for existing rating {rating_id} for edit {edit_id} by rater {rater_worker_id} as it is up-to-date")
                return True
        else:
            print(f"Inserting new rating for edit {edit_id} by rater {rater_worker_id}")
            rating_id = db.insert_rating(
                **record
            )
            return True

        # db.ratings.update_one(
        #     {"_id": rating_id},
        #     {"$set": rating_kwargs}
        # )
        
        
    except Exception as e:
        print(f"Error ingesting record: {e}")
        return False


def ingest_groundtruth_from_directory(db: DatabaseManager, gt_directory: str) -> tuple:
    """
    Ingest all ground truth JSONL files from a directory.
    
    Args:
        db: DatabaseManager instance
        gt_directory: Path to ground truth directory
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    if not osp.exists(gt_directory):
        print(f"Ground truth directory does not exist: {gt_directory}")
        return 0, 0
    
    jsonl_files = find_jsonl_files(gt_directory)
    
    if not jsonl_files:
        print(f"No JSONL files found in {gt_directory}")
        return 0, 0
    
    print(f"Found {len(jsonl_files)} JSONL files in {gt_directory}")
    
    successful = 0
    failed = 0
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file}")
        records = load_jsonl(jsonl_file)
        print(f"  Loaded {len(records)} records")
        
        for record in records:
            if ingest_groundtruth_record(db, record):
                successful += 1
            else:
                failed += 1
    
    return successful, failed


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if groundtruth_dir is in config
    if "groundtruth_dir" not in config:
        print("Error: 'groundtruth_dir' not found in config")
        return
    
    gt_config = config["groundtruth_dir"]
    if gt_config.get("type") != "local":
        print(f"Error: Only 'local' type is supported for groundtruth_dir, got: {gt_config.get('type')}")
        return
    
    gt_paths = gt_config.get("paths", [])
    if not gt_paths:
        print("Error: No paths specified in groundtruth_dir")
        return
    
    # Initialize database
    db = DatabaseManager(config)
    
    total_successful = 0
    total_failed = 0
    
    # Process each ground truth directory
    for gt_path in gt_paths:
        print(f"Processing ground truth directory: {gt_path}")
        
        successful, failed = ingest_groundtruth_from_directory(db, gt_path)
        total_successful += successful
        total_failed += failed
    
    # Print summary
    print(f"Total records successfully ingested: {total_successful}")
    print(f"Total records failed: {total_failed}")
    
    db.close_connection()


if __name__ == "__main__":
    main()
