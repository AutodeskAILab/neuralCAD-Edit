import subprocess
import os
import os.path as osp
from src.utils.args import parse_args
from src.utils.process_config import load_config
from src.utils.db import DatabaseManager
import json
import torch
import whisperx

def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )

def get_single_video_length(in_fn):
    try:
        # use ffprobe to get the video duration in seconds
        call_string = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {in_fn}"
        duration = subprocess.check_output(call_string, shell=True).strip()
        duration = float(duration)
    except subprocess.CalledProcessError:
        duration = 0
    except ValueError:
        duration = 0
    return duration
    

def extract_single_video(config, in_fn, out_fn, format):

    res = format["resolution"]
    fps = format["fps"]
    keep_audo = format.get("audio")

    if osp.exists(out_fn):
        return True

    try:
        # use ffprobe to get the original resolution
        call_string = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {in_fn}"
        original_resolution = subprocess.check_output(call_string, shell=True).strip().decode('utf-8')
        original_width, original_height = map(int, original_resolution.split(','))
        
        target_height = res
        target_width = original_width * target_height // original_height
        if target_width % 2 != 0:
            target_width += 1

        audio_str = " -an " if not keep_audo else " "

        call_string = f"ffmpeg -i {in_fn} -vf scale={target_width}:{target_height} -q:v 5 -r {fps}{audio_str}{out_fn}"
        subprocess.run(call_string, shell=True, check=True)
        print("Done with {}".format(out_fn))
        return True

    except subprocess.CalledProcessError as e:
        print(e.output)
        return False


class whisper_x_wrapper():
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available()  else "cpu"
        self.model = whisperx.load_model("large-v2", self.device, compute_type="int8", language="en")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)

    def extract_single_transcript(self, in_fn, out_fn, corrected_transcript=None, burn_in=False, overwrite=False):

        segments = None
        transcript = None

        print(f"In fn: {in_fn}, out fn: {out_fn}")

        if osp.exists(out_fn) and not overwrite:
            return None
        os.makedirs(out_fn, exist_ok=True)

        basename = osp.basename(out_fn)
        segment_json_fn = osp.join(out_fn, f"{basename}_segments.json")
        json_fn = osp.join(out_fn, f"{basename}_transcript.json")
        srt_fn = osp.join(out_fn, f"{basename}_subtitles.srt")
        burn_in_fn = osp.join(out_fn, f"{basename}_burn_in.mp4")
        tmp_audio_fn = osp.join(out_fn, f"{basename}_tmp.wav")


        try:
            call_string = f"ffmpeg -i {in_fn} -vn {tmp_audio_fn}"
            subprocess.run(call_string, shell=True, check=True)

            audio = whisperx.load_audio(tmp_audio_fn)
            
        except subprocess.CalledProcessError as e:
            print(e.output)
            return None
        except Exception as e:
            print(e)
            return None

        try:
            os.remove(tmp_audio_fn)
        except Exception as e:
            print(e)

        


        if corrected_transcript:
            segments = corrected_transcript
        elif osp.exists(segment_json_fn):
            with open(segment_json_fn, "r") as f:
                segments = json.load(f)
        else:
            result = self.model.transcribe(audio, batch_size=1)
            segments = result["segments"]
            with open(segment_json_fn, "w") as f:
                json.dump(segments, f, indent=4)


        if not segments:
            return None

        transcript = whisperx.align(segments, self.model_a, self.metadata, audio, self.device, return_char_alignments=False)

        # change all np.float64 to normal floats in nested transcript
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(i) for i in obj]
            elif isinstance(obj, float):
                return float(obj)
            return obj

        transcript = convert_floats(transcript)

        print(transcript)

        with open(json_fn, "w") as f:
            json.dump(transcript, f, indent=4)


        with open(srt_fn, "w") as f:
            decimal_marker = ","
            always_include_hours = True
            for i, segment in enumerate(transcript["segments"]):
                start = format_timestamp(segment["start"], decimal_marker=decimal_marker, always_include_hours=always_include_hours)
                end = format_timestamp(segment["end"], decimal_marker=decimal_marker, always_include_hours=always_include_hours)
                text = segment["text"].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


        if burn_in:
            call_string = f"ffmpeg -y -i {in_fn} -vf subtitles={srt_fn} -c:a copy {burn_in_fn}"
            try:
                subprocess.run(call_string, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(e.output)

        return transcript

def extract_all_videos(config):
    dbm = DatabaseManager(config)

    all_requests_iterator = dbm.requests.find()

    whisperx_model = whisper_x_wrapper(config)
    
    for request in all_requests_iterator:

        if "instructions" not in request or not request["instructions"]:
            print(f"Skipping request {request['_id']} because it has no instructions")
            continue

        in_fn = osp.join(dbm.root_dir, request["instructions"])

        request_id = request["_id"]

        # re-encode
        for format in config["video_formats"]:
            audio_str = "_audio" if format.get("audio", False) else ""
            out_fn_vid = osp.join(dbm.root_dir, dbm.video_dir, f"{request_id}_{format["fps"]}_{format["resolution"]}{audio_str}.mp4")

            result = extract_single_video(config, in_fn, out_fn_vid, format)

            if result:
                key = f"{format['fps']}_{format['resolution']}{audio_str}"

                dbm.requests.update_one({"_id": request_id}, {"$set": {key: dbm.strip_root_dir(out_fn_vid)}})


        # get video length
        video_length = get_single_video_length(in_fn)
        video_start_time_unix = request["end_time"] - video_length

        # extract transcript
        out_fn_transcript = osp.join(dbm.root_dir, dbm.video_dir, f"{request_id}_transcript")

        corrected_transcript = request.get("corrected_transcript", None)
        if corrected_transcript:
            transcript = whisperx_model.extract_single_transcript(in_fn, out_fn_transcript, corrected_transcript=corrected_transcript)
            if transcript:
                dbm.requests.update_one({"_id": request_id}, {"$set": {"corrected_transcript": transcript}})
        transcript = whisperx_model.extract_single_transcript(in_fn, out_fn_transcript, corrected_transcript=None)
        if transcript:        
            dbm.requests.update_one({"_id": request_id}, {"$set": {"transcript": transcript}})

        # # delete the transcript folder
        # if osp.exists(out_fn_transcript):
        #     os.remove(out_fn_transcript)

        # delete the original video file if it exists in the database video directory - it shouldn't be shared
        if request["instructions"].startswith(dbm.video_dir):
            if osp.exists(in_fn):
                os.remove(in_fn)
            dbm.requests.update_one({"_id": request_id}, {"$set": {"instructions": ""}})

        # create events_video_time to go from unix timestamp to video time
        events = request["events"]
        for event in events:
            event["timestamp_relative_to_vid_start"] = event["timestamp"] - video_start_time_unix

        dbm.requests.update_one({"_id": request_id}, {"$set": {"events": events}})

    dbm.print_db_summary()

def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    extract_all_videos(config)


if __name__ == "__main__":
    main()