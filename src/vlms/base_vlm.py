from abc import ABC, abstractmethod
import os
import os.path as osp
import random
import subprocess
import json
import shutil
import cv2

class GenerateResponseResult:
    def __init__(self, response_json=None, response_text=None, token_counts=None, thinking_text=None):
        self.response_json = response_json
        self.response_text = response_text
        self.token_counts = token_counts
        self.thinking_text = thinking_text

class BaseVLM(ABC):
    """
    A base class for Vision-Language Models (VLMs) that can load videos.
    """

    def __init__(self, config: dict, cache: bool = True):
        """
        Initializes the VLM with a specified backend.

        Args:
            backend (str): The backend to use for video processing. Default is "decord".
        """
        self.backend = config["backend"]
        assert self.backend in ["decord", "ffmpeg", "cv2"], f"Unsupported backend: {self.backend}"

        if self.backend == "decord":
            import decord
            self.decord = decord
        elif self.backend == "ffmpeg":
            pass
        elif self.backend == "cv2":
            import cv2
            self.cv2 = cv2

        self.cache = cache

    def load_video_ffmpeg(self, video_path: str) -> str:
        return video_path
    
    def load_video(self, video_path: str) -> list:
        """
        Loads a video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: A list of frames (as numpy arrays) from the video.
        """
        if self.backend == "decord":

            vr = self.decord.VideoReader(video_path)
            n_in_frames = len(vr)
            in_fps = float(vr.get_avg_fps())

            subsample_every = self.config.get("subsample_every", 1)


            out_fps = float(self.config["fps"])
            n_out_frames = int(n_in_frames * out_fps / in_fps / subsample_every)

            frame_indexes = [int(i * in_fps / out_fps) for i in range(n_out_frames)]

            if self.config.get("max_frames", -1) > 0:
                frame_indexes = frame_indexes[:self.config["max_frames"]]

            # frame_indexes = [int(i * in_fps / out_fps) for i in range(n_in_frames)]
            frames = [vr[i].asnumpy() for i in frame_indexes]

            if self.config.get("downsample_vertical_resolution", None):
                # resize frames to have the specified vertical resolution while maintaining aspect ratio
                vertical_resolution = self.config["downsample_vertical_resolution"]
                frames_resized = []
                for frame in frames:
                    h, w, c = frame.shape
                    aspect_ratio = w / h
                    new_h = vertical_resolution
                    new_w = int(aspect_ratio * new_h)
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    frames_resized.append(resized_frame)
                frames = frames_resized

            return frames
        elif self.backend == "ffmpeg":
            return video_path

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @abstractmethod
    def create_messages(self, inputs: list, sys=None) -> list:
        """
        Creates messages from interleaved inputs such as video paths, images, and text.

        Args:
            inputs (list): A list of interleaved inputs (e.g., video paths, image paths, text).
        Returns:
            list: A list of formatted messages for the VLM.
        """
        pass


    @abstractmethod
    def generate_response(self, messages: list, output_path=None, return_token_counts=False) -> GenerateResponseResult:
        """
        Abstract method to generate a response based on the provided messages.

        Args:
            messages (list): A list of formatted messages for the VLM.

        Returns:
            str: The generated response.
        """
        pass

    def load_brep(self, brep_path: str) -> dict:
        """
        Loads a BREP file.

        Args:
            brep_path (str): Path to the BREP file.

        Returns:
            dict: A dictionary containing the loaded BREP data.
        """

        extension = os.path.splitext(brep_path)[1][1:]
        with open(brep_path, 'r') as f:
            brep_data = f.read()
        return {"type": extension, "content": brep_data}
  

    def run_single_edit_file_only(self, db, request_id, output_path: str) -> None:
        """
        Runs a single edit on the video and saves the output.

        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to save the edited video.
        """

        request = db.requests.find_one({"_id": request_id})
        if request is None:
            raise ValueError(f"Request ID {request_id} not found in the database.")
        
        brep_fn = db.breps.find_one({"_id": request["brep_start"]})[self.config["extension"]]
        if brep_fn is None:
            raise ValueError(f"BREP file not found for request ID {request_id}.")
        if not os.path.exists(brep_fn):
            raise ValueError(f"BREP file does not exist: {brep_fn}")


        brep = self.load_brep(brep_fn)
        fps = f"Video is at {self.config['fps']} fps."
        video_path = request[f"{self.config['fps']}_{self.config['resolution']}"]
        if video_path is None:
            raise ValueError(f"Video path not found for request ID {request_id}.")
        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        messages = self.create_messages([video_path, f"brep file: {brep}", self.config["prompt"], fps], sys=self.config["system_prompt"])
        response = self.generate_response(messages, output_path=output_path).response_json
        self.clean_up(messages)
        return response
    

    def run_rating_video_images(self, db, edit_id, output_path) -> None:
        """
        Runs a rating on the video and saves the output.

        Args:
            edit_instruction (dict): Edit instruction containing the video path and other parameters.
            brep_path_start (str): Path to the starting BREP file.
            brep_path_end (str): Path to the ending BREP file.
            output_path (str): Path to save the rating in a json file.
        """
        edit_entry = db.edits.find_one({"_id": edit_id})

        request_entry = db.requests.find_one({"_id": edit_entry["request"]})

        if request_entry is None:
            return None

        start_brep_id = db.breps.find_one({"_id": request_entry["brep_start"]})
        start_breps = db.get_brep_images(start_brep_id["_id"], views=self.config["views_request"])
        start_breps = [osp.join(db.root_dir, start_brep) for start_brep in start_breps]

        end_brep_id = db.breps.find_one({"_id": edit_entry["brep_end"]})
        end_breps = db.get_brep_images(end_brep_id["_id"], views=self.config["views_edit"])
        end_breps = [osp.join(db.root_dir, end_brep) for end_brep in end_breps]

        audio_str = "_audio" if self.config.get("audio", False) else ""
        video_path = request_entry[f"{self.config['fps']}_{self.config['resolution']}{audio_str}"]
        video_path = osp.join(db.root_dir, video_path) if video_path else None

        if 'text' in request_entry and request_entry['text']:
            request_text = f"Instruction text: \n{request_entry['text']}"
        else:
            request_text = ""

        # print(video_path, *start_breps, *end_breps)

        fps = f"Video is at {self.config['fps']} fps. "
        start = "Initial cad model: "
        end = "Edited cad model: "
        messages = self.create_messages(
            [
                video_path, 
                request_text,
                start, 
                *start_breps,
                end,
                *end_breps,
                self.config["prompt"],
                fps
            ],
            sys=self.config["system_prompt"])
        response = self.generate_response(messages, output_path=output_path).response_json
        self.clean_up(messages)
        return response


    # def run_ranking_video_images(self, edit_instruction: dict, image_path_start: str, image_paths_end: list[str], output_path: str) -> None:
    def run_ranking_video_images(self, db, request_id, output_path: str):
        """
        Runs a rating on the video and saves the output.

        Args:
            edit_instruction (dict): Edit instruction containing the video path and other parameters.
            brep_path_start (str): Path to the starting BREP file.
            brep_path_end (str): Path to the ending BREP file.
            output_path (str): Path to save the rating in a json file.
        """
        fps = f"Video is at {self.config['fps']} fps. "
        start = "Initial cad model: "

        # print(f"Request ID: {request_id}")

        request_entry = db.requests.find_one({"_id": request_id})

        audio_str = "_audio" if self.config.get("audio", False) else ""
        video_path = request_entry[f"{self.config['fps']}_{self.config['resolution']}{audio_str}"]
        video_path = osp.join(db.root_dir, video_path) if video_path else None

        transcript = request_entry.get("transcript", None)
        if transcript is None:
            transcript_text = ""
        else:
            transcript_text = f'\nInstruction transcript: {transcript["segments"]}\n'
        
        start_breps = db.breps.find_one({"_id": request_entry["brep_start"]})
        start_breps = db.get_brep_images(start_breps["_id"], views=self.config["views_request"])
        start_breps = [osp.join(db.root_dir, start_brep) for start_brep in start_breps if start_brep]

        edit_iterator = db.edits.find({"request": request_id})

        edits_list = list(edit_iterator)
        
        # shuffle the edits list
        random.shuffle(edits_list)

        messages_list = [
                video_path,
                transcript_text,
                fps,
                start, 
                *start_breps,
                self.config["prompt"],
            ]
        
        for i in range(len(edits_list)):
            messages_list.append(f"Edit {i}:")
            images = db.breps.find_one({"_id": edits_list[i]["brep_end"]})
            images = db.get_brep_images(images["_id"], views=self.config["views_edit"])
            for image in images:
                image = osp.join(db.root_dir, image)
                if not image:
                    raise ValueError(f"BREP file not found for edit ID {edits_list[i]["_id"]}.")
                messages_list.append(image)

        messages = self.create_messages(
            messages_list,
            sys=self.config["system_prompt"])
        response = self.generate_response(messages, output_path=output_path).response_json
        self.clean_up(messages)

        ranking = []
        try:
            if response is not None:
                for idx in response["ranking"]:
                    ranking.append(edits_list[idx]["_id"])
        except TypeError as e:
            print(f"Error processing response: {e}")

        return ranking
    

    def run_edit_cot_gen(self, db, edit_id, output_path) -> None:
        """
        Runs a rating on the video and saves the output.

        Args:
            edit_instruction (dict): Edit instruction containing the video path and other parameters.
            brep_path_start (str): Path to the starting BREP file.
            brep_path_end (str): Path to the ending BREP file.
            output_path (str): Path to save the rating in a json file.
        """
        edit_entry = db.edits.find_one({"_id": edit_id})

        request_entry = db.requests.find_one({"_id": edit_entry["request"]})

        if request_entry is None:
            return None

        start_brep_id = db.breps.find_one({"_id": request_entry["brep_start"]})
        start_breps = db.get_brep_images(start_brep_id["_id"], views=self.config["views_request"])
        start_breps = [osp.join(db.root_dir, start_brep) for start_brep in start_breps]

        end_brep_id = db.breps.find_one({"_id": edit_entry["brep_end"]})
        end_breps = db.get_brep_images(end_brep_id["_id"], views=self.config["views_edit"])
        end_breps = [osp.join(db.root_dir, end_brep) for end_brep in end_breps]

        audio_str = "_audio" if self.config.get("audio", False) else ""
        video_path = request_entry[f"{self.config['fps']}_{self.config['resolution']}{audio_str}"]
        video_path = osp.join(db.root_dir, video_path) if video_path else None

        transcript = request_entry.get("transcript", None)

        if transcript is None:
            transcript = []
        else:
            transcript = transcript["segments"]
        events = edit_entry.get("events", [])

        frames_dir = osp.join(db.root_dir, edit_entry["frames_dir"])

        if not osp.exists(frames_dir):
            frames = []
        else:
            frames = [f for f in os.listdir(frames_dir)]
            frames = [f for f in frames if any(f.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])]
            frames = [osp.join(db.root_dir, frames_dir, frame) for frame in frames]


        combined_events_and_frames = []

        image_timestamps = [os.path.splitext(f)[0] for f in frames]
        image_timestamps = [os.path.basename(ts) for ts in image_timestamps]
        image_timestamps = [f.split('_')[-1] for f in image_timestamps]
        image_timestamps = [float(ts) for ts in image_timestamps]

        event_timestamps = [event['timestamp'] for event in events]

        # Combine and sort by timestamp. Output is a list of the ordered frame or event
        ts_frames = [(ts, f) for ts, f in zip(image_timestamps, frames)]
        ts_events = [(ts, e) for ts, e in zip(event_timestamps, events)]
        combined = ts_frames + ts_events
        combined.sort(key=lambda x: x[0])
        combined_events_and_frames = [str(item[1]) for item in combined]
        combined_ids = [f"ID: {i}" for i in range(len(combined_events_and_frames))]
        combined_events_and_frames = [val for pair in zip(combined_ids, combined_events_and_frames) for val in pair]

        message_list = []
        if "request" in self.config["fields"]:
            message_list.extend([
                video_path, 
                f"Edit instruction video is at {self.config['fps']} fps.",
                f"Transcript: {transcript}",
                "Initial cad model: ", 
                *start_breps,
            ])

        if "edit" in self.config["fields"]:
            message_list.extend([
                "Edited cad model: ", 
                *end_breps,
            ])

        if "frames" in self.config["fields"]:
            message_list.extend([
                "Images during the edit process: ",
                *frames,
            ])

        if "events" in self.config["fields"]:
            message_list.extend([
                f"Autodesk Fusion events: {events}",
            ])

        if "frame_event_interleaved" in self.config["fields"]:
            message_list.extend([
                "Images and events during the edit process: ",
                *combined_events_and_frames,
            ])

        message_list.append(
                self.config["prompt"],
        )

        messages = self.create_messages(
            message_list,
            sys=self.config["system_prompt"])
        
        response = self.generate_response(messages, output_path=output_path).response_json
        self.clean_up(messages)

        return response
    
    def run_rating_gen(self, db, edit_id, output_path: str) -> None:
        edit_entry = db.edits.find_one({"_id": edit_id})
        request_entry = db.requests.find_one({"_id": edit_entry["request"]})

        if request_entry is None:
            return None

        start_brep_id = db.breps.find_one({"_id": request_entry["brep_start"]})
        if start_brep_id:
            start_breps = db.get_brep_images(start_brep_id["_id"], views=self.config["views_request"])
            start_breps = [osp.join(db.root_dir, start_brep) for start_brep in start_breps]
        else:
            start_breps = []

        end_brep_id = db.breps.find_one({"_id": edit_entry["brep_end"]})
        end_breps = db.get_brep_images(end_brep_id["_id"], views=self.config["views_edit"])
        end_breps = [osp.join(db.root_dir, end_brep) for end_brep in end_breps]

        text_prompt = request_entry.get("text", "")
        if not text_prompt:
            text_prompt = request_entry.get("prompt", "")
        start = f"Requested generation: {text_prompt}."
        end = "Edited cad model: "
        messages = self.create_messages(
            [
                start,
                *start_breps,
                end,
                *end_breps,
                self.config["prompt"],
            ],
            sys=self.config["system_prompt"])
        response = self.generate_response(messages, output_path=output_path).response_json
        self.clean_up(messages)
        return response

    def clean_up(self, messages) -> None:
        pass


##########
########## Harnesses-specific functions
##########  
    def read_text_file(self, instruction_text_file):
        with open(instruction_text_file, 'r') as f:
            instruction_text = f.read()
        return instruction_text
    
    def load_task_info_dict(self, row):
        messages = []
        for k, v in row.items():
            messages.append(f"{k}: ")
            if v is None or v == "":
                messages.append("None")
            else:
                messages.append(v)

        return messages
    
    def visual_update_loop(self, instruction_text, task_info_dict, harness_script_file, output_dir, max_iters=10, run_function=None, conversation_instruction="", output_script_key=None):
        if run_function is None:
            raise ValueError("run_function must be provided to visual_update_loop")
        
        messages = []

        messages.append(conversation_instruction)

        harness_script = self.read_text_file(harness_script_file)

        messages.append(harness_script)

        messages.append("Instruction:")
        messages.append(instruction_text)
        messages.append("Task information:")
        messages.extend(self.load_task_info_dict(task_info_dict))
        messages.append("Last output rendering: None (this is the first iteration)")

        iters_remaining = max_iters

        token_count_accumulator = {
        }

        iter_count = 0
        while iters_remaining > 0:
            messages.append(f"Iterations remaining: {iters_remaining}")
            whole_response = self.generate_response(self.create_messages(messages, sys=self.config["system_prompt"]), return_token_counts=True)

            response = whole_response.response_json
            token_counts = whole_response.token_counts            


            for k in token_counts.keys():
                token_count_accumulator[k] = token_count_accumulator.get(k, 0) + token_counts[k]

            if isinstance(response, str):
                # find the part wrapped in ```json and ```. It might be in the middle of the response
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                try:
                    response = json.loads(response)
                except json.JSONDecodeError as e:
                    print(f"Error parsing response as json: {e}")
                    response = {}

            script = response.get("my_cad_function", "")
            complete = response.get("complete", False)

            if complete:
                print("Task completed.")
                break

            # run the script in FreeCAD
            program_output = run_function(script, harness_script_file, output_dir)

            # Logging iteration outputs
            iteration_dir = os.path.join(output_dir, "iteration_output")
            os.makedirs(iteration_dir, exist_ok=True)
            iteration_text_file = os.path.join(iteration_dir, f"{iter_count}_response.txt")

            with open(iteration_text_file, 'w') as f:
                f.write("Thinking text:\n")
                f.write(whole_response.thinking_text)
                f.write("\n\n")
                # write the script with correct indentation
                f.write("Generated script:\n")
                f.write(script)
                # write the program output
                f.write("\n\nProgram output:\n")
                f.write(program_output)
                # copy the output image with the correct index in the filename
                image_fn = osp.join(output_dir, "tmp.png")
                if os.path.exists(image_fn):
                    shutil.copy(image_fn, osp.join(iteration_dir, f"{iter_count}_output.png"))


            messages.append("Program output from last iteration:")
            messages.append(program_output)
            messages.append("Visual output from last iteration:")
            
            image_fn = osp.join(output_dir, "tmp.png")
            if os.path.exists(image_fn):
                messages.append(image_fn)
            else:
                messages.append("None")
            messages.append("The function that you produced from the last iteration:")
            messages.append(script)

            # print(messages)
            iters_remaining -= 1
            iter_count += 1

        input_tokens = float(token_count_accumulator["input_tokens"])
        output_tokens = float(token_count_accumulator["output_tokens"]) + float(token_count_accumulator.get("thinking_tokens", 0))
        token_count_accumulator["cost_estimate"] = self.config["1m_token_cost_input"] * input_tokens / 1e6 + self.config["1m_token_cost_output"] * output_tokens / 1e6

        return_dict = {
            "token_counts": token_count_accumulator,
        }
        if output_script_key is not None:
            return_dict[output_script_key] = script
        return return_dict

    
##########
########## CadQuery specific methods for VLMs that generate CadQuery scripts
########## 
    def run_cadquery_script(self, script: str, harness_script_file: str, output_dir: str, input_file: str = None) -> str:
        # save script to temp file
        temp_script_file = os.path.join(output_dir, "temp_script.py")
        with open(temp_script_file, 'w') as f:
            f.write(script)

        cmd = [
            "python",
            harness_script_file,
            "--output_dir",
            output_dir,
            "--function_file",
            temp_script_file
        ]

        if input_file is not None:
            cmd.extend(["--input_file", input_file])

        print(f"Running CadQuery script with command: {' '.join(cmd)}")
        # run command and capture output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)

        # remove all stdout lines before the one that starts with "Loading function from"
        for line in result.stdout.splitlines():
            if line.startswith("Loading function from"):
                break
            else:
                result.stdout = result.stdout.replace(line + "\n", "")

        return f"stdout: {result.stdout}"
    
    def cadquery_script(self, instruction_text, task_info_dict, harness_script_file, output_dir, max_iters=10):
        conversation_instruction = """
        Given the following instruction and task information, generate a CadQuery Python function that accomplishes the task.
        The script that runs the your function is provided for reference, as well as an example function, but you should only create your version of the function my_cad_function. Do not include any of the rest of the script.
        Once you have generated the function, it will be executed in CadQuery. You will then be provided with a rendering of the CAD model created by your function, and any prints, debug or error messages.
        You will then have the opportunity to refine your function based on this feedback, and it will be re-executed.
        Continue this process until the task is complete, or you reach the maximum number of iterations.

        If an input file is provided as a brep_start_path in the task information, you should use .step. For these cases, you might need to print out debug information once the model is loaded in your first iteration.

        Return a json object with two fields. Do not include any other text outside of the json object in your response:
        'complete': true if the task has been completed. IMPORTANT: This should only be judged by looking at the output from the last iteration, not whether a function has been returned this iteration. If there is no valid output that corresponds to the instruction, the task is not complete. The first iteration can never be complete. If this is true, then the current function will NOT be executed, and the function from the last iteration will be used.
        'my_cad_function': CadQuery python function as a string. Use the same definition and arguments as the example.
        Script:

        """

        return self.visual_update_loop(
            instruction_text=instruction_text,
            task_info_dict=task_info_dict,
            harness_script_file=harness_script_file,
            output_dir=output_dir,
            max_iters=max_iters,
            run_function=self.run_cadquery_script,
            conversation_instruction=conversation_instruction
        )
    