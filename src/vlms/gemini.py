from .base_vlm import BaseVLM, GenerateResponseResult
import os
from google import genai
import json

from google.genai import types

class VLM(BaseVLM):
    """
    A Vision-Language Model (VLM) implementation that interacts with the Gemini API.
    """

    def __init__(self, config: dict, cache: bool = True):
        """
        Initializes the GeminiVLM with a Gemini API key.

        Args:
            gemini_api_key (str): The API key for Gemini.
        """

        super().__init__(config=config, cache=cache)
        self.config = config

        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
        self.client = genai.Client(api_key=self.gemini_api_key)


    def create_messages(self, inputs, sys=None):
        """
        Creates a list of messages for the Gemini API.

        Args:
            inputs (list): List of input frames.
            sys (str, optional): System message. Defaults to None.

        Returns:
            list: List of messages formatted for the Gemini API.
        """

        parts = []
        for i, part in enumerate(inputs):
            if part.endswith(".mp4"):
                # resized video
                video_bytes = open(part, "rb").read()
                parts.append(types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')))
            elif part.endswith(".png"):
                # image
                image_bytes = open(part, "rb").read()
                parts.append(types.Part(inline_data=types.Blob(data=image_bytes, mime_type='image/png')))
            elif part.endswith(".jpg") or part.endswith(".jpeg"):
                image_bytes = open(part, "rb").read()
                parts.append(types.Part(inline_data=types.Blob(data=image_bytes, mime_type='image/jpeg')))
            else:
                parts.append(types.Part(text=part))
                
        contents = types.Content(parts=parts)
        return contents


    def generate_response(self, messages: str, output_path=None, return_token_counts=False) -> str:
        if self.cache and output_path is not None and os.path.exists(output_path):
            with open(output_path, "r") as f:
                full_response = None
                thinking_text = None
                response = f.read()
        else:

            generation_config_args = {
                "system_instruction": self.config["system_prompt"],
                "temperature": 0.0,
                "max_output_tokens": 20000
            }
            if "thinking_level" in self.config:
                generation_config_args["thinking_config"] = types.ThinkingConfig(thinking_level=self.config["thinking_level"], include_thoughts=True)

            generation_config = types.GenerateContentConfig(**generation_config_args)

            full_response = self.client.models.generate_content(
                model=self.config["model"],
                config=generation_config,
                contents=messages,
            )

            try:
                thinking_text = [part for part in full_response.parts if part.thought]
            except Exception:
                pass

            response = full_response.text

            # get rid of ```json and '''
            if response.startswith("```json"):
                # remove first and last lines
                response = response.split("\n")[1:-1]
                response = "\n".join(response)

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            pass

        if output_path is not None:
            if output_path.endswith(".json"):
                with open(output_path, "w") as f:
                    json.dump(response, f, indent=4)
            else:
                with open(output_path, "w") as f:
                    f.write(response)
        

        response_object = GenerateResponseResult(
            response_json=response,
            response_text=full_response.text if full_response else str(response),
            thinking_text=str(thinking_text) if thinking_text else ""
        )

        if return_token_counts:
            token_count_dict = {
                "input_tokens": full_response.usage_metadata.prompt_token_count,
                "output_tokens": full_response.usage_metadata.candidates_token_count,
                "thinking_tokens": full_response.usage_metadata.thoughts_token_count,
                "total_tokens": full_response.usage_metadata.total_token_count,
            }
            # set any token count to 0 if it is None
            for k in token_count_dict.keys():
                if token_count_dict[k] is None:
                    token_count_dict[k] = 0
            response_object.token_counts = token_count_dict




        return response_object