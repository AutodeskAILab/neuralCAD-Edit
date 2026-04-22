import openai
from .base_vlm import BaseVLM, GenerateResponseResult
import os
import base64
import json
import io
from PIL import Image

class VLM(BaseVLM):
    """
    A Vision-Language Model (VLM) implementation that interacts with the OpenAI API.
    """

    def __init__(self, config: dict, cache: bool = True):
        """
        Initializes the OpenAIVLM with an OpenAI API key.

        Args:
            openai_api_key (str): The API key for OpenAI.
        """

        super().__init__(config=config)
        self.config = config

        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.openai_api_key
        self.client = openai.OpenAI(api_key=self.openai_api_key)

    def load_video(self, video_path: str) -> dict:
        np_frames = super().load_video(video_path)

        # convert to jpg in memory and then to base64
        base64_frames = []
        for frame in np_frames:
            pil_img = Image.fromarray(frame)
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            base64_frame = base64.b64encode(buff.getvalue()).decode('utf-8')
            base64_frames.append(base64_frame)
        return base64_frames

    def load_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        return base64_image
    
    def create_messages(self, inputs, sys=None):
        messages = []
        if sys is not None:
            messages.append({"role": "system", "content": sys})

        # make user message
        user_message = {"role": "user", "content": []}
        for i, part in enumerate(inputs):
            if part.endswith(".mp4"):
                frames = self.load_video(part)
                for frame in frames:
                    user_message["content"].append({
                        "type": "input_image",
                        "image_url":  f"data:image/png;base64,{frame}"
                    })
            elif part.endswith(".png"):
                image = self.load_image(part)
                user_message["content"].append({
                    "type": "input_image",
                    "image_url":  f"data:image/png;base64,{image}"
                })
            elif part.endswith(".jpg") or part.endswith(".jpeg"):
                image = self.load_image(part)
                user_message["content"].append({
                    "type": "input_image",
                    "image_url":  f"data:image/jpeg;base64,{image}"
                })
            else:
                user_message["content"].append({
                    "type": "input_text",
                    "text": part
                })
        messages.append(user_message)
        return messages



    def extract_reasoning_summaries(self, resp):
        summaries = []
        if resp is None:
            return ""
        for item in resp.output:
            if getattr(item, "type", None) == "reasoning":
                # item.summary is a list[Summary] (or None)
                for s in (item.summary or []):
                    # s is a Summary object with .text and .type
                    summaries.append(s.text)

        return "\n".join(summaries)


    def generate_response(self, messages: str, output_path=None, return_token_counts=False) -> str:
        """
        Sends a prompt to the OpenAI API and receives a response.

        Args:
            messages: The messages to send to the OpenAI API.
            output_path: Optional path to cache/save the response
            return_token_counts: Whether to return token count information

        Returns:
            str or tuple: The response from the OpenAI API, optionally with token counts.
        """
        if self.cache and output_path is not None and os.path.exists(output_path):
            with open(output_path, "r") as f:
                response = f.read()
            full_response = None  # No token counts available from cache
        else:

            create_response_args = {
                "model": self.config["model"],
                # "messages": messages,
                "input": messages,
            }
            if "reasoning_level" in self.config:
                create_response_args["reasoning"] = {
                    "effort": self.config["reasoning_level"],
                    "summary": "auto"
                }

            full_response = self.client.responses.create(**create_response_args)
            response = full_response.output_text.strip()


            # get rid of ```json and ``` (same as Gemini)
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
                    if isinstance(response, dict):
                        json.dump(response, f, indent=4)
                    else:
                        f.write(str(response))
        
        reasoning = self.extract_reasoning_summaries(full_response)

        response_object = GenerateResponseResult(
            response_json=response,
            response_text=full_response.output_text.strip() if full_response else response,
            thinking_text=reasoning  # OpenAI API does not provide thinking text
            
        )

        if return_token_counts:
        
            if full_response is not None:
                token_count_dict = {
                    "input_tokens": full_response.usage.input_tokens,
                    "output_tokens": full_response.usage.output_tokens,
                    "total_tokens": full_response.usage.total_tokens,
                }
            else:
                # Return empty token counts if response was cached
                token_count_dict = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            response_object.token_counts = token_count_dict

        return response_object