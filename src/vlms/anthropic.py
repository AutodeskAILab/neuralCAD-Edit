import anthropic
from .base_vlm import BaseVLM, GenerateResponseResult
import os
import base64
import json
import io
from PIL import Image

class VLM(BaseVLM):
    """
    A Vision-Language Model (VLM) implementation that interacts with the Anthropic Claude API.
    """

    def __init__(self, config: dict, cache: bool = True):
        """
        Initializes the AnthropicVLM with an Anthropic API key.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            cache (bool): Whether to cache responses. Defaults to True.
        """

        super().__init__(config=config, cache=cache)
        self.config = config

        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    def load_video(self, video_path: str) -> list:
        """Load video and convert frames to base64 format."""
        np_frames = super().load_video(video_path)

        # Convert to PNG in memory and then to base64
        base64_frames = []
        for frame in np_frames:
            pil_img = Image.fromarray(frame)
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            base64_frame = base64.b64encode(buff.getvalue()).decode('utf-8')
            base64_frames.append(base64_frame)
        return base64_frames

    def load_image(self, image_path: str) -> str:
        """Load image file and convert to base64."""
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        return base64_image
    
    def create_messages(self, inputs, sys=None):
        """
        Creates a list of messages for the Anthropic API.

        Args:
            inputs (list): List of input frames/text.
            sys (str, optional): System message. Defaults to None.

        Returns:
            list: List of messages formatted for the Anthropic API.
        """
        content = []
        
        for i, part in enumerate(inputs):
            if part is None or part == "":
                continue


            if part.endswith(".mp4"):
                frames = self.load_video(part)
                for frame in frames:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": frame
                        }
                    })
            elif part.endswith(".png"):
                image = self.load_image(part)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image
                    }
                })
            elif part.endswith(".jpg") or part.endswith(".jpeg"):
                image = self.load_image(part)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image
                    }
                })
            else:
                content.append({
                    "type": "text",
                    "text": part
                })
        
        messages = [{"role": "user", "content": content}]
        return messages

    def generate_response(self, messages: list, output_path=None, return_token_counts=False) -> str:
        """
        Sends a prompt to the Anthropic API and receives a response.

        Args:
            messages (list): The messages to send to the Anthropic API.
            output_path: Optional path to cache/save the response
            return_token_counts: Whether to return token count information

        Returns:
            str or tuple: The response from the Anthropic API, optionally with token counts.
        """
        if self.cache and output_path is not None and os.path.exists(output_path):
            with open(output_path, "r") as f:
                response = f.read()
            full_response = None  # No token counts available from cache
        else:
            # Prepare system message
            system_message = self.config.get("system_prompt", "")

            create_kwargs = {
                "model": self.config["model"],
                "max_tokens": self.config.get("max_tokens", 20000),
                "system": system_message,
                "messages": messages
            }
            if "thinking" in self.config:
                create_kwargs["thinking"] = self.config["thinking"]
            else:
                create_kwargs["temperature"] = self.config.get("temperature", 0.0)
    
            
            full_response = self.client.messages.create(**create_kwargs)
            
            text_blocks = [block for block in full_response.content if block.type == 'text']
            response = "\n".join([block.text for block in text_blocks])

            thinking_blocks = [block for block in full_response.content if block.type == 'thinking']
            thinking_text = "\n".join([block.thinking for block in thinking_blocks])

            # Handle JSON code blocks (same as other implementations)
            if response.startswith("```json"):
                # Remove first and last lines
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
        
        response_object = GenerateResponseResult(
            response_json=response,
            response_text=full_response if full_response else response,
            thinking_text=thinking_text if 'thinking_text' in dir() else ""
        )

        if return_token_counts:
            if full_response is not None:
                token_count_dict = {
                    "input_tokens": full_response.usage.input_tokens,
                    "output_tokens": full_response.usage.output_tokens,
                    "total_tokens": full_response.usage.input_tokens + full_response.usage.output_tokens,
                }
            else:
                token_count_dict = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            response_object.token_counts = token_count_dict

        return response_object