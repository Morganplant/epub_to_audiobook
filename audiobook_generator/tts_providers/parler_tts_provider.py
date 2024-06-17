from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

class ParlerTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        super().__init__(config)
        self.device = self._setup_device()
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(self.device, dtype=torch.float16 if self.device != "cpu" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    def _setup_device(self):
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        elif torch.xpu.is_available():
            return "xpu"
        return "cpu"

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_file, audio_arr, self.model.config.sampling_rate)

    def validate_config(self):
        pass  # Add any specific validation for Parler TTS configuration

    def estimate_cost(self, total_chars):
        return 0  # Implement cost estimation logic if applicable

    def get_break_string(self):
        return "\n"  # Define how text breaks should be handled

    def get_output_file_extension(self):
        return "wav"  # Define the output file format
