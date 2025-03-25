from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langchain_ibm import ChatWatsonx
from pydantic.v1 import SecretStr
from typing import Any

import requests

from langflow.inputs import DropdownInput, IntInput, SecretStrInput, StrInput, FloatInput, SliderInput
from langflow.field_typing.range_spec import RangeSpec
from langflow.schema.dotdict import dotdict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class WatsonxComponent(LCModelComponent):
    display_name = "IBM watsonx.ai"
    description = "Generate text using IBM watsonx.ai foundation models."
    beta = False
    
    _default_models = ["ibm/granite-3-2b-instruct", "ibm/granite-3-8b-instruct", "ibm/granite-13b-instruct-v2"]

    inputs = [
        *LCModelComponent._base_inputs,
        DropdownInput(
            name="url",
            display_name="watsonx API Endpoint",
            info="The base URL of the API.",
            value=None,
            options=[
                "https://us-south.ml.cloud.ibm.com",
                "https://eu-de.ml.cloud.ibm.com",
                "https://eu-gb.ml.cloud.ibm.com",
                "https://au-syd.ml.cloud.ibm.com",
                "https://jp-tok.ml.cloud.ibm.com",
                "https://ca-tor.ml.cloud.ibm.com",
            ],
            real_time_refresh=True,
        ),
        StrInput(
            name="project_id",
            display_name="watsonx project id",
            advanced=False,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="The API Key to use for the model.",
            advanced=False,
            required=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=[],
            value=None,
            dynamic=True,
            required=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate.",
            range_spec=RangeSpec(min=1, max=4096),
        ),
        IntInput(
            name="min_tokens",
            display_name="Min Tokens",
            advanced=True,
            info="The minimum number of tokens to generate.",
            range_spec=RangeSpec(min=0, max=2048),
        ),
        DropdownInput(
            name="decoding_method",
            display_name="Decoding method",
            advanced=True,
            options=["greedy", "sample"],
            value="greedy",
        ),
        FloatInput(
            name="repetition_penalty",
            display_name="Repetition Penalty",
            advanced=True,
            info="Penalty for repetition in generation.",
            range_spec=RangeSpec(min=1.0, max=2.0),
        ),
        IntInput(
            name="random_seed",
            display_name="Random Seed",
            advanced=True,
            info="The random seed for the model.",
        ),
        SliderInput(
            name="top_p",
            display_name="Top P",
            advanced=True,
            info="The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus.",
            range_spec=RangeSpec(min=0, max=1),
            field_type="float",
        ),
        SliderInput(
            name="top_k",
            display_name="Top K",
            advanced=True,
            info="Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.",
            range_spec=RangeSpec(min=1, max=100),
            field_type="int",
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            advanced=True,
            info="Controls randomness, higher values increase diversity.",
            range_spec=RangeSpec(min=0, max=2),
            field_type="float",
        ),
        StrInput(
            name="stop_sequence",
            display_name="Stop Sequence",
            advanced=True,
            info="A sequence where the generation should stop.",
            field_type="str",
        ),
    ]
    
    @staticmethod
    def fetch_models(base_url: str) -> list[str]:
        """Fetch available models from the watsonx.ai API."""
        try:
            endpoint = f"{base_url}/ml/v1/foundation_model_specs"
            params = {"version": "2024-09-16", "filters": "function_text_chat,!lifecycle_withdrawn"}
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model["model_id"] for model in data.get("resources", [])]
            return sorted(models)
        except Exception:
            logger.exception("Error fetching models. Using default models.")
            return WatsonxComponent._default_models

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        """Update model options when URL or API key changes."""
        logger.info("Updating build config. Field name: %s, Field value: %s", field_name, field_value)

        if field_name == "url" and field_value:
            try:
                models = self.fetch_models(base_url=build_config.url.value)
                build_config.model_name.options = models
                if build_config.model_name.value:
                    build_config.model_name.value = models[0]
                info_message = f"Updated model options: {len(models)} models found in {build_config.url.value}"
                logger.info(info_message)
            except Exception:
                logger.exception("Error updating model options.")

def build_model(self) -> LanguageModel:

        return ChatWatsonx(
            apikey=SecretStr(self.api_key).get_secret_value(),
            url=self.url,
            project_id=self.project_id,
            model_id=self.model_name,
            params={},
            streaming=self.stream,
        )
