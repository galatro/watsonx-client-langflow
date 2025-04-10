from typing import Any
import requests

from langchain_ibm import WatsonxEmbeddings

from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import IntInput, DictInput, DropdownInput, StrInput, SecretStrInput
from langflow.schema.dotdict import dotdict

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from pydantic.v1 import SecretStr

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WatsonxAIEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "IBM watsonx.ai Embeddings"
    description = "Generate embeddings using watsonx.ai models."
    name = "IBMwatsonxEmbeddings"

    _default_models = ["sentence-transformers/all-minilm-l12-v2",
                       "ibm/slate-125m-english-rtrvr-v2",
                       "ibm/slate-30m-english-rtrvr-v2",
                       "intfloat/multilingual-e5-large"]

    inputs = [
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
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="The API Key to use for the model.",
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
        DictInput(
            name="return_options",
            display_name="Return Options",
            advanced=True,
            value={"input_text": True}
        ),
        IntInput(
            name="truncate_input_tokens",
            display_name="Truncate Input Tokens",
            advanced=True,
            value=200,
        )
    ]

    @staticmethod
    def fetch_models(base_url: str) -> list[str]:
        """Fetch available models from the watsonx.ai API."""
        try:
            endpoint = f"{base_url}/ml/v1/foundation_model_specs"
            params = {"version": "2024-09-16",
                      "filters": "function_embedding,!lifecycle_withdrawn:and"}
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model["model_id"] for model in data.get("resources", [])]
            return sorted(models)
        except Exception:
            logger.exception("Error fetching models")
            return WatsonxAIEmbeddingsComponent._default_models

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        """Update model options when URL or API key changes."""
        logger.info(
            "Updating build config. Field name: %s, Field value: %s", field_name, field_value)

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

    def build_embeddings(self) -> Embeddings:
        creds = Credentials(
            api_key=SecretStr(self.api_key).get_secret_value(),
            url=self.url,
        )
        watsonx_client = APIClient(
            credentials=creds, project_id=self.project_id)
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: self.truncate_input_tokens,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: self.return_options
        }
        return WatsonxEmbeddings(
            model_id=self.model_name,
            params=embed_params,
            watsonx_client=watsonx_client
        )
