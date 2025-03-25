from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import IntInput, DictInput, DropdownInput, StrInput, SecretStrInput

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from pydantic.v1 import SecretStr


class WatsonxAIEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "IBM watsonx.ai Embeddings"
    description = "Generate embeddings using watsonx.ai models."
    name = "IBMwatsonxEmbeddings"
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