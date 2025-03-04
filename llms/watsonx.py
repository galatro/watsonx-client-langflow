from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langchain_ibm import WatsonxLLM
from pydantic.v1 import SecretStr


from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langflow.inputs import DropdownInput, IntInput, SecretStrInput, StrInput, FloatInput, SliderInput
from langflow.field_typing.range_spec import RangeSpec



class WatsonxComponent(LCModelComponent):
    display_name = "Watsonx"
    description = "Watsonx foundation models"
    beta = False
    inputs = [
        *LCModelComponent._base_inputs,
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=[
                "codellama/codellama-34b-instruct-hf",
                "google/flan-ul2",
                "ibm/granite-13b-instruct-v2",
                "ibm/granite-20b-code-instruct",
                "ibm/granite-20b-multilingual",
                "ibm/granite-3-2-8b-instruct",
                "ibm/granite-3-2b-instruct",
                "ibm/granite-3-8b-instruct",
                "ibm/granite-34b-code-instruct",
                "ibm/granite-3b-code-instruct",
                "meta-llama/llama-3-2-11b-vision-instruct",
                "meta-llama/llama-3-2-1b-instruct",
                "meta-llama/llama-3-2-3b-instruct",
                "meta-llama/llama-3-2-90b-vision-instruct",
                "meta-llama/llama-3-3-70b-instruct",
                "meta-llama/llama-3-405b-instruct",
            ],
            value="meta-llama/llama-3-3-70b-instruct",
        ),
        StrInput(
            name="url",
            display_name="watsonx API Endpoint",
            advanced=True,
            info="The base URL of the API.",
            value="https://us-south.ml.cloud.ibm.com",
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
        ]

    def build_model(self) -> LanguageModel:
        creds = Credentials(
            api_key=SecretStr(
                self.api_key).get_secret_value(),
            url=self.url,
        )



        model = ModelInference(
            model_id=self.model_name,
            params={},
            credentials=creds,
            project_id=self.project_id,
        )

        return WatsonxLLM(watsonx_model=model)
