# watsonx Custom Model for Langflow

This repository contains a custom Langflow component for integrating IBM watsonx.ai foundation models into Langflow. It provides a configurable interface for interacting with watsonx models using Langflow's component system.

## Features
- Supports multiple watsonx foundation models.
- Allows fine-grained control over generation parameters like `max_tokens`, `temperature`, `top_k`, and `top_p`.
- Securely manages API keys with `SecretStrInput`.
- Provides dropdowns, sliders, and text inputs for easy configuration.

## Installation

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install langflow langchain_ibm ibm-watsonx-ai pydantic
```

## Usage

### Setting Up the Component in Langflow

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd <repository>
   ```
2. **Create the Custom Component Directory Structure:**
   ```
   /app/custom_components/
   └── llms/
       └── watsonx_component.py
   ```
3. **Configure Langflow to Recognize the Custom Component:**
   - Set the `LANGFLOW_COMPONENTS_PATH` environment variable to point to your custom components directory:
     ```bash
     export LANGFLOW_COMPONENTS_PATH=/app/custom_components/
     ```
   - Alternatively, set this variable within your Python script:
     ```python
     import os
     os.environ['LANGFLOW_COMPONENTS_PATH'] = '/app/custom_components/'
     ```
4. **Launch Langflow:**
   ```bash
   langflow run --components-path=/app/custom_components/ 
   ```
   - Access the Langflow interface at `http://localhost:7860`.

5. **Utilize the watsonx Component in Your Workflow:**
   - Locate the "LLMs" category in the components sidebar.
   - Drag and drop the "watsonx" component into your workflow canvas.
   - Configure the component’s parameters, such as `model_name`, `api_key`, and `url`.
   - Connect the watsonx component to other components in your workflow.

6. **Execute and Test Your Workflow:**
   - Run the workflow to ensure the watsonx component functions correctly.
   - Monitor the outputs and adjust configurations as needed.

## Configuration

| Parameter            | Description                                              |
|---------------------|------------------------------------------------------|
| `model_name`       | Selectable dropdown of supported watsonx models.    |
| `api_key`         | Secret API key for authentication.                   |
| `url`             | Endpoint URL for watsonx API.                        |
| `project_id`      | Project ID associated with the API key.              |
| `max_tokens`      | Maximum number of tokens to generate.                |
| `temperature`     | Controls randomness; higher values increase diversity. |
| `top_k`           | Number of top tokens to sample from.                  |
| `top_p`           | Probability mass cutoff for token selection.          |

## Example Code

Here's an example of how to initialize and use the component:

```python
from custom_component import WatsonxComponent

watsonx_model = WatsonxComponent()
model = watsonx_model.build_model()
response = model.invoke("Hello, watsonx!")
print(response)
```

## Contributing

Feel free to fork this repository and submit pull requests. Make sure to follow best practices and test your changes before submitting.

## License

This project is licensed under the MIT License.

