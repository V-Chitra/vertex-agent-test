import vertexai
from vertexai.language_models import TextGenerationModel
from flask import Flask, request
from flask import jsonify
import requests
from vertexai.preview import reasoning_engines

PROJECT_ID = "chitra-agent-project"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
STAGING_BUCKET = "gs://grcv-bucket"  # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

#### defining all tools
class SimpleAdditionApp:
    def query(self, a: int, b: int) -> str:
        """Query the application.

        Args:
            a: The first input number
            b: The second input number

        Returns:
            int: The additional result.
        """

        return f"{int(a)} + {int(b)} is {int(a + b)}"

# Locally test
# app = SimpleAdditionApp()
# app.query(a=1, b=2)

# Create a remote app with reasoning engine.
# This may take 1-2 minutes to finish.
reasoning_engine = reasoning_engines.ReasoningEngine.create(
    SimpleAdditionApp(),
    display_name="Demo Addition App",
    description="A simple demo addition app",
    requirements=[],
    extra_packages=[],
)

##### end of tools

parameters = {
    "temperature": 0.5,
    "max_output_tokens": 256,
    "top_k": 3,
    "top_p": 0.5
}
model = TextGenerationModel.from_pretrained("text-bison@001")

app = Flask(__name__)

@app.route('/predict', methods= ['POST'])
def predict():
    prompt = request.data
    prompt = "this is something I'm trying"
    response = model.predict(prompt, **parameters)
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8080, host='0.0.0.0', debug=True)
