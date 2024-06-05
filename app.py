import vertexai
from vertexai.language_models import TextGenerationModel
from flask import Flask, request
from flask import jsonify
import requests
from vertexai.preview import reasoning_engines


vertexai.init(project="chitra-v-project", location="us-central1")
#### defining all tools
def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
):
    """Retrieves the exchange rate between two currencies on a specified date."""
    import requests

    response = requests.get(
        f"https://api.frankfurter.app/{currency_date}",
        params={"from": currency_from, "to": currency_to},
    )
    return response.json()

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


##### defining agentic relationships

agent = reasoning_engines.LangchainAgent(
    model=model,
    tools=[get_exchange_rate],
)

remote_agent = reasoning_engines.ReasoningEngine.create(
    agent,
    requirements=[
        "google-cloud-aiplatform==1.51.0",
        "langchain==0.1.20",
        "langchain-google-vertexai==1.0.3",
        "cloudpickle==3.0.0",
        "pydantic==2.7.1",
        "requests",
    ],
)

def query():
    remote_agent.query(
    input="What's the exchange rate from US dollars to Swedish currency today?"
)
###### end of agentic relationships

if __name__ == "__main__":
    app.run(port=8080, host='0.0.0.0', debug=True)
