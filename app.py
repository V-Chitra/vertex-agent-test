import vertexai
from vertexai.language_models import TextGenerationModel
from flask import Flask, request

vertexai.init(project="chitra-v-project", location="us-central1")

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
    response = model.predict(prompt, **parameters)
    return response

if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0', debug=True)
