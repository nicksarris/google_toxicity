import warnings
warnings.filterwarnings('ignore')

import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
from service.bert_service import calculate_toxicity

app = Flask(__name__)
api = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/toxicity', methods=["POST"])
def toxicity():
    toxic_string = [str(request.get_json()["inputText"])]
    toxicity = str(calculate_toxicity(toxic_string)[0][0])
    output_toxicity = {
        'toxicityScore': toxicity
    }
    return json.dumps(output_toxicity)

def main():
    app.run(host='0.0.0.0', port=3001, threaded=True)

if __name__ == '__main__':
    main()

