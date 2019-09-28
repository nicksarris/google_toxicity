import warnings
warnings.filterwarnings('ignore')

import json
from flask import Flask, request
from flask_cors import CORS, cross_origin

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from service.bert_service import calculate_toxicity
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert import BertTokenizer, BertConfig

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
api = CORS(app, resources={r"/*": {"origins": "*"}})

class MyBertClassifier(BertPreTrainedModel):

    def __init__(self, config, num_aux_targets):
        super(MyBertClassifier, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, 1)
        self.linear_aux_out = nn.Linear(config.hidden_size, num_aux_targets)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        h_conc_linear1  = F.relu(self.linear(pooled_output))
        h_conc_linear1 = self.dropout(h_conc_linear1)

        hidden = pooled_output + h_conc_linear1
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out

bert_model_path = "./service/uncased_L-12_H-768_A-12/"
trained_model_path = "./service/bert_pytorch_model.bin"
bert_config = BertConfig(bert_model_path + 'bert_config.json')

device = torch.device('cpu')
model = MyBertClassifier(bert_config, 6)
model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
model.to(device)

for param in model.parameters():
    param.requires_grad = False

model.eval()

@app.route('/toxicity', methods=["POST"])
@cross_origin(origin='*',headers=['Content- Type','Authorization'])
def toxicity():
    toxic_string = [str(request.get_json()["inputText"])]
    toxicity = str(calculate_toxicity(model, toxic_string)[0][0])
    output_toxicity = {
        'toxicityScore': toxicity
    }
    return json.dumps(output_toxicity)

def main():
    app.run(host='0.0.0.0', port=3001, threaded=True)

if __name__ == '__main__':
    main()
