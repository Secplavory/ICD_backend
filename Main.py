from model.ModelClasses import Segment_BERT_Layer, F1_Score
import tensorflow as tf
import flask
import json

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

app = flask.Flask(__name__)

code_structure = None
with open("icd_data/code_structure.json") as json_file:
    code_structure = json.load(json_file)

BERT_model = tf.keras.models.load_model(
    'model/BERT_AutoSplit.h5',
    custom_objects = {
            "Segment_BERT_Layer": Segment_BERT_Layer,
            "F1_Score": F1_Score
        }
)

@app.route("/", methods=["GET"])
def index():
    return "ICD_backend active!!!"

@app.route('/', methods=["POST"])
def predict():
    json_data = flask.request.json
    predict = BERT_model.predict([json_data["inputs"]])
    data_list = []
    for index, pred in enumerate(predict[0]):
        data_list.append(dict(
            code = code_structure[str(index)]['code'],
            title = code_structure[str(index)]['short_title'],
            description = code_structure[str(index)]['long_title'],
            percentage = "{}".format(round(float(pred), 4))
        ))

    return flask.jsonify(data_list)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
