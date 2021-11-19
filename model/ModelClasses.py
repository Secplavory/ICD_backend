import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, thresholds=[
        0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
        0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
        0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
    ], name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision_block = []
        self.recall_block = []
        for threshold in thresholds:
            self.precision_block.append(tf.keras.metrics.Precision(thresholds=threshold))
            self.recall_block.append(tf.keras.metrics.Recall(thresholds=threshold))
        self.f1 = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        best_f1 = 0.0
        for index in range(len(self.precision_block)):
            p = self.precision_block[index](y_true, y_pred)
            r = self.recall_block[index](y_true, y_pred)
            f1 = 2 * ((p * r) / (p + r + 1e-6))
            if best_f1 < f1:
                best_f1 = f1

        # since f1 is a variable, we use assign
        self.f1.assign(best_f1)

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        for index in range(len(self.precision_block)):
            self.precision_block[index].reset_state()
            self.recall_block[index].reset_state()
        self.f1.assign(0)

class Segment_BERT_Layer(tf.keras.layers.Layer):
    
    def __init__(self, preprocess_path, bert_path):
        super(Segment_BERT_Layer, self).__init__()
        self.preprocess_path = preprocess_path
        self.bert_path = bert_path
        preprocessor = hub.load(self.preprocess_path)
        self.tokenize = hub.KerasLayer(preprocessor.tokenize)
        self.bert_pack = hub.KerasLayer(preprocessor.bert_pack_inputs, arguments=dict(seq_length=512))
        self.encoder = hub.KerasLayer(self.bert_path, trainable=True)
        self.bert_drop = tf.keras.layers.Dropout(0.2)
        self.classifier = tf.keras.layers.Dense(50)
    
    def get_config(self):
        return {"preprocess_path": self.preprocess_path, "bert_path": self.bert_path}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
    def build(self, input_shape):
        pass
    
    def call(self, inputs):
        tokenizes = self.tokenize(inputs)
        premises = [self.bert_pack([tokenizes[:, index:index+510]]) for index in range(0, 2550, 510)]
        encodes = [self.encoder(data) for data in premises]
        encodes_drop = [self.bert_drop(data['pooled_output']) for data in encodes]
        logits = [self.classifier(data) for data in encodes_drop]
        
        result = None
        for data in logits:
            temp = data
            result = temp if result is None else tf.math.maximum(result, temp)

        return result
