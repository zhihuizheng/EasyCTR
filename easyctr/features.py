import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import io
import pickle
import os
import logging
import json
from .preprocess import Tokenizer, Normalizer, LabelEncoderExt


class FeatureMap(object):
    def __init__(self, dataset_id, data_dir):
        self.data_dir = data_dir # used in embedding layer for pretrained emb
        self.dataset_id = dataset_id
        self.num_fields = 0
        self.num_features = 0
        self.input_length = 0
        self.feature_specs = OrderedDict()

    def set_feature_index(self):
        logging.info("Set feature index...")
        idx = 0
        for feature, feature_spec in self.feature_specs.items():
            self.feature_specs[feature]["index"] = idx
            idx += 1
        self.input_length = idx

    def get_feature_index(self, feature_type=None):
        feature_indexes = []
        if feature_type is not None:
            if not isinstance(feature_type, list):
                feature_type = [feature_type]
            feature_indexes = [feature_spec["index"] for feature, feature_spec in self.feature_specs.items()
                               if feature_spec["type"] in feature_type]
        return feature_indexes

    def load(self, json_file):
        logging.info("Load feature_map from json: " + json_file)
        with io.open(json_file, "r", encoding="utf-8") as fd:
            feature_map = json.load(fd, object_pairs_hook=OrderedDict)
        if feature_map["dataset_id"] != self.dataset_id:
            raise RuntimeError("dataset_id={} does not match to feature_map!".format(self.dataset_id))
        self.num_fields = feature_map["num_fields"]
        self.num_features = feature_map.get("num_features", None)
        self.input_length = feature_map.get("input_length", None)
        self.feature_specs = OrderedDict(feature_map["feature_specs"])

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        if not os.path.exists(os.path.dirname(json_file)):
            os.makedirs(os.path.dirname(json_file))
        feature_map = OrderedDict()
        feature_map["dataset_id"] = self.dataset_id
        feature_map["num_fields"] = self.num_fields
        feature_map["num_features"] = self.num_features
        feature_map["input_length"] = self.input_length
        feature_map["feature_specs"] = self.feature_specs
        with open(json_file, "w") as fd:
            json.dump(feature_map, fd, indent=4)


class FeatureEncoder(object):
    def __init__(self, 
                 feature_cols=[], 
                 label_col={}, 
                 dataset_id=None, 
                 data_root="../data/", 
                 **kwargs):
        logging.info("Set up feature encoder...")
        self.data_dir = os.path.join(data_root, dataset_id)
        self.pickle_file = os.path.join(self.data_dir, "feature_encoder.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_cols = self._complete_label_cols(label_col)
        self.feature_map = FeatureMap(dataset_id, self.data_dir)
        self.encoders = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def _complete_label_cols(self, label_col):
        full_label_cols = []
        name_or_namelist = label_col["name"]
        if isinstance(name_or_namelist, list):
            for _name in name_or_namelist:
                _col = label_col.copy()
                _col["name"] = _name
                full_label_cols.append(_col)
        else:
            full_label_cols.append(label_col)
        return full_label_cols

    def read_csv(self, data_path):
        logging.info("Reading file: " + data_path)
        all_cols = self.feature_cols + self.label_cols
        dtype_dict = dict((x["name"], eval(x["dtype"]) if isinstance(x["dtype"], str) else x["dtype"]) 
                          for x in all_cols)
        ddf = pd.read_csv(data_path, dtype=dtype_dict, memory_map=True)
        return ddf

    def preprocess(self, ddf, fill_na=True):
        logging.info("Preprocess feature columns...")
        all_cols = self.label_cols + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            if fill_na and name in ddf.columns and ddf[name].isnull().values.any():
                ddf[name] = self._fill_na(col, ddf[name])
            if "preprocess" in col and col["preprocess"] != "":
                preprocess_fn = getattr(self, col["preprocess"])
                ddf[name] = preprocess_fn(ddf, name)
        active_cols = [col["name"] for col in self.label_cols] + [col["name"] for col in self.feature_cols if col["active"]]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na(self, col, series):
        na_value = col.get("na_value")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] in ["str", str]:
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign na_value!".format(col["name"]))

    def fit(self, ddf, min_categr_count=1, num_buckets=10, **kwargs):           
        logging.info("Fit feature encoder...")
        self.feature_map.num_fields = 0
        for col in self.feature_cols:
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                name = col["name"]
                self.fit_feature_col(col, ddf[name].values, 
                                     min_categr_count=min_categr_count,
                                     num_buckets=num_buckets)
                self.feature_map.num_fields += 1
        self.feature_map.set_feature_index()
        self.save_pickle(self.pickle_file)
        self.feature_map.save(self.json_file)
        logging.info("Set feature encoder done.")

    def fit_feature_col(self, feature_column, feature_values, min_categr_count=1, num_buckets=10):
        name = feature_column["name"]
        feature_type = feature_column["type"]
        feature_source = feature_column.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}
        # if "min_categr_count" in feature_column:
        #     min_categr_count = feature_column["min_categr_count"]
        #     self.feature_map.feature_specs[name]["min_categr_count"] = min_categr_count
        if "embedding_dim" in feature_column:
            self.feature_map.feature_specs[name]["embedding_dim"] = feature_column["embedding_dim"]
        if feature_type == "numeric":
            normalizer_name = feature_column.get("normalizer", None)
            if normalizer_name is not None:
                normalizer = Normalizer(normalizer_name)
                normalizer.fit(feature_values)
                self.encoders[name + "_normalizer"] = normalizer
            self.feature_map.num_features += 1
        elif feature_type == "categorical":
            encoder = feature_column.get("encoder", "")
            if encoder != "":
                self.feature_map.feature_specs[name]["encoder"] = encoder
            if encoder == "":
                labelencoder = LabelEncoderExt()
                labelencoder.fit(feature_values)
                self.encoders[name + "_labelencoder"] = labelencoder
                self.feature_map.num_features += len(labelencoder.classes_)
                self.feature_map.feature_specs[name]["vocab_size"] = len(labelencoder.classes_)
        elif feature_type == "sequence":
            cname = name.lstrip('hist_') # ??????????????????????????????????????????
            separator = feature_column.get("separator", "-")
            na_value = feature_column.get("na_value", "")
            max_seq_len = feature_column.get("max_seq_len", 10)
            padding = feature_column.get("padding", "post")
            self.feature_map.feature_specs[name]["cname"] = cname
            self.feature_map.feature_specs[name]["separator"] = separator
            self.feature_map.feature_specs[name]["na_value"] = na_value
            self.feature_map.feature_specs[name]["max_seq_len"] = max_seq_len
            self.feature_map.feature_specs[name]["padding"] = padding
        else:
            raise NotImplementedError("feature_col={}".format(feature_column))
        
    # def fit_feature_col(self, feature_column, feature_values, min_categr_count=1, num_buckets=10):
    #     name = feature_column["name"]
    #     feature_type = feature_column["type"]
    #     feature_source = feature_column.get("source", "")
    #     self.feature_map.feature_specs[name] = {"source": feature_source,
    #                                             "type": feature_type}
    #     if "min_categr_count" in feature_column:
    #         min_categr_count = feature_column["min_categr_count"]
    #         self.feature_map.feature_specs[name]["min_categr_count"] = min_categr_count
    #     if "embedding_dim" in feature_column:
    #         self.feature_map.feature_specs[name]["embedding_dim"] = feature_column["embedding_dim"]
    #     if feature_type == "numeric":
    #         normalizer_name = feature_column.get("normalizer", None)
    #         if normalizer_name is not None:
    #             normalizer = Normalizer(normalizer_name)
    #             normalizer.fit(feature_values)
    #             self.encoders[name + "_normalizer"] = normalizer
    #         self.feature_map.num_features += 1
    #     elif feature_type == "categorical":
    #         encoder = feature_column.get("encoder", "")
    #         if encoder != "":
    #             self.feature_map.feature_specs[name]["encoder"] = encoder
    #         if encoder == "":
    #             tokenizer = Tokenizer(min_freq=min_categr_count,
    #                                   na_value=feature_column.get("na_value", ""))
    #             if "share_embedding" in feature_column:
    #                 self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
    #                 tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
    #             else:
    #                 if self.is_share_embedding_with_sequence(name):
    #                     tokenizer.fit_on_texts(feature_values, use_padding=True)
    #                 else:
    #                     tokenizer.fit_on_texts(feature_values, use_padding=False)
    #             if "pretrained_emb" in feature_column:
    #                 logging.info("Loading pretrained embedding: " + name)
    #                 self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
    #                 self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
    #                 tokenizer.load_pretrained_embedding(name,
    #                                                     feature_column["pretrained_emb"],
    #                                                     feature_column["embedding_dim"],
    #                                                     os.path.join(self.data_dir, "pretrained_{}.h5".format(name)),
    #                                                     feature_dtype=feature_column.get("dtype"),
    #                                                     freeze_emb=feature_column.get("freeze_emb", True))
    #             if tokenizer.use_padding: # update to account pretrained keys
    #                 self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
    #             self.encoders[name + "_tokenizer"] = tokenizer
    #             self.feature_map.num_features += tokenizer.vocab_size
    #             self.feature_map.feature_specs[name]["vocab_size"] = tokenizer.vocab_size
    #         elif encoder == "numeric_bucket":
    #             num_buckets = feature_column.get("num_buckets", num_buckets)
    #             qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
    #             qtf.fit(feature_values)
    #             boundaries = qtf.quantiles_[1:-1]
    #             self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
    #             self.feature_map.num_features += num_buckets
    #             self.encoders[name + "_boundaries"] = boundaries
    #         elif encoder == "hash_bucket":
    #             num_buckets = feature_column.get("num_buckets", num_buckets)
    #             uniques = Counter(feature_values)
    #             num_buckets = min(num_buckets, len(uniques))
    #             self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
    #             self.feature_map.num_features += num_buckets
    #             self.encoders[name + "_num_buckets"] = num_buckets
    #     elif feature_type == "sequence":
    #         encoder = feature_column.get("encoder", "MaskedAveragePooling")
    #         splitter = feature_column.get("splitter", " ")
    #         na_value = feature_column.get("na_value", "")
    #         max_len = feature_column.get("max_len", 0)
    #         padding = feature_column.get("padding", "post")
    #         tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter,
    #                               na_value=na_value, max_len=max_len, padding=padding)
    #         if "share_embedding" in feature_column:
    #             if feature_column.get("max_len") is None:
    #                 tokenizer.fit_on_texts(feature_values, use_padding=True) # Have to get max_len even share_embedding
    #             self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
    #             tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
    #         else:
    #             tokenizer.fit_on_texts(feature_values, use_padding=True)
    #         if "pretrained_emb" in feature_column:
    #             logging.info("Loading pretrained embedding: " + name)
    #             self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
    #             self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
    #             tokenizer.load_pretrained_embedding(name,
    #                                                 feature_column["pretrained_emb"],
    #                                                 feature_column["embedding_dim"],
    #                                                 os.path.join(self.data_dir, "pretrained_{}.h5".format(name)),
    #                                                 feature_dtype=feature_column.get("dtype"),
    #                                                 freeze_emb=feature_column.get("freeze_emb", True))
    #         self.encoders[name + "_tokenizer"] = tokenizer
    #         self.feature_map.num_features += tokenizer.vocab_size
    #         self.feature_map.feature_specs[name].update({"encoder": encoder,
    #                                                      "padding_idx": tokenizer.vocab_size - 1,
    #                                                      "vocab_size": tokenizer.vocab_size,
    #                                                      "max_len": tokenizer.max_len})
    #     else:
    #         raise NotImplementedError("feature_col={}".format(feature_column))

    def transform(self, ddf):

        def pad(inputs, default='Unknown', max_seq_len=10, padding='post'):
            # ?????????pad????????????????????????0?????????
            if len(inputs) < max_seq_len:
                inputs = [default] * (max_seq_len - len(inputs)) + inputs
            else:
                inputs = inputs[-max_seq_len:]
            return inputs

        def concat(inputs, sep='-'):
            return sep.join(map(str, inputs))

        logging.info("Transform feature columns...")
        for feature, feature_spec in self.feature_map.feature_specs.items():
            feature_type = feature_spec["type"]
            if feature_type == "numeric":
                numeric_array = ddf.loc[:, feature].fillna(0).apply(lambda x: float(x)).values
                normalizer = self.encoders.get(feature + "_normalizer")
                if normalizer:
                    numeric_array = normalizer.normalize(numeric_array)
                ddf.loc[:, feature] = numeric_array
            elif feature_type == "categorical":
                encoder = self.encoders.get(feature + "_labelencoder")
                if encoder:
                    ddf.loc[:, feature] = encoder.transform(ddf.loc[:, feature].values)
            elif feature_type == "sequence":
                encoder = self.encoders.get(feature_spec['cname'] + "_labelencoder")
                separator = feature_spec['separator']
                max_seq_len = feature_spec['max_seq_len']
                padding = feature_spec['padding']
                if encoder:
                    ddf.loc[:, feature] = [concat(encoder.transform(pad(value.split(separator), max_seq_len=max_seq_len, padding=padding)))
                                           for value in ddf.loc[:, feature].values]
        for label_col in self.label_cols:
            label_name = label_col["name"]
            if ddf[label_name].dtype != np.float64:
                ddf.loc[:, label_name] = ddf.loc[:, label_name].apply(lambda x: float(x))
        return ddf

    def load_pickle(self, pickle_file=None):
        """ Load feature encoder from cache """
        if pickle_file is None:
            pickle_file = self.pickle_file
        logging.info("Load feature_encoder from pickle: " + pickle_file)
        if os.path.exists(pickle_file):
            pickled_feature_encoder = pickle.load(open(pickle_file, "rb"))
            if pickled_feature_encoder.feature_map.dataset_id == self.feature_map.dataset_id:
                return pickled_feature_encoder
        raise IOError("pickle_file={} not valid.".format(pickle_file))

    def save_pickle(self, pickle_file):
        logging.info("Pickle feature_encoder: " + pickle_file)
        if not os.path.exists(os.path.dirname(pickle_file)):
            os.makedirs(os.path.dirname(pickle_file))
        pickle.dump(self, open(pickle_file, "wb"))

    def load_json(self, json_file):
        self.feature_map.load(json_file)

    # # TODO: ?????????????????????
    # def get_feature_name_dict(self):
    #     numeric_feature_names = []
    #     categorical_feature_names = []
    #     sequence_feature_names = []
    #     label_names = [label_col['name'] for label_col in self.label_cols]
    #     for feature, feature_spec in self.feature_map.feature_specs.items():
    #         if feature_spec['type'] == 'numeric':
    #             numeric_feature_names.append(feature)
    #         elif feature_spec['type'] == 'categorical':
    #             categorical_feature_names.append(feature)
    #         elif feature_spec['type'] == 'sequence':
    #             sequence_feature_names.append(feature)
    #     return {
    #         'numeric_feature_names': numeric_feature_names,
    #         'categorical_feature_names': categorical_feature_names,
    #         'sequence_feature_names': sequence_feature_names,
    #         'label_names': label_names
    #     }
    #
    # def get_vocab_size(self, feature_name):
    #     feature_spec = self.feature_map.feature_specs.get(feature_name, None)
    #     if not feature_spec:
    #         raise RuntimeError("unknown feature name={}!".format(feature_name))
    #     if feature_spec["type"] != "categorical":
    #         raise RuntimeError("feature type={} does not have vocab size!".format(feature_spec["type"]))
    #     return feature_spec["vocab_size"]
