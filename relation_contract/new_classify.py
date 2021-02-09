# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import eval
import collections
import csv
import os
import bert_normal
import optimization
import tokenization
import tensorflow as tf
import numpy as np
import HeadSelectionScores
from deal_data import deal_data
from deal_data import process_single_data_to_example
import tf_metrics
flags = tf.flags
import matplotlib.pyplot as plt
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "data/semeval2018/multi",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "F:/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "semeval", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "F:/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "tmp/bertsp_muti/",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "F:/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 2, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 2, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 80,
                   "Total number of training epochs to perform.")


flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "max_num_relations", 12,
    "Maximum number of relation within the text")

flags.DEFINE_integer(
    "max_distance", 2,
    "The max_distance for entity relative position")


class Extras(object):
    """
    Extra objact to pass entity related features to the model function
    """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, locations, labels, num_relations, joins_id = None, scoringMatrix = None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: unused in the entity task
          locations: entity localtion
          labels: relation label.tsv
          num_relations: number of relations in the text
          label.tsv: (Optional) string. The label.tsv of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.locations = locations
        self.labels = labels
        self.num_relations = num_relations
        self.joins_id = joins_id
        self.scoringMatrix = scoringMatrix


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,
                 loc, mas, e1_mas, e2_mas, cls_mask,
                 label_id, scoringMatrix):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.loc = loc
        self.mas = mas
        self.e1_mas = e1_mas
        self.e2_mas = e2_mas
        self.cls_mask = cls_mask
        self.label_id = label_id
        self.scoringMatrix = scoringMatrix


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, label_list):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, label_list):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SemEvalProcessor(DataProcessor):
    """Processor for the SemEval data set (GLUE version)."""

    def get_train_examples(self, data_dir, label_list):
        """See base class."""
        data, relation_list = deal_data("train")
        return self._create_examples(
            data, "train", relation_list)

    def get_dev_examples(self, data_dir, label_list):
        """See base class."""
        data, relation_list = deal_data("dev")
        return self._create_examples(
            data, "dev", relation_list)

    def get_test_examples(self, data_dir):
        """See base class."""
        data, relation_list = deal_data("test")
        return self._create_examples(
            data, "test", relation_list)

    def get_labels(self, data_dir):
        """See base class."""
        label_list = []
        filein = open(os.path.join(data_dir, "label.tsv"))
        for line in filein:
            label = line.strip()
            label_list.append(tokenization.convert_to_unicode(label))
        return label_list

    def _create_examples(self, lines, set_type, label_list):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines.data):
            guid = "%s-%s" % (set_type, i)
            text_a, locations, labels, num_relations = process_single_data_to_example(line)
            # join_id = get_join_id(text_a, locations, labels, num_relations, label_list)# 将join_id 变成在句子转换成token之后进行计算
            # examples.append(
            #     InputExample(guid=guid, text_a=text_a, text_b=None,
            #                  locations=locations, labels=labels, num_relations=num_relations, joins_id = join_id))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,
                             locations=locations, labels=labels, num_relations=num_relations))
        return examples

def get_join_id(token, locations, labels, mapping, label_list):
    '''
    构造句子中的label对应的matrix。
    :return: scoring_matrix_gold
    '''
    joins_id = []
    head_id, labels_id = get_head_and_label_id(token, locations, labels, label_list, mapping)
    for index in range(len(head_id)):
        join_id = []

        for label_index in range(len(head_id[index])):
            if (len(head_id) * len(label_list) < head_id[index][label_index] * len(label_list) + labels_id[index][label_index]):
                x =123
            join_id.append(head_id[index][label_index] * len(label_list) + labels_id[index][label_index])
        joins_id.append(join_id)
    return joins_id

def get_head_and_label_id(token, locations, labels, label_list, mapping):
    '''
    获取关系实体中第一个实体对应的第二个实体的位置
    :param text_a:
    :param locations:
    :return:
    '''
    head_id = [[i] for i in range(len(token))] # 计算加上SEP和CLS的长度
    labels_id = [[label_list.index("N")] for i in range(len(token))]

    for index, (e1, e2) in enumerate(locations):
        if mapping[-1] < e1[0] or mapping[-1] < e2[0]:
            continue
        first_position = mapping.index(e1[0]) #第一个实体的第一个词mapping位置
        last_position = mapping.index(e2[0]) #第二个实体的第一个词mapping位置
        # if first_position >= len(head_id) or last_position >= len(head_id):
        #     continue

        tmp_list = head_id[first_position] #获取与第一个实体有关系的词的位置
        tmp_relation_list = labels_id[first_position] #获取与第一个实体有关系的词的关系
        if len(tmp_list) == 1 and tmp_list[0] == first_position: #如果指向自己，就清空
            tmp_list = []
            tmp_relation_list = []
        tmp_list.append(last_position) #添加新的词的位置
        tmp_relation_list.append(find(labels[index], label_list))#添加新的关系
        head_id[first_position] = tmp_list # 修改
        labels_id[first_position] = tmp_relation_list
    return head_id, labels_id



def find(label_name, label_list):
    '''
    寻找label对应的下标
    :param label_id:
    :param label_list:
    :return:
    '''
    return label_list.index(label_name)


class ACEProcessor(DataProcessor):
    """Processor for the SemEval data set (GLUE version)."""

    def get_train_examples(self, data_dir, label_list):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ACE.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ACE.dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ACE.test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        label_list = []
        filein = open(os.path.join(data_dir, "ACE.label.tsv.tsv"))
        for line in filein:
            label = line.strip()
            label_list.append(tokenization.convert_to_unicode(label))
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            # if i == 0:
            #   continue



            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            # bert裁剪到126+2
            if len(text_a.split(" ")) > 126:
                text_a = text_a.split(" ")[0:126].join(" ")
                x = 123
            num_relations = int((len(line) - 1) / 7)
            locations = list()
            labels = list()
            for j in range(num_relations):
                label = tokenization.convert_to_unicode(line[j * 7 + 1])
                labels.append(label)
                # (lo, hi)
                entity_pos1 = (int(line[j * 7 + 2]) + 1, int(line[j * 7 + 3]) + 1)
                entity_pos2 = (int(line[j * 7 + 5]) + 1, int(line[j * 7 + 6]) + 1)
                # [((lo1,hi1), (lo2, hi2))]
                locations.append((entity_pos1, entity_pos2))
            if len(locations) == 0:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,
                             locations=locations, labels=labels, num_relations=num_relations))

        return examples


def find_lo_hi(mapping, value):
    """
    find the boundary of a value in a list
    will return (0,0) if no such value in the list
    """
    try:
        lo = mapping.index(value)
        hi = min(len(mapping) - 1 - mapping[::-1].index(value), FLAGS.max_seq_length)
        return (lo, hi)
    except:
        return (0, 0)


def convert_entity_row(mapping, loc, max_distance):
    """
    convert an entity span(lo,hi) to a relative distance vector of shape [max_seq_length]
    """
    lo, hi = loc
    res = [max_distance] * FLAGS.max_seq_length
    mas = [0] * FLAGS.max_seq_length
    for i in range(FLAGS.max_seq_length):
        if i < len(mapping):
            val = mapping[i]
            if val < lo - max_distance:
                res[i] = max_distance
            elif val < lo:
                res[i] = lo - val
            elif val <= hi:
                res[i] = 0
                mas[i] = 1
            elif val <= hi + max_distance:
                res[i] = val - hi + max_distance
            else:
                res[i] = 2 * max_distance
        else:
            res[i] = 2 * max_distance
    return res, mas


def prepare_extra_data(mapping, locs, max_distance):
    res = np.zeros([FLAGS.max_seq_length, FLAGS.max_seq_length], dtype=np.int8)
    mas = np.zeros([FLAGS.max_seq_length, FLAGS.max_seq_length], dtype=np.int8)

    e1_mas = np.zeros([FLAGS.max_num_relations, FLAGS.max_seq_length], dtype=np.int8)
    e2_mas = np.zeros([FLAGS.max_num_relations, FLAGS.max_seq_length], dtype=np.int8)

    entities = set()
    for loc in locs:
        entities.add(loc[0])
        entities.add(loc[1])

    for e in entities:
        (lo, hi) = e
        relative_position, _ = convert_entity_row(mapping, e, max_distance)
        sub_lo1, sub_hi1 = find_lo_hi(mapping, lo)
        sub_lo2, sub_hi2 = find_lo_hi(mapping, hi)
        if sub_lo1 == 0 and sub_hi1 == 0:
            continue
        if sub_lo2 == 0 and sub_hi2 == 0:
            continue
        # col
        res[:, sub_lo1:sub_hi2 + 1] = np.expand_dims(relative_position, -1)
        mas[1:, sub_lo1:sub_hi2 + 1] = 1

    for e in entities:
        (lo, hi) = e
        relative_position, _ = convert_entity_row(mapping, e, max_distance)
        sub_lo1, sub_hi1 = find_lo_hi(mapping, lo)
        sub_lo2, sub_hi2 = find_lo_hi(mapping, hi)
        if sub_lo1 == 0 and sub_hi1 == 0:
            continue
        if sub_lo2 == 0 and sub_hi2 == 0:
            continue
        # row
        res[sub_lo1:sub_hi2 + 1, :] = relative_position
        mas[sub_lo1:sub_hi2 + 1, 1:] = 1

    for idx, (e1, e2) in enumerate(locs):
        # if idx >= 10:
        #     continue
        # 不记得为什么加的这个判断，所以注释掉。
        # e1
        (lo, hi) = e1
        _, mask = convert_entity_row(mapping, e1, max_distance)
        e1_mas[idx] = mask
        # e2
        (lo, hi) = e2
        _, mask = convert_entity_row(mapping, e2, max_distance)
        e2_mas[idx] = mask

    return res, mas, e1_mas, e2_mas

def get_scoringMatrix(joint_ids, relation_list):
    scoringMatrix = np.zeros((FLAGS.max_seq_length, FLAGS.max_seq_length * len(relation_list)))
    # [len(seq), len(seq) * len(relation)]
    for tokenIdx in range(len(joint_ids)):

        tokenHeads = joint_ids[tokenIdx]
        for head in tokenHeads:
            # print (str(tokenIdx)+ " "+ str(head))
            scoringMatrix[tokenIdx, head] = 1  # 填充关系score数组
    return scoringMatrix


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    #bert

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a, mapping_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]



    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # 使用处理后的token进行标签矩阵的构建
    example.joins_id = get_join_id(tokens, example.locations, example.labels, mapping_a, label_list)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    if len(example.joins_id) >= 128:
        x =123
    scoringMatrix = get_scoringMatrix(example.joins_id, label_list)
    example.scoringMatrix = scoringMatrix
    loc, mas, e1_mas, e2_mas = prepare_extra_data(mapping_a, example.locations, FLAGS.max_distance)
    label_id = [label_map[label] for label in example.labels]
    label_id = label_id + [0] * (FLAGS.max_num_relations - len(label_id))
    cls_mask = [1] * example.num_relations + [0] * (FLAGS.max_num_relations - example.num_relations)
    if len(cls_mask) != FLAGS.max_num_relations:
        print("error")


    np.set_printoptions(edgeitems=15)
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("loc:")
        tf.logging.info("\n" + str(loc))
        tf.logging.info("mas:")
        tf.logging.info("\n" + str(mas))
        tf.logging.info("e1_mas:")
        tf.logging.info("\n" + str(e1_mas))
        tf.logging.info("e2_mas:")
        tf.logging.info("\n" + str(e2_mas))
        tf.logging.info("cls_mask:")
        tf.logging.info("\n" + str(cls_mask))
        tf.logging.info("labels: %s" % " ".join([str(x) for x in label_id]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        loc=loc.flatten(),
        mas=mas.flatten(),
        e1_mas=e1_mas.flatten(),
        e2_mas=e2_mas.flatten(),
        cls_mask=cls_mask,
        label_id=label_id,
        scoringMatrix=scoringMatrix)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f



        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["loc"] = create_int_feature(feature.loc)
        features["mas"] = create_int_feature(feature.mas)
        features["e1_mas"] = create_int_feature(feature.e1_mas)
        features["e2_mas"] = create_int_feature(feature.e2_mas)
        features["cls_mask"] = create_int_feature(feature.cls_mask)
        features["label_ids"] = create_int_feature(feature.label_id)
        features["scoringMatrix"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.scoringMatrix.astype(np.float32).tostring()]))
        features['matrix_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.scoringMatrix.shape)))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, relation_num):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "loc": tf.FixedLenFeature([seq_length * seq_length], tf.int64),
        "mas": tf.FixedLenFeature([seq_length * seq_length], tf.int64),
        "e1_mas": tf.FixedLenFeature([FLAGS.max_num_relations * seq_length], tf.int64),
        "e2_mas": tf.FixedLenFeature([FLAGS.max_num_relations * seq_length], tf.int64),
        "cls_mask": tf.FixedLenFeature([FLAGS.max_num_relations], tf.int64),
        "label_ids": tf.FixedLenFeature([FLAGS.max_num_relations], tf.int64),
        "scoringMatrix" : tf.FixedLenFeature((), tf.string), #[seq_length, seq_length * relation_num]
        "matrix_shape": tf.FixedLenFeature([2], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if name == "scoringMatrix":
                t = tf.decode_raw(example[name], tf.float32)
                shape = example['matrix_shape']
                t = tf.reshape(t, shape)
            elif t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, extras, scoring_matrix_gold):
    """Creates a classification model."""
    model = bert_normal.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,

        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_sequence_output()
    batch_size = output_layer.shape[0].value
    from_seq_length = output_layer.shape[1].value
    hidden_size = output_layer.shape[2].value

    #大小
    rel_scores = HeadSelectionScores.getHeadSelectionScores(output_layer, hidden_size, 16, num_labels,
                                             dropout_keep_in_prob=1.0)  # [16, 348, 348*6]

    lossREL = tf.nn.sigmoid_cross_entropy_with_logits(logits=rel_scores,
                                                      labels=scoring_matrix_gold)  # 通过label和logits的值的对比对模型进行对比
    obj = tf.reduce_sum(lossREL)
    probas = tf.nn.sigmoid(rel_scores)  # 将输出压缩到0~1
    predictedRel = tf.round(probas)  # 四舍五入 转成0或1
    raw_perturb = tf.gradients(obj, output_layer)[0]  # [batch, L, dim] 通过反向传播更新embeding中的参数并返回
    normalized_per = tf.nn.l2_normalize(raw_perturb, axis=[1, 2])  # 加入L2范式，防止过拟合
    perturb = 0.05 * tf.sqrt(tf.cast(tf.shape(output_layer)[2], tf.float32)) * tf.stop_gradient(
        normalized_per)
    perturb_inputs = output_layer + perturb
    rel_scores = HeadSelectionScores.getHeadSelectionScores(perturb_inputs, hidden_size, 16, num_labels,
                                             dropout_keep_in_prob=1.0, reuse = True)  # [16, 348, 348*6]
    lossREL = tf.nn.sigmoid_cross_entropy_with_logits(logits=rel_scores,
                                            labels=scoring_matrix_gold)

    obj += tf.reduce_sum(lossREL)
    actualRel = tf.round(scoring_matrix_gold)
    return (obj, predictedRel, actualRel, rel_scores)



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, labels_list):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
            # tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        extras = Extras()
        extras.loc = features["loc"]
        extras.mas = features["mas"]
        extras.e1_mas = features["e1_mas"]
        extras.e2_mas = features["e2_mas"]
        extras.cls_mask = features["cls_mask"]
        scoring_matrix_gold = features["scoringMatrix"]
        extras.max_distance = FLAGS.max_distance


        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, predictedRel, actualRel, rel_scores) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, extras, scoring_matrix_gold)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = bert_normal.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
            # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                 init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # evaluator = eval.chunkEvaluator(labels_list, ner_chunk_eval="boundaries", rel_chunk_eval="boundaries")
            def metric_fn(loss, predicted_rel, actual_rel):
                # evaluator.add(predicted_rel, actual_rel)
                # pre, recall, f1 = evaluator.printInfo()
                # return {
                #     "eval_pre": pre,
                #     "eval_recall": recall,
                #     "eval_f1" : f1
                # }
                precision = tf.metrics.precision(
                    labels=actual_rel, predictions=predicted_rel)
                recall = tf.metrics.recall(
                    labels=actual_rel, predictions=predicted_rel)
                f1_score = tf.contrib.metrics.f1_score(
                    labels=actual_rel, predictions=predicted_rel)

                loss_new = tf.metrics.mean(values=loss, weights=None)

                # for metric_name, op in metrics.items():  # tensorboard
                #     tf.summary.scalar(metric_name, op[1])
                return {
                    "eval_precision": precision,
                    "eval_loss": loss_new,
                    "recall" : recall,
                    "f1_score" : f1_score
                }

            # total_loss, predictedRel, actualRel, rel_scores
            eval_metrics = (metric_fn,
                            [total_loss, predictedRel, actualRel])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn
            )
        else:
            # B 10 num_labels
            # logits = tf.reshape(predictedRel, [-1, FLAGS.max_num_relations, num_labels])
            # B 10
            predictions = predictedRel
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features

class EvaluationHook(tf.train.SessionRunHook):
    def __init__(self, **wrapper):
        self.estimator = wrapper['estimator']
        self.eval_steps = wrapper['eval_steps']
        self.eval_input_fn = wrapper['eval_input_fn']
        self.train_steps = wrapper['train_steps']
        self.step = 0
        self.eval_precision = []
        self.recall = []
        self.f1_score = []
        self.epoch = []
        self.loss = []

    def after_run(self, run_context, run_values):
        self.step += 1
        if (self.step % self.train_steps == 0):   #it means an epoch
            epoch = self.step // self.train_steps
            result = self.estimator.evaluate(input_fn=self.eval_input_fn, steps=self.eval_steps)
            print("epoch:{}, eval_precision: {}".format(epoch, result["eval_precision"]))
            print("epoch:{}, eval_loss: {}".format(epoch, result["eval_loss"]))
            print("epoch:{}, recall: {}".format(epoch, result["recall"]))
            print("epoch:{}, f1_score: {}".format(epoch, result["f1_score"]))
            self.recall.append(result["recall"])
            self.eval_precision.append(result["eval_precision"])
            self.f1_score.append(result["f1_score"])
            self.epoch.append(epoch)
            self.loss.append(result["eval_loss"])
            # plt.plot(self.epoch, self.f1_score, label="f1")
            # plt.plot(self.recall, self.epoch, label="recall")
            # plt.plot(self.epoch, self.eval_precision, label="eval_precision")
            # plt.plot(eval_hook.eval_precision, eval_hook.eval_precision)
            plt.plot(self.epoch, self.loss, label="loss")
            plt.title('result')

            plt.savefig("bert_mutihead.png")
            plt.show()



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "semeval": SemEvalProcessor,
        "ace": ACEProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)


    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = bert_normal.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels(FLAGS.data_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir, label_list)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        labels_list=label_list)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            relation_num = len(label_list))
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir, label_list)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            relation_num = len(label_list))
        if FLAGS.do_train:
            eval_hook = EvaluationHook(eval_steps=eval_steps, train_steps=1001, estimator=estimator,
                                       eval_input_fn=eval_input_fn)

            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[eval_hook])



    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir, label_list)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            relation_num=len(label_list))
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True,
            relation_num= len(label_list)
            )

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        num_actual_eval_examples = len(predict_examples)
        real_label_list = []
        length = 0


        evaluator = eval.chunkEvaluator(label_list)

        # with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for prediction, example in zip(result, predict_examples):
            # length += len(example.labels)
            evaluator.add(prediction, example.scoringMatrix)
            # for x in prediction[:example.num_relations]:
            #     pre_label.append(label_list[int(x)])
            #     writer.write(label_list[int(x)] + '\n')
            # writer.write('\n')
        evaluator.printInfo()
        print(length)
        # from sklearn.metrics import classification_report
        #
        # # Y_test = np.array(Y_test).reshape(len(Y_test), -1)
        # # enc = OneHotEncoder()
        # # enc.fit(Y_test)
        # # targets = enc.transform(Y_test).toarray()
        # print(classification_report(real_label_list, pre_label))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
