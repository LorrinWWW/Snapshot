import pandas as pd
import numpy as np
import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 input_ids, input_mask, segment_ids, label_id

    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


        
class allTHUperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "all.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "all_test.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "all_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example  

class MIXperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "mix_all.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "all_test.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "all_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[0]
            label = line[1]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example


class THUperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "train_012.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_012.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_012.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example
    
class THUperocessor1(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "train_345_15W.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_345_4W.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_345_4W.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example

class THUperocessor2(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "train_678_13W.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_678_4W.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_678_4W.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example
    
class THUperocessor3(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "train_101112.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_101112.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "test_101112.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example
    
class AGperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "AG_news/train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "AG_news/test.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "AG_news/test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example
    
class AMZperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "amazon/train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "amazon/test.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "amazon/test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[0] + '- '+ line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example
    
class YELPperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "yelp/train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "yelp/test.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "yelp/test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [1, 2, 3, 4,5]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[0]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example

class DBPperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "dbpedia/train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "dbpedia/test.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "dbpedia/test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1] + '-' + line[2]
            label = line[0]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example
    
class sgperocessor(DataProcessor):

    def input_file(self, data_path):
        dataset = pd.read_csv(data_path)
        dataset = np.array(dataset)
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "sogou_012_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "sogou_012_test.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.input_file(os.path.join(data_dir, "sogou_012_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        example = []
        for i, line in enumerate(dataset):
            guid = '%s-%d' % (set_type, i)
            text_a = line[1]
            label = line[2]
            example.append(InputExample(guid=guid, text_a=text_a, label=label))
        return example

def convert_example_to_feature(example_row, pad_token=0,
sequence_a_segment_id=0, sequence_b_segment_id=1,
cls_token_segment_id=1, pad_token_segment_id=0,
mask_padding_with_zero=True):
    example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    examples = [(example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id) for example in examples]

    process_count = cpu_count() - 2

    with Pool(process_count) as p:
        features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=100), total=len(examples)))


    return features


processors = {
    "thu1": THUperocessor,
    "thu2": THUperocessor1,
    "thu3": THUperocessor2,
    "thu4": THUperocessor3,
    "allthu":allTHUperocessor,
    "AG": AGperocessor,
    "amazon": AMZperocessor,
    "yelp":YELPperocessor,
    "DBP":DBPperocessor,
    "allmix":MIXperocessor,
}