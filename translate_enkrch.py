from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

EOS = text_encoder.EOS_ID

_ENKR_SUBTITLE_TRAIN_DATASETS=[
  [
    "MultiUN_dataset", # dummy url
    ("train/input.txt",
     "train/output.txt")
  ]
]

_ENKR_SUBTITLE_TEST_DATASETS=[
  [
    "MultiUN_dataset", # dummy url
    ("dev/eng_val.txt",
     "dev/kor_val.txt")
  ]
]

def compile_data_from_txt(tmp_dir, datasets, filename):
  filename = os.path.join(tmp_dir, filename)
  with tf.gfile.GFile(filename + ".lang1", mode="w") as lang1_resfile:
    with tf.gfile.GFile(filename + ".lang2", mode="w") as lang2_resfile:
      for dataset in datasets:
        """
        datasets = [[url0, (file0.lang1, file0.lang2)], [url1, (file1.lang1,
        file1.lang2)], [url2, (file2.lang1, file2.lang2)], ... ]
        """
        lang1_filename, lang2_filename = dataset[1]
        lang1_filepath = os.path.join(tmp_dir, lang1_filename)
        lang2_filepath = os.path.join(tmp_dir, lang2_filename)

        if not (os.path.exists(lang1_filepath) and os.path.exists(lang2_filepath)):
          raise ValueError("%s, or %s file not found" % (lang1_filepath, lang2_filepath))
        
        with tf.gfile.GFile(lang1_filepath, mode="r") as lang1_file:
          with tf.gfile.GFile(lang2_filepath, mode="r") as lang2_file:
            line1, line2 = lang1_file.readline(), lang2_file.readline()
            while line1 or line2:
              lang1_resfile.write(line1.strip() + "\n")
              lang2_resfile.write(line2.strip() + "\n")
              line1, line2 = lang1_file.readline(), lang2_file.readline()

  return filename

@registry.register_problem("translate_zero_shot_exp")
class TranslateZeroShotExperiment(translate.TranslateProblem):
  
  @property
  def targeted_vocab_size(self):
    return 90000 # 32768

  @property
  def source_vocab_name(self):
    return "vocab_zero_input.%d" % self.targeted_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab_zero_output.%d" % self.targeted_vocab_size

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return 32

  def generator(self, data_dir, tmp_dir, train):
    datasets = _ENKR_SUBTITLE_TRAIN_DATASETS if train else _ENKR_SUBTITLE_TEST_DATASETS
    source_datasets = [os.path.join(data_dir, path[1][0]) for path in datasets]
    target_datasets = [os.path.join(data_dir, path[1][1]) for path in datasets]

    source_vocab = generator_utils.get_or_generate_txt_vocab(data_dir, self.source_vocab_name, self.targeted_vocab_size, source_datasets)
    target_vocab = generator_utils.get_or_generate_txt_vocab(data_dir, self.target_vocab_name, self.targeted_vocab_size, target_datasets)

    tag = "train" if train else "dev"
    data_path = compile_data_from_txt(tmp_dir, datasets, "zero_shot_enkrch_tok_%s" % tag)

    return translate.bi_vocabs_token_generator(data_path + ".lang1", data_path + ".lang2", source_vocab, target_vocab, EOS)

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }

"""
@registry.register_problem("translate_zero_shot_exp")
class TranslateKrenSubtitleSep32k(translate.TranslateProblem):
  
  @property
  def targeted_vocab_size(self):
    return 90000 # 32768

  @property
  def source_vocab_name(self):
    return "vocab_zero_input.%d" % self.targeted_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab_zero_output.%d" % self.targeted_vocab_size

  @property
  def input_space_id(self):
    return 32

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  def generator(self, data_dir, tmp_dir, train):
    datasets = _ENKR_SUBTITLE_TRAIN_DATASETS if train else _ENKR_SUBTITLE_TEST_DATASETS
    datasets = [[dataset[0], (dataset[1][1], dataset[1][0])] for dataset in datasets]
    source_datasets = [os.path.join(data_dir, path[1][0]) for path in datasets]
    target_datasets = [os.path.join(data_dir, path[1][1]) for path in datasets]

    source_vocab = generator_utils.get_or_generate_txt_vocab(data_dir, self.source_vocab_name, self.targeted_vocab_size, source_datasets)
    target_vocab = generator_utils.get_or_generate_txt_vocab(data_dir, self.target_vocab_name, self.targeted_vocab_size, target_datasets)

    tag = "train" if train else "dev"
    data_path = compile_data_from_txt(tmp_dir, datasets, "zero_shot_enkrch_tok_%s" % tag)

    return translate.bi_vocabs_token_generator(data_path + ".lang1", data_path + ".lang2", source_vocab, target_vocab, EOS)

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }
"""

from tensor2tensor.models import transformer

@registry.register_hparams
def enkrch_hparams():
    hparams = transformer.transformer_base_single_gpu()  # Or whatever you'd like to build off.
    hparams.learning_rate = 0.04
    return hparams
