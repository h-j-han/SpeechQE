import os
import re
import random
import logging
import hashlib
import langcodes
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, BinaryIO

import torch
import datasets
from transformers import WhisperFeatureExtractor

logger = logging.getLogger(__name__)


def str2int(input_string, min_value=0, max_value=1000000):
  if input_string is None:
    return "None"
  # Calculate SHA-256 hash value of the input string
  hash_object = hashlib.sha256(input_string.encode())
  hash_hex = hash_object.hexdigest()

  # Convert hash value to an integer within the specified range
  scaled_integer = min_value + int(hash_hex, 16) % (max_value - min_value + 1)

  return scaled_integer


def set_preprocess_repo_name(tokenizer, lower, chars_to_ignore_regex):
  return f"tok{os.path.basename(tokenizer.name_or_path)}_lower{lower}_regex{str2int(chars_to_ignore_regex)}"


def process_dataset(
    batch,
    tokenizer,
    instruction,
    audio_column_name="audio",
    text_column_name="sentence",
    chars_to_ignore_regex=None,
    lower=True,
    from_file=False,
    predifined_inst=False,
    lid_in_inst=False,
    training=True,
    chatml_prompt_template=False,
    indicate_audio_in_lid=True,
    label_all_input_ids=False,
    output_lang="en",
):
  if predifined_inst:
    instruction = batch["inst"]
  if chatml_prompt_template:
    user_prefix = "<|im_start|>user\n"
  else:
    user_prefix = "###[Human]: "
  if lid_in_inst:
    assert "lang" in batch, "lang is required for predifined_inst"  # source lang
    lid_suffix = " audio" if indicate_audio_in_lid else ""
    lang_full_name = langcodes.Language.get(batch["lang"]).display_name("en")
    input_ids_text = f"{user_prefix}{instruction}\n{lang_full_name}{lid_suffix}: "
  else:
    input_ids_text = f"{user_prefix}{instruction} "
  input_ids = tokenizer(input_ids_text).input_ids
  attention_mask = [1] * len(input_ids)
  if label_all_input_ids:
    labels = input_ids
  else:
    labels = [-100] * len(input_ids)

  if from_file:
    audio_path = batch[audio_column_name]
    try:
      info = sf.info(audio_path)
      is_readable = True
    except:
      logger.warning(f"audio_path={audio_path} is not readable")
      is_readable = False
  else:
    is_readable = True

  raw_text = str(batch[text_column_name])
  if chars_to_ignore_regex is not None:
    raw_text = re.sub(chars_to_ignore_regex, "", raw_text)
  if lower:
    raw_text = raw_text.lower()
  batch["target_text"] = raw_text + " "

  suffix_input_ids, suffix_attention_mask, suffix_labels = [], [], []
  if "suffix" in batch and batch["suffix"] is not None:
    new_input_ids = tokenizer(batch["suffix"]).input_ids[1:]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    if label_all_input_ids:
      suffix_labels += new_input_ids
    else:
      suffix_labels += [-100] * len(new_input_ids)
  if chatml_prompt_template:
    assitant_prefix = "<|im_end|>\n<|im_start|>assistant\n"
  else:
    assitant_prefix = "\n\n\n###[Assistant]:"
  new_input_ids = tokenizer(assitant_prefix).input_ids[1:]  # remove bos token
  suffix_input_ids += new_input_ids
  suffix_attention_mask += [1] * len(new_input_ids)
  if label_all_input_ids:
    suffix_labels += new_input_ids
  else:
    suffix_labels += [-100] * len(new_input_ids)
  if training:
    ### response
    new_input_ids = tokenizer(batch["target_text"]).input_ids[1:]
    new_input_ids += [tokenizer.eos_token_id]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += new_input_ids

  batch["input_ids"] = input_ids
  batch["attention_mask"] = attention_mask
  batch["labels"] = labels
  batch["suffix_input_ids"] = suffix_input_ids
  batch["suffix_attention_mask"] = suffix_attention_mask
  batch["suffix_labels"] = suffix_labels
  if from_file:
    batch["audio_path"] = audio_path
  else:
    batch["raw_audio"] = batch[audio_column_name]["array"]
  batch["is_readable"] = is_readable
  return batch


def load_speech_text_paired_dataset(
    dataroot=None,
    manifest_files=None,
    tokenizer=None,
    instruction="",
    dataset_name=None,
    dataset_config_name=None,
    split_name=None,
    token=None,
    audio_column_name="audio",
    text_column_name="sentence",
    chars_to_ignore_regex=None,
    load_from_cache_file=True,
    num_proc=8,
    partial_sample=-1,
    sampling_rate=16000,
    from_file=False,
    lower=True,
    predifined_inst=False,
    lid_in_inst=False,
    training=True,
    chatml_prompt_template=False,
    indicate_audio_in_lid=True,
    label_all_input_ids=False,
    base_audio_path=None,
):
  if manifest_files is not None:
    assert dataroot is not None, "dataroot is required for manifest_files"
    repo_suffix = set_preprocess_repo_name(tokenizer, lower,
                                           chars_to_ignore_regex)
    logger.warning(f"repo_suffix={repo_suffix}")
    # remove .tsv or .json
    repo_name = f"processed_{repo_suffix}_{manifest_files}".replace(
        ".tsv", "").replace(".json", "")
    logger.warning(f"repo_name={repo_name}")
    if (os.path.exists(os.path.join(dataroot, repo_name.replace("*", "all")))
        and load_from_cache_file):
      logger.warning("load processed dataset")
      dataset = datasets.load_from_disk(
          os.path.join(dataroot, repo_name.replace("*", "all")))
      return dataset

    logger.warning(
        f"load dataset from scratch from {dataroot}/{manifest_files}")

    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        dataroot,
        data_files=manifest_files_list,
        split="train",
        streaming=False)
  else:
    logger.warning(
        f"load processed dataset from hub {dataset_name=} {dataset_config_name=} {split_name=}"
    )
    raw_dataset = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        split=split_name,
        token=token,
    )
    if base_audio_path is not None:

      def add_prefix(example):
        example["path"] = os.path.join(base_audio_path, example["path"])
        return example
      raw_dataset = raw_dataset.map(add_prefix)

    if partial_sample > 0:
      logger.warning(f"Slicing dataset partial_sample={partial_sample}")
      raw_dataset = raw_dataset.select(range(partial_sample))
    logger.warning(f"Casting to sample rate {sampling_rate}")
    raw_dataset.cast_column(
        audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate))
  dataset = raw_dataset.map(
      process_dataset,
      fn_kwargs={
          "tokenizer": tokenizer,
          "instruction": instruction,
          "audio_column_name": audio_column_name,
          "text_column_name": text_column_name,
          "chars_to_ignore_regex": chars_to_ignore_regex,
          "from_file": from_file,
          "lower": lower,
          "predifined_inst": predifined_inst,
          "lid_in_inst": lid_in_inst,
          "training": training,
          "chatml_prompt_template": chatml_prompt_template,
          "indicate_audio_in_lid": indicate_audio_in_lid,
          "label_all_input_ids": label_all_input_ids,
      },
      # remove_columns=raw_dataset.column_names,
      load_from_cache_file=load_from_cache_file,
      num_proc=num_proc,
  )

  def is_path_readable(flag):
    return flag

  blen = len(dataset)
  print(f"Before filter is_readable {len(dataset)=}")
  dataset = dataset.filter(is_path_readable, input_columns=["is_readable"])
  print(f"{len(dataset)=}, filtered out {blen - len(dataset)}")
  assert len(dataset), f"No data found in the dataset { len(dataset)=}"

  if from_file and load_from_cache_file and dataroot is not None:
    logger.info(f"Save preprocessed disc to {repo_name=}")
    dataset.save_to_disk(os.path.join(dataroot, repo_name.replace("*", "all")))

  return dataset


def collate_tokens(values: List[List[int]], pad_id: int):
  size = max(len(v) for v in values)
  batch_size = len(values)
  res = torch.LongTensor(batch_size, size).fill_(pad_id)

  def copy_tensor(src, dst):
    assert dst.numel() == src.numel()
    dst.copy_(src)

  for i, v in enumerate(values):
    copy_tensor(torch.LongTensor(v), res[i][:len(v)])

  return res


def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
  meta = path_or_fp.split(":")
  if len(meta) == 3 and (meta[0].endswith(".wav") or meta[0].endswith(".flac")):
    path_or_fp = meta[0]
    start = int(meta[1])
    frames = int(meta[2])
  else:
    path_or_fp = path_or_fp

  if isinstance(path_or_fp, str):
    ext = Path(path_or_fp).suffix
    if ext in [".wav", ".flac", ".ogg", ".mp3"]:
      pass
    else:
      raise ValueError(f"Unsupported audio format: {ext}")

  try:
    import soundfile as sf
  except ImportError:
    raise ImportError(
        "Please install soundfile to load WAV/FLACC/OGG/MP3 audios")
  waveform, sample_rate = sf.read(
      path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start)
  waveform = waveform.T

  waveform, sample_rate = convert_waveform(
      waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
  if not normalization:
    waveform *= 2**15
  if not always_2d:
    waveform = waveform.squeeze(axis=0)
  return waveform


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
  """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
  try:
    import torchaudio.sox_effects as ta_sox
  except ImportError:
    raise ImportError("Please install torchaudio: pip install torchaudio")

  effects = []
  if normalize_volume:
    effects.append(["gain", "-n"])
  if to_sample_rate is not None and to_sample_rate != sample_rate:
    effects.append(["rate", f"{to_sample_rate}"])
  if to_mono and waveform.shape[0] > 1:
    effects.append(["channels", "1"])
  if len(effects) > 0:
    is_np_input = isinstance(waveform, np.ndarray)
    _waveform = torch.from_numpy(waveform) if is_np_input else waveform
    converted, converted_sample_rate = ta_sox.apply_effects_tensor(
        _waveform, sample_rate, effects)
    if is_np_input:
      converted = converted.numpy()
    return converted, converted_sample_rate
  return waveform, sample_rate


@dataclass
class SpeechTextPairedDataCollator:
  """
    Data collator that will dynamically pad the inputs received.
    """

  pad_id: int = 0
  sampling_rate: int = 16000
  extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()
  from_file: bool = False

  def __call__(self, samples: List[Dict]):
    input_ids = [sample["input_ids"] for sample in samples]
    attention_mask = [sample["attention_mask"] for sample in samples]
    labels = [sample["labels"] for sample in samples]
    suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
    suffix_attention_mask = [
        sample["suffix_attention_mask"] for sample in samples
    ]
    suffix_labels = [sample["suffix_labels"] for sample in samples]

    input_ids = collate_tokens(input_ids, self.pad_id)
    attention_mask = collate_tokens(attention_mask, 0)
    labels = collate_tokens(labels, -100)
    suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
    suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)
    suffix_labels = collate_tokens(suffix_labels, -100)

    if self.from_file:
      raw_speech = [
          get_waveform(
              sample["audio_path"], output_sample_rate=self.sampling_rate)
          for sample in samples
      ]
    else:
      raw_speech = [sample["raw_audio"] for sample in samples]
    speech_inputs = self.extractor(
        raw_speech,
        sampling_rate=self.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "suffix_input_ids": suffix_input_ids,
        "suffix_attention_mask": suffix_attention_mask,
        "suffix_labels": suffix_labels,
        "speech_values": speech_inputs.input_features,
        "speech_attention_mask": speech_inputs.attention_mask,
    }

