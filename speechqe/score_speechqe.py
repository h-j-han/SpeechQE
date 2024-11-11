import os
import re
import sys
import json
import logging
import argparse
import langcodes
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, field, asdict

import torch
from datasets import load_dataset, Audio
from transformers import GenerationConfig, HfArgumentParser
from transformers import LlamaTokenizer, WhisperFeatureExtractor
from speechqe.models.modeling_speechqe import SpeechQEModel
from speechqe.arguments import ModelArguments, DataTrainingArguments, GenerationArguments
from speechqe.speech_text_paired_dataset import (
    get_waveform,
    collate_tokens,
    load_speech_text_paired_dataset,
)

logger = logging.getLogger(__name__)

generation_config = GenerationConfig(
    max_new_tokens=500,
    min_new_tokens=1,
    do_sample=False,
    temperature=0.1,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)


def main():
  parser = HfArgumentParser(
      (ModelArguments, DataTrainingArguments, GenerationArguments))
  model_args, data_args, gen_args = parser.parse_args_into_dataclasses()
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      handlers=[logging.StreamHandler(sys.stdout)],
  )
  logger.setLevel(logging.INFO)

  logger.info(f"Generation parameters {gen_args}")
  logger.info(f"Model parameters {model_args}")
  logger.info(f"Dataset parameters {data_args}")

  torch_dtype = None
  if gen_args.dtype == 'bf16':
    torch_dtype = torch.bfloat16
  elif gen_args.dtype == 'fp16':
    torch_dtype = torch.float16
  elif gen_args.dtype == 'fp32':
    torch_dtype = torch.float32

  tokenizer = LlamaTokenizer.from_pretrained(gen_args.speechqe_model)
  extractor = WhisperFeatureExtractor.from_pretrained(gen_args.speechqe_model)

  pad_token_id = 0 if "<pad>" not in tokenizer.get_vocab(
  ) else tokenizer.pad_token_id
  generation_config.update(
      **{
          "max_new_tokens": gen_args.max_new_tokens,
          "min_new_tokens": gen_args.min_new_tokens,
          "do_sample": gen_args.do_sample,
          "temperature": gen_args.temperature,
          "top_p": gen_args.top_p,
          "pad_token_id": pad_token_id,
          "bos_token_id": tokenizer.bos_token_id,
          "eos_token_id": tokenizer.eos_token_id,
      })

  batch_size = gen_args.batch_size
  audio_column_name = data_args.audio_column_name
  text_column_name = data_args.text_column_name
  sampling_rate = extractor.sampling_rate
  chars_to_ignore_regex = (f'[{"".join(data_args.chars_to_ignore)}]'
                           if data_args.chars_to_ignore is not None else None
                          )  # chars_to_ignore_regex = '[,?.!\-\;\:"“%‘”�—’…–]'
  lower = data_args.lower
  instruction = data_args.instruction
  lid_in_inst = data_args.lid_in_inst

  raw_data = load_speech_text_paired_dataset(
      dataroot=data_args.dataroot,
      manifest_files=data_args.manifest_files,
      tokenizer=tokenizer,
      instruction=data_args.instruction,
      dataset_name=data_args.dataset_name,
      dataset_config_name=data_args.dataset_config_name,
      split_name=data_args.test_split_name,
      token=data_args.token,
      audio_column_name=data_args.audio_column_name,
      text_column_name=data_args.text_column_name,
      chars_to_ignore_regex=chars_to_ignore_regex,
      load_from_cache_file=data_args.load_from_cache_file,
      num_proc=data_args.preprocessing_num_workers,
      sampling_rate=extractor.sampling_rate,
      from_file=data_args.speech_from_file_path,
      lower=data_args.lower,
      predifined_inst=data_args.predifined_inst,
      lid_in_inst=data_args.lid_in_inst,
      chatml_prompt_template=data_args.chatml_prompt_template,
      indicate_audio_in_lid=data_args.indicate_audio_in_lid,
      base_audio_path=data_args.base_audio_path,
      training=False,
  )
  print(f'{len(raw_data)=}')
  assert len(raw_data), "No data found in the test set"

  model = SpeechQEModel.from_pretrained(
      gen_args.speechqe_model, torch_dtype=torch_dtype, device_map="cuda")
  model.eval()

  def map_to_pred(samples):
    input_ids = samples["input_ids"]
    # print(tokenizer.decode(input_ids[0]))
    input_ids = collate_tokens(input_ids, pad_token_id).cuda()
    suffix_input_ids = samples["suffix_input_ids"]
    # print(tokenizer.decode(suffix_input_ids[0]))
    suffix_input_ids = collate_tokens(suffix_input_ids, pad_token_id).cuda()
    if data_args.speech_from_file_path:
      raw_speech = [
          get_waveform(sample, output_sample_rate=sampling_rate)
          for sample in samples["audio_path"]
      ]
    else:
      raw_speech = samples["speech"]
    speech_inputs = extractor(
        raw_speech,
        sampling_rate=extractor.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )
    speech_values = speech_inputs.input_features.cuda().to(
        dtype=model.dtype)  # only match dtype in embedding
    speech_attention_mask = speech_inputs.attention_mask.cuda()
    output = model.generate(
        input_ids=input_ids,
        suffix_input_ids=suffix_input_ids,
        speech_values=speech_values,
        speech_attention_mask=speech_attention_mask,
        generation_config=generation_config,
    )
    samples["pred_strings"] = tokenizer.batch_decode(
        output, skip_special_tokens=True)
    # print(f'Pred  raw: {samples["pred_strings"][0]}')
    # print(f'Refe  raw: {samples["target_text"][0]}')
    return samples

  remove_columns = None if data_args.speech_from_file_path else ["speech"]
  result = raw_data.map(
      map_to_pred,
      batched=True,
      batch_size=batch_size,
      remove_columns=remove_columns)
  df = pd.DataFrame(result)
  # save df to tsv
  output_repo = "outputs"
  if data_args.manifest_files is not None:
    output_file = f'speechqe.{data_args.manifest_files}'
  else:
    output_file = f'speechqe.{data_args.test_split_name}-{data_args.dataset_config_name}.tsv'
  output_path = os.path.join(output_repo, output_file)
  print(f"saved to {output_path}")
  df['pred_strings'].to_csv(output_path, sep="\t", index=False)
  # make df['pred_strings'] float
  num_out_of_format = 0
  for i, row in df.iterrows():
    try:
      float(row['pred_strings'])
    except ValueError:
      num_out_of_format += 1
      print(f"{i}th row['pred_strings']: {row['pred_strings']}")
  if num_out_of_format > 0:
    # manual post processing for "100%" output example
    df['pred_strings'] = df['pred_strings'].apply(
        lambda x: str(float(x.replace("%", "", 1)) / 100.0)
        if x.replace("%", "", 1).isdigit() else x)
    df['pred_strings'] = df['pred_strings'].astype(str)  #just make sure all str
    # if pred_string cannot be converted to  float, then make it zero
    df['pred_strings_post'] = df['pred_strings'].apply(
        lambda x: float(x) if x.strip().replace(".", "", 1).isdigit() else 0)
    # now we have all float format
  else:
    df['pred_strings_post'] = df['pred_strings']
  df['pred_strings_post'] = df['pred_strings_post'].astype(float)
  # get corrlation betwee score and pred_strings in df
  if "metric_score_xcomet-xl" in df.columns:
    corr1 = df['pred_strings_post'].corr(
        df['metric_score_xcomet-xl'], method="spearman")
  if "metric_score_metricx-23-xl" in df.columns:
    corr2 = df['pred_strings_post'].corr(
        df['metric_score_metricx-23-xl'], method="spearman")
  if "humanda" in df.columns:
    corr3 = df['pred_strings_post'].corr(df['humanda'], method="spearman")
  if gen_args.output_file_name is not None:
    output_file_name = gen_args.output_file_name
  else:
    current_time = datetime.now().strftime("%y%m%d%H%M")
    output_file_name = output_file.replace(".tsv", "")
  output_json = {
      "corr_xcomet":
          None if "metric_score_xcomet-xl" not in df.columns else corr1,
      "corr_metricx":
          None if "metric_score_metricx-23-xl" not in df.columns else corr2 *
          -1.0,
      "corr_humanda":
          None if "humanda" not in df.columns else corr3,
      "prediction":
          result["pred_strings"],
      "pred_strings_float":
          df['pred_strings_post'].tolist(),
      "metric_score_xcomet-xl":
          result["metric_score_xcomet-xl"]
          if "metric_score_xcomet-xl" in df.columns else None,
      "metric_score_metricx-23-xl":
          result["metric_score_metricx-23-xl"]
          if "metric_score_metricx-23-xl" in df.columns else None,
      "humanda":
          result["humanda"] if "humanda" in df.columns else None,
      "gen_args":
          asdict(gen_args),
      "generation_config":
          generation_config.to_dict(),
      "model_args":
          asdict(model_args),
      "data_args":
          asdict(data_args),
  }
  with open(os.path.join(output_repo, f"{output_file_name}.json"), "w") as f:
    json.dump(output_json, f)


if __name__ == "__main__":
  main()
