import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple


def list_field(default=None, metadata=None):
  return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
  """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

  llama_model: str = field(
      default="Unbabel/TowerInstruct-7B-v0.2",
      metadata={"help": "the path of base model"},
  )
  whisper_model: str = field(
      default='path/to/downloaded/whisper/whisper-large-v2',
      metadata={"help": "the path of downloaded whisper model"},
  )
  init_model: str = field(
      default=None,
      metadata={"help": "to initialize asr adaptor from a pre-trained model"},
  )
  cache_dir: str = field(
      default=None,
      metadata={
          "help":
              "Where do you want to store the pretrained models downloaded from huggingface.co"
      },
  )
  qe_model: str = field(
      default="Unbabel/XCOMET-XL",
      metadata={"help": "the path of Quality estimation model"},
  )
  peft: str = field(
      default=None,
      metadata={"help": "If True, train with qlora. If false, use lora"},
  )
  tokenizer: str = field(
      default=None, metadata={"help": "The name of the tokenizer"})

  fix_adaptor: bool = field(
      default=False,
      metadata={"help": "If True, fix the adaptor during lora training"},
  )


@dataclass
class DataTrainingArguments:
  """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

  dataset_name: str = field(
      default="mozilla-foundation/common_voice_6_1",
      metadata={
          "help":
              "The configuration name of the dataset to use (via the datasets library)."
      },
  )
  data_dir: str = field(
      default=None,
      metadata={"help": "Specify data_dir for load_datasets. e.g. covost2"},
  )
  dataset_config_name: str = field(
      default="tr",
      metadata={
          "help":
              "The configuration name of the dataset to use (via the datasets library)."
      },
  )
  token: str = field(
      default=None,
      metadata={
          "help": (
              "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
              "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
          )
      },
  )
  train_split_name: str = field(
      default="train",
      metadata={
          "help": (
              "The name of the training data set split to use (via the datasets library). Defaults to "
              "'train'")
      },
  )
  test_split_name: str = field(
      default="test",
      metadata={
          "help":
              "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
      },
  )
  eval_split_name: str = field(
      default="validation",
      metadata={
          "help":
              "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
      },
  )
  eval_sample_number: Optional[int] = field(
      default=100,
      metadata={
          "help":
              "not using all test data but only using the number of samples to eval during training"
      },
  )
  eval_metrics: List[str] = list_field(
      default=["wer"],
      metadata={
          "help":
              "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"
      },
  )
  audio_column_name: str = field(
      default="path",
      metadata={
          "help":
              "The name of the dataset column containing the audio data. Defaults to 'audio'"
      },
  )
  text_column_name: str = field(
      default="sentence",
      metadata={
          "help":
              "The name of the dataset column containing the text data. Defaults to 'text'"
      },
  )
  chars_to_ignore: Optional[List[str]] = list_field(
      default=None,
      metadata={"help": "A list of characters to remove from the transcripts."},
  )
  lower: Optional[bool] = field(
      default=True,
      metadata={"help": "Lowercase the transcripts."},
  )
  dataroot: str = field(
      default=None,
      metadata={"help": "the root to load dataset"},
  )
  base_audio_path: str = field(
      default=None,
      metadata={"help": "the base path to load audio"},
  )
  manifest_files: str = field(
      default=None,
      metadata={
          "help": "The name of the training unit text paired set split to use."
      },
  )
  instruction: str = field(
      default="Transcribe this audio",
      metadata={
          "help":
              "The text prefix instruction before speech input, default None"
      },
  )
  suffix_inst: str = field(
      default="",
      metadata={"help": "suffix after the audio and before the assistant"},
  )
  predifined_inst: Optional[bool] = field(
      default=True,
      metadata={
          "help": ("If True, use predefined instruction from manifest file."
                   "If False, use arg.instruction.")
      },
  )
  lid_in_inst: Optional[bool] = field(
      default=True,
      metadata={
          "help":
              "If True, indicate language in instruction. e.g. ~instructions~\nEnglish:"
      },
  )
  chatml_prompt_template: Optional[bool] = field(
      default=True,
      metadata={
          "help":
              "use chatml prompt template of <|im_start|>user~instructions~\nEnglish: <|im_end|>assistant"
      },
  )
  indicate_audio_in_lid: Optional[bool] = field(
      default=False,
      metadata={
          "help":
              "If True, indicate audio after language in instruction. e.g. ~instructions~\nEnglish audio:"
      },
  )
  preprocessing_num_workers: int = field(
      default=1,
      metadata={
          "help": "The number of processes to use for the preprocessing."
      },
  )
  max_seq_len: int = field(
      default=150,
      metadata={
          "help": "The number of processes to use for the preprocessing."
      },
  )
  load_from_cache_file: Optional[bool] = field(
      default=False,
      metadata={"help": "Overwrite the cached preprocessed datasets or not."},
  )
  label_all_input_ids: Optional[bool] = field(
      default=False,
      metadata={
          "help":
              "label only for the answers, if True, all inputs will be trained."
      },
  )
  speech_from_file_path: Optional[bool] = field(
      default=True,
      metadata={
          "help":
              "read speech data from the file path."
      },
  )


@dataclass
class GenerationArguments:
  speechqe_model: str = field(
      default="h-j-han/SpeechQE-TowerInstruct-7B-en2de",
      metadata={"help": "Path to the SpeechQE model"},
  )
  lang: str = field(
      default="en",
      metadata={
          "help": "language of the input text or speech. Default is 'en'"
      },
  )
  #   src_lang: str = field(
  #       default=None,
  #       metadata={
  #           "help": "language of the input text or speech. Default is 'en'"
  #       },
  #   )
  tgt_lang: str = field(
      default="en",
      metadata={
          "help": "language of the output text or speech. Default is 'en'"
      },
  )
  max_new_tokens: int = field(
      default=50, metadata={"help": "max new tokens for generation"})
  min_new_tokens: int = field(
      default=1, metadata={"help": "min new tokens for generation"})
  do_sample: bool = field(
      default=False,
      metadata={
          "help":
              "whether do sample. For ST task, we will use greedy search to ensure stable output"
      },
  )
  temperature: float = field(
      default=0.1, metadata={"help": "temperature for generation"})
  top_p: float = field(default=0.75, metadata={"help": "top_p for generation"})
  batch_size: int = field(default=1, metadata={"help": "batch_size"})
  output_file_name: str = field(
      default=None,
      metadata={"help": "name of the json file to be saved in the model repo."},
  )
  wn_wer: bool = field(
      default=False,
      metadata={
          "help":
              "whether to use Whisper normalizer for WER calculation. Default is False."
      },
  )
  eval_bleu: bool = field(
      default=False,
      metadata={
          "help":
              "For speech translation task, whether to use BLEU for evaluation. Default is False."
      },
  )
  dtype: str = field(
      default="bf16",
      metadata={"help": "dtype for generation"},
  )
  task: str = field(
      default=None,
      metadata={"help": "task, this is for text"},
  )
  output_repo: str = field(
      default=None,
      metadata={"help": "output repo normally llama_model but if specified"},
  )