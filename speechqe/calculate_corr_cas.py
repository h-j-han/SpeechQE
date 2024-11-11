import os
import pickle
import random
import logging
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)
logger.disabled = True


def main():
  for asr_system in l_asr_system:
    tabstr = ""
    df = pd.DataFrame(columns=["qe", "metric"])
    for st_system in l_st_system:
      new_df = pd.DataFrame(columns=["qe", "metric"])
      # qe
      cas_file_name = f'cas.A.{asr_system}.T.{st_system}.{dataset_name.replace("/", "_")}.{config}.{split}'
      qe_repo = 'data/cas_speechqe' if dataset_name != "ACL" else 'data/acl'
      if "google" in qe_model:
        qe_file_name = f'qe.{qe_model.replace("/","_")}.{cas_file_name}.tsv'
        qe_path = os.path.join(qe_repo, qe_file_name)
        assert os.path.exists(qe_path), f"{qe_path} does not exist"
        # load tsv file
        qedf = pd.read_csv(qe_path, sep="\t")
        new_df["qe"] = qedf["score"] * -1
      elif "blaser" in qe_model:
        qe_file_name = f'qe-t2t.{qe_model.replace("/","_")}.{cas_file_name}.tsv'
        qe_path = os.path.join(qe_repo, qe_file_name)
        assert os.path.exists(qe_path), f"{qe_path} does not exist"
        # load tsv file
        qedf = pd.read_csv(qe_path, sep="\t")
        new_df["qe"] = qedf["score"]
      else:
        qe_file_name = f'qe.{qe_model.replace("/","_")}.{cas_file_name}.pkl'
        qe_path = os.path.join(qe_repo, qe_file_name)
        assert os.path.exists(qe_path), f"{qe_path} does not exist"
        with open(qe_path, "rb") as f:
          qe = pickle.load(f)
        logger.info(f"{qe_file_name=} {qe.system_score=}")
        new_df["qe"] = qe.scores

      if (metric_model == "Unbabel/wmt23-cometkiwi-da-xl" or
          metric_model == "Unbabel/wmt22-cometkiwi-da"):
        # metric
        metric_repo = 'data/acl'
        metric_file_name = f'qe.{metric_model.replace("/","_")}.st.{st_system}.{dataset_name.replace("/", "_")}.{config}.{split}.pkl'
        metric_path = os.path.join(metric_repo, metric_file_name)
        assert os.path.exists(metric_path), f"{metric_path} does not exist"
        with open(metric_path, "rb") as f:
          metric = pickle.load(f)
        logger.info(f"{metric_file_name=} {metric.system_score=}")
        new_df["metric"] = metric.scores
      elif "Unbabel" in metric_model:
        # metric
        metric_repo = f'data/metric'
        metric_file_name = f'metric.{metric_model.replace("/","_")}.st.{st_system}.{dataset_name.replace("/", "_")}.{config}.{split}.pkl'
        metric_path = os.path.join(metric_repo, metric_file_name)
        assert os.path.exists(metric_path), f"{metric_path} does not exist"
        with open(metric_path, "rb") as f:
          metric = pickle.load(f)
        logger.info(f"{metric_file_name=} {metric.system_score=}")
        new_df["metric"] = metric.scores
      elif (metric_model == "google/metricx-23-xl-v2p0" or
            metric_model == "google/metricx-23-xxl-v2p0"):
        # metric
        metric_repo = f'data/metric'
        metric_file_name = f'metric.{metric_model.replace("/","_")}.st.{st_system}.{dataset_name.replace("/", "_")}.{config}.{split}.tsv'
        metric_path = os.path.join(metric_repo, metric_file_name)
        assert os.path.exists(metric_path), f"{metric_path} does not exist"
        # load tsv
        metricdf = pd.read_csv(metric_path, sep="\t")
        # get average
        logger.info(f'{metric_file_name=} {metricdf["score"].mean()=}')
        new_df["metric"] = metricdf["score"] * -1
      elif "iwslt" in metric_model:
        metric_repo = f'data/acl'
        metric_file_name = f'st.{st_system}.{dataset_name.replace("/", "_")}.{config}.{split}.tsv'
        metric_path = os.path.join(metric_repo, metric_file_name)
        assert os.path.exists(metric_path), f"{metric_path} does not exist"
        # load tsv
        metricdf = pd.read_csv(metric_path, sep="\t")
        # get average
        logger.info(f'{metric_file_name=} {metricdf["raw"].mean()=}')
        new_df["metric"] = metricdf["raw"]
      else:
        raise ValueError(f"{metric_model=} not supported")
      # corr
      corr = new_df["metric"].corr(new_df["qe"], method=corr_methods)
      tabstr += str(corr) + " "
      logger.info(f" {len(new_df)=}, {corr=}")
      # cat
      df = pd.concat([df if not df.empty else None, new_df])
      logger.info(f" {len(df)=}")

    tabstr = f"{asr_system} " + str(df["metric"].corr(
        df["qe"], method=corr_methods))
    print(df["metric"].corr(df["qe"], method=corr_methods))


if __name__ == "__main__":
  for metric_model in [
      "Unbabel/XCOMET-XL",
      "google/metricx-23-xl-v2p0",
  ]:
    for qe_model in [
        "Unbabel/XCOMET-XL",
        "google/metricx-23-xl-v2p0",
        "blaser_2_0_qe",
    ]:
      split = "test0-500"
      config = "es_en"
      dataset_name = "covost2"
      corr_methods = "spearman"
      lang = config.split("_")[0]
      print(
          f"{corr_methods=},{qe_model=},{metric_model=},{dataset_name=},{config=} "
      )
      l_asr_system = [
          "gold",
          "openai_whisper-large-v3",
          # "openai_whisper-large-v2",
          # "openai_whisper-large",
          # "openai_whisper-medium",
          # "openai_whisper-small",
          # "openai_whisper-base",
          # # "openai_whisper-tiny",
      ]
      l_st_system = [
          "openai_whisper-large-v3",
          "openai_whisper-large-v2",
          # "openai_whisper-large",
          "openai_whisper-medium",
          "openai_whisper-small",
          "openai_whisper-base",
          # "openai_whisper-tiny",
      ]

      main()

  for metric_model in [
      "Unbabel/XCOMET-XL",
      "google/metricx-23-xl-v2p0",
  ]:
    print(f"{metric_model=}")
    for qe_model in [
        "Unbabel/XCOMET-XL",
        "google/metricx-23-xl-v2p0",
        "blaser_2_0_qe",
    ]:
      corr_methods = "spearman"
      split = "test0-500"
      config = "en_de"
      dataset_name = "covost2"
      lang = config.split("_")[0]
      print(
          f"{corr_methods=},{qe_model=},{metric_model=},{dataset_name=},{config=} "
      )
      l_asr_system = [
          "gold",
          "openai_whisper-large-v3",
          # "openai_whisper-large-v2",
          # "openai_whisper-large",
          # "openai_whisper-medium",
          # "openai_whisper-small",
          # "openai_whisper-base",
          # "openai_whisper-tiny",
      ]
      l_st_system = [
          # "facebook_seamless-m4t-v2-large",
          "facebook_hf-seamless-m4t-large",
          "facebook_hf-seamless-m4t-medium",
          "facebook_s2t-wav2vec2-large-en-de",
          "facebook_s2t-medium-mustc-multilingual-st",
          "facebook_s2t-small-mustc-en-de-st",
          # "facebook_s2t-small-covost2-en-de-st",
      ]
      main()

  for corr_methods in ["spearman"]:
    split = "test"
    config = "en-de"
    dataset_name = "ACL"
    lang = config.split("-")[0]
    for qe_model in [
        "Unbabel/XCOMET-XL",
        "Unbabel/XCOMET-XXL",
        "google/metricx-23-xl-v2p0",
        "google/metricx-23-xxl-v2p0",
        "Unbabel/wmt23-cometkiwi-da-xl",
        "Unbabel/wmt22-cometkiwi-da",
        "Unbabel/wmt20-comet-qe-da",
        # "blaser_2_0_qe",
    ]:
      print(f"{corr_methods=},{dataset_name=},{qe_model=}")
      metric_model = "iwslt"  # human DA label
      l_asr_system = [
          "gold",
          "iwslt23da",
      ]
      l_st_system = [
          "iwslt23da",
      ]

      main()
