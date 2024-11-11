# [SpeechQE: Estimating the Quality of Direct Speech Translation](https://aclanthology.org/2024.emnlp-main.1218) at EMNLP2024  
We formulate the task of quality estimation for speech translation (SpeechQE), construct a benchmark, and evaluate a family of systems based on cascaded and end-to-end architectures.

We provide our E2E model on Huggingface Hub.
The provided models corresponds to "TowerInstruct-LoRA+Adapter-pt-Fixed" in the paper.
|SpeechQE for | E2E Model | Trained Domain
|---|---|---|
|English-to-German Speech Translation |[h-j-han/SpeechQE-TowerInstruct-7B-en2de](https://huggingface.co/h-j-han/SpeechQE-TowerInstruct-7B-en2de)| CoVoST2|
|Spanish-to-English Speech Translation  |[h-j-han/SpeechQE-TowerInstruct-7B-es2en](https://huggingface.co/h-j-han/SpeechQE-TowerInstruct-7B-es2en)|CoVoST2|

## Benchmarks and Training Corpus for SpeechQE
In [SpeechQE-CoVoST2](https://huggingface.co/datasets/h-j-han/SpeechQE-CoVoST2), 
we subsample about 80k segments from the training set and 500 from the dev and test of [facebook/covost2](https://huggingface.co/datasets/facebook/covost2), then run seven different direct ST models to generate the ST hypotheses. So, the `test` split consists of 3500 instances(500*7). We also provide splits for each translation model.


## Environment Setup
```bash
$ conda create -n speechqe Python=3.11 pytorch=2.0.1  pytorch-cuda=11.7 torchvision torchaudio -c pytorch -c nvidia
$ conda activate speechqe
$ pip install -r requirements.txt
```

## Cascaded SpeechQE
We use [Unbabel/XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) and [google/metricx-23-xl-v2p0](https://huggingface.co/google/metricx-23-xl-v2p0) for cascaded SpeechQE systems.
ASR system we mainly report in the cascaded system is [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).

We provide all the result data in `data` folder, where `data/cas_speechqe` is results of cascaded SpeechQE with the input of \[audio and ST hypothesis\] while `data/metric` is the automatic quality labels with the input of \[gold transcription, gold reference, ST hypothesis\].
We also provide a code to calculate the correlations of cascaded SpeechQE systems.
```bash
$ python speechqe/calculate_corr_cas.py
```

## End-to-End SpeechQE
The model we provide is trained with two phase steps.
First step is to train the model in ST and ASR tasks, only updating the adapter.
Second step is to train SpeechQE task while we fix the pre-trained adapter in the previous step and LoRA fine-tuning the TowerInstruct.
### Download Common Voice
Download the audio data from Common Voice.
Here, we use [mozilla-foundation/common_voice_4_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_4_0).
```python
import datasets
cv4en = datasets.load_dataset(
    "mozilla-foundation/common_voice_4_0", "en", cache_dir='path/to/cv4/download',
)
```
### Train
The training code and training corpus will be provided later. However, if you want those quickly, please do not hesitate to ping me (hjhan@umd.edu)!

### Eval
We provide SpeechQE benchmark: [h-j-han/SpeechQE-CoVoST2](https://huggingface.co/datasets/h-j-han/SpeechQE-CoVoST2).
BASE_AUDIO_PATH is the path of downloaded Common Voice dataset.
Please refer to `./scripts/eval_mt.sh` for full commands.
```bash
$ python speechqe/score_speechqe.py \
    --speechqe_model=h-j-han/SpeechQE-TowerInstruct-7B-en2de \
    --dataset_name=h-j-han/SpeechQE-CoVoST2 \
    --base_audio_path=$BASE_AUDIO_PATH \
    --dataset_config_name=en2de \
    --test_split_name=test_seamlar
 # for simple test run
```
or 
```bash
$ ./scripts/score_spechqe.sh
```

## SpeechQE Correlation with Human Direct Assessment Score
We compare the output quality scores from SpeechQE systems with human direct assessment (DA) scores from the IWSLT-ACL test set from [IWSLT/da2023](https://huggingface.co/datasets/IWSLT/da2023).


```bash
$ python speechqe/score_speechqe.py \
    --dataroot=data/acl \
    --manifest_files=test_ACL-iwslt23da-humandasc-en2de_fixedinst.tsv \
    --speechqe_model=h-j-han/SpeechQE-TowerInstruct-7B-en2de
```


For cascaded, we use the ASR output provided by [Salesky et al.(2023)](https://aclanthology.org/2023.iwslt-1.2/).
We tried Whisper ASR systems, but the output quality was not acceptable, likely due to the IWSLT23-ACL set being out-of-domain and covering highly technical NLP topics. The ASR provided is Azure API speech-to-text service, which we believe performs comparably to SOTA ASR models.
The result of cascaded system on IWSLT-ACL test set and related data can be found in `data/acl`.


## Reference
Please find details in the [ACL](https://aclanthology.org/2024.emnlp-main.1218) paper or [arXiv](https://arxiv.org/abs/2410.21485) paper.
```
@inproceedings{han-etal-2024-speechqe,
    title = "{S}peech{QE}: Estimating the Quality of Direct Speech Translation",
    author = "Han, HyoJung  and
      Duh, Kevin  and
      Carpuat, Marine",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1218",
    pages = "21852--21867",
    abstract = "Recent advances in automatic quality estimation for machine translation have exclusively focused on written language, leaving the speech modality underexplored. In this work, we formulate the task of quality estimation for speech translation (SpeechQE), construct a benchmark, and evaluate a family of systems based on cascaded and end-to-end architectures. In this process, we introduce a novel end-to-end system leveraging pre-trained text LLM. Results suggest that end-to-end approaches are better suited to estimating the quality of direct speech translation than using quality estimation systems designed for text in cascaded systems. More broadly, we argue that quality estimation of speech translation needs to be studied as a separate problem from that of text, and release our [data and models](https://github.com/h-j-han/SpeechQE) to guide further research in this space.",
}
```
