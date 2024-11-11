from transformers import PretrainedConfig, LlamaConfig, WhisperConfig


class SpeechQEConfig(PretrainedConfig):

  def __init__(self,
               whisper_config=None,
               llama_config=None,
               conv_kernel_sizes="5,5,5",
               adapter_inner_dim=512,
               **kwargs):
    super().__init__(**kwargs)

    if whisper_config is None:
      whisper_config = {}

    if llama_config is None:
      llama_config = {}

    self.whisper_config = WhisperConfig(**whisper_config).to_dict()
    self.llama_config = LlamaConfig(**llama_config).to_dict()

    self.conv_kernel_sizes = conv_kernel_sizes
    self.adapter_inner_dim = adapter_inner_dim
