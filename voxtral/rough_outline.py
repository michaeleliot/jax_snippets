# Take in audio
# Convert that audio to tensor
# Input that tensor to the voxtral model
# Then take in a query text
# Convert that query text to a tensor
# Pass that into the model too
import jax.numpy as jnp
from datasets import load_dataset
from jax import device_get, jit
from transformers import WhisperProcessor

from whisper_jax import FlaxWhisperForConditionalGeneration
import jax
from flax import linen as nn

from typing import Callable

import jax
import jax.random as jrand
import torch
from transformers import AutoTokenizer, MistralForCausalLM

jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from mistral_v0_2.model.mistral_lm import convert_mistral_lm_params, shard_mistral_lm_params
from mistral_v0_2.lib.generate import generate

def answerQuestion(audio, queryText):
  ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
  sample = ds[0]["audio"]

  audioEmbedding = audioToText(sample)
  audioEmbedding = downSample(audioEmbedding)

  callModelAudio(audioEmbedding)
  output = callModelText(["what is the point of this text"])
  return output


def audioToText(sample):
  # load the processor and model
  processor = WhisperProcessor.from_pretrained("openai/whisper-base")
  model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
      "openai/whisper-base", _do_init=False
  )

  def generate_fn(input_features, params):
      pred_ids = model.generate(
          input_features, 
          task="transcribe", 
          return_timestamps=False, 
          max_length=model.config.max_length, 
          params=params,
      )
      return pred_ids.sequences

  # jit the generate function for speed
  p_generate = jit(generate_fn)

  # load a dummy sample from the LibriSpeech dataset


  # pre-process: convert the audio array to log-mel input features
  input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features

  # Always pass params when calling model.encode!
  output = model.encode(input_features=input_features, params=params)
  last_hidden_state = output.last_hidden_state

  # print(last_hidden_state)
  # print(last_hidden_state.shape)

  # # run the forward pass (JIT compiled the first time it is called)
  # pred_ids = p_generate(input_features, params)

  # # post-process: convert tokens ids to text string
  # transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)
  # print(transcription)
  return last_hidden_state


def downSample(audioEmbedding):
  class AudioAdapter(nn.Module):
        """
        Audio adapter module that downsamples audio embeddings using a 1D
        convolutional layer for a 4x temporal downsampling.
        """
        embedding_dim: int

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            """
            Performs the forward pass of the adapter.
            
            Args:
                x (jnp.ndarray): The input tensor from the audio encoder.
                                Expected shape: (batch_size, sequence_length, embedding_dim)
            
            Returns:
                jnp.ndarray: The downsampled output tensor.
                            Expected shape: (batch_size, new_sequence_length, embedding_dim)
            """
            # Flax Conv layer expects channel-last format (NLC), which the input
            # tensor (1, 1500, 512) already is, so no permutation is needed.
            downsampler = nn.Conv(
                features=self.embedding_dim,
                kernel_size=(4,),
                strides=(4,)
            )
            x = downsampler(x)
            return x

  # Example Usage:
  # Define the input tensor based on your specifications
  input_tensor = audioEmbedding

  # Generate a random key for parameter initialization
  key = jax.random.PRNGKey(0)

  # Instantiate the adapter with the specified embedding dimension
  embedding_dim = input_tensor.shape[2]
  adapter = AudioAdapter(embedding_dim=embedding_dim)

  # Initialize the model parameters
  params = adapter.init(key, input_tensor)['params']

  # Perform a forward pass
  output_tensor = adapter.apply({'params': params}, input_tensor)

  print(f"Original input tensor shape: {input_tensor.shape}")
  print(f"Downsampled output tensor shape: {output_tensor.shape}")


def callModelAudio(input_features):
    model_dir = 'mistral-hf-7B-v0.2'  # convert first with 'Mistral 7B v0.2 Parameter Conversion' part in README
    model = MistralForCausalLM.from_pretrained(model_dir)
    output = model(*input_features)

    return output


def callModelText(sentences):
    jax.distributed.initialize()
    model_dir = 'mistral-hf-7B-v0.2'  # convert first with 'Mistral 7B v0.2 Parameter Conversion' part in README
    model = MistralForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer.pad_token = tokenizer.eos_token

    if jax.local_device_count() == 8:
        # if it's V3-8, load on CPU first to avoid OOM
        cpu_device = jax.devices('cpu')[0]
        with jax.default_device(cpu_device):
            params = convert_mistral_lm_params(model)
    elif jax.local_device_count() == 4:
        # if it's V4-32
        params = convert_mistral_lm_params(model)
    params = shard_mistral_lm_params(params)

    max_new_tokens = 32
    max_length = 64
    key = jrand.key(42)
    key, subkey = jrand.split(key)

    output_ids = generate(params, tokenizer, sentences, max_length, max_new_tokens)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return output