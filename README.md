# PicoGPT
Accompanying blog post: [GPT in 60 Lines of Numpy](https://jaykmody.com/blog/gpt-from-scratch/)

---

You've seen [openai/gpt-2](https://github.com/openai/gpt-2).

You've seen [karpathy/minGPT](https://github.com/karpathy/mingpt).

You've even seen [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)!

But have you seen [picoGPT](https://github.com/jaymody/picoGPT)??!?

`picoGPT` is an unnecessarily tiny and minimal implementation of [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in plain [NumPy](https://numpy.org). The entire forward pass code is [40 lines of code](https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py#L3-L41).

picoGPT features:
* Fast? ❌ Nah, picoGPT is megaSLOW 🐌
* Training code? ❌ Error, 4️⃣0️⃣4️⃣ not found
* Batch inference? ❌ picoGPT is civilized, single file line, one at a time only
* top-p sampling? ❌ top-k? ❌ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Readable? `gpt2.py` ✅ `gpt2_pico.py` ❌
* Smol??? ✅✅✅✅✅✅ YESS!!! TEENIE TINY in fact 🤏

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer, taken straight from their [gpt-2 repo](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `gpt2.py` contains the actual GPT model and generation code which we can run as a python script.
* `gpt2_pico.py` is the same as `gpt2.py`, but in even fewer lines of code. Why? Because why not 😎👍.

#### Dependencies
```bash
poetry install
```
Tested on `Python 3.11.7`.

#### Usage

If you're on MacOS you MUST use devcontainer in order to run the project.

```bash
python gpt2.py "Alan Turing theorized that computers would one day become"
```

Which generates

```
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

You can also control the number of tokens to generate, the model size (one of `["124M", "355M", "774M", "1558M"]`), and the directory to save the models:

```bash
python gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --model_size "124M" \
    --models_dir "models"
```

## Components

### Encoder

encoder is the `BPE tokenizer` used by GPT-2:
```python
ids = encoder.encode("Not all heroes wear capes.")
ids
# [3673, 477, 10281, 5806, 1451, 274, 13]

encoder.decode(ids)
# "Not all heroes wear capes."
```

Using the vocabulary of the tokenizer (stored in encoder.decoder), we can take a peek at what the actual tokens look like:
```python
[encoder.decoder[i] for i in ids]
# ['Not', 'Ġall', 'Ġheroes', 'Ġwear', 'Ġcap', 'es', '.']
```

Notice, sometimes our tokens are words (e.g. `Not`), sometimes they are words but with a space in front of them (e.g. `Ġall`, the **Ġ represents a space**), sometimes there are part of a word (e.g. capes is split into `Ġcap` and `es`), and sometimes they are punctuation (e.g. `.`).

One nice thing about BPE is that it can encode any arbitrary string. If it encounters something that is not present in the vocabulary, it just breaks it down into substrings it does understand:
```python
[encoder.decoder[i] for i in encoder.encode("zjqfl")]
['z', 'j', 'q', 'fl']
```

To get the size of the vocabulary:
```
len(encoder.decoder)
# 50257
```

The vocabulary, as well as the byte-pair merges which determines how strings are broken down, is obtained by training the tokenizer. When we load the tokenizer, we're loading the already trained vocab and byte-pair merges from some files, which were downloaded alongside the model files when we ran `load_encoder_hparams_and_params`. See `models/124M/encoder.json` (the vocabulary) and `models/124M/vocab.bpe` (byte-pair merges).

### Hyperparameters

`hparams` is a dictionary that contains the hyper-parameters of our model:

```python
hparams
{
  "n_vocab": 50257, # number of tokens in our vocabulary
  "n_ctx": 1024, # maximum possible sequence length of the input
  "n_embd": 768, # embedding dimension (determines the "width" of the network)
  "n_head": 12, # number of attention heads (n_embd must be divisible by n_head)
  "n_layer": 12 # number of layers (determines the "depth" of the network)
}
```

We'll use these symbols in our code's comments to show the underlying shape of things. We'll also use `n_seq` to denote the length of our input sequence (i.e. `n_seq = len(inputs)`).