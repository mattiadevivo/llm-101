# GPT

Stands for **General Pre-trained Transformer**, it's a type of neural network architecture based on the [Transformer](https://arxiv.org/pdf/1706.03762.pdf).
TL;DR:

- **Generative**: A GPT generates text.
- **Pre-trained**: A GPT is trained on lots of text from books, the internet, etc ...
- **Transformer**: A GPT is a decoder-only transformer neural network.

## References

[GPT From scratch](https://jaykmody.com/blog/gpt-from-scratch/)
[How GPT3 works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

## How GPT works

A language model generates text, we can optionally provide some input which influences the output that is generated from what the model “learned” during its training period where it scanned vast amounts of text.

**Training** is the process of exposing the model to lots of text.

During the training we provide the model billions of text token and every time we provide a part of sentence as input and we make the LLM calculate a **label** (the next work of the sentence), if it's wrong we calculate the error in its prediction and update the model so next time it makes a better prediction. This process is repeated millions of times.

GPT3 usually generates output one token at a time.

The untrained model starts with random parameters. Training finds values that lead to better predictions.

GPT3 is 2048 tokens wide. That is its “context window”. That means it has 2048 tracks along which tokens are processed.

But, how the GPT model processes a word and produces another word? High level steps:

- Convert the word to a vector (list of numbers) representing the word
- Compute prediction
- Convert resulting vector to word

The important calculations of the GPT3 occur inside its stack of 96 transformer decoder layers. All these layers are the 'depth` in **deep learning**. Each of these layers has its own 1.8B parameter to make its calculations. That is where the “magic” happens. This is a high-level view of that process:

- every input word flows into the model and produces an output, one per time
- the last input produces the first valid output that is then used as next token input.

## What is a GPT

Large Language Models (LLMs) are just HPTs under the hood, what makes them special is they happen to have billions of parameters and are trained on lots of data.

Fundamentally, a GPT **generates text given a prompt**. Even with this very simple API (input = text, output = text), a well-trained GPT can do some pretty awesome stuff like write your emails, summarize a book, give you instagram caption ideas,

## Input/Output

The function signature for a GPT looks roughly like this:

```python
def gpt(inputs: list[int]) -> list[list[float]]:
    # inputs has shape [n_seq]
    # output has shape [n_seq, n_vocab]
    output = # beep boop neural network magic
    return output
```

The input is some text represented by a sequence of integers that map to tokens in the text:

```python
# integers represent tokens in our text, for example:
# text   = "not all heroes wear capes":
# tokens = "not"  "all" "heroes" "wear" "capes"
inputs =   [1,     0,    2,      4,     6]
```

Tokens are sub-pieces of the text, which are produced using some kind of **tokenizer**. We can map tokens to integers using a **vocabulary**:

```python
# the index of a token in the vocab represents the integer id for that token
# i.e. the integer id for "heroes" would be 2, since vocab[2] = "heroes"
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]

# a pretend tokenizer that tokenizes on whitespace
tokenizer = WhitespaceTokenizer(vocab)

# the encode() method converts a str -> list[int]
ids = tokenizer.encode("not all heroes wear") # ids = [1, 0, 2, 4]

# we can see what the actual tokens are via our vocab mapping
tokens = [tokenizer.vocab[i] for i in ids] # tokens = ["not", "all", "heroes", "wear"]

# the decode() method converts back a list[int] -> str
text = tokenizer.decode(ids) # text = "not all heroes wear"
```

In short:

- We have a string.
- We use a tokenizer to break it down into smaller pieces called tokens.
- We use a vocabulary to map those tokens to integers.

In practice, we use more advanced methods of tokenization than simply splitting by whitespace, such as Byte-Pair Encoding or WordPiece, but the principle is the same:

- There is a vocab that maps string tokens to integer indices
- There is an encode method that converts str -> list[int]
- There is a decode method that converts list[int] -> str[2]

**Output**

The output is a 2D array, where output[i][j] is the model's predicted probability that the token at vocab[j] is the next token inputs[i+1]. For example:

```python
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)
#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[0] =  [0.75    0.1     0.0       0.15    0.0   0.0    0.0  ]
# given just "not", the model predicts the word "all" with the highest probability

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[1] =  [0.0     0.0      0.8     0.1    0.0    0.0   0.1  ]
# given the sequence ["not", "all"], the model predicts the word "heroes" with the highest probability

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[-1] = [0.0     0.0     0.0     0.1     0.0    0.05  0.85  ]
# given the whole sequence ["not", "all", "heroes", "wear"], the model predicts the word "capes" with the highest probability
```

To get a next token prediction for the whole sequence, we simply take the token with the highest probability in `output[-1]`:

```python
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)
next_token_id = np.argmax(output[-1]) # next_token_id = 6
next_token = vocab[next_token_id] # next_token = "capes"
```

Taking the token with the highest probability as our prediction is known as greedy decoding or greedy sampling.

The task of predicting the next logical word in a sequence is called language modeling. As such, we can call a GPT a language model.

Generating a single word is cool and all, but what about entire sentences, paragraphs, etc ...?

## Generating text

### Autoregressive

We can generate full sentences by iteratively getting the next token prediction from our model. At each iteration, we append the predicted token back into the input:

```python
def generate(inputs, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate): # auto-regressive decode loop
        output = gpt(inputs) # model forward pass
        next_id = np.argmax(output[-1]) # greedy sampling
        inputs.append(int(next_id)) # append prediction to input
    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

input_ids = [1, 0] # "not" "all"
output_ids = generate(input_ids, 3) # output_ids = [2, 4, 6]
output_tokens = [vocab[i] for i in output_ids] # "heroes" "wear" "capes"
```

This process of predicting a future value (regression), and adding it back into the input (auto), is why you might see a GPT described as **autoregressive**.

### Sampling

We can introduce some stochasticity (randomness) to our generations by sampling from the probability distribution instead of being greedy:

```python
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # hats
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # pants
```

This allows us to generate different sentences given the same input. When combined with techniques like top-k, top-p, and temperature, which modify the distribution prior to sampling, the quality of our outputs is greatly increased. These techniques also introduce some hyperparameters that we can play around with to get different generation behaviors (for example, increasing temperature makes our model take more risks and thus be more "creative").

## Training

**Loss function**: Machine learning algorithms learn through different methods, but a fundamental component of the learning process of machine learning algorithms and models is the **loss function**. The loss function is a mathematical process that quantifies the error margin between a model's prediction and the actual target value.

We train a GPT like any other neural network, using gradient descent with respect to some loss function. In the case of a GPT, we take the cross entropy loss over the language modeling task:

```python
def lm_loss(inputs: list[int], params) -> float:
    # the labels y are just the input shifted 1 to the left
    #
    # inputs = [not,     all,   heros,   wear,   capes]
    #      x = [not,     all,   heroes,  wear]
    #      y = [all,  heroes,     wear,  capes]
    #
    # of course, we don't have a label for inputs[-1], so we exclude it from x
    #
    # as such, for N inputs, we have N - 1 langauge modeling example pairs
    x, y = inputs[:-1], inputs[1:]

    # forward pass
    # all the predicted next token probability distributions at each position
    output = gpt(x, params)

    # cross entropy loss
    # we take the average over all N-1 examples
    loss = np.mean(-np.log(output[y]))

    return loss

def train(texts: list[list[str]], params) -> float:
    for text in texts:
        inputs = tokenizer.encode(text)
        loss = lm_loss(inputs, params)
        gradients = compute_gradients_via_backpropagation(loss, params)
        params = gradient_descent_update_step(gradients, params)
    return params
```

During each iteration of the training loop:

1. We compute the language modeling loss for the given input text example
2. The loss determines our gradients, which we compute via backpropagation
3. We use the gradients to update our model parameters such that the loss is minimized (gradient descent)

Notice, we don't use explicitly labelled data. Instead, we are able to produce the input/label pairs from just the raw text itself. This is referred to as self-supervised learning.

Self-supervision enables us to massively scale train data, just get our hands on as much raw text as possible and throw it at the model. For example, GPT-3 was trained on 300 billion tokens of text from the internet and books.

Of course, you need a sufficiently large model to be able to learn from all this data, which is why GPT-3 has 175 billion parameters and probably cost between $1m-10m in compute cost to train.

This **self-supervised** training step is called **pre-training**, since we can reuse the "pre-trained" models weights to further train the model on downstream tasks, such as classifying if a tweet is toxic or not. **Pre-trained models** are also sometimes called **foundation models**.

Training the model on downstream tasks is called **fine-tuning**, since the model weights have already been pre-trained to understand language, it's just being fine-tuned to the specific task at hand.

The "pre-training on a general task + fine-tuning on a specific task" strategy is called **transfer learning**.

## Prompting

With GPT2 and GPT3 papers we realized that a GPT model pre-trained on enough data with enough parametes is capable of performing any arbitrary task **by itself** without fine-tuning, this is referred as **in-context learning**, because the model is using just the context of the prompt to perform the task.

Generating text given a prompt is also referred to as **conditional generation**, since our model is generating some output conditioned on some input.

GPTs are not limited to NLP tasks. You can condition the model on anything you want. For example, you can turn a GPT into a chatbot (i.e. **ChatGPT**) by conditioning it on the conversation history. You can also further condition the chatbot to behave a certain way by prepending the prompt with some kind of description.
