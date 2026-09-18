# build an LLM from scratch

a learning project for implementing a GPT-style large language model from first principles.

the project follows sebastian raschka's book, [*build a large language model (from scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) and its [official code repository](https://github.com/rasbt/LLMs-from-scratch).

## the roadmap

1. **set up the basics** -- install pytorch, learn tensors and automatic differentiation, and build a small training loop.

2. **prepare text data** -- load raw text, tokenize it with byte pair encoding (BPE), and convert tokens into token IDs.

3. **build the data loader** -- split token IDs into fixed-length input windows, shift each window by one token to create targets, and batch the pairs with a `Dataset` and `DataLoader`.

4. **create embeddings** -- learn a token embedding for every token ID and a positional embedding for every position, then add them together.

5. **implement self-attention** -- create queries, keys, and values; calculate scaled dot-product attention; and use the attention weights to build context vectors.

6. **make attention causal** -- mask future tokens so the model can only use earlier tokens when predicting the next one.

7. **use multi-head attention** -- run several attention heads in parallel, combine their context vectors, and project the result back into the embedding space.

8. **assemble a transformer block** -- add layer normalization, a feed-forward network, residual connections, and dropout around multi-head attention.

9. **build the GPT model** -- stack transformer blocks, add a final layer normalization, and use a linear output head to predict a probability for every vocabulary token.

10. **generate text** -- repeatedly feed the model its current context, select the next token from its output, and append that token to the sequence.

11. **pretrain on unlabeled text** -- train the model to predict the next token, measure training and validation loss, and save model checkpoints.

12. **fine-tune for classification** -- reuse the pretrained model, replace its output head, and train it to classify labeled text such as spam or not spam.

13. **fine-tune for instructions** -- format instruction-response examples, train the model to produce helpful responses, and evaluate its outputs.

## notebooks

- `00_pytorch_fundamentals.ipynb` -- pytorch foundations
- `01_pytorch_training_example.ipynb` -- a simple training loop
- `02_working_with_text_data.ipynb` -- tokenization and data loading
- `03_attention_mechanisms.ipynb` -- self-attention and multi-head attention
- `04_gpt_architecture.ipynb` -- the GPT model architecture
- `05_pretraining.ipynb` -- pretraining the model on text
- `06_classification_finetuning.ipynb` -- fine-tuning a pretrained model for text classification

## shared code

`notebooks/functions.py` contains reusable data-loading, attention, GPT-model, and text-generation code used by the notebooks.
