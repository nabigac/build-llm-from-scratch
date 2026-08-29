# build an LLM from scratch

a learning project for implementing a gpt-style large language model from first principles.

the project follows and is based on sebastian raschka's book, *build a large language model (from scratch)*.

high-level process: 
1. **load raw text** -- read the training data into one python string
2. **tokenize the text** -- split the text into tokens using bpe tokenizer
3. **convert tokens into token IDs** -- map every token to an integer from the tokenizer's vocabulary

4. **create input windows** -- select fixed-length sequences of the token IDs for the model's inputs
5. **create target windows** -- shift each input window one token forward to create expected next-token answers
6. **store windows in a Dataset** -- make input-target pairs accessible by index
7. **load batches with a DataLoader** -- group several input-target pairs into batches

8. **look up token embeddings** -- convert each token ID into a learned numerical vector
9. **look up positional embeddings** -- create a vector representing each token's position in the sequence
10. **combine the embeddings** -- add token embeddings and position embeddings together to form input embeddings

11. **create queries** -- multiply the input embeddings by the query weight matrix
12. **create keys** -- multiply the input embeddings by the key weight matrix
13. **create values** -- multiply the input embeddings by the value weight matrix

14. **calculate attention scores** -- compare every query with every key using matrix multiplication
15. **scale the attention scores** -- divide by the square root of the key dimension
16. **calculate attention weights** -- apply softmax so every row contains positive values that sum to 1
17. **calculate context vectors** -- use the attention weights to combine the value vectors    
