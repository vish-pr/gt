# Design objective

We need to find representation of word/token in a vector.
"word is defined by its neighbour"
But any simple attempt to learn this will result in mode collapse, all words having same representation satisfy this.

## Duality
We can use duality to solve this.
Vector A: for a word superposition of it's neighbours
Vector B: Basis vector for a word which when combined make A, or from A gives how much it has B.

We can use surprise to learn A and B without leaking them. Like current GPT like language model.

# Design

Word have two neighbours.
F1(A) = A1      # left neighbour
F2(A) = A2     # right neighbour

A1 and A2 are initial guesses of neighbours superposition, they can be passed through two different encoder only causal (left for A1 and right for A2) transformers to get A1' and A2' which is more confident guess of neighbour.

We can force model to be more confident after each layer.

F3(B) = B1    # basis vector if it is left neighbour
F4(B) = B2   # basis vector if it is right neighbour


Currently transformer do a dot product of A and B, to get output distribution.
This is a big constraint on B vector. Forces superposition of basis vectors in A to be close together. For instance a word can be followed by two different words which are not semantically close in a particular case, this reduces model power to express this. 

To solve this we should do decomposition like Fourier transform to find basis vector. This will give model freedom to have A' be superposition of unrelated (far away in B) words.


# Personalize

## Lora layers
## Document embedding

To get embedding of a text, add token/s which can see whole text, but text can not see tokens, pass them through model, collect these extra token and pass them to model again.
This time all tokens can see extra tokens, but they can not see text. Now predicted text should be equal to actual text.

### To make a better decoder
We can use approach of paper: https://github.com/jxmorris12/vec2text/tree/c5988d6e7c361ad097ed41c4746d7a28699f0fb8

## Properties of these embeddings.

We can cluster them. We can move them while generating text in direction we want to generate (positive, critique, fun, hallucination.)