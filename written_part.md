(1) In Assignment 4 we used 256-dimensional word embeddings (eword = 256),
while in this assignment, it turns out that a character embedding size of 50 suffices (echar = 50).
In 1-2 sentences, explain one reason why the embedding size used for character-level embeddings is
typically lower than that used for word embeddings.

A: Vocabulary size for word embedding is much higher to vocab size for characters, 
which implies dimensionality of char embeddings can be lower.

(2) Write down the total number of parameters in the character-based embedding
model (Figure 2), then do the same for the word-based lookup embedding model (Figure 1). Write
each answer as a single expression (though you may show working) in terms of echar, k, eword,
Vword (the size of the word-vocabulary in the lookup embedding model) and Vchar (the size of the
character-vocabulary in the character-based embedding model).

__Answer__

for character model: `echar*mword*k*Vchar`

for word model: `eword*k*Vword`

Character model has 256*50000/(96*20*50) ~ 150-200 times less parameters.


(3) In step 3 of the character-based embedding model, instead of using a 1D convnet, we could have used a RNN instead (e.g. feed the sequence of characters into a bidirectional
LSTM and combine the hidden states using max-pooling). Explain one advantage of using a convolutional architecture rather than a recurrent architecture for this purpose, making it clear how
the two contrast. Below is an example answer; you should give a similar level of detail and choose
a different advantage.

__Answer__

1D Convnet has small number of parameters, which is usually filter size by number of filters. By contrast, a RNN 
requires larger hidden state size to to learn details properly, making the overall model larger and slower to train.

(4)  In lectures we learned about both max-pooling and average-pooling. For each
pooling method, please explain one advantage in comparison to the other pooling method. For
each advantage, make it clear how the two contrast, and write to a similar level of detail as in the
example given in the previous question.

Max pooling is used to detect overall presence of specific "feature", while average pooling can measure 
average quantity of the feature. 