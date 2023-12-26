# Idea
Given a token it predict next token before last layer.
For some tokens context of last 1-3 token is enough.

So we can merge these tokens.
Try to predict next/prev token with context of 1 token, then 2, 4, 8, 16, .... so on.
1 token predict next one token.
2 token predict next two tokens.
4 token predict next 4 tokens....


What is a token?
Objective is to predict next token, even at higher layers, where it has context of whole text.
Each token tries to keep information only needed to predict next word.
That is why even at higher level all token do not look almost same, because they are trying to be relevant at that point in text.

Often few initial token are sacrificed to be class token. Because
* They do not have enough previous token to predict next word, so anyway they are going to be loss making token, so use it to store other class level information.

Better would be.
Last layer output should be what is intent of remaining document.
And first layer out put would be just the context of current word, so next can be predicted.
when next is predicted we update most the first layer output and least the last layer output.


Better model would be that 


If it is predictable,  we can merge these tokens.