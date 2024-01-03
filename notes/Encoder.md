# Language Encoder
## Byte Pair Encoding
There are lots of encode and they work at some characters at a time. This is obviously wrong when looking at examples
like unicode has un-... which is because un is common prefinx and has meaning but not for unicode code which should have uni as root.

So we need a character level encoder.


# Position Encoder
The word "he" needs to refer something in past. It may just behind it or far behind but most probably first male noun when looking backwards. And not anything forward.
But a simple position encoding will make it tough to differ between just behind and just forward.

So each token has search radius which can be sharp at token position else smooth.

It can also be programmatic like find first such occurence, etc.

Given a token we can calculate query of what it is looking and similarly what can look at it, key.
Maybe key is just information trimming to a category for a specific, so it can match better.

Then for each token we have search probability space by position vector.

So given a query we will have a space defined for it.

Properties
* symmetry: needed by some tokens and should not be there for other tokens, so overall optimising for symmetry is wrong.
* monotonically decreasing from token of interest. How it will decrease is function of token.
* Relative, translation invariance.



# Super Encoder

We can build an encode which can map to any other encoder.

Lets assume encoder + 1 layer as target of this character level encoder.

TODO
* Training data
* Get other models first layer
* execute it
* for this encoder + converter layer should be same.
* Have different converter layer for each target model.
* AIM: this converter layer should not add infomation
	* This is unclear what reforms vector and what adds information. TODO maths (maybe a linear transformation does not add information)


