## Missing meta learning
It is frequceny based learning (monkey see monkey do). We have optimisation like RLHF and DPO which are still example based, but solve frequency of examples problem. But models can not learn from other meta learning (other than putting it in context, maybe).

Like adding a very concise set of introductions (not examples) in training set will not make model a good story teller.
Or saying thou shall not kill. Will cover all scenarios of model harming human (in andorid). Instead we might need to train on examples of good and harmful behaviour, which always leaves way to hack by using examples on which it is not trained, and can not be generalised from trained examples. 

## Single token coherence
This is not true, I believe model has information to predict multiple tokens at some time with high confidence, we do not have way to use it.
Also there might be multiple higher coherence / topic which can be applied to keep prediction in a scope. Like being humorous/good/bad/coding/google search over next word prediction to choose from. This may result some common word lying in multiple category when one is appied.