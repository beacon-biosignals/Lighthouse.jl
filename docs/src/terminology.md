# Terminology

* _sample_: a piece of data to be classified by the model, or a labelled piece of training/test/validation data.
* _classes_: the set of possible class labels which the model attempts to predict.
* _voters_: the individual sources of labelled data, such as human labellers. Each voter may supply a "vote" for a class label for a sample.
* _votes_: the matrix of votes corresponding to a set of data, whose rows correspond to the index of a sample in a set of data, whose columns correspond to voters, and whose values are the indices of class labels (i.e. numbers in `1:length(classes)`). E.g. if 2 voters have voted on ten samples, then `votes` is a 10 by 2 matrix of integers. If a voter has not voted on a particular sample, any value outside `1:length(classes)` may be supplied to indicate this.
* _elected class_: the class elected by the voters. By default in [`learn!`](@ref),
    the elected class is chosen by a simple majority of the votes with ties broken randomly.
* _predicted class_: the class predicted by the model for a given input.
