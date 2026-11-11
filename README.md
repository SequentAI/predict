# predict

`predict` is a premise selector for hammers: given the features of a goal, it
ranks the theorems of a library by how likely they are to help prove it. It
offers k-nearest neighbours (`knn`), naive Bayes (`nbayes`), a TF-IDF cosine
filter (`mepo`) and random forests (`rforest`).

## Provenance and licence

The sources were taken from CoqHammer's `src/predict/` directory,
https://github.com/lukaszcz/coqhammer, at commit
`27b80e8f2aa1a53d8d2a0642302736bad14c20ed`. The first commit of this
repository contains those files unmodified, with CoqHammer's `LICENSE`.

`predict` is by Łukasz Czajka and Cezary Kaliszyk and is distributed under the
GNU Lesser General Public License, version 2.1 (see `LICENSE`). Every change
after the first commit is ours and is distributed under the same licence.
