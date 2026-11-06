
= 1 Introduction

In this project we train variational autoencoders (VAEs) on the states of the planning domain Blocksworld @slaney2001blocks, to see whether the latent units they keep active reflect a structure of the data that we know in advance. A VAE with more latents than it needs can leave some of them inactive, namely their posterior mean hardly changes with the input. Burda et al. found far fewer _active units_ than latent dimensions, and adding dimensions did not raise their number @burda2016iwae. Asperti, for whom this sparsity is a form of self-regularisation, suggested that "for a given architecture, there seems to exist an intrinsic internal dimension of data" @asperti2018sparsity[p. 1]. Asperti and Trentin later found that the count also depends on the learning rate, and warned against reading it as the dimension of the data manifold @asperti2020balancing.

To see what the count reflects we need data whose structure is known before training. Disentanglement studies use synthetic images and their generative factors for this @higgins2017beta @locatello2019challenging, and we take Blocksworld instead, whose states are sets of binary facts tied by logical constraints, so that all the _legal states_ can be listed. The theory that links active units to the dimension of the data assumes continuous data @dai2019diagnosing @zheng2022manifold, and our states are discrete, so for us the question is empirical. We organise the report around the four questions of the project proposal.


- *Q1. Can the "right number" be fixed in advance?* Not uniquely, since the states can be described by anything from five dependent block variables to all of their binary facts, so before training we fixed several possible dimensions (Section 3).
- *Q2. How many variables stay active, and what does the count depend on?* At β = 1, a median of about half of the 32 latents in five of our six Blocksworld settings, and the count moves with the learning rate in one network, the encoding and redundancy in the input (Sections 5.1–5.4).
- *Q3. Does each active variable correspond to one fact?* Not one to one (Section 5.5).
- *Q4. Are generated states legal?* Often not, mostly because a block is not in exactly one place or a derived fact contradicts the positions (Section 5.6).

