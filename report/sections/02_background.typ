
= 2 Background

*Active units.* Following Burda et al. @burda2016iwae[p. 7], we call a latent unit _active_ when the variance over the data of its posterior mean is above 0.01. Approximating the averaged KL term, and assuming zero mean, Asperti derived that it is smallest when this variance and the mean posterior variance add up to one (Eq. 13 of @asperti2018sparsity), which Asperti and Trentin call the variance law @asperti2020balancing[p. 199442].

*The reconstruction–KL balance.* The β-VAE @higgins2017beta minimises
$ cal(L)_β = D + β R, $
where the distortion $D$ is the reconstruction error, on our states the binary cross-entropy summed over the facts, and the rate $R$ is the average Kullback–Leibler (KL) term @alemi2018broken. Their sum $D + R$ is the negative evidence lower bound (ELBO). At evaluation we decode the posterior mean and threshold each output at 0.5, so a model can reconstruct every state exactly even with a positive distortion.

In an experiment of Asperti, a latent variable collapsed when its gain in reconstruction became smaller than its KL divergence @asperti2019collapse[p. 6]. When the bits are independent, a decoder that ignores the code and outputs the frequency of each bit is already as good as any other, so at β = 1 a code with zero rate is optimal @chen2017vlae. Other optima can keep information, though, and training does not have to end in collapse @alemi2018broken. Copying a bit changes this, since a fair bit copied $c$ times costs such a decoder $c ln 2$ nats, while sending it perfectly through the code costs at least $ln 2$ of rate. Copies can therefore make informative codes worth their cost, but this does not tell us how many units become active, or whether each takes one bit.
