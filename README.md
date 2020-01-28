# bayes_count
See the bayes.ipynb for bayesian inference of protein function from high-throughput assays.

## Biological Motivation:
Understanding the effects of mutations on a protein sequence is not just a fundamental question in evolutionary biology but also at the core of therapeutic protein drug design. We have measured the effect of 1000s of protein sequence variants in order to learn about such mutation effects. We're trying to understand whether current pairwise abstractions of interacting proteins are useful in predicting the actual next 2 mutational steps across a protein interface.


## Motivation for performing Bayesian analysis:

**1. Interpretability**:

Inherently, we are interested in whether and how much a given mutation rescues the growth rate, compared to the wt sequence, which is the posterior Bayesian credible interval. The frequentist confidence interval would ask something along the lines of whether a particular mutant differs in it's growth rate (reject the null hypothesis or not).

A frequentist confidence interval is not a probability distribution over your parameter of interest. Instead it constructs an interval, which, under repeated sampling of datasets from your null distribution, would contain the true (fixed) parameter a particular fraction of times (with a confidence level 95%). This is different from what you're usually interested in, which is actually a Bayesian credible interval over your parameter (ie. of the posterior probability). Of course practically, these could be very similar.

 see for example: https://www.ncbi.nlm.nih.gov/pubmed/26620955
 
 and Andrew Gelman and Deborah Nolan's Book: teaching statistics: a Bag of tricks.
 nice viz of confidence intervals: https://seeing-theory.brown.edu/frequentist-inference/index.html#section2

**2.Philosophical**: satisfying the likelihood principle.

Frequentist null hypothesis significance testing asks the question: What is the probability of observing data at least as extreme as my observed data under a given null hypothesis, H0. The issues is that H0 influences all the possible datasets that I could've observed, which depends on my experimental design such as my stopping rule for data collection. This means there are multiple p-values for a given sequence, depending on H0.
    
A concrete example might be trying to assess whether an observed sequence of coin tosses (HTHTTTH) comes from a fair coin. Depending on whether the experiment was conducted until a certain number of coin flips were performed, or until a certain number of heads was observed, the space of possible observed coin toss sequences differs. The probability of a particular sequence in the former follows a binomial probability distribution, versus the latter a negative binomial distribution. These will give different p-values.
    David MacKay puts this weirdness clearly: Experimenter A who conducts the experiment inside of a room, should come to the same conclusion as experimenter B, who just watches the outcomes of the experiment through the window without knowing the experimental design.
   
 
**3. Practical**:

Bayesian modeling allows us to specify complex structures in the data generating process, and infer our parameters of interest by integrating out the parameters that we are not interested in. It also allows for hierarchical models that come with all the benefits of shrinkage and pooling of information across groups. See the classical example of estimating treatment effect among different small groups: https://statmodeling.stat.columbia.edu/2014/01/21/everything-need-know-bayesian-statistics-learned-eight-schools/
