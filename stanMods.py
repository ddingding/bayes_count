# stan model to infer difference in read ratio growths between mutants and wt synonymous protein sequences
diff_r_model = """
data{
    // synonymous count data
    int<lower=0> K; // number of synonymous mutants
    int<lower=0> c_pre[K]; // counts synonymous before selection
    int<lower=0> c_aft[K]; // counts synonymous after selection

    // mutant data.
    int<lower=0> Km; // number of observed aa mutants
    int<lower=0> N; // number of total codon observations
    int<lower=0> c_pre_m[N]; //all codon observations pre
    int<lower=0> c_aft_m[N];
    int s[N]; //amino acid mutant group number [1 1 2 2 2 3 3 3 3 ...]
}

parameters {

    // params for synonymous mutants
    real<lower=0> alpha; // gamma param for f_pre
    real<lower=0> beta; //  gamma param
    vector<lower=0>[K] f_pre; // define vector of length K
    real mu; // mean of growth rates of synonymous mutants
    real<lower=0> sigma; // variance of growth rates
    real<lower=0> r; //real r

    // params for mutants
    real<lower=0> alpha_m; // gamma params for f_pre mutant
    real<lower=0> beta_m;
    
    vector<lower=0>[N] f_pre_m;
    vector<lower=0>[Km] r_m; // aa mutant growth rates

}

transformed parameters {
    vector<lower=0>[K] f_aft; // for synonymous
    vector<lower=0>[N] f_aft_m; // poisson param after 
    f_aft = r * f_pre; //elementwise
    
    for (n in 1:N){
        //g = s[n];
        f_aft_m[n] = r_m[s[n]] * f_pre_m[n];
    } 

}

model {
    
    // for the mutant r_m
    mu ~ normal(1,1);
    sigma ~ uniform(0,10);
    r ~ normal(mu,sigma);

    // for the synonymous counts
    for (i in 1:K) {
        f_pre[i] ~ gamma(alpha, beta);
        c_pre[i] ~ poisson(f_pre[i]);
        c_aft[i] ~ poisson(f_aft[i]);
    }

    // mutant counts sample
    for (k in 1:Km){
        r_m[k] ~ normal(mu, sigma); 
    }
    
    // mutant observations sample
    for (n in 1:N){
        f_pre_m[n] ~ gamma(alpha_m, beta_m);
        c_pre_m[n] ~ poisson(f_pre_m[n]);
        c_aft_m[n] ~ poisson(f_aft_m[n]);
    }
    
}

generated quantities {
    // for PPC

    vector[Km] diff_r;
    
    //synonymous counts
    int<lower=0> c_pre_rep[K];
    int<lower=0> c_aft_rep[K];
    
    //mutant counts
    int<lower=0> c_pre_m_rep[N];
    int<lower=0> c_aft_m_rep[N];

    diff_r = r_m - r;

    // generate synonymous pre and post counts
    for (i in 1:K) {
        c_pre_rep[i] = poisson_rng(f_pre[i]);
        c_aft_rep[i] = poisson_rng(f_aft[i]);
    }
    
    for (i in 1:N) {
        c_pre_m_rep[i] = poisson_rng(f_pre_m[i]);
        c_aft_m_rep[i] = poisson_rng(f_aft_m[i]);
    }
}
"""
