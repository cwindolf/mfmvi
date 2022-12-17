import numpy as np
import torch
from scipy.stats import poisson
import torch.distributions
from . import formulas as rx


log2pi = torch.log(torch.tensor(2.0 * torch.pi))

# from UDN
PositivePoisson = lambda p: torch.distributions.TransformedDistribution(
    torch.distributions.Poisson(p, validate_args=False),
    torch.distributions.AffineTransform(1, 1),
)


# -- CAVI specific formulas


def qK_(lambd, alphatK, NtK, N, alpha0, delta=0.95):
    assert alphatK.shape == NtK.shape
    m_old, m_ = alphatK.shape
    assert m_old == m_

    # compute what the beginning of qK would be proportional to
    digamma_alphatK_Ntk = torch.digamma(alphatK) * NtK
    # this is the K factorial part of the Poisson. turns into sum of log.
    triu_log_t = torch.triu(
        torch.log(torch.arange(1, m_old + 1))[:, None] * torch.ones(m_old)
    )
    sum_term = torch.sum(digamma_alphatK_Ntk - triu_log_t, axis=0)
    log_qK = torch.arange(1, m_old + 1) * torch.log(lambd) + sum_term

    # compute m_delta. we work on the log scale.
    po = poisson(lambd)
    for m in range(m_old):
        # need to bound here with log!
        log_po_tail = torch.tensor(po.logsf(m))
        tail_prob_log = N * torch.digamma(alpha0 + N) + log_po_tail
        head_prob_log = torch.logsumexp(log_qK[:m])
        log_Zm = torch.logsumexp(torch.tensor([head_prob_log, tail_prob_log]))
        tail_prob_qm_log = tail_prob_log - log_Zm
        if tail_prob_qm_log < torch.log(torch.tensor(1 - delta)):
            log_qK = log_qK[:m]
            break
    else:
        # we never got there. let's extend using the prior.
        pad_q = []
        while True:
            m += 1
            log_po_tail = torch.log(torch.tensor(po.sf(m)))
            tail_prob_log = N * torch.digamma(alpha0 + N) + log_po_tail
            new_logpmf = po.logpmf(m)
            pad_q.append()
            head_prob_log = torch.logsumexp(
                torch.tensor([head_prob_log, new_logpmf])
            )
            log_Zm = torch.logsumexp(
                torch.tensor([head_prob_log, tail_prob_log])
            )
            tail_prob_qm_log = tail_prob_log - log_Zm
            if tail_prob_qm_log < torch.log(torch.tensor(1 - delta)):
                break

        # build the final qK
        log_qK = torch.cat([log_qK, torch.tensor(pad_q)])

    log_qK = log_qK - torch.logsumexp(log_qK)
    print(f"{log_qK.shape=}")

    return torch.exp(log_qK), log_qK


# -- optimization steps


def E_step(
    N,
    lambd,
    alpha0,
    beta0,
    m0p,
    betak,
    nut,
    mkp,
    Wkpp,
    alphatK,
    qK,
    Xnp,
    delta=0.95,
):
    mahalnk = rx.mahalnk_(nut, Xnp, mkp, Wkpp)
    logLamTildek = rx.logLamTildek_(nut, Wkpp)
    Lnt = rx.Lnt_(beta0, m0p, betak, nut, Wkpp, Xnp, logLamTildek, mkp, mahalnk)
    rntK, rnt, Nt, NtK = rx.NtK_rnt_Nt_(qK, Lnt, alphatK)
    alphatK = rx.alphatK_(alpha0, NtK)
    qK, log_qK = qK_(lambd, alphatK, NtK, N, alpha0, delta=delta)
    return rnt, Nt, qK, log_qK, logLamTildek, mahalnk, rntK


def M_step(beta0, nu0, m0p, W0inv, rnk, Nk, Xnp):
    Nxbarkp, xbarkp = rx.xbarkp_(rnk, Nk, Xnp)
    NSkpp, Skpp = rx.Skpp_(rnk, Nk, Xnp, xbarkp=None)
    betak = rx.betak_(beta0, Nk)
    nut = rx.nut_(nu0, Nk)
    mkp = rx.mkp_(beta0, m0p, betak, Nk, Nxbarkp)
    Wkpp, Winvkpp = rx.Wkpp_(nu0, beta0, m0p, Nk, xbarkp, NSkpp, W0inv)
    return mkp, betak, Wkpp, nut, Skpp


def elbo_(
    beta0,
    nu0,
    m0p,
    W0inv,
    qK,
    pk,
    log_qK,
    Nk,
    logLamTildek,
    nut,
    betak,
    Skpp,
    Wkpp,
    mahalnk,
    alpha0,
    alphatK,
    rntK,
    mkp,
):
    p = Wkpp.shape[2]

    # likelihood term
    E_lnpXZmuLambda = torch.sum(
        Nk
        * (
            # K
            logLamTildek
            # K
            - p / betak
            - nut
            * (
                torch.einsum(
                    "kii->k", torch.einsum("kde,kef->kdf", Skpp, Wkpp)
                )
                + mahalnk
            )
            - p * log2pi
        )
    )

    # q/p(K) dkl
    log_pk = pk.log_prob(torch.arange(qK.shape[0]))
    Dkl_K = torch.sum(qK * (log_pk - log_qK))

    # KL terms for Y
    _1tK = torch.ones_like(alphatK)
    Y_p = torch.distributions.Gamma(alpha0 * _1tK, _1tK)
    Y_q = torch.distributions.Gamma(torch.triu(alphatK), _1tK, validate_args=False)
    Y_kltk = torch.distributions.kl_divergence(Y_q, Y_p)
    Y_kl = torch.einsum("tk,k->", Y_kltk, qK)

    # KL for Z
    alpha_hat = alphatK.sum(axis=0)
    dlogpq = (
        torch.digamma(alphatK)
        - torch.digamma(alpha_hat)[None, :]
        - torch.log(rntK)
    )
    # this guy has lower triangle of bad stuff
    dlogpq = torch.triu(dlogpq)
    Zkl = torch.einsum("k,ntk,ntk->", qK, rntK, dlogpq)

    # kl for mu/Lambda
    dm = mkp - m0p
    ElogpmuLambda = torch.sum(
        0.5 * (nu0 - p) * logLamTildek
        - 0.5 * p * beta0 / betak
        - 0.5 * beta0 * nut * torch.einsum("kp,kpr,kr->k", dm, Wkpp, dm)
        - 0.5 * nut * torch.einsum("kii->k", torch.einsum("de,kef->kdf", W0inv, Wkpp))
    )
    ElogqmuLambda = torch.sum(
        torch.distributions.Wishart(nut, precision_matrix=Wkpp).entropy().sum()
        - 0.5 * logLamTildek
        - 0.5 * p * torch.log(betak / (2. * torch.pi))
    )
    muLambdakl = ElogpmuLambda - ElogqmuLambda

    elbo = E_lnpXZmuLambda + Dkl_K + Y_kl + Zkl + muLambdakl

    return elbo


def cavi(lambd, alpha0, beta0, m0p, nu0, W0inv, Xnp, delta=0.95, n_steps=20):
    N, p = Xnp.shape

    pK = PositivePoisson(lambd)

    # initialize qK
    m_delta = 1 + poisson(lambd).ppf(delta)
    qK = torch.tensor(poisson(lambd).pmf(np.arange(m_delta)))

    betak = beta0 * torch.ones(m_delta)
    mkp = torch.normal(torch.zeros(m_delta, p), 1)
    Wkpp = torch.linalg.inv(W0inv)[None, :, :] * torch.ones(m_delta)[:, None, None]
    alphatK = alpha0 * torch.triu(torch.ones(m_delta, m_delta))
    betak = beta0 * torch.ones(m_delta)
    nut = nu0 * torch.ones(m_delta)

    elbos = []
    for _ in range(n_steps):
        rnt, Nt, qK, log_qK, logLamTildek, mahalnk, rntK = E_step(
            N,
            lambd,
            alpha0,
            beta0,
            m0p,
            betak,
            nut,
            mkp,
            Wkpp,
            alphatK,
            qK,
            Xnp,
            delta=delta,
        )
        mkp, betak, Wkpp, nut, Skpp = M_step(beta0, nu0, m0p, W0inv, rnt, Nt, Xnp)
        elbo = elbo_(
            beta0,
            nu0,
            m0p,
            W0inv,
            qK,
            pK,
            log_qK,
            Nt,
            logLamTildek,
            nut,
            betak,
            Skpp,
            Wkpp,
            mahalnk,
            alpha0,
            alphatK,
            rntK,
            mkp,
        )
        elbos.append(elbo)

    return elbos
