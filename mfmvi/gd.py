import torch.distributions
from torch import nn
from torch.nn.functional import softplus
from scipy.stats import poisson
import matplotlib.pyplot as plt
import time

from . import formulas as rx
from .cavi import elbo_ as cavi_elbo_, PositivePoisson


def softplus_inverse(x):
    """log(exp(x) - 1)"""
    return torch.where(x > 10, x, x.expm1().log())


class TruncatedPoisson(nn.Module):
    """
    Variational posterior approximation q(L) which is a Truncated Poisson.
    Used to adapt the depth during training.
    Parameters
    -------
    initial_nu_L: float, default 2.0
        Initial value of the variational parameter nu_L. nu_L is almost equal to the
        mean of the TruncatedPoisson, so the defaults 2.0 starts with two layers.
    truncation_quantile: float (between 0.0 and 1.0), default 0.95
        Truncation level of the Truncated Poisson, recommended to leave at 0.95
    """

    def __init__(
        self, initial_nu_L: float = 2.0, truncation_quantile: float = 0.95
    ):
        super().__init__()
        self.truncation_quantile = truncation_quantile
        self._nu_L = nn.Parameter(
            softplus_inverse(torch.tensor(float(initial_nu_L)))
        )

    @property
    def nu_L(self):
        """Returns the variational parameter nu_L, which is reparametrized to be positive."""
        return nn.Softplus()(self._nu_L)

    def compute_depth(self):
        p = poisson(self.nu_L.item())
        for a in range(int(self.nu_L.item()) + 1, 10000):
            if p.cdf(a) >= self.truncation_quantile:
                return a + 1
        raise Exception()

    def probability_vector(self):
        depth = self.compute_depth()
        ks = torch.arange(
            0, depth, dtype=self._nu_L.dtype, device=self._nu_L.device
        )
        alpha_L = (ks * self.nu_L.log() - torch.lgamma(ks + 1)).exp()
        # alpha_L = torch.cat(
        #     [torch.zeros(1, device=ks.device, dtype=ks.dtype), alpha_L]
        # )
        return alpha_L / alpha_L.sum()

    def mean(self):
        proba = self.probability_vector().cpu()
        return (proba * torch.arange(len(proba))).sum().item()


def elbo_(
    lambd,
    alpha0,
    beta0,
    m0p,
    nu0,
    W0inv,
    pK,
    qK,
    mkp,
    Wkpp_chol,
    alphatK_unnorm,
    betak_unnorm,
    nut_unnorm,
    Xnp,
):
    p = mkp.shape[1]

    # get pdfs of K
    qk_ = qK.probability_vector()
    mq = qk_.shape[0]
    pk_ = pK.log_prob(torch.arange(mq, dtype=qk_.dtype))

    # reparameterize
    betak = 1.0 + softplus(betak_unnorm)
    nut = 1.0 + p + softplus(nut_unnorm)
    alphatK = torch.triu(1 + softplus(alphatK_unnorm)) + alpha0 * torch.tril(
        torch.ones_like(alphatK_unnorm), diagonal=-1
    )
    Wkpp_diag = torch.diag_embed(torch.diagonal(Wkpp_chol, dim1=1, dim2=2))
    Wkpp_chol_pos = torch.tril(Wkpp_chol, diagonal=-1) + softplus(Wkpp_diag)
    Wkpp = torch.einsum("kab,kcb->kac", Wkpp_chol_pos, Wkpp_chol_pos)

    # compute Z-related quantities
    mahalnk = rx.mahalnk_(nut[:mq], Xnp, mkp[:mq], Wkpp[:mq])
    logLamTildek = rx.logLamTildek_(nut[:mq], Wkpp[:mq])
    Lnt = rx.Lnt_(
        beta0, m0p, betak[:mq], nut[:mq], Wkpp[:mq], Xnp, logLamTildek, mkp[:mq], mahalnk
    )
    rntK, rnt, Nk, NtK = rx.NtK_rnt_Nt_(qk_, Lnt, alphatK[:mq, :mq])
    Nxbarkp, xbarkp = rx.xbarkp_(rnt, Nk, Xnp)
    NSkpp, Skpp = rx.Skpp_(rnt, Nk, Xnp, xbarkp=xbarkp)

    log_qk_ = torch.log(qk_ + (qk_ <= 0).double())

    return qk_, rnt, cavi_elbo_(
        beta0,
        nu0,
        m0p,
        W0inv,
        qk_,
        pK,
        log_qk_,
        Nk,
        logLamTildek,
        nut[:mq],
        betak[:mq],
        Skpp,
        Wkpp[:mq],
        xbarkp,
        alpha0,
        alphatK[:mq, :mq],
        rntK,
        mkp[:mq],
    )


# y = 1 + softplus(x)
# x = sofplus_inverse(y - 1)


def gd(
    lambd,
    alpha0,
    beta0,
    m0p,
    nu0,
    W0inv,
    Xnp,
    delta=0.95,
    n_steps=20,
    trunc=False,
    max_K=50,
):
    N, p = Xnp.shape
    print("x")

    lambd = torch.tensor(float(lambd))
    alpha0 = torch.tensor(float(alpha0))
    beta0 = torch.tensor(float(beta0))
    nu0 = torch.tensor(float(nu0))

    # pK = PositivePoisson(lambd)
    pK = torch.distributions.Poisson(lambd)
    qK = TruncatedPoisson(lambd, truncation_quantile=delta)

    mkp = torch.tensor(torch.normal(torch.zeros(max_K, p), 1), requires_grad=True)
    Wkpp_chol = torch.tensor(
        torch.linalg.cholesky(
            torch.linalg.inv(W0inv)[None, :, :] * torch.ones(max_K)[:, None, None]
        ),
        requires_grad=True,
    )
    alphatK_unnorm = torch.tensor(
        softplus_inverse(
            alpha0 * torch.ones(max_K, max_K) - 1
        ),
        requires_grad=True,
    )
    betak_unnorm = torch.tensor(softplus_inverse(beta0 * torch.ones(max_K) - 1), requires_grad=True)
    nut_unnorm = torch.tensor(softplus_inverse(nu0 * torch.ones(max_K) - 1 - p), requires_grad=True)
    torch.autograd.set_detect_anomaly(True)
    opt = torch.optim.Adam(
        [
            mkp,
            Wkpp_chol,
            alphatK_unnorm,
            betak_unnorm,
            nut_unnorm,
            qK._nu_L,
        ],
        # max_iter=10000,
        # lr=0.001,
    )

    elbos = []
    tic = time.time()
    for jjj in range(10_000):
        opt.zero_grad()
        qk_, rnt, elbo = elbo_(
            lambd,
            alpha0,
            beta0,
            m0p,
            nu0,
            W0inv,
            pK,
            qK,
            mkp,
            Wkpp_chol,
            alphatK_unnorm,
            betak_unnorm,
            nut_unnorm,
            Xnp,
        )
        if not (jjj + 1) % 250:
            print(jjj, f"{(time.time() - tic)/60:0.2f}m, {elbo=}")
            plt.figure()
            plt.plot(elbos)
            plt.show()
            plt.close("all")
        if torch.isnan(elbo):
            raise ValueError
        loss = -elbo
        loss.backward()
        elbos.append(elbo.detach().numpy())
        opt.step()

    # reparameterize
    betak = 1.0 + softplus(betak_unnorm)
    nut = 1.0 + p + softplus(nut_unnorm)
    alphatK = torch.triu(1 + softplus(alphatK_unnorm)) + alpha0 * torch.tril(
        torch.ones_like(alphatK_unnorm), diagonal=-1
    )
    Wkpp_diag = torch.diag_embed(torch.diagonal(Wkpp_chol, dim1=1, dim2=2))
    Wkpp_chol_pos = torch.tril(Wkpp_chol, diagonal=-1) + softplus(Wkpp_diag)
    Wkpp = torch.einsum("kab,kcb->kac", Wkpp_chol_pos, Wkpp_chol_pos)

    return elbos, dict(
        betak=betak,
        nut=nut,
        alphatK=alphatK,
        Wkpp=Wkpp,
        qk_=qk_,
        rnt=rnt,
    )
