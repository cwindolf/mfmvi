import torch


torch.set_default_dtype(torch.double)
log2 = torch.log(torch.tensor(2.0))


debug = False


def dprint(*args, **kwargs):
    if debug:
        print(*args, **kwargs)


# -- E-step formulas


def Lnt_(beta0, m0p, betak, nut, Wkpp, Xnp, logLamTildek, mkp, mahalnk):
    k, p, p_ = Wkpp.shape
    assert p == p_
    return 0.5 * (
        p / betak[None, :]
        + logLamTildek[None, :]
        + mahalnk
    )


def rntK_K_(alphatK_K, Lnt):
    rhont = torch.digamma(alphatK_K)[None, :] + Lnt
    rntK_K = torch.softmax(rhont)
    NtK_K = rntK_K.sum(axis=0)
    return NtK_K, rntK_K


def NtK_rnt_Nt_(qK, Lnt, alphatK):
    m_hat_delta = qK.shape[0]
    assert alphatK.shape == (m_hat_delta, m_hat_delta)
    assert Lnt.shape[1] == m_hat_delta

    # index is t <= K, which is the upper triangle
    # fill the lower triangle with -inf so they vanish after softmax
    rho_ntK = torch.digamma(alphatK[None, :, :] + Lnt[:, :, None])
    rho_ntK = rho_ntK - torch.tril(torch.inf * torch.ones_like(rho_ntK), diagonal=-1)
    # softmax over the t dimension
    rntK = torch.softmax(rho_ntK, dim=1)

    # take our expectations and sums that we want
    rnt = torch.einsum("ntk,k->nt", rntK, qK)
    Nt = rnt.sum(axis=0)
    NtK = rntK.sum(axis=0)

    return rntK, rnt, Nt, NtK


def alphatK_(alpha0, NtK):
    return alpha0 + NtK


# -- M-step formulas


def xbarkp_(rnk, Nk, Xnp):
    Nxbarkp = rnk.T @ Xnp
    dprint(f"{Nxbarkp=}")
    xbarkp = Nxbarkp.clone()
    dprint(f"{xbarkp[Nk > 0].shape}")
    dprint(f"{xbarkp[Nk > 0].shape}")
    xbarkp[Nk > 0] /= Nk[Nk > 0, None]
    return Nxbarkp, xbarkp


def Skpp_(rnk, Nk, Xnp, xbarkp=None):
    if xbarkp is None:
        xbarkp = xbarkp_(rnk, Nk, Xnp)
    Xcent = Xnp[:, None, :] - xbarkp[None, :, :]
    dprint(f"{Xcent.shape=}")
    NSkpp = torch.einsum("ik,ikd,ike->kde", rnk, Xcent, Xcent)
    dprint(f"{NSkpp.max()=} {16*torch.abs(NSkpp).max()=}")
    dprint(f"{NSkpp.shape=}")
    dprint(f"{torch.det(NSkpp)=}")
    Skpp = NSkpp.clone()
    dprint(f"{Skpp[Nk > 0].shape=}")
    dprint(f"{Nk[:, None, None].shape}")
    Skpp[Nk > 0] /= Nk[Nk > 0, None, None]
    dprint(f"{torch.det(Skpp)=}")
    return NSkpp, Skpp


def betak_(beta0, Nk):
    return beta0 + Nk


def nut_(nu0, Nk):
    return nu0 + Nk + 1


def mkp_(beta0, m0p, betak, Nk, Nxbarkp):
    return ((beta0 * m0p)[None, :] + Nxbarkp) / betak[:, None]


def Wkpp_(nu0, beta0, m0p, Nk, xbarkp, NSkpp, W0inv):
    bonk = beta0 * Nk / (beta0 + Nk)
    dprint(f"{bonk=}")
    dxbarkp = xbarkp - m0p[None, :]
    # dprint(f"{dxbarkp=}")
    covfix = (bonk[:, None, None] * dxbarkp[:, :, None]) * dxbarkp[:, None, :]
    # dprint(f"{covfix=}")
    dprint(f"{torch.det(W0inv)=}")
    dprint(f"{torch.det(NSkpp)=}")
    Winvkpp = W0inv[None, :, :] + NSkpp + covfix
    dprint(f"{torch.det(Winvkpp)=}")
    dprint(f"{torch.det(Winvkpp / Nk.max())=}")
    # W = torch.linalg.inv(Winvkpp / Nk.max()) / Nk.max()
    Wkpp = torch.linalg.inv(Winvkpp)
    dprint(f"{torch.det(Wkpp)=}")
    return Wkpp, Winvkpp


def logLamTildek_(nut, Wkpp):
    k, p, p_ = Wkpp.shape
    assert p == p_
    return (
        torch.log(torch.det(Wkpp))
        + torch.sum(
            torch.digamma(
                0.5 * (nut[:, None] - torch.arange(p, dtype=nut.dtype)[None, :])
            ),
            axis=1,
        )
        + p * log2
    )


def mahalnk_(nut, Xnp, mkp, Wkpp):
    Xcentnkpt = Xnp[:, None, :] - mkp[None, :, :]
    dprint(f"{Xcentnkpt=}")
    mahalnk = nut * torch.einsum("nkd,kde,nke->nk", Xcentnkpt, Wkpp, Xcentnkpt)
    return mahalnk


# -- ELBO formulas


def E_log_pmuLambdak_(nu0, beta0, m0p, ):
    pass


def neg_E_log_qmuLambdak_():
    pass
