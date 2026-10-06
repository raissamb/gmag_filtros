#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESSAMENTO UNIFICADO DE CAMPOS POTENCIAIS — GRAVIDADE / MAGNETOMETRIA
VERSÃO PARA ENTRADA JÁ GRIDADA EM UTM
=========================================================================

ENTRADA
-------
Arquivo texto .dat/.xyz com exatamente 3 colunas:

    X_UTM   Y_UTM   CAMPO

Unidades:
    X, Y = metros

    GRAVIDADE:
        CAMPO = mGal

    MAGNETOMETRIA:
        CAMPO = nT


REGRA CENTRAL DESTA VERSÃO
--------------------------
O arquivo de entrada JÁ DEVE SER UMA GRADE REGULAR COMPLETA EM UTM.

O código:
    - NÃO usa griddata;
    - NÃO cria uma nova grade;
    - NÃO adiciona linha ou coluna;
    - NÃO faz extrapolação por nearest;
    - usa DX e DY informados explicitamente pelo usuário como valores autoritativos;
    - reorganiza os pontos apenas por ordenação X/Y + reshape;
    - preserva EXATAMENTE o número de pontos da entrada em todas
      as saídas em grade.

Exemplo:
    entrada = 73.341 pontos
    saída   = 73.341 pontos


FLUXO
-----
                            ENTRADA X Y Z
                                  │
                         pergunta: GRAV ou MAG?
                                  │
                    verifica grade regular completa
                                  │
                    recebe DX e DY informados pelo usuário
                                  │
                      pergunta: ORIGINAL ou RESIDUAL?
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
             ORIGINAL                        RESIDUAL PRONTA
                 │                                 │
       suavização auxiliar                         │
      somente para o regional                      │
                 │                                 │
        regional Gaussiano                         │
                 │                                 │
 residual = original - regional                    │
                 └────────────────┬────────────────┘
                                  │
                         RESIDUAL DE TRABALHO
                                  │
                    suavização pré-derivadas
                                  │
            ┌─────────┬───────────┼──────────┬──────────┐
            │         │           │          │          │
          THDR       Tilt        1VD        ASA       Strike
            │
            ├── THDR log10
            │
            └───────────────────────────────────────────────┐
                                                            │
 GRAVIDADE                                                  │
     └── 2VD auxiliar                                       │
                                                            │
 MAGNETOMETRIA                                              │
     ├── Theta-map                                          │
     └── RTP regularizado opcional                          │
             ├── THDR                                       │
             ├── Tilt                                       │
             ├── ASA                                        │
             └── Theta                                      │
                                                            │
 AMBOS                                                      │
     residual ──> upward continuation                       │
                     └──> THDR em cada altura <─────────────┘


TOPOGRAFIA / BATIMETRIA
-----------------------
Não entram matematicamente neste processamento.
São controles externos para interpretação.


OBSERVAÇÃO SOBRE BORDAS
-----------------------
Efeitos de borda provenientes da separação regional-residual podem
permanecer nos mapas. O recorte interpretativo das bordas deve ser feito
posteriormente de forma consistente com o procedimento adotado no estudo.


DEPENDÊNCIAS
------------
numpy
scipy
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter


# ======================================================================
# CONFIGURAÇÕES
# ======================================================================

EPS = 1e-12
PAD_FACTOR = 1

# Percentis usados para mascarar Strike em regiões de gradiente fraco.
PERCENTIS_THDR = [80, 90]

# Continuação para cima aplicada ao RESIDUAL.
UPWARD_HEIGHTS_KM = [10.0, 20.0, 40.0]

# Defaults de processamento.
# DX e DY voltam a ser fornecidos explicitamente pelo usuário.
DEFAULTS = {
    "GRAV": {
        "dx_m": 2785.869840357143,
        "dy_m": 2780.067275,
        "sigma_pre_reg": 1.0,
        "sigma_reg_km": 100.0,
        "sigma_deriv": 2.0,
        "sigma_2vd": 5.0,
        "unit": "mGal",
        "dunit": "mGal/km",
    },
    "MAG": {
        "dx_m": 5608.985010357143,
        "dy_m": 5668.691753846157,
        "sigma_pre_reg": 1.0,
        "sigma_reg_km": 100.0,
        "sigma_deriv": 1.0,
        "sigma_2vd": 3.0,
        "unit": "nT",
        "dunit": "nT/km",
    },
}


# ======================================================================
# FUNÇÕES DE INTERAÇÃO
# ======================================================================

def ask_float(msg, default):
    s = input(f"{msg} [{default}]: ").strip().replace(",", ".")

    if s == "":
        return float(default)

    return float(s)


def ask_yes_no(msg, default=False):
    d = "s" if default else "n"

    s = input(
        f"{msg} [s/n; padrão={d}]: "
    ).strip().lower()

    if s == "":
        return default

    return s.startswith("s")


# ======================================================================
# FUNÇÕES DE SALVAMENTO / CONTROLE
# ======================================================================

def save_grid(path, X, Y, Z, header, n_expected):
    """
    Salva uma grade X/Y/Z.
    Confere obrigatoriamente que o número de pontos permaneceu inalterado.
    """

    arr = np.column_stack(
        (
            X.ravel(),
            Y.ravel(),
            Z.ravel()
        )
    )

    if arr.shape[0] != n_expected:
        raise RuntimeError(
            f"ERRO DE CONTAGEM ao salvar {os.path.basename(path)}:\n"
            f"esperado = {n_expected}\n"
            f"obtido   = {arr.shape[0]}"
        )

    np.savetxt(
        path,
        arr,
        fmt="%.8e",
        header=header,
        comments="# "
    )

    print(
        f"Salvo: {os.path.basename(path)} | "
        f"{arr.shape[0]} pontos"
    )


def save_original_order(path, x, y, z, header, n_expected):
    """
    Salva nos X/Y exatamente na ordem original do arquivo de entrada.
    """

    arr = np.column_stack(
        (
            x,
            y,
            z
        )
    )

    if arr.shape[0] != n_expected:
        raise RuntimeError(
            f"ERRO DE CONTAGEM ao salvar {os.path.basename(path)}"
        )

    np.savetxt(
        path,
        arr,
        fmt="%.8e",
        header=header,
        comments="# "
    )


def stats(name, z, unit=""):
    v = np.asarray(z)
    v = v[np.isfinite(v)]

    print("-" * 76)
    print(name)

    if v.size == 0:
        print("Sem valores finitos.")
        return

    p = np.percentile(
        v,
        [1, 5, 50, 95, 99]
    )

    u = f" {unit}" if unit else ""

    print(
        f"Min={v.min():.6e}{u} | "
        f"P01={p[0]:.6e}{u} | "
        f"P05={p[1]:.6e}{u}"
    )

    print(
        f"Med={p[2]:.6e}{u} | "
        f"P95={p[3]:.6e}{u} | "
        f"P99={p[4]:.6e}{u} | "
        f"Max={v.max():.6e}{u}"
    )


# ======================================================================
# GRADE EXATA — SEM INTERPOLAÇÃO
# ======================================================================

def build_exact_grid(x, y, z, dx_m, dy_m):
    """
    Reconstrói a matriz 2D EXATAMENTE a partir de uma grade regular completa.

    Não interpola.
    Não extrapola.
    Não cria novos nós.
    NÃO estima DX/DY.

    DX e DY são informados pelo usuário e tratados como valores
    autoritativos para todas as derivadas e operações espectrais.

    A função apenas confere se as diferenças entre coordenadas consecutivas
    são compatíveis com os DX/DY fornecidos.

    Retorna:
        Xg, Yg, G
        ordem usada para organizar o arquivo
        nx, ny
    """

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    nx = len(x_unique)
    ny = len(y_unique)

    n = len(z)

    if nx < 2 or ny < 2:
        raise ValueError(
            "A grade precisa possuir pelo menos 2 valores distintos "
            "em X e em Y."
        )

    n_grid = nx * ny

    print("")
    print("=" * 76)
    print("VALIDAÇÃO DA GRADE DE ENTRADA")
    print("=" * 76)
    print(f"Pontos no arquivo : {n}")
    print(f"X únicos          : {nx}")
    print(f"Y únicos          : {ny}")
    print(f"nx × ny           : {n_grid}")
    print(f"DX informado      : {dx_m:.12f} m")
    print(f"DY informado      : {dy_m:.12f} m")

    if n_grid != n:
        raise ValueError(
            "\nO arquivo NÃO representa uma grade retangular completa.\n"
            f"Número de pontos = {n}\n"
            f"nx × ny           = {n_grid}\n\n"
            "Esta versão não interpola dados ausentes.\n"
            "Use uma entrada já gridada e completa."
        )

    # Validação contra os valores fornecidos pelo usuário.
    # Não estimamos nem substituímos DX/DY.
    dxs = np.diff(x_unique)
    dys = np.diff(y_unique)

    # DX e DY informados pelo usuário são AUTORITATIVOS para os cálculos.
    # Aqui apenas diagnosticamos pequenas variações numéricas nas coordenadas.
    #
    # Uma grade gerada/convertida em software pode apresentar pequenas diferenças
    # entre coordenadas consecutivas por arredondamento ou exportação, mesmo tendo
    # sido construída com um passo nominal fixo.
    #
    # O código NÃO substitui DX/DY pelos valores medidos no arquivo.
    # As derivadas/FFT continuam usando exatamente dx_m e dy_m fornecidos.

    dev_dx = np.max(np.abs(dxs - dx_m))
    dev_dy = np.max(np.abs(dys - dy_m))

    rel_dx = dev_dx / abs(dx_m)
    rel_dy = dev_dy / abs(dy_m)

    print(f"Menor ΔX no arquivo : {np.min(dxs):.12f} m")
    print(f"Maior ΔX no arquivo : {np.max(dxs):.12f} m")
    print(f"Desvio máx. de DX   : {dev_dx:.6f} m ({100.0*rel_dx:.6f} %)")

    print(f"Menor ΔY no arquivo : {np.min(dys):.12f} m")
    print(f"Maior ΔY no arquivo : {np.max(dys):.12f} m")
    print(f"Desvio máx. de DY   : {dev_dy:.6f} m ({100.0*rel_dy:.6f} %)")

    # Critério apenas para detectar um DX/DY claramente incorreto.
    # 0,1 % é muito maior que o jitter numérico observado na sua grade
    # (~0,004 % em X), mas ainda impede aceitar um espaçamento errado.
    MAX_REL_SPACING_ERROR = 1e-3  # 0,1 %

    if rel_dx > MAX_REL_SPACING_ERROR:
        raise ValueError(
            "\nO DX informado difere demais do espaçamento da grade.\n"
            f"DX informado = {dx_m:.12f} m\n"
            f"Desvio relativo máximo = {100.0*rel_dx:.6f} %\n"
            "Revise o DX informado."
        )

    if rel_dy > MAX_REL_SPACING_ERROR:
        raise ValueError(
            "\nO DY informado difere demais do espaçamento da grade.\n"
            f"DY informado = {dy_m:.12f} m\n"
            f"Desvio relativo máximo = {100.0*rel_dy:.6f} %\n"
            "Revise o DY informado."
        )

    if rel_dx > 0 or rel_dy > 0:
        print(
            "AVISO: existem pequenas variações numéricas nas coordenadas, "
            "mas DX/DY informados serão usados EXATAMENTE nos cálculos."
        )

    # Ordena primeiro por Y e, dentro de cada Y, por X.
    order = np.lexsort(
        (
            x,
            y
        )
    )

    xs = x[order]
    ys = y[order]
    zs = z[order]

    Xg, Yg = np.meshgrid(
        x_unique,
        y_unique
    )

    # Confirma que cada combinação X/Y existe exatamente uma vez.
    atol_x = max(1e-6, abs(dx_m) * 1e-8)
    atol_y = max(1e-6, abs(dy_m) * 1e-8)

    if not np.allclose(
        xs,
        Xg.ravel(),
        rtol=0.0,
        atol=atol_x
    ):
        raise ValueError(
            "A sequência de X não corresponde a uma grade regular completa. "
            "Pode haver duplicatas ou nós ausentes."
        )

    if not np.allclose(
        ys,
        Yg.ravel(),
        rtol=0.0,
        atol=atol_y
    ):
        raise ValueError(
            "A sequência de Y não corresponde a uma grade regular completa. "
            "Pode haver duplicatas ou nós ausentes."
        )

    G = zs.reshape(
        ny,
        nx
    )

    print(f"Dimensão da matriz : {ny} linhas × {nx} colunas")
    print("STATUS             : grade regular completa — OK")
    print("=" * 76)

    return (
        Xg,
        Yg,
        G,
        order,
        nx,
        ny
    )



def grid_to_original_order(Zg, order):
    """
    Converte um produto 2D, organizado em ordem Y/X,
    de volta à ordem original do arquivo de entrada.

    Não interpola.
    """

    flat_sorted = Zg.ravel()

    out = np.empty_like(
        flat_sorted,
        dtype=float
    )

    out[order] = flat_sorted

    return out


# ======================================================================
# FFT
# ======================================================================

def prep_fft(
    G,
    pad_factor=1,
    remove_mean=True
):
    ny, nx = G.shape

    py = int(
        ny * pad_factor
    )

    px = int(
        nx * pad_factor
    )

    mean = (
        float(
            np.mean(G)
        )
        if remove_mean
        else 0.0
    )

    Gp = np.pad(
        G - mean,
        (
            (py, py),
            (px, px)
        ),
        mode="reflect"
    )

    return (
        Gp,
        mean,
        py,
        px,
        ny,
        nx
    )


def kgrid(
    Gp,
    dx_km,
    dy_km
):
    ny, nx = Gp.shape

    kx = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            nx,
            d=dx_km
        )
    )

    ky = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            ny,
            d=dy_km
        )
    )

    KX, KY = np.meshgrid(
        kx,
        ky
    )

    K = np.sqrt(
        KX**2
        + KY**2
    )

    return (
        KX,
        KY,
        K
    )


def crop(
    Zp,
    py,
    px,
    ny,
    nx
):
    return Zp[
        py:py + ny,
        px:px + nx
    ]


# ======================================================================
# DERIVADAS / CONTINUAÇÃO
# ======================================================================

def vertical_derivative(
    G,
    dx_km,
    dy_km,
    order=1
):
    Gp, _, py, px, ny, nx = prep_fft(
        G,
        PAD_FACTOR,
        True
    )

    _, _, K = kgrid(
        Gp,
        dx_km,
        dy_km
    )

    FG = np.fft.fft2(
        Gp
    )

    Zp = np.real(
        np.fft.ifft2(
            (K**order)
            * FG
        )
    )

    return crop(
        Zp,
        py,
        px,
        ny,
        nx
    )


def upward(
    G,
    dx_km,
    dy_km,
    height_km
):
    """
    Continuação para cima no domínio de Fourier:

        U(k,h) = exp(-k h)
    """

    Gp, mean, py, px, ny, nx = prep_fft(
        G,
        PAD_FACTOR,
        True
    )

    _, _, K = kgrid(
        Gp,
        dx_km,
        dy_km
    )

    H = np.exp(
        -K
        * height_km
    )

    FG = np.fft.fft2(
        Gp
    )

    Zp = (
        np.real(
            np.fft.ifft2(
                H
                * FG
            )
        )
        + mean
    )

    return crop(
        Zp,
        py,
        px,
        ny,
        nx
    )


# ======================================================================
# FILTROS COMUNS
# ======================================================================

def common_filters(
    G,
    dx_km,
    dy_km
):
    """
    Produtos comuns à gravidade e magnetometria.
    """

    dGdy, dGdx = np.gradient(
        G,
        dy_km,
        dx_km
    )

    # Total Horizontal Derivative
    thdr = np.sqrt(
        dGdx**2
        + dGdy**2
    )

    # Primeira derivada vertical
    vdr = vertical_derivative(
        G,
        dx_km,
        dy_km,
        1
    )

    # Tilt Angle
    tilt_rad = np.arctan2(
        vdr,
        thdr + EPS
    )

    tilt_deg = np.degrees(
        tilt_rad
    )

    # Analytic Signal Amplitude
    asa = np.sqrt(
        dGdx**2
        + dGdy**2
        + vdr**2
    )

    # Azimute do máximo gradiente horizontal
    az = (
        np.degrees(
            np.arctan2(
                dGdx,
                dGdy
            )
        )
        + 360.0
    ) % 360.0

    # Strike axial perpendicular ao gradiente
    strike = (
        az
        + 90.0
    ) % 180.0

    return {
        "THDR": thdr,
        "THDR_LOG10": np.log10(
            thdr + EPS
        ),
        "VDR": vdr,
        "TILT_RAD": tilt_rad,
        "TILT_DEG": tilt_deg,
        "ASA": asa,
        "ASA_LOG10": np.log10(
            asa + EPS
        ),
        "AZIMUTH": az,
        "STRIKE": strike,
    }


def theta_map(
    thdr,
    asa
):
    return np.clip(
        thdr
        / (
            asa
            + EPS
        ),
        0.0,
        1.0
    )


# ======================================================================
# STRIKE / MÁSCARAS
# ======================================================================

def strike_mask_and_classes(
    strike,
    thdr,
    percentile
):
    threshold = np.nanpercentile(
        thdr,
        percentile
    )

    mask = np.where(
        thdr >= threshold,
        strike,
        np.nan
    )

    cls = np.full_like(
        strike,
        np.nan,
        dtype=float
    )

    valid = (
        thdr
        >= threshold
    )

    ns = (
        (
            strike >= 0.0
        )
        &
        (
            strike < 22.5
        )
    ) | (
        (
            strike >= 157.5
        )
        &
        (
            strike <= 180.0
        )
    )

    ne = (
        (strike >= 22.5)
        &
        (strike < 67.5)
    )

    ew = (
        (strike >= 67.5)
        &
        (strike < 112.5)
    )

    nw = (
        (strike >= 112.5)
        &
        (strike < 157.5)
    )

    cls[
        valid & ns
    ] = 1.0

    cls[
        valid & ne
    ] = 2.0

    cls[
        valid & ew
    ] = 3.0

    cls[
        valid & nw
    ] = 4.0

    return (
        mask,
        cls,
        threshold
    )


# ======================================================================
# RTP MAGNÉTICO — OPCIONAL
# ======================================================================

def rtp_regularized(
    G,
    dx_km,
    dy_km,
    inc,
    dec,
    damping=0.20,
    mag_inc=None,
    mag_dec=None
):
    """
    RTP espectral regularizado diagnóstico.

    Somente para MAG.

    Não deve ser apresentado como reprodução exata de um método
    específico de RTP de baixa latitude sem validação metodológica
    adicional.
    """

    if mag_inc is None:
        mag_inc = inc

    if mag_dec is None:
        mag_dec = dec

    I = np.radians(
        inc
    )

    D = np.radians(
        dec
    )

    Im = np.radians(
        mag_inc
    )

    Dm = np.radians(
        mag_dec
    )

    Gp, mean, py, px, ny, nx = prep_fft(
        G,
        PAD_FACTOR,
        True
    )

    KX, KY, K = kgrid(
        Gp,
        dx_km,
        dy_km
    )

    FG = np.fft.fft2(
        Gp
    )

    # x = leste; y = norte
    HF = (
        KX * np.sin(D)
        + KY * np.cos(D)
    )

    HM = (
        KX * np.sin(Dm)
        + KY * np.cos(Dm)
    )

    QF = (
        K * np.sin(I)
        + 1j
        * np.cos(I)
        * HF
    )

    QM = (
        K * np.sin(Im)
        + 1j
        * np.cos(Im)
        * HM
    )

    DEN = (
        QF
        * QM
    )

    NUM = (
        K**2
    )

    REG = (
        damping
        * NUM
    )**2

    H = np.zeros_like(
        DEN,
        dtype=complex
    )

    m = (
        K > 0.0
    )

    H[m] = (
        NUM[m]
        * np.conj(
            DEN[m]
        )
        /
        (
            np.abs(
                DEN[m]
            )**2
            + REG[m]
            + EPS
        )
    )

    H[~m] = (
        1.0
        + 0.0j
    )

    Zp = (
        np.real(
            np.fft.ifft2(
                H
                * FG
            )
        )
        + mean
    )

    return (
        crop(
            Zp,
            py,
            px,
            ny,
            nx
        ),
        H
    )


# ======================================================================
# README
# ======================================================================

def create_readme(
    folder,
    kind,
    cfg,
    input_is_residual,
    n_points,
    nx,
    ny,
    dx,
    dy,
    sigma_pre_reg,
    sigma_reg_km,
    sigma_deriv,
    rtp_enabled
):
    text = f"""PROCESSAMENTO UNIFICADO GRAV/MAG — GRADE EXATA
=================================================

TIPO:
    {kind}

UNIDADE:
    {cfg['unit']}

NÚMERO DE PONTOS:
    entrada = {n_points}
    saída em grade = {n_points}

DIMENSÕES:
    nx = {nx}
    ny = {ny}
    nx * ny = {nx * ny}

ESPAÇAMENTO INFORMADO PELO USUÁRIO:
    DX = {dx:.9f} m
    DY = {dy:.9f} m

ENTRADA RESIDUAL PRONTA:
    {'SIM' if input_is_residual else 'NÃO'}

IMPORTANTE
----------
Nenhuma reinterpolação foi executada.
Nenhum novo nó foi criado.
Nenhum preenchimento nearest foi usado.

FLUXO
-----
Se campo original:
    original
    -> suavização auxiliar
    -> regional
    -> residual = original - regional

Se residual pronta:
    residual de entrada
    -> filtros

Em ambos:
    residual
    -> suavização pré-derivadas
    -> THDR / Tilt / 1VD / ASA / Strike

    residual
    -> upward continuation
    -> THDR em cada altura

GRAV:
    + 2VD auxiliar

MAG:
    + Theta-map
    + RTP regularizado opcional:
      {'ATIVADO' if rtp_enabled else 'DESATIVADO'}

PARÂMETROS
----------
Sigma pré-regional:
    {sigma_pre_reg} células

Sigma regional:
    {sigma_reg_km} km

Sigma pré-derivadas:
    {sigma_deriv} células

COMPARAÇÃO PRIORITÁRIA GRAV x MAG
---------------------------------
Residual x Residual
THDR x THDR
Tilt x Tilt
ASA x ASA
1VD x 1VD
Upward + THDR x Upward + THDR

BORDAS
------
Efeitos de borda associados à separação regional-residual devem ser
tratados posteriormente pelo mesmo recorte interpretativo adotado
para os dois campos.
"""

    with open(
        os.path.join(
            folder,
            "README_PROCESSAMENTO.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            text
        )


# ======================================================================
# INTERAÇÃO INICIAL
# ======================================================================

print("=" * 76)
print("PROCESSAMENTO UNIFICADO — GRAVIDADE / MAGNETOMETRIA")
print("ENTRADA JÁ GRIDADA EM UTM — SEM GRIDDATA")
print("=" * 76)

print("")
print("1 = GRAVIDADE")
print("2 = MAGNETOMETRIA")

resp = input(
    "Qual tipo de dado será processado? [1/2]: "
).strip().lower()

if resp in {
    "1",
    "g",
    "grav",
    "gravidade"
}:
    KIND = "GRAV"

elif resp in {
    "2",
    "m",
    "mag",
    "magnetometria"
}:
    KIND = "MAG"

else:
    raise ValueError(
        "Escolha 1 para GRAV ou 2 para MAG."
    )

cfg = DEFAULTS[
    KIND
]

print("")
print("ESPAÇAMENTO DA GRADE")
print("-" * 76)
DX_M = ask_float(
    "DX em metros",
    cfg["dx_m"]
)
DY_M = ask_float(
    "DY em metros",
    cfg["dy_m"]
)
print(
    "DX e DY informados pelo usuário serão usados diretamente "
    "nas derivadas e operações espectrais."
)

file_in = input(
    "Digite o nome/caminho do arquivo X Y Z: "
).strip()

if not file_in:
    raise ValueError(
        "Informe o arquivo de entrada."
    )

if not os.path.exists(
    file_in
):
    raise FileNotFoundError(
        f"Arquivo não encontrado: {file_in}\n"
        f"Pasta atual: {os.getcwd()}"
    )

print("")
print("O arquivo contém:")
print("1 = CAMPO ORIGINAL/Bouguer -> gera regional e residual")
print("2 = RESIDUAL JÁ PRONTA -> começa diretamente nos filtros")

mode = input(
    "Escolha [1/2]: "
).strip()

INPUT_IS_RESIDUAL = (
    mode == "2"
)

# Parâmetros de processamento.
SIGMA_DERIV = ask_float(
    "Suavização pré-derivadas, em células",
    cfg["sigma_deriv"]
)

if INPUT_IS_RESIDUAL:

    SIGMA_PRE_REG = 0.0
    SIGMA_REG_KM = 0.0

else:

    SIGMA_PRE_REG = ask_float(
        "Suavização auxiliar antes do regional, em células",
        cfg["sigma_pre_reg"]
    )

    SIGMA_REG_KM = ask_float(
        "Sigma do regional Gaussiano, em km",
        cfg["sigma_reg_km"]
    )

# RTP somente MAG.
RTP_ENABLED = False
INC_DEG = None
DEC_DEG = None
RTP_DAMPINGS = [
    0.10,
    0.20,
    0.30
]

if KIND == "MAG":

    print("")
    print("RTP MAGNÉTICO")
    print("-" * 76)

    RTP_ENABLED = ask_yes_no(
        "Ativar RTP regularizado diagnóstico?",
        False
    )

    if RTP_ENABLED:

        INC_DEG = ask_float(
            "Inclinação magnética I (graus)",
            0.0
        )

        DEC_DEG = ask_float(
            "Declinação magnética D (graus)",
            0.0
        )


# ======================================================================
# LEITURA
# ======================================================================

data = np.loadtxt(
    file_in
)

if (
    data.ndim != 2
    or data.shape[1] < 3
):
    raise ValueError(
        "O arquivo precisa ter pelo menos 3 colunas: X Y Z."
    )

# Usa somente as três primeiras colunas.
x = data[:, 0].astype(float)
y = data[:, 1].astype(float)
z = data[:, 2].astype(float)

valid = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(z)
)

x = x[valid]
y = y[valid]
z = z[valid]

N_INPUT = len(
    z
)

if (
    np.nanmax(
        np.abs(x)
    ) <= 360.0
    and
    np.nanmax(
        np.abs(y)
    ) <= 90.0
):
    raise ValueError(
        "X/Y parecem longitude/latitude. "
        "Converta previamente para uma grade métrica UTM/projetada."
    )


# ======================================================================
# RECONSTRUÇÃO EXATA DA GRADE
# ======================================================================

(
    Xg,
    Yg,
    G,
    ORDER,
    NX,
    NY
) = build_exact_grid(
    x,
    y,
    z,
    DX_M,
    DY_M
)

dx_km = (
    DX_M
    / 1000.0
)

dy_km = (
    DY_M
    / 1000.0
)

base = os.path.splitext(
    os.path.basename(
        file_in
    )
)[0]

outdir = (
    f"{base}_{KIND}_processamento_grid_exata"
)

os.makedirs(
    outdir,
    exist_ok=True
)

print("")
print(f"Tipo            : {KIND}")
print(f"Unidade         : {cfg['unit']}")
print(f"Pontos entrada  : {N_INPUT}")
print(f"Pontos saída    : {Xg.size}")
print(f"Pasta de saída  : {outdir}")


# ======================================================================
# CAMPO DE ENTRADA
# ======================================================================

save_grid(
    os.path.join(
        outdir,
        f"00_{KIND}_campo_entrada_grade_exata.dat"
    ),
    Xg,
    Yg,
    G,
    (
        "X_UTM_m Y_UTM_m "
        f"{KIND}_campo_{cfg['unit']}"
    ),
    N_INPUT
)


# ======================================================================
# REGIONAL / RESIDUAL
# ======================================================================

if INPUT_IS_RESIDUAL:

    G_res = G.copy()
    G_reg = None

    print("")
    print(
        "Entrada já residual: "
        "etapa regional-residual ignorada."
    )

else:

    print("")
    print("=" * 76)
    print("SEPARAÇÃO REGIONAL / RESIDUAL")
    print("=" * 76)

    # Suavização usada SOMENTE para estabilizar a estimativa do regional.
    G_aux = gaussian_filter(
        G,
        sigma=SIGMA_PRE_REG,
        mode="nearest"
    )

    save_grid(
        os.path.join(
            outdir,
            f"01_{KIND}_suavizacao_auxiliar_regional_grade_exata.dat"
        ),
        Xg,
        Yg,
        G_aux,
        (
            "X_UTM_m Y_UTM_m "
            f"{KIND}_aux_{cfg['unit']}"
        ),
        N_INPUT
    )

    sx = (
        SIGMA_REG_KM
        / dx_km
    )

    sy = (
        SIGMA_REG_KM
        / dy_km
    )

    G_reg = gaussian_filter(
        G_aux,
        sigma=(
            sy,
            sx
        ),
        mode="nearest"
    )

    # PONTO CENTRAL:
    # residual = CAMPO ORIGINAL - REGIONAL
    G_res = (
        G
        - G_reg
    )

    save_grid(
        os.path.join(
            outdir,
            f"02_{KIND}_regional_gaussiano_grade_exata.dat"
        ),
        Xg,
        Yg,
        G_reg,
        (
            "X_UTM_m Y_UTM_m "
            f"{KIND}_regional_{cfg['unit']}"
        ),
        N_INPUT
    )


save_grid(
    os.path.join(
        outdir,
        f"03_{KIND}_residual_grade_exata.dat"
    ),
    Xg,
    Yg,
    G_res,
    (
        "X_UTM_m Y_UTM_m "
        f"{KIND}_residual_{cfg['unit']}"
    ),
    N_INPUT
)

stats(
    f"{KIND} residual",
    G_res,
    cfg["unit"]
)


# ======================================================================
# SUAVIZAÇÃO PRÉ-DERIVADAS
# ======================================================================

if SIGMA_DERIV > 0:

    Gf = gaussian_filter(
        G_res,
        sigma=SIGMA_DERIV,
        mode="nearest"
    )

else:

    Gf = G_res.copy()

save_grid(
    os.path.join(
        outdir,
        f"04_{KIND}_residual_para_filtros_grade_exata.dat"
    ),
    Xg,
    Yg,
    Gf,
    (
        "X_UTM_m Y_UTM_m "
        f"{KIND}_residual_pre_derivatives_{cfg['unit']}"
    ),
    N_INPUT
)


# ======================================================================
# FILTROS COMUNS
# ======================================================================

print("")
print("=" * 76)
print("FILTROS COMUNS APLICADOS AO RESIDUAL")
print("=" * 76)

F = common_filters(
    Gf,
    dx_km,
    dy_km
)

products = [
    (
        f"10_{KIND}_THDR_residual_grade_exata.dat",
        F["THDR"],
        cfg["dunit"]
    ),
    (
        f"10_{KIND}_THDR_log10_residual_grade_exata.dat",
        F["THDR_LOG10"],
        "log10"
    ),
    (
        f"11_{KIND}_TILT_residual_graus_grade_exata.dat",
        F["TILT_DEG"],
        "degrees"
    ),
    (
        f"12_{KIND}_1VD_residual_grade_exata.dat",
        F["VDR"],
        cfg["dunit"]
    ),
    (
        f"13_{KIND}_ASA_residual_grade_exata.dat",
        F["ASA"],
        cfg["dunit"]
    ),
    (
        f"13_{KIND}_ASA_log10_residual_grade_exata.dat",
        F["ASA_LOG10"],
        "log10"
    ),
    (
        f"15_{KIND}_AZIMUTE_residual_graus_grade_exata.dat",
        F["AZIMUTH"],
        "degrees"
    ),
    (
        f"15_{KIND}_STRIKE_residual_graus_grade_exata.dat",
        F["STRIKE"],
        "degrees"
    ),
]

for fname, arr, unit in products:

    save_grid(
        os.path.join(
            outdir,
            fname
        ),
        Xg,
        Yg,
        arr,
        (
            "X_UTM_m Y_UTM_m "
            f"{fname} [{unit}]"
        ),
        N_INPUT
    )


# ======================================================================
# THETA — SOMENTE MAG
# ======================================================================

THETA = None

if KIND == "MAG":

    THETA = theta_map(
        F["THDR"],
        F["ASA"]
    )

    save_grid(
        os.path.join(
            outdir,
            "14_MAG_THETA_residual_grade_exata.dat"
        ),
        Xg,
        Yg,
        THETA,
        (
            "X_UTM_m Y_UTM_m "
            "MAG_Theta_dimensionless"
        ),
        N_INPUT
    )


# ======================================================================
# 2VD — SOMENTE GRAV
# ======================================================================

if KIND == "GRAV":

    print("")
    print(
        "Calculando 2VD auxiliar da gravidade..."
    )

    G2 = gaussian_filter(
        G_res,
        sigma=cfg["sigma_2vd"],
        mode="nearest"
    )

    SVD = vertical_derivative(
        G2,
        dx_km,
        dy_km,
        2
    )

    save_grid(
        os.path.join(
            outdir,
            "16_GRAV_2VD_residual_grade_exata.dat"
        ),
        Xg,
        Yg,
        SVD,
        (
            "X_UTM_m Y_UTM_m "
            "GRAV_2VD_mGal_per_km2"
        ),
        N_INPUT
    )


# ======================================================================
# STRIKE MASCARADO / CLASSES
# ======================================================================

print("")
print(
    "Gerando Strike mascarado por THDR..."
)

for p in PERCENTIS_THDR:

    smask, classes, threshold = strike_mask_and_classes(
        F["STRIKE"],
        F["THDR"],
        p
    )

    save_grid(
        os.path.join(
            outdir,
            f"15_{KIND}_STRIKE_mask_THDR_P{p}_grade_exata.dat"
        ),
        Xg,
        Yg,
        smask,
        (
            "X_UTM_m Y_UTM_m "
            f"{KIND}_Strike_P{p}"
        ),
        N_INPUT
    )

    save_grid(
        os.path.join(
            outdir,
            f"15_{KIND}_STRIKE_classes_THDR_P{p}_grade_exata.dat"
        ),
        Xg,
        Yg,
        classes,
        (
            "X_UTM_m Y_UTM_m "
            f"{KIND}_StrikeClasses_P{p}"
        ),
        N_INPUT
    )

    print(
        f"THDR P{p}: "
        f"{threshold:.6e} "
        f"{cfg['dunit']}"
    )


# ======================================================================
# UPWARD CONTINUATION — AMBOS
# ======================================================================

print("")
print("=" * 76)
print("UPWARD CONTINUATION DO RESIDUAL")
print("=" * 76)

for h in UPWARD_HEIGHTS_KM:

    Gup = upward(
        G_res,
        dx_km,
        dy_km,
        h
    )

    # THDR do campo continuado.
    Fu = common_filters(
        Gup,
        dx_km,
        dy_km
    )

    tag = (
        str(h)
        .replace(
            ".0",
            ""
        )
        .replace(
            ".",
            "p"
        )
    )

    save_grid(
        os.path.join(
            outdir,
            f"20_{KIND}_RESIDUAL_upward_{tag}km_grade_exata.dat"
        ),
        Xg,
        Yg,
        Gup,
        (
            "X_UTM_m Y_UTM_m "
            f"{KIND}_residual_upward_{h}km_{cfg['unit']}"
        ),
        N_INPUT
    )

    save_grid(
        os.path.join(
            outdir,
            f"21_{KIND}_THDR_RESIDUAL_upward_{tag}km_grade_exata.dat"
        ),
        Xg,
        Yg,
        Fu["THDR"],
        (
            "X_UTM_m Y_UTM_m "
            f"{KIND}_THDR_upward_{h}km_{cfg['dunit']}"
        ),
        N_INPUT
    )


# ======================================================================
# RTP — SOMENTE MAG
# ======================================================================

if (
    KIND == "MAG"
    and
    RTP_ENABLED
):

    print("")
    print("=" * 76)
    print("RTP REGULARIZADO DIAGNÓSTICO")
    print("=" * 76)

    for damping in RTP_DAMPINGS:

        Grtp, H = rtp_regularized(
            G_res,
            dx_km,
            dy_km,
            INC_DEG,
            DEC_DEG,
            damping=damping
        )

        tag = str(
            damping
        ).replace(
            ".",
            "p"
        )

        save_grid(
            os.path.join(
                outdir,
                f"30_MAG_RTP_RESIDUAL_d{tag}_grade_exata.dat"
            ),
            Xg,
            Yg,
            Grtp,
            (
                "X_UTM_m Y_UTM_m "
                f"MAG_RTP_nT_damping_{damping}"
            ),
            N_INPUT
        )

        if SIGMA_DERIV > 0:

            Grtpf = gaussian_filter(
                Grtp,
                sigma=SIGMA_DERIV,
                mode="nearest"
            )

        else:

            Grtpf = Grtp

        Fr = common_filters(
            Grtpf,
            dx_km,
            dy_km
        )

        Tr = theta_map(
            Fr["THDR"],
            Fr["ASA"]
        )

        save_grid(
            os.path.join(
                outdir,
                f"31_MAG_THDR_RTP_RESIDUAL_d{tag}_grade_exata.dat"
            ),
            Xg,
            Yg,
            Fr["THDR"],
            "X_UTM_m Y_UTM_m MAG_THDR_RTP_nT_per_km",
            N_INPUT
        )

        save_grid(
            os.path.join(
                outdir,
                f"32_MAG_TILT_RTP_RESIDUAL_d{tag}_graus_grade_exata.dat"
            ),
            Xg,
            Yg,
            Fr["TILT_DEG"],
            "X_UTM_m Y_UTM_m MAG_TILT_RTP_degrees",
            N_INPUT
        )

        save_grid(
            os.path.join(
                outdir,
                f"33_MAG_ASA_RTP_RESIDUAL_d{tag}_grade_exata.dat"
            ),
            Xg,
            Yg,
            Fr["ASA"],
            "X_UTM_m Y_UTM_m MAG_ASA_RTP_nT_per_km",
            N_INPUT
        )

        save_grid(
            os.path.join(
                outdir,
                f"34_MAG_THETA_RTP_RESIDUAL_d{tag}_grade_exata.dat"
            ),
            Xg,
            Yg,
            Tr,
            "X_UTM_m Y_UTM_m MAG_THETA_RTP_dimensionless",
            N_INPUT
        )

        gain = np.abs(
            H
        )

        gain = gain[
            np.isfinite(
                gain
            )
        ]

        print(
            f"RTP damping={damping}: "
            f"P95={np.percentile(gain,95):.3f}, "
            f"P99={np.percentile(gain,99):.3f}, "
            f"MAX={gain.max():.3f}"
        )


# ======================================================================
# PRODUTOS PRINCIPAIS NA ORDEM X/Y ORIGINAL
# ======================================================================

print("")
print("=" * 76)
print("SALVANDO PRINCIPAIS PRODUTOS NA ORDEM ORIGINAL X/Y")
print("=" * 76)

main_original = {
    "90_residual": G_res,
    "91_THDR": F["THDR"],
    "92_TILT_graus": F["TILT_DEG"],
    "93_ASA": F["ASA"],
    "94_1VD": F["VDR"],
    "95_STRIKE_graus": F["STRIKE"],
}

if (
    KIND == "MAG"
    and
    THETA is not None
):
    main_original[
        "96_THETA"
    ] = THETA

for tag, arr in main_original.items():

    arr_original = grid_to_original_order(
        arr,
        ORDER
    )

    save_original_order(
        os.path.join(
            outdir,
            f"{tag}_{KIND}_originalXY.dat"
        ),
        x,
        y,
        arr_original,
        (
            "X_UTM_m Y_UTM_m "
            f"{tag}_{KIND}"
        ),
        N_INPUT
    )


# ======================================================================
# README
# ======================================================================

create_readme(
    outdir,
    KIND,
    cfg,
    INPUT_IS_RESIDUAL,
    N_INPUT,
    NX,
    NY,
    DX_M,
    DY_M,
    SIGMA_PRE_REG,
    SIGMA_REG_KM,
    SIGMA_DERIV,
    RTP_ENABLED
)


# ======================================================================
# CONFERÊNCIA FINAL
# ======================================================================

print("")
print("=" * 76)
print("CONFERÊNCIA FINAL")
print("=" * 76)

print(
    f"Pontos na entrada              : {N_INPUT}"
)

print(
    f"Pontos em cada saída de grade  : {Xg.size}"
)

print(
    f"Dimensão da grade              : {NY} × {NX}"
)

print(
    f"DX informado                   : {DX_M:.12f} m"
)

print(
    f"DY informado                   : {DY_M:.12f} m"
)

if (
    Xg.size
    == N_INPUT
    == NX * NY
):
    print(
        "STATUS: OK — número de pontos preservado exatamente."
    )

else:
    raise RuntimeError(
        "ERRO: a quantidade de pontos não foi preservada."
    )

print("")
print(
    f"Pasta de saída: {outdir}"
)

print("")
print("ABRIR PRIMEIRO NO SURFER:")

print(
    f"03_{KIND}_residual_grade_exata.dat"
)

print(
    f"10_{KIND}_THDR_residual_grade_exata.dat"
)

print(
    f"11_{KIND}_TILT_residual_graus_grade_exata.dat"
)

print(
    f"13_{KIND}_ASA_residual_grade_exata.dat"
)

print(
    f"12_{KIND}_1VD_residual_grade_exata.dat"
)

if KIND == "MAG":
    print(
        "14_MAG_THETA_residual_grade_exata.dat"
    )

if KIND == "GRAV":
    print(
        "16_GRAV_2VD_residual_grade_exata.dat [auxiliar]"
    )

print("")
print("ANÁLISE MULTIESCALA:")

print(
    f"20_{KIND}_RESIDUAL_upward_*km_grade_exata.dat"
)

print(
    f"21_{KIND}_THDR_RESIDUAL_upward_*km_grade_exata.dat"
)

print("")
print("COMPARE GRAV × MAG NESTA ORDEM:")

print(
    "1) Residual × Residual"
)

print(
    "2) THDR × THDR"
)

print(
    "3) Tilt × Tilt"
)

print(
    "4) ASA × ASA"
)

print(
    "5) 1VD × 1VD"
)

print(
    "6) Upward + THDR × Upward + THDR"
)

print("=" * 76)
