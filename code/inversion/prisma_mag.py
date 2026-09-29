import numpy as np
import matplotlib.pyplot as plt

def _dircos(incl, decl, azim):
    """
    Compute the direction cosines from inclination,
    declination, and azimuth.
    
    Parameters
    ----------
    incl : float
        Inclination in degrees, positive downward from the horizontal.
    
    decl : float
        Declination in degrees, positive east of true north.
    
    azim : float
        Azimuth of the x-axis in degrees, positive east of true north.
    
    Returns
    -------
    a, b, c : float
        Direction cosines.
    """

    d2rad = np.pi / 180.0
    xincl = incl * d2rad
    xdecl = decl * d2rad
    xazim = azim * d2rad

    a = np.cos(xincl) * np.cos(xdecl - xazim)
    b = np.cos(xincl) * np.sin(xdecl - xazim)
    c = np.sin(xincl)

    return a, b, c



def _infinity_z_box(x0, y0, z0, x1, y1, z1, x2, y2, mi, md, fi, fd, m, theta):
    """
    Compute the total-field magnetic anomaly produced by
    a rectangular prism that extends infinitely downward.
    
    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the observation point.
    
    x1, x2 : float
        Prism limits along the x-direction.
    
    y1, y2 : float
        Prism limits along the y-direction.
    
    z1 : float
        Coordinate of the top of the prism.
    
    mi, md : float
        Inclination and declination of the magnetization, in degrees.
    
    fi, fd : float
        Inclination and declination of the ambient magnetic field,
        in degrees.
    
    m : float
        Magnetization intensity in A/m.
    
    theta : float
        Azimuth of the x-axis, in degrees.
    
    Returns
    -------
    t : float
        Total-field magnetic anomaly in nT.
    """

    cm = 1.0e-7
    t2nt = 1.0e9

    # Cossenos diretores da magnetização
    ma, mb, mc = _dircos(mi, md, theta)

    # Cossenos diretores do campo magnético ambiente
    fa, fb, fc = _dircos(fi, fd, theta)

    # Combinações dos cossenos diretores
    fm1 = ma * fb + mb * fa
    fm2 = ma * fc + mc * fa
    fm3 = mb * fc + mc * fb

    fm4 = ma * fa
    fm5 = mb * fb
    fm6 = mc * fc

    # Distâncias horizontais entre os limites do prisma
    # e o ponto de observação
    alpha = np.array([
        x1 - x0,
        x2 - x0
    ], dtype=float)

    beta = np.array([
        y1 - y0,
        y2 - y0
    ], dtype=float)

    # Distância vertical
    h = z1 - z0

    hsq = h**2

    t = 0.0

    for i in range(2):

        alphasq = alpha[i]**2

        for j in range(2):

            if i == j:
                sign = 1.0
            else:
                sign = -1.0

            r0sq = alphasq + beta[j]**2 + hsq

            r0 = np.sqrt(r0sq)

            r0h = r0 * h

            alphabeta = alpha[i] * beta[j]

            arg1 = (r0 - alpha[i]) / (r0 + alpha[i])

            arg2 = (r0 - beta[j]) / (r0 + beta[j])

            arg3 = alphasq + r0h + hsq

            arg4 = r0sq + r0h - alphasq

            # Termos logarítmicos
            tlog = (
                0.5 * fm3 * np.log(arg1)
                + 0.5 * fm2 * np.log(arg2)
                - fm1 * np.log(r0 + h)
            )

            # Termos arco-tangente
            tatan = (
                -fm4 * np.arctan2(alphabeta, arg3)
                -fm5 * np.arctan2(alphabeta, arg4)
                +fm6 * np.arctan2(alphabeta, r0h)
            )

            t += sign * (tlog + tatan)

    # Conversão para nT
    t *= m * cm * t2nt

    return t

def mbox(x0, y0, z0, x1, y1, z1, x2, y2, z2, mi, md, fi, fd, m, theta):
    """
    Compute the total-field magnetic anomaly produced by a finite
    rectangular prism.

    The prism is bounded by x1 and x2 along the x-direction, by y1
    and y2 along the y-direction, and by z_top and z_bottom along the
    vertical direction. The z-axis is assumed positive downward.
    
    The total-field magnetic anomaly of a finite rectangular prism is
    computed by calling _infinity_z_box twice. The first call evaluates
    the anomaly produced by a semi-infinite prism extending downward
    from the top of the finite prism, while the second call evaluates
    the anomaly produced by a semi-infinite prism extending downward
    from its bottom. The response of the finite prism is then obtained
    by subtracting:
        t = t_top - t_bottom
    where 
        t_top       = _infinity_z_box(..., z_top=z1, ...) and
        t_bottom = _infinity_z_box(..., z_bottom=z2, ...).

    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the observation point.

    x1, x2 : float
        Minimum and maximum x-coordinates of the prism.

    y1, y2 : float
        Minimum and maximum y-coordinates of the prism.

    z_top : float
        Depth of the top surface of the prism.

    z_bottom : float
        Depth of the bottom surface of the prism.

    mi : float
        Inclination of the prism magnetization, in degrees.

    md : float
        Declination of the prism magnetization, in degrees.

    fi : float
        Inclination of the ambient magnetic field, in degrees.

    fd : float
        Declination of the ambient magnetic field, in degrees.

    m : float
        Magnetization intensity, in A/m.

    theta : float
        Azimuth of the x-axis, in degrees.

    Returns
    -------
    t : float
        Total-field magnetic anomaly produced by the finite prism,
        in nT.
    """
    
    if z2 <= z1:
        raise ValueError(
            "z_bottom must be greater than z_top "
            "because the z-axis is positive downward." )
    
    t_top      =_infinity_z_box(x0, y0, z0, x1, y1, z1, x2, y2, mi, md, fi, fd, m, theta)
    t_bottom=_infinity_z_box(x0, y0, z0, x1, y1, z2, x2, y2, mi, md, fi, fd, m, theta)
    return t_top-t_bottom
















def main():
    # ============================================================
    # Observation grid
    # ============================================================
    lim_min = 0.0
    lim_max = 64.0
    spacing = 1.0

    east  = np.arange(lim_min, lim_max + spacing, spacing)
    north= np.arange(lim_min, lim_max + spacing, spacing)
    E, N = np.meshgrid(east, north)

    # Observation elevation
    z0 = 0.0


    # ============================================================
    # Prism geometry
    # ============================================================

    # Prism dimensions:
    # width  = 20 m
    # length = 20 m
    # center = (30, 30) m

    x1 = 20.0
    x2 = 40.0
    y1 = 20.0
    y2 = 40.0
    # Top and bottom depths
    z1 = 1.0
    z2 = 3.0

    # ============================================================
    # Magnetic parameters
    # ============================================================

    # Geomagnetic field
    fi = 5.0       # inclination [degrees]
    fd = 10.0    # declination [degrees]

    # Induced magnetization
    #
    # Since there is no remanent magnetization, the direction
    # of the magnetization is the same as the geomagnetic field.

    mi = fi
    md = fd

    # Magnetization intensity [A/m]
    m = 0.35

    # Azimuth of the x-axis.
    #
    # The original MBOX formulation uses x pointing toward North
    # when theta = 0 deg. Therefore, inside MBOX we use:
    #
    #     x -> North
    #     y -> East
    #
    # The final map is plotted as East versus North.

    theta = 0.0


    # ============================================================
    # Compute magnetic anomaly
    # ============================================================

    magnetic_anomaly = np.zeros_like(E, dtype=float)

    for i in range(len(north)):
        for j in range(len(east)):
            x0 = N[i, j]
            y0 = E[i, j]
            magnetic_anomaly[i, j] = mbox(x0, y0, z0, x1, y1, z1, x2, y2, z2, mi, md, fi, fd, m, theta)
            

    # ============================================================
    # 2-D map
    # ============================================================

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(E, N, magnetic_anomaly, levels=30, cmap="jet")
    plt.colorbar(contour, label="Magnetic anomaly (nT)")

    # Prism outline
    plt.plot([20, 40, 40, 20, 20], [20, 20, 40, 40, 20],"k-", linewidth=1.5)
    plt.xlabel("E (m)")
    plt.ylabel("N (m)")
    plt.xlim(0, 64)
    plt.ylim(0, 64)
    plt.gca().set_aspect("equal")
    plt.title("Magnetic Anomaly")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()