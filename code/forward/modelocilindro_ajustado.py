import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==========================================
# PARÂMETROS FÍSICOS DA ANOMALIA (Figura 8)
# - Densidade: 3100 kg/m^3[cite: 1]
# - Suscetibilidade Magnética: 0.24 SI[cite: 1]
# ==========================================
DENSIDADE = 3100.0
SUSCETIBILIDADE = 0.24

# ==========================================
# GERAÇÃO DOS DADOS DO MODELO 1 (Cilindro Simples)
# ==========================================
nx, ny = 25, 25
x_edges = np.linspace(-4000, 4000, nx + 1)
y_edges = np.linspace(-4000, 4000, ny + 1)

xc = 0.5 * (x_edges[:-1] + x_edges[1:])
yc = 0.5 * (y_edges[:-1] + y_edges[1:])
Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
R_xy = np.sqrt(Xc**2 + Yc**2)

# Parâmetros Modelo 1: Raio = 2.5 km (2500 m), Z fundo = -7100 m, Z topo = -133 m[cite: 1]
mask_m1 = (R_xy <= 2500)
z_base_m1 = -7100.0
z_top_m1 = -133.0

# Escrevendo o arquivo "modelo_cilindro.dat"
with open("modelo_cilindro.dat", "w") as f1:
    f1.write("# X_centro(m) Y_centro(m) Z_base(m) Z_topo(m) Densidade(kg/m^3) Suscetibilidade(SI)\n")
    for i in range(nx):
        for j in range(ny):
            if mask_m1[i, j]:
                f1.write(f"{xc[i]:.2f} {yc[j]:.2f} {z_base_m1:.2f} {z_top_m1:.2f} {DENSIDADE} {SUSCETIBILIDADE}\n")

# ==========================================
# GERAÇÃO DOS DADOS DO MODELO 2 (Complexo / Base Alargada)
# ==========================================
nx_m2, ny_m2 = 30, 30
x_edges_m2 = np.linspace(-6000, 6000, nx_m2 + 1)
y_edges_m2 = np.linspace(-6000, 6000, ny_m2 + 1)
xc_m2 = 0.5 * (x_edges_m2[:-1] + x_edges_m2[1:])
yc_m2 = 0.5 * (y_edges_m2[:-1] + y_edges_m2[1:])
Xc_m2, Yc_m2 = np.meshgrid(xc_m2, yc_m2, indexing='ij')
R_xy_m2 = np.sqrt(Xc_m2**2 + Yc_m2**2)

# Parte central (R <= 2500 m): vai da base inferior (-2650 m) até o topo (-300 m) de forma contínua
mask_m2_central = (R_xy_m2 <= 2500)
z_base_central = -2650.0
z_top_central = -300.0

# Parte externa alargada (2500 m < R <= 5000 m): vai da base (-2650 m) até -1600 m[cite: 1]
mask_m2_externa = (R_xy_m2 > 2500) & (R_xy_m2 <= 5000)
z_base_externa = -2650.0
z_top_externa = -1600.0

# Escrevendo o arquivo "modelo_complexo.dat"
with open("modelo_complexo.dat", "w") as f2:
    f2.write("# X_centro(m) Y_centro(m) Z_base(m) Z_topo(m) Densidade(kg/m^3) Suscetibilidade(SI)\n")
    # Escreve a parte central contínua (do fundo ao topo)
    for i in range(nx_m2):
        for j in range(ny_m2):
            if mask_m2_central[i, j]:
                f2.write(f"{xc_m2[i]:.2f} {yc_m2[j]:.2f} {z_base_central:.2f} {z_top_central:.2f} {DENSIDADE} {SUSCETIBILIDADE}\n")
    # Escreve a parte externa alargada
    for i in range(nx_m2):
        for j in range(ny_m2):
            if mask_m2_externa[i, j]:
                f2.write(f"{xc_m2[i]:.2f} {yc_m2[j]:.2f} {z_base_externa:.2f} {z_top_externa:.2f} {DENSIDADE} {SUSCETIBILIDADE}\n")

# ==========================================
# GERAÇÃO DO ARQUIVO DE RELATÓRIO
# ==========================================
with open("report_modelocilindro_ajustado.txt", "w") as f3:
    f3.write("=== RELATÓRIO DO MODELO DE ANOMALIA GEOFÍSICA ===\n\n")
    f3.write("Descrição do Programa:\n")
    f3.write("Este programa em Python gera malhas de dados tridimensionais (modelos numéricos) para a representação de estruturas geológicas/anomalias (como cilindro simples e modelo complexo de base alargada). ")
    f3.write("Ele calcula as coordenadas espaciais, limites de profundidade (topo e base), densidade e suscetibilidade magnética, exportando os resultados em arquivos de texto formatados e renderizando visualizações interativas em 3D usando voxels.\n\n")
    f3.write("Parâmetros Magnéticos:\n")
    f3.write("- Campo magnético indutor (campo externo regional): declinação de 3,5° e inclinação de 73,5°.\n")
    f3.write("- Magnetização remanente predominantemente medida em amostras: declinação de 345° e inclinação de 60°.\n")

print("Arquivos 'modelo_cilindro.dat', 'modelo_complexo.dat' e 'report_modelocilindro_ajustado.txt' gerados com sucesso!")

# ==========================================
# RENDERIZAÇÃO DA JANELA 3D (Prismas Verticais)
# ==========================================
fig = plt.figure(figsize=(14, 7))

# Subplot Modelo 1
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Modelo 1: Cilindro Uniforme[cite: 1]")
mask_m1_3d = mask_m1[:, :, np.newaxis]
Z_edges_m1 = np.array([z_base_m1, z_top_m1])
X1, Y1, Z1 = np.meshgrid(x_edges, y_edges, Z_edges_m1, indexing='ij')
ax1.voxels(X1, Y1, Z1, mask_m1_3d, facecolors='red', edgecolors='k', linewidth=0.2, alpha=0.8)
ax1.set_xlabel('Eixo X (m)')
ax1.set_ylabel('Eixo Y (m)')
ax1.set_zlabel('Profundidade Z (m)')
ax1.set_zlim(-8000, 200)

# Subplot Modelo 2
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Modelo 2: Base Alargada[cite: 1]")

# Renderizar parte central contínua (da base ao topo) - Verde
mask_m2_central_3d = mask_m2_central[:, :, np.newaxis]
Z_edges_central = np.array([z_base_central, z_top_central])
X2_c, Y2_c, Z2_c = np.meshgrid(x_edges_m2, y_edges_m2, Z_edges_central, indexing='ij')
ax2.voxels(X2_c, Y2_c, Z2_c, mask_m2_central_3d, facecolors='forestgreen', edgecolors='k', linewidth=0.2, alpha=0.8)

# Renderizar parte externa alargada - Verde
mask_m2_externa_3d = mask_m2_externa[:, :, np.newaxis]
Z_edges_externa = np.array([z_base_externa, z_top_externa])
X2_e, Y2_e, Z2_e = np.meshgrid(x_edges_m2, y_edges_m2, Z_edges_externa, indexing='ij')
ax2.voxels(X2_e, Y2_e, Z2_e, mask_m2_externa_3d, facecolors='limegreen', edgecolors='k', linewidth=0.2, alpha=0.8)

ax2.set_xlabel('Eixo X (m)')
ax2.set_ylabel('Eixo Y (m)')
ax2.set_zlabel('Profundidade Z (m)')
ax2.set_zlim(-3500, 200)

plt.tight_layout()
plt.show()