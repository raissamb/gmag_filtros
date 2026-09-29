import numpy as np

a = np.array([[1, 2, 2, 4], 
              [1, 5, 3, 2], 
              [3, 2, 1, 3], 
              [6, 2, 1, 4], 
              [3, 1, 2, 5]])

ata = a.T @ a

npar=4
nobs=5
ata_calc = np.zeros((npar,npar))
row=np.zeros(npar)

for i in range(nobs):
    for j in range(npar):
        row[j]=a[i,j]

    for j in range(npar):
        ata_calc[j,j] += row[j]**2
        for k in range(j+1,npar):
            ata_calc[j,k] += row[j]*row[k]

for i in range(npar):
    for j in range(i+1,npar):
        ata_calc[j,i] = ata_calc[i,j]


print("A =")
print(a)

print("\nA^T A (numpy)=")
print(ata)

print("\nA^T A (def)=")
print(ata_calc)