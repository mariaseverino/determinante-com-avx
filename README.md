## Determinante de Matriz 4x4

### Este programa calcula o determinante de uma matriz 4x4 utilizando instruções AVX.

## :pushpin: Algumas instruções

```bash
# carrega da memória
__m256d vet1 = _mm256_loadu_pd(vetor1);
__m256d vet2 = _mm256_loadu_pd(vetor2);

# define os valores
__m256d vetor = _mm256_set_pd(0.0, 0.0, 1.0, 0.0);

# efetua uma soma 
_mm256_add_pd(vet1, vet2);

# efetua uma subtração 
_mm256_sub_pd(vet1, vet2);

# efetua uma multiplicação 
_mm256_mul_pd(vet1, vet2);

# embaralha os elementos
_mm256_permute_pd
_mm256_permute2f128_pd
```

Para mais instruções acesse a [documentação](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX).

## :rocket: Como usar

```bash
# Clone este repositorio
$ git clone https://github.com/mariaseverino/determinante-avx.git

# Vá até a pasta do repositorio
$ cd determinante-avx

# No terminal execute
$ gcc -mavx determinante.c -o determinante
$ .\determinante
```
