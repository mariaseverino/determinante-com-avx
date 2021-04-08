#include <immintrin.h>
#include <stdio.h>

/* metodo que retorna apenas o valor da primeira posicao do vetor (vetor[0]) */
__m256d zeraPrimeira(__m256d vetor){
    __m256d v = _mm256_set_pd(0.0, 0.0, 0.0, 1.0);
    return _mm256_mul_pd(vetor, v);
}

/* metodo que retorna apenas o valor da segunda posicao (vetor[1])
deslocado para a primeira posicao do vetor (vetor[0]) */
__m256d zeraSegunda(__m256d vetor){
    __m256d v = _mm256_set_pd(0.0, 0.0, 1.0, 0.0);
    __m256d b = _mm256_mul_pd(vetor, v);
    v = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    return _mm256_hadd_pd(b, v);
}

/* metodo que retorna apenas o valor da terceira posicao (vetor[2])
deslocado para a primeira posicao do vetor (vetor[0]) */
__m256d zeraTerceira(__m256d vetor){
    __m256d v = _mm256_set_pd(0.0, 1.0, 0.0, 0.0);
    __m256d c = _mm256_mul_pd(vetor, v);
    c = _mm256_permute_pd(c,5);
    c = _mm256_permute2f128_pd(c,c,1);
    return _mm256_permute_pd(c,5);

}

/* metodo que retorna apenas o valor da quarta posicao (vetor[3])
deslocado para a primeira posicao do vetor (vetor[0]) */
__m256d zeraQuarta(__m256d vetor){
    __m256d v = _mm256_set_pd(1.0, 0.0, 0.0, 0.0);
    __m256d d = _mm256_mul_pd(vetor, v);
    d = _mm256_permute2f128_pd(d,d,1);
    return _mm256_permute_pd(d,5);
}

/* metodo que retorna o determinante de uma matriz menor obtida com a eliminacao do vetor 1 
multiplicada pelo valor inserido na 4 posicao do vetor 1 (vetor1[3]) */
__m256d parte1(__m256d vetor1, __m256d vetor2, __m256d vetor3, __m256d vetor4){

    __m256d p = zeraQuarta(vetor1);

    __m256d neg = _mm256_set_pd(-1.0, -1.0, -1.0, -1.0);
    p = _mm256_mul_pd(p,neg);

    /* multiplicacao dos elementos da diagonal principal e das demais diagonais a direta dela */
    __m256d p1 = _mm256_mul_pd(zeraPrimeira(vetor2), zeraSegunda(vetor3));
    p1 = _mm256_mul_pd(p1,zeraTerceira(vetor4));

    __m256d p2 = _mm256_mul_pd(zeraSegunda(vetor2), zeraTerceira(vetor3));
    p2 = _mm256_mul_pd(p2,zeraPrimeira(vetor4));

    __m256d p3 = _mm256_mul_pd(zeraPrimeira(vetor3), zeraSegunda(vetor4));
    p3 = _mm256_mul_pd(p3,zeraTerceira(vetor2));

    /* soma do produto das tres diagonais */
    __m256d parte1 = _mm256_add_pd(p1,p2);
    parte1 = _mm256_add_pd(parte1,p3);


    /* multiplicacao dos elementos da diagonal secundaria e das demais diagonais a direta dela */
    __m256d p4 = _mm256_mul_pd(zeraTerceira(vetor2), zeraSegunda(vetor3));
    p4 = _mm256_mul_pd(p4,zeraPrimeira(vetor4));

    __m256d p5 = _mm256_mul_pd(zeraSegunda(vetor2), zeraPrimeira(vetor3));
    p5 = _mm256_mul_pd(p5,zeraTerceira(vetor4));

    __m256d p6 = _mm256_mul_pd(zeraPrimeira(vetor2), zeraTerceira(vetor3));
    p6 = _mm256_mul_pd(p6,zeraSegunda(vetor4));

    /* soma do produto das tres diagonais */
    __m256d parte2 = _mm256_add_pd(p4,p5);
    parte2 = _mm256_add_pd(parte2,p6);

    /* subtracao dos resultados */
    __m256d result = _mm256_sub_pd(parte1,parte2);

    return _mm256_mul_pd(p,result);
}

/* metodo que retorna o determinante de uma matriz menor obtida com a eliminacao do vetor 2 
multiplicada pelo valor inserido na 4 posicao do vetor 2 (vetor2[3]) */
__m256d parte2(__m256d vetor1, __m256d vetor2, __m256d vetor3, __m256d vetor4){

    __m256d p = zeraQuarta(vetor2);
    
    /* multiplicacao dos elementos da diagonal principal e das demais diagonais a direta dela */
    __m256d p1 = _mm256_mul_pd(zeraPrimeira(vetor1), zeraSegunda(vetor3));
    p1 = _mm256_mul_pd(p1,zeraTerceira(vetor4));

    __m256d p2 = _mm256_mul_pd(zeraSegunda(vetor1), zeraTerceira(vetor3));
    p2 = _mm256_mul_pd(p2,zeraPrimeira(vetor4));

    __m256d p3 = _mm256_mul_pd(zeraPrimeira(vetor3), zeraSegunda(vetor4));
    p3 = _mm256_mul_pd(p3,zeraTerceira(vetor1));


    /* soma do produto das tres diagonais */
    __m256d parte1 = _mm256_add_pd(p1,p2);
    parte1 = _mm256_add_pd(parte1,p3);

    
    /* multiplicacao dos elementos da diagonal secundaria e das demais diagonais a direta dela */
    __m256d p4 = _mm256_mul_pd(zeraTerceira(vetor1), zeraSegunda(vetor3));
    p4 = _mm256_mul_pd(p4,zeraPrimeira(vetor4));

    __m256d p5 = _mm256_mul_pd(zeraSegunda(vetor1), zeraPrimeira(vetor3));
    p5 = _mm256_mul_pd(p5,zeraTerceira(vetor4));

    __m256d p6 = _mm256_mul_pd(zeraPrimeira(vetor1), zeraTerceira(vetor3));
    p6 = _mm256_mul_pd(p6,zeraSegunda(vetor4));

    
    /* soma do produto das tres diagonais */
    __m256d parte2 = _mm256_add_pd(p4,p5);
    parte2 = _mm256_add_pd(parte2,p6);

    /* subtracao dos resultados */
    __m256d result = _mm256_sub_pd(parte1,parte2);

    
    return _mm256_mul_pd(p,result);

}
 
/* metodo que retorna o determinante de uma matriz menor obtida com a eliminacao do vetor 3 
multiplicada pelo valor inserido na 4 posicao do vetor 3 (vetor3[3]) */
__m256d parte3(__m256d vetor1, __m256d vetor2, __m256d vetor3, __m256d vetor4){

    __m256d p = zeraQuarta(vetor3);

    __m256d neg = _mm256_set_pd(-1.0, -1.0, -1.0, -1.0);
    p = _mm256_mul_pd(p,neg);

    /* multiplicacao dos elementos da diagonal principal e das demais diagonais a direta dela */
    __m256d p1 = _mm256_mul_pd(zeraPrimeira(vetor1), zeraSegunda(vetor2));
    p1 = _mm256_mul_pd(p1,zeraTerceira(vetor4));

    __m256d p2 = _mm256_mul_pd(zeraSegunda(vetor1), zeraTerceira(vetor2));
    p2 = _mm256_mul_pd(p2,zeraPrimeira(vetor4));

    __m256d p3 = _mm256_mul_pd(zeraPrimeira(vetor2), zeraSegunda(vetor4));
    p3 = _mm256_mul_pd(p3,zeraTerceira(vetor1));

    
    /* soma do produto das tres diagonais */
    __m256d parte1 = _mm256_add_pd(p1,p2);
    parte1 = _mm256_add_pd(parte1,p3);


    /* multiplicacao dos elementos da diagonal secundaria e das demais diagonais a direta dela */
    __m256d p4 = _mm256_mul_pd(zeraTerceira(vetor1), zeraSegunda(vetor2));
    p4 = _mm256_mul_pd(p4,zeraPrimeira(vetor4));

    __m256d p5 = _mm256_mul_pd(zeraSegunda(vetor1), zeraPrimeira(vetor2));
    p5 = _mm256_mul_pd(p5,zeraTerceira(vetor4));

    __m256d p6 = _mm256_mul_pd(zeraPrimeira(vetor1), zeraTerceira(vetor2));
    p6 = _mm256_mul_pd(p6,zeraSegunda(vetor4));

    
    /* soma do produto das tres diagonais */
    __m256d parte2 = _mm256_add_pd(p4,p5);
    parte2 = _mm256_add_pd(parte2,p6);

    /* subtracao dos resultados */
    __m256d result = _mm256_sub_pd(parte1,parte2);

    return _mm256_mul_pd(p,result);

}

/* metodo que retorna o determinante de uma matriz menor obtida com a eliminacao do vetor 4 
multiplicada pelo valor inserido na 4 posicao do vetor 4 (vetor4[3]) */
__m256d parte4(__m256d vetor1, __m256d vetor2, __m256d vetor3, __m256d vetor4){

    __m256d p = zeraQuarta(vetor4);

    /* multiplicacao dos elementos da diagonal principal e das demais diagonais a direta dela */
    __m256d p1 = _mm256_mul_pd(zeraPrimeira(vetor1), zeraSegunda(vetor2));
    p1 = _mm256_mul_pd(p1,zeraTerceira(vetor3));

    __m256d p2 = _mm256_mul_pd(zeraSegunda(vetor1), zeraTerceira(vetor2));
    p2 = _mm256_mul_pd(p2,zeraPrimeira(vetor3));

    __m256d p3 = _mm256_mul_pd(zeraPrimeira(vetor2), zeraSegunda(vetor3));
    p3 = _mm256_mul_pd(p3,zeraTerceira(vetor1));

    /* soma do produto das tres diagonais */
    __m256d parte1 = _mm256_add_pd(p1,p2);
    parte1 = _mm256_add_pd(parte1,p3);

    /* multiplicacao dos elementos da diagonal secundaria e das demais diagonais a direta dela */
    __m256d p4 = _mm256_mul_pd(zeraTerceira(vetor1), zeraSegunda(vetor2));
    p4 = _mm256_mul_pd(p4,zeraPrimeira(vetor3));

    __m256d p5 = _mm256_mul_pd(zeraSegunda(vetor1), zeraPrimeira(vetor2));
    p5 = _mm256_mul_pd(p5,zeraTerceira(vetor3));

    __m256d p6 = _mm256_mul_pd(zeraPrimeira(vetor1), zeraTerceira(vetor2));
    p6 = _mm256_mul_pd(p6,zeraSegunda(vetor3));


    /* soma do produto das tres diagonais */
    __m256d parte2 = _mm256_add_pd(p4,p5);
    parte2 = _mm256_add_pd(parte2,p6);

    /* subtracao dos resultados */
    __m256d result = _mm256_sub_pd(parte1,parte2);

    
    return _mm256_mul_pd(p,result);

}

int main(){

    double vet1[4], vet2[4], vet3[4], vet4[4];

    printf("Este programa calcula o determinante de uma matriz 4x4\nInsira os valores da matriz\n");

    for (int i=0; i<4; i++){
        scanf("%lf", &vet1[i]);
    }

    for (int i=0; i<4; i++){
        scanf("%lf", &vet2[i]);
    }

    for (int i=0; i<4; i++){
        scanf("%lf", &vet3[i]);
    }

    for (int i=0; i<4; i++){
        scanf("%lf", &vet4[i]);
    }

    __m256d vetor1 = _mm256_loadu_pd(vet1);
    __m256d vetor2 = _mm256_loadu_pd(vet2);
    __m256d vetor3 = _mm256_loadu_pd(vet3);
    __m256d vetor4 = _mm256_loadu_pd(vet4);

    /* soma dos quatro determinantes de matrizes menores obtidos */
    __m256d det = _mm256_add_pd(parte1(vetor1, vetor2, vetor3, vetor4),parte2(vetor1, vetor2, vetor3, vetor4));
    det = _mm256_add_pd(det,parte3(vetor1, vetor2, vetor3, vetor4));
    det = _mm256_add_pd(det,parte4(vetor1, vetor2, vetor3, vetor4));
    

    double* d = (double*)&det;
  	printf("O determinante da matriz é: %lf\n", d[0]);


    return 0;
}