#include <immintrin.h> //AVX AVX2
#include <stdio.h>
#include <stdlib.h> //malloc+free
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include "function.h"

//重复加密轮
//Round0
//Round1
//Round2
//Round3

//128比特的NONCE按照特定方式放入256比特的变量中
__m256i From128To256(__m128i src128)
{
    //src128是原始128比特的NONCE值
    __m128i cover0 = _mm_set_epi32(0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000);
    __m128i cover1 = _mm_set_epi32(0x00FF0000, 0x00FF0000, 0x00FF0000, 0x00FF0000);
    __m128i cover2 = _mm_set_epi32(0x0000FF00, 0x0000FF00, 0x0000FF00, 0x0000FF00);
    __m128i cover3 = _mm_set_epi32(0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF);
    __m128i src128_0 = _mm_and_si128(src128, cover0); //0  4  8  12
    __m128i src128_1 = _mm_and_si128(src128, cover1); //1  5  9  13
    __m128i src128_2 = _mm_and_si128(src128, cover2); //2  6  10 14
    __m128i src128_3 = _mm_and_si128(src128, cover3); //3  7  11  15
    //__m128i(32×4) ---> __m256i(64×4) 零扩展
    //__m256i _mm256_cvtepu32_epi64 (__m128i a)
    __m256i dst256_0 = _mm256_cvtepu32_epi64(src128_0);
    __m256i dst256_1 = _mm256_cvtepu32_epi64(src128_1);
    dst256_1 = _mm256_slli_epi64(dst256_1, 8);
    __m256i dst256_2 = _mm256_cvtepu32_epi64(src128_2);
    dst256_2 = _mm256_slli_epi64(dst256_2, 16);
    __m256i dst256_3 = _mm256_cvtepu32_epi64(src128_3);
    dst256_3 = _mm256_slli_epi64(dst256_3, 24);
    //__m256i ---> __m256d
    __m256d row0 = _mm256_castsi256_pd(dst256_0);
    __m256d row1 = _mm256_castsi256_pd(dst256_1);
    __m256d row2 = _mm256_castsi256_pd(dst256_2);
    __m256d row3 = _mm256_castsi256_pd(dst256_3);
    __m256d Yee0 = _mm256_shuffle_pd(row1, row0, 0x00);
    __m256d Yee2 = _mm256_shuffle_pd(row1, row0, 0x0F);
    __m256d Yee1 = _mm256_shuffle_pd(row3, row2, 0x00);
    __m256d Yee3 = _mm256_shuffle_pd(row3, row2, 0x0F);
    row0 = _mm256_permute2f128_pd(Yee3, Yee2, 0x31);
    row1 = _mm256_permute2f128_pd(Yee1, Yee0, 0x31);
    row2 = _mm256_permute2f128_pd(Yee3, Yee2, 0x20);
    row3 = _mm256_permute2f128_pd(Yee1, Yee0, 0x20);
    //__m256d ---> __m256i
    dst256_0 = _mm256_castpd_si256(row0);
    dst256_1 = _mm256_castpd_si256(row1);
    dst256_1 = _mm256_srli_epi64(dst256_1, 8);
    dst256_2 = _mm256_castpd_si256(row2);
    dst256_2 = _mm256_srli_epi64(dst256_2, 16);
    dst256_3 = _mm256_castpd_si256(row3);
    dst256_3 = _mm256_srli_epi64(dst256_3, 24);
    //联接：dst256_0 & dst256_1 & dst256_2 & dst256_3
    __m256i dst256;
    dst256 = _mm256_xor_si256(dst256_0, dst256_1);
    dst256 = _mm256_xor_si256(dst256, dst256_2);
    dst256 = _mm256_xor_si256(dst256, dst256_3);

    return dst256;
}

//256比特按照特定方式放入128比特的变量中
__m128i From256To128(__m256i src256)
{
    __m256i cover0 = _mm256_set_epi64x(0x00000000FF000000, 0x00000000FF000000, 0x00000000FF000000, 0x00000000FF000000);
    __m256i cover1 = _mm256_set_epi64x(0x0000000000FF0000, 0x0000000000FF0000, 0x0000000000FF0000, 0x0000000000FF0000);
    __m256i cover2 = _mm256_set_epi64x(0x000000000000FF00, 0x000000000000FF00, 0x000000000000FF00, 0x000000000000FF00);
    __m256i cover3 = _mm256_set_epi64x(0x00000000000000FF, 0x00000000000000FF, 0x00000000000000FF, 0x00000000000000FF);
    __m256i src256_0 = _mm256_and_si256(src256, cover0); //0  1  2  3
    __m256i src256_1 = _mm256_and_si256(src256, cover1); //4  5  6  7
    __m256i src256_2 = _mm256_and_si256(src256, cover2); //8  9  10 11
    __m256i src256_3 = _mm256_and_si256(src256, cover3); //12  13  14  15

    __m256i dst256_0 = src256_0;
    __m256i dst256_1 = _mm256_slli_epi64(src256_1, 8);
    __m256i dst256_2 = _mm256_slli_epi64(src256_1, 16);
    __m256i dst256_3 = _mm256_slli_epi64(src256_1, 24);

    //__m256i ---> __m256d
    __m256d row0 = _mm256_castsi256_pd(dst256_0);
    __m256d row1 = _mm256_castsi256_pd(dst256_1);
    __m256d row2 = _mm256_castsi256_pd(dst256_2);
    __m256d row3 = _mm256_castsi256_pd(dst256_3);
    __m256d Yee0 = _mm256_shuffle_pd(row1, row0, 0x00);
    __m256d Yee2 = _mm256_shuffle_pd(row1, row0, 0x0F);
    __m256d Yee1 = _mm256_shuffle_pd(row3, row2, 0x00);
    __m256d Yee3 = _mm256_shuffle_pd(row3, row2, 0x0F);
    row0 = _mm256_permute2f128_pd(Yee3, Yee2, 0x31);
    row1 = _mm256_permute2f128_pd(Yee1, Yee0, 0x31);
    row2 = _mm256_permute2f128_pd(Yee3, Yee2, 0x20);
    row3 = _mm256_permute2f128_pd(Yee1, Yee0, 0x20);
    //__m256d ---> __m256i
    dst256_0 = _mm256_castpd_si256(row0);
    dst256_1 = _mm256_castpd_si256(row1);
    dst256_1 = _mm256_srli_epi64(dst256_1, 8);
    dst256_2 = _mm256_castpd_si256(row2);
    dst256_2 = _mm256_srli_epi64(dst256_2, 16);
    dst256_3 = _mm256_castpd_si256(row3);
    dst256_3 = _mm256_srli_epi64(dst256_3, 24);
    //联接：dst256_0 & dst256_1 & dst256_2 & dst256_3
    __m256i dst256;
    __m128i dst128;
    dst256 = _mm256_xor_si256(dst256_0, dst256_1);
    dst256 = _mm256_xor_si256(dst256, dst256_2);
    dst256 = _mm256_xor_si256(dst256, dst256_3);
    unsigned int* value = (unsigned int*)&dst256;
    dst128 = _mm_set_epi32(value[6], value[4], value[2], value[0]);

    return dst128;
}

//ROUND0
__m256i* ENC_ROUND0(__m256i pt[32], __m256i k[8])
{
    //轮密钥在每一轮加密前可以任意赋值
    //第1、2、3列明文与轮密钥异或运算（轮密钥变量可以用来暂时存放异或运算的中间值）
    //第0比特（low）
    __m256i k0 = _mm256_xor_si256(k[0], pt[8]);
    k0 = _mm256_xor_si256(k0, pt[16]);
    k0 = _mm256_xor_si256(k0, pt[24]);
    //第1比特
    __m256i k1 = _mm256_xor_si256(k[1], pt[9]);
    k1 = _mm256_xor_si256(k1, pt[17]);
    k1 = _mm256_xor_si256(k1, pt[25]);
    //第2比特
    __m256i k2 = _mm256_xor_si256(k[2], pt[10]);
    k2 = _mm256_xor_si256(k2, pt[18]);
    k2 = _mm256_xor_si256(k2, pt[26]);
    //第3比特
    __m256i k3 = _mm256_xor_si256(k[3], pt[11]);
    k3 = _mm256_xor_si256(k3, pt[19]);
    k3 = _mm256_xor_si256(k3, pt[27]);
    //第4比特
    __m256i k4 = _mm256_xor_si256(k[4], pt[12]);
    k4 = _mm256_xor_si256(k4, pt[20]);
    k4 = _mm256_xor_si256(k4, pt[28]);
    //第5比特
    __m256i k5 = _mm256_xor_si256(k[5], pt[13]);
    k5 = _mm256_xor_si256(k5, pt[21]);
    k5 = _mm256_xor_si256(k5, pt[29]);
    //第6比特
    __m256i k6 = _mm256_xor_si256(k[6], pt[14]);
    k6 = _mm256_xor_si256(k6, pt[22]);
    k6 = _mm256_xor_si256(k6, pt[30]);
    //第7比特（high）
    __m256i k7 = _mm256_xor_si256(k[7], pt[15]);
    k7 = _mm256_xor_si256(k7, pt[23]);
    k7 = _mm256_xor_si256(k7, pt[31]);
    //S(x)=I(A1×x+C1)×A2+C2=B[M(T×A1×x+T×C1)]+C2
    //( 按位逻辑操作：XOR/AND )
    //变量k0-k7的值进入S盒（保留k0-k7的值不覆盖），中间值暂存入不同的变量
    //I(1.linear XOR/28+22)
    //T×A1×x (XOR/28)
    __m256i y13 = k1;
    __m256i n1 = _mm256_xor_si256(k4, k2);
    __m256i n2 = _mm256_xor_si256(k3, k0);
    __m256i n3 = _mm256_xor_si256(n2, k1);
    __m256i y10 = _mm256_xor_si256(n1, k7);
    __m256i y3 = _mm256_xor_si256(n3, k5);
    __m256i n4 = _mm256_xor_si256(y10, k5);
    __m256i y5 = _mm256_xor_si256(k6, k2);
    __m256i n5 = _mm256_xor_si256(k7, k4);
    __m256i y21 = _mm256_xor_si256(y3, n1);
    __m256i y14 = _mm256_xor_si256(n4, k6);
    __m256i n6 = _mm256_xor_si256(n5, k6);
    __m256i n7 = _mm256_xor_si256(k1, k0);
    __m256i y6 = _mm256_xor_si256(y21, n7);
    __m256i y17 = _mm256_xor_si256(n3, k7);
    __m256i y0 = _mm256_xor_si256(y17, k0);
    __m256i y1 = _mm256_xor_si256(y10, y6);
    __m256i y2 = _mm256_xor_si256(y5, k1);
    __m256i y7 = _mm256_xor_si256(y10, k0);
    __m256i y8 = _mm256_xor_si256(y5, y3);
    __m256i y9 = _mm256_xor_si256(n5, n3);
    __m256i y4 = _mm256_xor_si256(y9, y2);
    __m256i y11 = _mm256_xor_si256(n3, n1);
    __m256i y12 = _mm256_xor_si256(y14, k1);
    __m256i y15 = _mm256_xor_si256(n4, k1);
    __m256i y16 = _mm256_xor_si256(y21, k1);
    __m256i y18 = _mm256_xor_si256(n7, n6);
    __m256i y19 = _mm256_xor_si256(y3, n6);
    __m256i y20 = _mm256_xor_si256(k7, k2);
    //T×C1=(0100 0011 0001 0001 011010)
    //初始化2个常量（4×64bit的全1和全0）
    __m256i c0 = _mm256_set_epi64x(0, 0, 0, 0);//全0
    __m256i c1 = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);//全1
    //T×A1×x+T×C1 (y0-y21与立即数异或运算)
    //y0 = _mm256_xor_si256(c0, y0);
    y1 = _mm256_xor_si256(c1, y1);
    //y2 = _mm256_xor_si256(c0, y2);
    //y3 = _mm256_xor_si256(c0, y3);
    //y4 = _mm256_xor_si256(c0, y4);
    //y5 = _mm256_xor_si256(c0, y5);
    y6 = _mm256_xor_si256(c1, y6);
    y7 = _mm256_xor_si256(c1, y7);
    //y8 = _mm256_xor_si256(c0, y8);
    //y9 = _mm256_xor_si256(c0, y9);
    //y10 = _mm256_xor_si256(c0, y10);
    y11 = _mm256_xor_si256(c1, y11);
    //y12 = _mm256_xor_si256(c0, y12);
    //y13 = _mm256_xor_si256(c0, y13);
    //y14 = _mm256_xor_si256(c0, y14);
    y15 = _mm256_xor_si256(c1, y15);
    //y16 = _mm256_xor_si256(c0, y16);
    y17 = _mm256_xor_si256(c1, y17);
    y18 = _mm256_xor_si256(c1, y18);
    //y19 = _mm256_xor_si256(c0, y19);
    y20 = _mm256_xor_si256(c1, y20);
    //y21 = _mm256_xor_si256(c0, y21);
    //I(2.non-linear XOR&AND/62)
    //(1)
    __m256i t2 = _mm256_and_si256(y12, y15);
    __m256i t3 = _mm256_and_si256(y3, y6);
    __m256i t4 = _mm256_xor_si256(t3, t2);
    __m256i t5 = _mm256_and_si256(y4, y0);
    __m256i t6 = _mm256_xor_si256(t5, t2);
    __m256i t7 = _mm256_and_si256(y13, y16);
    __m256i t8 = _mm256_and_si256(y5, y1);
    __m256i t9 = _mm256_xor_si256(t8, t7);
    __m256i t10 = _mm256_and_si256(y2, y7);
    __m256i t11 = _mm256_xor_si256(t10, t7);
    __m256i t12 = _mm256_and_si256(y9, y11);
    __m256i t13 = _mm256_and_si256(y14, y17);
    __m256i t14 = _mm256_xor_si256(t13, t12);
    __m256i t15 = _mm256_and_si256(y8, y10);
    __m256i t16 = _mm256_xor_si256(t15, t12);
    __m256i t17 = _mm256_xor_si256(t4, t14);
    __m256i t18 = _mm256_xor_si256(t6, t16);
    __m256i t19 = _mm256_xor_si256(t9, t14);
    __m256i t20 = _mm256_xor_si256(t11, t16);
    __m256i t21 = _mm256_xor_si256(t17, y20);
    __m256i t22 = _mm256_xor_si256(t18, y19);
    __m256i t23 = _mm256_xor_si256(t19, y21);
    __m256i t24 = _mm256_xor_si256(t20, y18);
    //(2)
    __m256i t25 = _mm256_xor_si256(t21, t22);
    __m256i t26 = _mm256_and_si256(t21, t23);
    __m256i t27 = _mm256_xor_si256(t24, t26);
    __m256i t28 = _mm256_and_si256(t25, t27);
    __m256i t29 = _mm256_xor_si256(t28, t22);
    __m256i t30 = _mm256_xor_si256(t23, t24);
    __m256i t31 = _mm256_xor_si256(t22, t26);
    __m256i t32 = _mm256_and_si256(t31, t30);
    __m256i t33 = _mm256_xor_si256(t32, t24);
    __m256i t34 = _mm256_xor_si256(t23, t33);
    __m256i t35 = _mm256_xor_si256(t27, t33);
    __m256i t36 = _mm256_and_si256(t24, t35);
    __m256i t37 = _mm256_xor_si256(t36, t34);
    __m256i t38 = _mm256_xor_si256(t27, t36);
    __m256i t39 = _mm256_and_si256(t29, t38);
    __m256i t40 = _mm256_xor_si256(t25, t39);
    //(3)
    __m256i t41 = _mm256_xor_si256(t40, t37);
    __m256i t42 = _mm256_xor_si256(t29, t33);
    __m256i t43 = _mm256_xor_si256(t29, t40);
    __m256i t44 = _mm256_xor_si256(t33, t37);
    __m256i t45 = _mm256_xor_si256(t42, t41);
    __m256i z0 = _mm256_and_si256(t44, y15);
    __m256i z1 = _mm256_and_si256(t37, y6);
    __m256i z2 = _mm256_and_si256(t33, y0);
    __m256i z3 = _mm256_and_si256(t43, y16);
    __m256i z4 = _mm256_and_si256(t40, y1);
    __m256i z5 = _mm256_and_si256(t29, y7);
    __m256i z6 = _mm256_and_si256(t42, y11);
    __m256i z7 = _mm256_and_si256(t45, y17);
    __m256i z8 = _mm256_and_si256(t41, y10);
    __m256i z9 = _mm256_and_si256(t44, y12);
    __m256i z10 = _mm256_and_si256(t37, y3);
    __m256i z11 = _mm256_and_si256(t33, y4);
    __m256i z12 = _mm256_and_si256(t43, y13);
    __m256i z13 = _mm256_and_si256(t40, y5);
    __m256i z14 = _mm256_and_si256(t29, y2);
    __m256i z15 = _mm256_and_si256(t42, y9);
    __m256i z16 = _mm256_and_si256(t45, y14);
    __m256i z17 = _mm256_and_si256(t41, y8);
    //到此只需要保留z0-z17的值以完成线性部分的运算（clear?）
    //I(3.linear XOR/30)
    //I(A1×x+C1)×A2
    __m256i g1 = _mm256_xor_si256(z9, z15);
    __m256i g2 = _mm256_xor_si256(z6, z10);
    __m256i g3 = _mm256_xor_si256(z13, z14);
    __m256i g4 = _mm256_xor_si256(g2, g3);
    __m256i g5 = _mm256_xor_si256(z0, z1);
    __m256i g6 = _mm256_xor_si256(g1, z17);
    __m256i g7 = _mm256_xor_si256(g4, z7);
    __m256i g8 = _mm256_xor_si256(z4, z5);
    __m256i g9 = _mm256_xor_si256(g5, z8);
    __m256i g10 = _mm256_xor_si256(g6, g7);
    __m256i g11 = _mm256_xor_si256(g8, z11);
    __m256i g12 = _mm256_xor_si256(g1, z16);
    __m256i g13 = _mm256_xor_si256(g9, g12);
    __m256i g14 = _mm256_xor_si256(g8, g13);
    __m256i g15 = _mm256_xor_si256(z3, z4);
    __m256i g16 = _mm256_xor_si256(z12, z13);
    __m256i g17 = _mm256_xor_si256(z15, z16);
    __m256i g18 = _mm256_xor_si256(z0, z2);
    __m256i g19 = _mm256_xor_si256(z7, z10);
    __m256i s7 = _mm256_xor_si256(g16, g17);
    __m256i g20 = _mm256_xor_si256(g11, g18);
    __m256i s6 = _mm256_xor_si256(g7, g20);
    __m256i s5 = _mm256_xor_si256(g5, g10);
    __m256i s4 = _mm256_xor_si256(g2, g14);
    __m256i s3 = _mm256_xor_si256(g6, z11);
    __m256i s2 = _mm256_xor_si256(g10, g15);
    __m256i g21 = _mm256_xor_si256(g9, g11);
    __m256i s1 = _mm256_xor_si256(g4, g21);
    __m256i g22 = _mm256_xor_si256(g14, g15);
    __m256i s0 = _mm256_xor_si256(g19, g22);
    //I(A1×x+C1)×A2+C2
    //C2=(11010011)
    //s0-s7与立即数异或运算，结果暂存于s0-s7
    s7 = _mm256_xor_si256(c1, s7);
    s6 = _mm256_xor_si256(c1, s6);
    //s5 = _mm256_xor_si256(c0, s5);
    s4 = _mm256_xor_si256(c1, s4);
    //s3 = _mm256_xor_si256(c0, s3);
    //s2 = _mm256_xor_si256(c0, s2);
    s1 = _mm256_xor_si256(c1, s1);
    s0 = _mm256_xor_si256(c1, s0);
    //循环移位
    //<<<64 (10 01 00 11)  147
    //__m256i ss7 = _mm256_permute4x64_epi64(s7, 147);
    //__m256i ss6 = _mm256_permute4x64_epi64(s6, 147);
    //__m256i ss5 = _mm256_permute4x64_epi64(s5, 147);
    //__m256i ss4 = _mm256_permute4x64_epi64(s4, 147);
    //__m256i ss3 = _mm256_permute4x64_epi64(s3, 147);
    //__m256i ss2 = _mm256_permute4x64_epi64(s2, 147);
    //__m256i ss1 = _mm256_permute4x64_epi64(s1, 147);
    //__m256i ss0 = _mm256_permute4x64_epi64(s0, 147);
    __m256i shuffle1 = _mm256_set_epi8(30, 29, 28, 31, 26, 25, 24, 27, 22, 21, 20, 23, 18, 17, 16, 19, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3);
    __m256i shuffle2 = _mm256_set_epi8(29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    __m256i shuffle3 = _mm256_set_epi8(28, 31, 30, 29, 24, 27, 26, 25, 20, 23, 22, 21, 16, 19, 18, 17, 12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1);
    __m256i ss7 = _mm256_shuffle_epi8(s7, shuffle1);
    __m256i ss6 = _mm256_shuffle_epi8(s6, shuffle1);
    __m256i ss5 = _mm256_shuffle_epi8(s5, shuffle1);
    __m256i ss4 = _mm256_shuffle_epi8(s4, shuffle1);
    __m256i ss3 = _mm256_shuffle_epi8(s3, shuffle1);
    __m256i ss2 = _mm256_shuffle_epi8(s2, shuffle1);
    __m256i ss1 = _mm256_shuffle_epi8(s1, shuffle1);
    __m256i ss0 = _mm256_shuffle_epi8(s0, shuffle1);
    //<<<128 (01 00 11 10)  78
    //__m256i sss7 = _mm256_permute4x64_epi64(s7, 78);
    //__m256i sss6 = _mm256_permute4x64_epi64(s6, 78);
    //__m256i sss5 = _mm256_permute4x64_epi64(s5, 78);
    //__m256i sss4 = _mm256_permute4x64_epi64(s4, 78);
    //__m256i sss3 = _mm256_permute4x64_epi64(s3, 78);
    //__m256i sss2 = _mm256_permute4x64_epi64(s2, 78);
    //__m256i sss1 = _mm256_permute4x64_epi64(s1, 78);
    //__m256i sss0 = _mm256_permute4x64_epi64(s0, 78);
    __m256i sss7 = _mm256_shuffle_epi8(s7, shuffle2);
    __m256i sss6 = _mm256_shuffle_epi8(s6, shuffle2);
    __m256i sss5 = _mm256_shuffle_epi8(s5, shuffle2);
    __m256i sss4 = _mm256_shuffle_epi8(s4, shuffle2);
    __m256i sss3 = _mm256_shuffle_epi8(s3, shuffle2);
    __m256i sss2 = _mm256_shuffle_epi8(s2, shuffle2);
    __m256i sss1 = _mm256_shuffle_epi8(s1, shuffle2);
    __m256i sss0 = _mm256_shuffle_epi8(s0, shuffle2);
    //<<<192 (00 11 10 01)  57
    //__m256i ssss7 = _mm256_permute4x64_epi64(s7, 57);
    //__m256i ssss6 = _mm256_permute4x64_epi64(s6, 57);
    //__m256i ssss5 = _mm256_permute4x64_epi64(s5, 57);
    //__m256i ssss4 = _mm256_permute4x64_epi64(s4, 57);
    //__m256i ssss3 = _mm256_permute4x64_epi64(s3, 57);
    //__m256i ssss2 = _mm256_permute4x64_epi64(s2, 57);
    //__m256i ssss1 = _mm256_permute4x64_epi64(s1, 57);
    //__m256i ssss0 = _mm256_permute4x64_epi64(s0, 57);
    __m256i ssss7 = _mm256_shuffle_epi8(s7, shuffle3);
    __m256i ssss6 = _mm256_shuffle_epi8(s6, shuffle3);
    __m256i ssss5 = _mm256_shuffle_epi8(s5, shuffle3);
    __m256i ssss4 = _mm256_shuffle_epi8(s4, shuffle3);
    __m256i ssss3 = _mm256_shuffle_epi8(s3, shuffle3);
    __m256i ssss2 = _mm256_shuffle_epi8(s2, shuffle3);
    __m256i ssss1 = _mm256_shuffle_epi8(s1, shuffle3);
    __m256i ssss0 = _mm256_shuffle_epi8(s0, shuffle3);

    //循环移位产生的中间值暂时存入x0(low)-x7(high)
    //x7
    __m256i x7 = _mm256_xor_si256(s7, s5); //<<<2
    x7 = _mm256_xor_si256(x7, ss5); //<<<10
    x7 = _mm256_xor_si256(x7, sss5); //<<<18
    x7 = _mm256_xor_si256(x7, ssss7); //<<<24
    //x6
    __m256i x6 = _mm256_xor_si256(s6, s4); //<<<2
    x6 = _mm256_xor_si256(x6, ss4); //<<<10
    x6 = _mm256_xor_si256(x6, sss4); //<<<18
    x6 = _mm256_xor_si256(x6, ssss6); //<<<24
    //x5
    __m256i x5 = _mm256_xor_si256(s5, s3); //<<<2
    x5 = _mm256_xor_si256(x5, ss3); //<<<10
    x5 = _mm256_xor_si256(x5, sss3); //<<<18
    x5 = _mm256_xor_si256(x5, ssss5); //<<<24
    //x4
    __m256i x4 = _mm256_xor_si256(s4, s2); //<<<2
    x4 = _mm256_xor_si256(x4, ss2); //<<<10
    x4 = _mm256_xor_si256(x4, sss2); //<<<18
    x4 = _mm256_xor_si256(x4, ssss4); //<<<24
    //x3
    __m256i x3 = _mm256_xor_si256(s3, s1); //<<<2
    x3 = _mm256_xor_si256(x3, ss1); //<<<10
    x3 = _mm256_xor_si256(x3, sss1); //<<<18
    x3 = _mm256_xor_si256(x3, ssss3); //<<<24
    //x2
    __m256i x2 = _mm256_xor_si256(s2, s0); //<<<2
    x2 = _mm256_xor_si256(x2, ss0); //<<<10
    x2 = _mm256_xor_si256(x2, sss0); //<<<18
    x2 = _mm256_xor_si256(x2, ssss2); //<<<24
    //x1
    __m256i x1 = _mm256_xor_si256(s1, ss7); //<<<2
    x1 = _mm256_xor_si256(x1, sss7); //<<<10
    x1 = _mm256_xor_si256(x1, ssss7); //<<<18
    x1 = _mm256_xor_si256(x1, ssss1); //<<<24
    //x0
    __m256i x0 = _mm256_xor_si256(s0, ss6); //<<<2
    x0 = _mm256_xor_si256(x0, sss6); //<<<10
    x0 = _mm256_xor_si256(x0, ssss6); //<<<18
    x0 = _mm256_xor_si256(x0, ssss0); //<<<24                                                                       
    //更新state（128bit）中第0列的内容（与原pt[0]-pt[7]的值做异或运算）
    //pt[0](low)-pt[7](high)
    pt[0] = _mm256_xor_si256(pt[0], x0);
    pt[1] = _mm256_xor_si256(pt[1], x1);
    pt[2] = _mm256_xor_si256(pt[2], x2);
    pt[3] = _mm256_xor_si256(pt[3], x3);
    pt[4] = _mm256_xor_si256(pt[4], x4);
    pt[5] = _mm256_xor_si256(pt[5], x5);
    pt[6] = _mm256_xor_si256(pt[6], x6);
    pt[7] = _mm256_xor_si256(pt[7], x7);

    return pt;
}

//ROUND1
__m256i* ENC_ROUND1(__m256i pt[32], __m256i k[8])
{
    //轮密钥在每一轮加密前可以任意赋值
    //第1、2、3列明文与轮密钥异或运算（轮密钥变量可以用来暂时存放异或运算的中间值）
    //第0比特（low）
    __m256i k0 = _mm256_xor_si256(k[0], pt[0]);
    k0 = _mm256_xor_si256(k0, pt[16]);
    k0 = _mm256_xor_si256(k0, pt[24]);
    //第1比特
    __m256i k1 = _mm256_xor_si256(k[1], pt[1]);
    k1 = _mm256_xor_si256(k1, pt[17]);
    k1 = _mm256_xor_si256(k1, pt[25]);
    //第2比特
    __m256i k2 = _mm256_xor_si256(k[2], pt[2]);
    k2 = _mm256_xor_si256(k2, pt[18]);
    k2 = _mm256_xor_si256(k2, pt[26]);
    //第3比特
    __m256i k3 = _mm256_xor_si256(k[3], pt[3]);
    k3 = _mm256_xor_si256(k3, pt[19]);
    k3 = _mm256_xor_si256(k3, pt[27]);
    //第4比特
    __m256i k4 = _mm256_xor_si256(k[4], pt[4]);
    k4 = _mm256_xor_si256(k4, pt[20]);
    k4 = _mm256_xor_si256(k4, pt[28]);
    //第5比特
    __m256i k5 = _mm256_xor_si256(k[5], pt[5]);
    k5 = _mm256_xor_si256(k5, pt[21]);
    k5 = _mm256_xor_si256(k5, pt[29]);
    //第6比特
    __m256i k6 = _mm256_xor_si256(k[6], pt[6]);
    k6 = _mm256_xor_si256(k6, pt[22]);
    k6 = _mm256_xor_si256(k6, pt[30]);
    //第7比特（high）
    __m256i k7 = _mm256_xor_si256(k[7], pt[7]);
    k7 = _mm256_xor_si256(k7, pt[23]);
    k7 = _mm256_xor_si256(k7, pt[31]);
    //S(x)=I(A1×x+C1)×A2+C2=B[M(T×A1×x+T×C1)]+C2
    //( 按位逻辑操作：XOR/AND )
    //变量k0-k7的值进入S盒（保留k0-k7的值不覆盖），中间值暂存入不同的变量
    //I(1.linear XOR/28+22)
    //T×A1×x (XOR/28)
    __m256i y13 = k1;
    __m256i n1 = _mm256_xor_si256(k4, k2);
    __m256i n2 = _mm256_xor_si256(k3, k0);
    __m256i n3 = _mm256_xor_si256(n2, k1);
    __m256i y10 = _mm256_xor_si256(n1, k7);
    __m256i y3 = _mm256_xor_si256(n3, k5);
    __m256i n4 = _mm256_xor_si256(y10, k5);
    __m256i y5 = _mm256_xor_si256(k6, k2);
    __m256i n5 = _mm256_xor_si256(k7, k4);
    __m256i y21 = _mm256_xor_si256(y3, n1);
    __m256i y14 = _mm256_xor_si256(n4, k6);
    __m256i n6 = _mm256_xor_si256(n5, k6);
    __m256i n7 = _mm256_xor_si256(k1, k0);
    __m256i y6 = _mm256_xor_si256(y21, n7);
    __m256i y17 = _mm256_xor_si256(n3, k7);
    __m256i y0 = _mm256_xor_si256(y17, k0);
    __m256i y1 = _mm256_xor_si256(y10, y6);
    __m256i y2 = _mm256_xor_si256(y5, k1);
    __m256i y7 = _mm256_xor_si256(y10, k0);
    __m256i y8 = _mm256_xor_si256(y5, y3);
    __m256i y9 = _mm256_xor_si256(n5, n3);
    __m256i y4 = _mm256_xor_si256(y9, y2);
    __m256i y11 = _mm256_xor_si256(n3, n1);
    __m256i y12 = _mm256_xor_si256(y14, k1);
    __m256i y15 = _mm256_xor_si256(n4, k1);
    __m256i y16 = _mm256_xor_si256(y21, k1);
    __m256i y18 = _mm256_xor_si256(n7, n6);
    __m256i y19 = _mm256_xor_si256(y3, n6);
    __m256i y20 = _mm256_xor_si256(k7, k2);
    //T×C1=(0100 0011 0001 0001 011010)
    //初始化2个常量（4×64bit的全1和全0）
    __m256i c0 = _mm256_set_epi64x(0, 0, 0, 0);//全0
    __m256i c1 = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);//全1
    //T×A1×x+T×C1 (y0-y21与立即数异或运算)
    //y0 = _mm256_xor_si256(c0, y0);
    y1 = _mm256_xor_si256(c1, y1);
    //y2 = _mm256_xor_si256(c0, y2);
    //y3 = _mm256_xor_si256(c0, y3);
    //y4 = _mm256_xor_si256(c0, y4);
    //y5 = _mm256_xor_si256(c0, y5);
    y6 = _mm256_xor_si256(c1, y6);
    y7 = _mm256_xor_si256(c1, y7);
    //y8 = _mm256_xor_si256(c0, y8);
    //y9 = _mm256_xor_si256(c0, y9);
    //y10 = _mm256_xor_si256(c0, y10);
    y11 = _mm256_xor_si256(c1, y11);
    //y12 = _mm256_xor_si256(c0, y12);
    //y13 = _mm256_xor_si256(c0, y13);
    //y14 = _mm256_xor_si256(c0, y14);
    y15 = _mm256_xor_si256(c1, y15);
    //y16 = _mm256_xor_si256(c0, y16);
    y17 = _mm256_xor_si256(c1, y17);
    y18 = _mm256_xor_si256(c1, y18);
    //y19 = _mm256_xor_si256(c0, y19);
    y20 = _mm256_xor_si256(c1, y20);
    //y21 = _mm256_xor_si256(c0, y21);
    //I(2.non-linear XOR&AND/62)
    //(1)
    __m256i t2 = _mm256_and_si256(y12, y15);
    __m256i t3 = _mm256_and_si256(y3, y6);
    __m256i t4 = _mm256_xor_si256(t3, t2);
    __m256i t5 = _mm256_and_si256(y4, y0);
    __m256i t6 = _mm256_xor_si256(t5, t2);
    __m256i t7 = _mm256_and_si256(y13, y16);
    __m256i t8 = _mm256_and_si256(y5, y1);
    __m256i t9 = _mm256_xor_si256(t8, t7);
    __m256i t10 = _mm256_and_si256(y2, y7);
    __m256i t11 = _mm256_xor_si256(t10, t7);
    __m256i t12 = _mm256_and_si256(y9, y11);
    __m256i t13 = _mm256_and_si256(y14, y17);
    __m256i t14 = _mm256_xor_si256(t13, t12);
    __m256i t15 = _mm256_and_si256(y8, y10);
    __m256i t16 = _mm256_xor_si256(t15, t12);
    __m256i t17 = _mm256_xor_si256(t4, t14);
    __m256i t18 = _mm256_xor_si256(t6, t16);
    __m256i t19 = _mm256_xor_si256(t9, t14);
    __m256i t20 = _mm256_xor_si256(t11, t16);
    __m256i t21 = _mm256_xor_si256(t17, y20);
    __m256i t22 = _mm256_xor_si256(t18, y19);
    __m256i t23 = _mm256_xor_si256(t19, y21);
    __m256i t24 = _mm256_xor_si256(t20, y18);
    //(2)
    __m256i t25 = _mm256_xor_si256(t21, t22);
    __m256i t26 = _mm256_and_si256(t21, t23);
    __m256i t27 = _mm256_xor_si256(t24, t26);
    __m256i t28 = _mm256_and_si256(t25, t27);
    __m256i t29 = _mm256_xor_si256(t28, t22);
    __m256i t30 = _mm256_xor_si256(t23, t24);
    __m256i t31 = _mm256_xor_si256(t22, t26);
    __m256i t32 = _mm256_and_si256(t31, t30);
    __m256i t33 = _mm256_xor_si256(t32, t24);
    __m256i t34 = _mm256_xor_si256(t23, t33);
    __m256i t35 = _mm256_xor_si256(t27, t33);
    __m256i t36 = _mm256_and_si256(t24, t35);
    __m256i t37 = _mm256_xor_si256(t36, t34);
    __m256i t38 = _mm256_xor_si256(t27, t36);
    __m256i t39 = _mm256_and_si256(t29, t38);
    __m256i t40 = _mm256_xor_si256(t25, t39);
    //(3)
    __m256i t41 = _mm256_xor_si256(t40, t37);
    __m256i t42 = _mm256_xor_si256(t29, t33);
    __m256i t43 = _mm256_xor_si256(t29, t40);
    __m256i t44 = _mm256_xor_si256(t33, t37);
    __m256i t45 = _mm256_xor_si256(t42, t41);
    __m256i z0 = _mm256_and_si256(t44, y15);
    __m256i z1 = _mm256_and_si256(t37, y6);
    __m256i z2 = _mm256_and_si256(t33, y0);
    __m256i z3 = _mm256_and_si256(t43, y16);
    __m256i z4 = _mm256_and_si256(t40, y1);
    __m256i z5 = _mm256_and_si256(t29, y7);
    __m256i z6 = _mm256_and_si256(t42, y11);
    __m256i z7 = _mm256_and_si256(t45, y17);
    __m256i z8 = _mm256_and_si256(t41, y10);
    __m256i z9 = _mm256_and_si256(t44, y12);
    __m256i z10 = _mm256_and_si256(t37, y3);
    __m256i z11 = _mm256_and_si256(t33, y4);
    __m256i z12 = _mm256_and_si256(t43, y13);
    __m256i z13 = _mm256_and_si256(t40, y5);
    __m256i z14 = _mm256_and_si256(t29, y2);
    __m256i z15 = _mm256_and_si256(t42, y9);
    __m256i z16 = _mm256_and_si256(t45, y14);
    __m256i z17 = _mm256_and_si256(t41, y8);
    //I(3.linear XOR/30)
    //I(A1×x+C1)×A2
    __m256i g1 = _mm256_xor_si256(z9, z15);
    __m256i g2 = _mm256_xor_si256(z6, z10);
    __m256i g3 = _mm256_xor_si256(z13, z14);
    __m256i g4 = _mm256_xor_si256(g2, g3);
    __m256i g5 = _mm256_xor_si256(z0, z1);
    __m256i g6 = _mm256_xor_si256(g1, z17);
    __m256i g7 = _mm256_xor_si256(g4, z7);
    __m256i g8 = _mm256_xor_si256(z4, z5);
    __m256i g9 = _mm256_xor_si256(g5, z8);
    __m256i g10 = _mm256_xor_si256(g6, g7);
    __m256i g11 = _mm256_xor_si256(g8, z11);
    __m256i g12 = _mm256_xor_si256(g1, z16);
    __m256i g13 = _mm256_xor_si256(g9, g12);
    __m256i g14 = _mm256_xor_si256(g8, g13);
    __m256i g15 = _mm256_xor_si256(z3, z4);
    __m256i g16 = _mm256_xor_si256(z12, z13);
    __m256i g17 = _mm256_xor_si256(z15, z16);
    __m256i g18 = _mm256_xor_si256(z0, z2);
    __m256i g19 = _mm256_xor_si256(z7, z10);
    __m256i s7 = _mm256_xor_si256(g16, g17);
    __m256i g20 = _mm256_xor_si256(g11, g18);
    __m256i s6 = _mm256_xor_si256(g7, g20);
    __m256i s5 = _mm256_xor_si256(g5, g10);
    __m256i s4 = _mm256_xor_si256(g2, g14);
    __m256i s3 = _mm256_xor_si256(g6, z11);
    __m256i s2 = _mm256_xor_si256(g10, g15);
    __m256i g21 = _mm256_xor_si256(g9, g11);
    __m256i s1 = _mm256_xor_si256(g4, g21);
    __m256i g22 = _mm256_xor_si256(g14, g15);
    __m256i s0 = _mm256_xor_si256(g19, g22);
    //I(A1×x+C1)×A2+C2
    //C2=(11010011)
    //s0-s7与立即数异或运算，结果暂存于s0-s7
    s7 = _mm256_xor_si256(c1, s7);
    s6 = _mm256_xor_si256(c1, s6);
    //s5 = _mm256_xor_si256(c0, s5);
    s4 = _mm256_xor_si256(c1, s4);
    //s3 = _mm256_xor_si256(c0, s3);
    //s2 = _mm256_xor_si256(c0, s2);
    s1 = _mm256_xor_si256(c1, s1);
    s0 = _mm256_xor_si256(c1, s0);
    //循环移位
    //<<<64 (10 01 00 11)  147
    //__m256i ss7 = _mm256_permute4x64_epi64(s7, 147);
    //__m256i ss6 = _mm256_permute4x64_epi64(s6, 147);
    //__m256i ss5 = _mm256_permute4x64_epi64(s5, 147);
    //__m256i ss4 = _mm256_permute4x64_epi64(s4, 147);
    //__m256i ss3 = _mm256_permute4x64_epi64(s3, 147);
    //__m256i ss2 = _mm256_permute4x64_epi64(s2, 147);
    //__m256i ss1 = _mm256_permute4x64_epi64(s1, 147);
    //__m256i ss0 = _mm256_permute4x64_epi64(s0, 147);
    __m256i shuffle1 = _mm256_set_epi8(30, 29, 28, 31, 26, 25, 24, 27, 22, 21, 20, 23, 18, 17, 16, 19, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3);
    __m256i shuffle2 = _mm256_set_epi8(29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    __m256i shuffle3 = _mm256_set_epi8(28, 31, 30, 29, 24, 27, 26, 25, 20, 23, 22, 21, 16, 19, 18, 17, 12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1);
    __m256i ss7 = _mm256_shuffle_epi8(s7, shuffle1);
    __m256i ss6 = _mm256_shuffle_epi8(s6, shuffle1);
    __m256i ss5 = _mm256_shuffle_epi8(s5, shuffle1);
    __m256i ss4 = _mm256_shuffle_epi8(s4, shuffle1);
    __m256i ss3 = _mm256_shuffle_epi8(s3, shuffle1);
    __m256i ss2 = _mm256_shuffle_epi8(s2, shuffle1);
    __m256i ss1 = _mm256_shuffle_epi8(s1, shuffle1);
    __m256i ss0 = _mm256_shuffle_epi8(s0, shuffle1);
    //<<<128 (01 00 11 10)  78
    //__m256i sss7 = _mm256_permute4x64_epi64(s7, 78);
    //__m256i sss6 = _mm256_permute4x64_epi64(s6, 78);
    //__m256i sss5 = _mm256_permute4x64_epi64(s5, 78);
    //__m256i sss4 = _mm256_permute4x64_epi64(s4, 78);
    //__m256i sss3 = _mm256_permute4x64_epi64(s3, 78);
    //__m256i sss2 = _mm256_permute4x64_epi64(s2, 78);
    //__m256i sss1 = _mm256_permute4x64_epi64(s1, 78);
    //__m256i sss0 = _mm256_permute4x64_epi64(s0, 78);
    __m256i sss7 = _mm256_shuffle_epi8(s7, shuffle2);
    __m256i sss6 = _mm256_shuffle_epi8(s6, shuffle2);
    __m256i sss5 = _mm256_shuffle_epi8(s5, shuffle2);
    __m256i sss4 = _mm256_shuffle_epi8(s4, shuffle2);
    __m256i sss3 = _mm256_shuffle_epi8(s3, shuffle2);
    __m256i sss2 = _mm256_shuffle_epi8(s2, shuffle2);
    __m256i sss1 = _mm256_shuffle_epi8(s1, shuffle2);
    __m256i sss0 = _mm256_shuffle_epi8(s0, shuffle2);
    //<<<192 (00 11 10 01)  57
    //__m256i ssss7 = _mm256_permute4x64_epi64(s7, 57);
    //__m256i ssss6 = _mm256_permute4x64_epi64(s6, 57);
    //__m256i ssss5 = _mm256_permute4x64_epi64(s5, 57);
    //__m256i ssss4 = _mm256_permute4x64_epi64(s4, 57);
    //__m256i ssss3 = _mm256_permute4x64_epi64(s3, 57);
    //__m256i ssss2 = _mm256_permute4x64_epi64(s2, 57);
    //__m256i ssss1 = _mm256_permute4x64_epi64(s1, 57);
    //__m256i ssss0 = _mm256_permute4x64_epi64(s0, 57);
    __m256i ssss7 = _mm256_shuffle_epi8(s7, shuffle3);
    __m256i ssss6 = _mm256_shuffle_epi8(s6, shuffle3);
    __m256i ssss5 = _mm256_shuffle_epi8(s5, shuffle3);
    __m256i ssss4 = _mm256_shuffle_epi8(s4, shuffle3);
    __m256i ssss3 = _mm256_shuffle_epi8(s3, shuffle3);
    __m256i ssss2 = _mm256_shuffle_epi8(s2, shuffle3);
    __m256i ssss1 = _mm256_shuffle_epi8(s1, shuffle3);
    __m256i ssss0 = _mm256_shuffle_epi8(s0, shuffle3);
    //循环移位产生的中间值暂时存入x0(low)-x7(high)
    //x7
    __m256i x7 = _mm256_xor_si256(s7, s5); //<<<2
    x7 = _mm256_xor_si256(x7, ss5); //<<<10
    x7 = _mm256_xor_si256(x7, sss5); //<<<18
    x7 = _mm256_xor_si256(x7, ssss7); //<<<24
    //x6
    __m256i x6 = _mm256_xor_si256(s6, s4); //<<<2
    x6 = _mm256_xor_si256(x6, ss4); //<<<10
    x6 = _mm256_xor_si256(x6, sss4); //<<<18
    x6 = _mm256_xor_si256(x6, ssss6); //<<<24
    //x5
    __m256i x5 = _mm256_xor_si256(s5, s3); //<<<2
    x5 = _mm256_xor_si256(x5, ss3); //<<<10
    x5 = _mm256_xor_si256(x5, sss3); //<<<18
    x5 = _mm256_xor_si256(x5, ssss5); //<<<24
    //x4
    __m256i x4 = _mm256_xor_si256(s4, s2); //<<<2
    x4 = _mm256_xor_si256(x4, ss2); //<<<10
    x4 = _mm256_xor_si256(x4, sss2); //<<<18
    x4 = _mm256_xor_si256(x4, ssss4); //<<<24
    //x3
    __m256i x3 = _mm256_xor_si256(s3, s1); //<<<2
    x3 = _mm256_xor_si256(x3, ss1); //<<<10
    x3 = _mm256_xor_si256(x3, sss1); //<<<18
    x3 = _mm256_xor_si256(x3, ssss3); //<<<24
    //x2
    __m256i x2 = _mm256_xor_si256(s2, s0); //<<<2
    x2 = _mm256_xor_si256(x2, ss0); //<<<10
    x2 = _mm256_xor_si256(x2, sss0); //<<<18
    x2 = _mm256_xor_si256(x2, ssss2); //<<<24
    //x1
    __m256i x1 = _mm256_xor_si256(s1, ss7); //<<<2
    x1 = _mm256_xor_si256(x1, sss7); //<<<10
    x1 = _mm256_xor_si256(x1, ssss7); //<<<18
    x1 = _mm256_xor_si256(x1, ssss1); //<<<24
    //x0
    __m256i x0 = _mm256_xor_si256(s0, ss6); //<<<2
    x0 = _mm256_xor_si256(x0, sss6); //<<<10
    x0 = _mm256_xor_si256(x0, ssss6); //<<<18
    x0 = _mm256_xor_si256(x0, ssss0); //<<<24                                                                       
    //更新state（128bit）中第1列的内容（与原pt[8]-pt[15]的值做异或运算）
    //pt[8](low)-pt[15](high)
    pt[8] = _mm256_xor_si256(pt[8], x0);
    pt[9] = _mm256_xor_si256(pt[9], x1);
    pt[10] = _mm256_xor_si256(pt[10], x2);
    pt[11] = _mm256_xor_si256(pt[11], x3);
    pt[12] = _mm256_xor_si256(pt[12], x4);
    pt[13] = _mm256_xor_si256(pt[13], x5);
    pt[14] = _mm256_xor_si256(pt[14], x6);
    pt[15] = _mm256_xor_si256(pt[15], x7);

    return pt;
}

//ROUND2
__m256i* ENC_ROUND2(__m256i pt[32], __m256i k[8])
{
    //轮密钥在每一轮加密前可以任意赋值
    //第1、2、3列明文与轮密钥异或运算（轮密钥变量可以用来暂时存放异或运算的中间值）
    //第0比特（low）
    __m256i k0 = _mm256_xor_si256(k[0], pt[0]);
    k0 = _mm256_xor_si256(k0, pt[8]);
    k0 = _mm256_xor_si256(k0, pt[24]);
    //第1比特
    __m256i k1 = _mm256_xor_si256(k[1], pt[1]);
    k1 = _mm256_xor_si256(k1, pt[9]);
    k1 = _mm256_xor_si256(k1, pt[25]);
    //第2比特
    __m256i k2 = _mm256_xor_si256(k[2], pt[2]);
    k2 = _mm256_xor_si256(k2, pt[10]);
    k2 = _mm256_xor_si256(k2, pt[26]);
    //第3比特
    __m256i k3 = _mm256_xor_si256(k[3], pt[3]);
    k3 = _mm256_xor_si256(k3, pt[11]);
    k3 = _mm256_xor_si256(k3, pt[27]);
    //第4比特
    __m256i k4 = _mm256_xor_si256(k[4], pt[4]);
    k4 = _mm256_xor_si256(k4, pt[12]);
    k4 = _mm256_xor_si256(k4, pt[28]);
    //第5比特
    __m256i k5 = _mm256_xor_si256(k[5], pt[5]);
    k5 = _mm256_xor_si256(k5, pt[13]);
    k5 = _mm256_xor_si256(k5, pt[29]);
    //第6比特
    __m256i k6 = _mm256_xor_si256(k[6], pt[6]);
    k6 = _mm256_xor_si256(k6, pt[14]);
    k6 = _mm256_xor_si256(k6, pt[30]);
    //第7比特（high）
    __m256i k7 = _mm256_xor_si256(k[7], pt[7]);
    k7 = _mm256_xor_si256(k7, pt[15]);
    k7 = _mm256_xor_si256(k7, pt[31]);
    //S(x)=I(A1×x+C1)×A2+C2=B[M(T×A1×x+T×C1)]+C2
    //( 按位逻辑操作：XOR/AND )
    //变量k0-k7的值进入S盒（保留k0-k7的值不覆盖），中间值暂存入不同的变量
    //I(1.linear XOR/28+22)
    //T×A1×x (XOR/28)
    __m256i y13 = k1;
    __m256i n1 = _mm256_xor_si256(k4, k2);
    __m256i n2 = _mm256_xor_si256(k3, k0);
    __m256i n3 = _mm256_xor_si256(n2, k1);
    __m256i y10 = _mm256_xor_si256(n1, k7);
    __m256i y3 = _mm256_xor_si256(n3, k5);
    __m256i n4 = _mm256_xor_si256(y10, k5);
    __m256i y5 = _mm256_xor_si256(k6, k2);
    __m256i n5 = _mm256_xor_si256(k7, k4);
    __m256i y21 = _mm256_xor_si256(y3, n1);
    __m256i y14 = _mm256_xor_si256(n4, k6);
    __m256i n6 = _mm256_xor_si256(n5, k6);
    __m256i n7 = _mm256_xor_si256(k1, k0);
    __m256i y6 = _mm256_xor_si256(y21, n7);
    __m256i y17 = _mm256_xor_si256(n3, k7);
    __m256i y0 = _mm256_xor_si256(y17, k0);
    __m256i y1 = _mm256_xor_si256(y10, y6);
    __m256i y2 = _mm256_xor_si256(y5, k1);
    __m256i y7 = _mm256_xor_si256(y10, k0);
    __m256i y8 = _mm256_xor_si256(y5, y3);
    __m256i y9 = _mm256_xor_si256(n5, n3);
    __m256i y4 = _mm256_xor_si256(y9, y2);
    __m256i y11 = _mm256_xor_si256(n3, n1);
    __m256i y12 = _mm256_xor_si256(y14, k1);
    __m256i y15 = _mm256_xor_si256(n4, k1);
    __m256i y16 = _mm256_xor_si256(y21, k1);
    __m256i y18 = _mm256_xor_si256(n7, n6);
    __m256i y19 = _mm256_xor_si256(y3, n6);
    __m256i y20 = _mm256_xor_si256(k7, k2);
    //T×C1=(0100 0011 0001 0001 011010)
    //初始化2个常量（4×64bit的全1和全0）
    __m256i c0 = _mm256_set_epi64x(0, 0, 0, 0);//全0
    __m256i c1 = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);//全1
    //T×A1×x+T×C1 (y0-y21与立即数异或运算)
    //y0 = _mm256_xor_si256(c0, y0);
    y1 = _mm256_xor_si256(c1, y1);
    //y2 = _mm256_xor_si256(c0, y2);
    //y3 = _mm256_xor_si256(c0, y3);
    //y4 = _mm256_xor_si256(c0, y4);
    //y5 = _mm256_xor_si256(c0, y5);
    y6 = _mm256_xor_si256(c1, y6);
    y7 = _mm256_xor_si256(c1, y7);
    //y8 = _mm256_xor_si256(c0, y8);
    //y9 = _mm256_xor_si256(c0, y9);
    //y10 = _mm256_xor_si256(c0, y10);
    y11 = _mm256_xor_si256(c1, y11);
    //y12 = _mm256_xor_si256(c0, y12);
    //y13 = _mm256_xor_si256(c0, y13);
    //y14 = _mm256_xor_si256(c0, y14);
    y15 = _mm256_xor_si256(c1, y15);
    //y16 = _mm256_xor_si256(c0, y16);
    y17 = _mm256_xor_si256(c1, y17);
    y18 = _mm256_xor_si256(c1, y18);
    //y19 = _mm256_xor_si256(c0, y19);
    y20 = _mm256_xor_si256(c1, y20);
    //y21 = _mm256_xor_si256(c0, y21);
    //I(2.non-linear XOR&AND/62)
    //(1)
    __m256i t2 = _mm256_and_si256(y12, y15);
    __m256i t3 = _mm256_and_si256(y3, y6);
    __m256i t4 = _mm256_xor_si256(t3, t2);
    __m256i t5 = _mm256_and_si256(y4, y0);
    __m256i t6 = _mm256_xor_si256(t5, t2);
    __m256i t7 = _mm256_and_si256(y13, y16);
    __m256i t8 = _mm256_and_si256(y5, y1);
    __m256i t9 = _mm256_xor_si256(t8, t7);
    __m256i t10 = _mm256_and_si256(y2, y7);
    __m256i t11 = _mm256_xor_si256(t10, t7);
    __m256i t12 = _mm256_and_si256(y9, y11);
    __m256i t13 = _mm256_and_si256(y14, y17);
    __m256i t14 = _mm256_xor_si256(t13, t12);
    __m256i t15 = _mm256_and_si256(y8, y10);
    __m256i t16 = _mm256_xor_si256(t15, t12);
    __m256i t17 = _mm256_xor_si256(t4, t14);
    __m256i t18 = _mm256_xor_si256(t6, t16);
    __m256i t19 = _mm256_xor_si256(t9, t14);
    __m256i t20 = _mm256_xor_si256(t11, t16);
    __m256i t21 = _mm256_xor_si256(t17, y20);
    __m256i t22 = _mm256_xor_si256(t18, y19);
    __m256i t23 = _mm256_xor_si256(t19, y21);
    __m256i t24 = _mm256_xor_si256(t20, y18);
    //(2)
    __m256i t25 = _mm256_xor_si256(t21, t22);
    __m256i t26 = _mm256_and_si256(t21, t23);
    __m256i t27 = _mm256_xor_si256(t24, t26);
    __m256i t28 = _mm256_and_si256(t25, t27);
    __m256i t29 = _mm256_xor_si256(t28, t22);
    __m256i t30 = _mm256_xor_si256(t23, t24);
    __m256i t31 = _mm256_xor_si256(t22, t26);
    __m256i t32 = _mm256_and_si256(t31, t30);
    __m256i t33 = _mm256_xor_si256(t32, t24);
    __m256i t34 = _mm256_xor_si256(t23, t33);
    __m256i t35 = _mm256_xor_si256(t27, t33);
    __m256i t36 = _mm256_and_si256(t24, t35);
    __m256i t37 = _mm256_xor_si256(t36, t34);
    __m256i t38 = _mm256_xor_si256(t27, t36);
    __m256i t39 = _mm256_and_si256(t29, t38);
    __m256i t40 = _mm256_xor_si256(t25, t39);
    //(3)
    __m256i t41 = _mm256_xor_si256(t40, t37);
    __m256i t42 = _mm256_xor_si256(t29, t33);
    __m256i t43 = _mm256_xor_si256(t29, t40);
    __m256i t44 = _mm256_xor_si256(t33, t37);
    __m256i t45 = _mm256_xor_si256(t42, t41);
    __m256i z0 = _mm256_and_si256(t44, y15);
    __m256i z1 = _mm256_and_si256(t37, y6);
    __m256i z2 = _mm256_and_si256(t33, y0);
    __m256i z3 = _mm256_and_si256(t43, y16);
    __m256i z4 = _mm256_and_si256(t40, y1);
    __m256i z5 = _mm256_and_si256(t29, y7);
    __m256i z6 = _mm256_and_si256(t42, y11);
    __m256i z7 = _mm256_and_si256(t45, y17);
    __m256i z8 = _mm256_and_si256(t41, y10);
    __m256i z9 = _mm256_and_si256(t44, y12);
    __m256i z10 = _mm256_and_si256(t37, y3);
    __m256i z11 = _mm256_and_si256(t33, y4);
    __m256i z12 = _mm256_and_si256(t43, y13);
    __m256i z13 = _mm256_and_si256(t40, y5);
    __m256i z14 = _mm256_and_si256(t29, y2);
    __m256i z15 = _mm256_and_si256(t42, y9);
    __m256i z16 = _mm256_and_si256(t45, y14);
    __m256i z17 = _mm256_and_si256(t41, y8);
    //到此只需要保留z0-z17的值以完成线性部分的运算（clear?）
    //I(3.linear XOR/30)
    //I(A1×x+C1)×A2
    __m256i g1 = _mm256_xor_si256(z9, z15);
    __m256i g2 = _mm256_xor_si256(z6, z10);
    __m256i g3 = _mm256_xor_si256(z13, z14);
    __m256i g4 = _mm256_xor_si256(g2, g3);
    __m256i g5 = _mm256_xor_si256(z0, z1);
    __m256i g6 = _mm256_xor_si256(g1, z17);
    __m256i g7 = _mm256_xor_si256(g4, z7);
    __m256i g8 = _mm256_xor_si256(z4, z5);
    __m256i g9 = _mm256_xor_si256(g5, z8);
    __m256i g10 = _mm256_xor_si256(g6, g7);
    __m256i g11 = _mm256_xor_si256(g8, z11);
    __m256i g12 = _mm256_xor_si256(g1, z16);
    __m256i g13 = _mm256_xor_si256(g9, g12);
    __m256i g14 = _mm256_xor_si256(g8, g13);
    __m256i g15 = _mm256_xor_si256(z3, z4);
    __m256i g16 = _mm256_xor_si256(z12, z13);
    __m256i g17 = _mm256_xor_si256(z15, z16);
    __m256i g18 = _mm256_xor_si256(z0, z2);
    __m256i g19 = _mm256_xor_si256(z7, z10);
    __m256i s7 = _mm256_xor_si256(g16, g17);
    __m256i g20 = _mm256_xor_si256(g11, g18);
    __m256i s6 = _mm256_xor_si256(g7, g20);
    __m256i s5 = _mm256_xor_si256(g5, g10);
    __m256i s4 = _mm256_xor_si256(g2, g14);
    __m256i s3 = _mm256_xor_si256(g6, z11);
    __m256i s2 = _mm256_xor_si256(g10, g15);
    __m256i g21 = _mm256_xor_si256(g9, g11);
    __m256i s1 = _mm256_xor_si256(g4, g21);
    __m256i g22 = _mm256_xor_si256(g14, g15);
    __m256i s0 = _mm256_xor_si256(g19, g22);
    //I(A1×x+C1)×A2+C2
    //C2=(11010011)
    //s0-s7与立即数异或运算，结果暂存于s0-s7
    s7 = _mm256_xor_si256(c1, s7);
    s6 = _mm256_xor_si256(c1, s6);
    //s5 = _mm256_xor_si256(c0, s5);
    s4 = _mm256_xor_si256(c1, s4);
    //s3 = _mm256_xor_si256(c0, s3);
    //s2 = _mm256_xor_si256(c0, s2);
    s1 = _mm256_xor_si256(c1, s1);
    s0 = _mm256_xor_si256(c1, s0);
    //循环移位
    //<<<64 (10 01 00 11)  147
    //__m256i ss7 = _mm256_permute4x64_epi64(s7, 147);
    //__m256i ss6 = _mm256_permute4x64_epi64(s6, 147);
    //__m256i ss5 = _mm256_permute4x64_epi64(s5, 147);
    //__m256i ss4 = _mm256_permute4x64_epi64(s4, 147);
    //__m256i ss3 = _mm256_permute4x64_epi64(s3, 147);
    //__m256i ss2 = _mm256_permute4x64_epi64(s2, 147);
    //__m256i ss1 = _mm256_permute4x64_epi64(s1, 147);
    //__m256i ss0 = _mm256_permute4x64_epi64(s0, 147);
    __m256i shuffle1 = _mm256_set_epi8(30, 29, 28, 31, 26, 25, 24, 27, 22, 21, 20, 23, 18, 17, 16, 19, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3);
    __m256i shuffle2 = _mm256_set_epi8(29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    __m256i shuffle3 = _mm256_set_epi8(28, 31, 30, 29, 24, 27, 26, 25, 20, 23, 22, 21, 16, 19, 18, 17, 12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1);
    __m256i ss7 = _mm256_shuffle_epi8(s7, shuffle1);
    __m256i ss6 = _mm256_shuffle_epi8(s6, shuffle1);
    __m256i ss5 = _mm256_shuffle_epi8(s5, shuffle1);
    __m256i ss4 = _mm256_shuffle_epi8(s4, shuffle1);
    __m256i ss3 = _mm256_shuffle_epi8(s3, shuffle1);
    __m256i ss2 = _mm256_shuffle_epi8(s2, shuffle1);
    __m256i ss1 = _mm256_shuffle_epi8(s1, shuffle1);
    __m256i ss0 = _mm256_shuffle_epi8(s0, shuffle1);
    //<<<128 (01 00 11 10)  78
    //__m256i sss7 = _mm256_permute4x64_epi64(s7, 78);
    //__m256i sss6 = _mm256_permute4x64_epi64(s6, 78);
    //__m256i sss5 = _mm256_permute4x64_epi64(s5, 78);
    //__m256i sss4 = _mm256_permute4x64_epi64(s4, 78);
    //__m256i sss3 = _mm256_permute4x64_epi64(s3, 78);
    //__m256i sss2 = _mm256_permute4x64_epi64(s2, 78);
    //__m256i sss1 = _mm256_permute4x64_epi64(s1, 78);
    //__m256i sss0 = _mm256_permute4x64_epi64(s0, 78);
    __m256i sss7 = _mm256_shuffle_epi8(s7, shuffle2);
    __m256i sss6 = _mm256_shuffle_epi8(s6, shuffle2);
    __m256i sss5 = _mm256_shuffle_epi8(s5, shuffle2);
    __m256i sss4 = _mm256_shuffle_epi8(s4, shuffle2);
    __m256i sss3 = _mm256_shuffle_epi8(s3, shuffle2);
    __m256i sss2 = _mm256_shuffle_epi8(s2, shuffle2);
    __m256i sss1 = _mm256_shuffle_epi8(s1, shuffle2);
    __m256i sss0 = _mm256_shuffle_epi8(s0, shuffle2);
    //<<<192 (00 11 10 01)  57
    //__m256i ssss7 = _mm256_permute4x64_epi64(s7, 57);
    //__m256i ssss6 = _mm256_permute4x64_epi64(s6, 57);
    //__m256i ssss5 = _mm256_permute4x64_epi64(s5, 57);
    //__m256i ssss4 = _mm256_permute4x64_epi64(s4, 57);
    //__m256i ssss3 = _mm256_permute4x64_epi64(s3, 57);
    //__m256i ssss2 = _mm256_permute4x64_epi64(s2, 57);
    //__m256i ssss1 = _mm256_permute4x64_epi64(s1, 57);
    //__m256i ssss0 = _mm256_permute4x64_epi64(s0, 57);
    __m256i ssss7 = _mm256_shuffle_epi8(s7, shuffle3);
    __m256i ssss6 = _mm256_shuffle_epi8(s6, shuffle3);
    __m256i ssss5 = _mm256_shuffle_epi8(s5, shuffle3);
    __m256i ssss4 = _mm256_shuffle_epi8(s4, shuffle3);
    __m256i ssss3 = _mm256_shuffle_epi8(s3, shuffle3);
    __m256i ssss2 = _mm256_shuffle_epi8(s2, shuffle3);
    __m256i ssss1 = _mm256_shuffle_epi8(s1, shuffle3);
    __m256i ssss0 = _mm256_shuffle_epi8(s0, shuffle3);
    //循环移位产生的中间值暂时存入x0(low)-x7(high)
    //x7
    __m256i x7 = _mm256_xor_si256(s7, s5); //<<<2
    x7 = _mm256_xor_si256(x7, ss5); //<<<10
    x7 = _mm256_xor_si256(x7, sss5); //<<<18
    x7 = _mm256_xor_si256(x7, ssss7); //<<<24
    //x6
    __m256i x6 = _mm256_xor_si256(s6, s4); //<<<2
    x6 = _mm256_xor_si256(x6, ss4); //<<<10
    x6 = _mm256_xor_si256(x6, sss4); //<<<18
    x6 = _mm256_xor_si256(x6, ssss6); //<<<24
    //x5
    __m256i x5 = _mm256_xor_si256(s5, s3); //<<<2
    x5 = _mm256_xor_si256(x5, ss3); //<<<10
    x5 = _mm256_xor_si256(x5, sss3); //<<<18
    x5 = _mm256_xor_si256(x5, ssss5); //<<<24
    //x4
    __m256i x4 = _mm256_xor_si256(s4, s2); //<<<2
    x4 = _mm256_xor_si256(x4, ss2); //<<<10
    x4 = _mm256_xor_si256(x4, sss2); //<<<18
    x4 = _mm256_xor_si256(x4, ssss4); //<<<24
    //x3
    __m256i x3 = _mm256_xor_si256(s3, s1); //<<<2
    x3 = _mm256_xor_si256(x3, ss1); //<<<10
    x3 = _mm256_xor_si256(x3, sss1); //<<<18
    x3 = _mm256_xor_si256(x3, ssss3); //<<<24
    //x2
    __m256i x2 = _mm256_xor_si256(s2, s0); //<<<2
    x2 = _mm256_xor_si256(x2, ss0); //<<<10
    x2 = _mm256_xor_si256(x2, sss0); //<<<18
    x2 = _mm256_xor_si256(x2, ssss2); //<<<24
    //x1
    __m256i x1 = _mm256_xor_si256(s1, ss7); //<<<2
    x1 = _mm256_xor_si256(x1, sss7); //<<<10
    x1 = _mm256_xor_si256(x1, ssss7); //<<<18
    x1 = _mm256_xor_si256(x1, ssss1); //<<<24
    //x0
    __m256i x0 = _mm256_xor_si256(s0, ss6); //<<<2
    x0 = _mm256_xor_si256(x0, sss6); //<<<10
    x0 = _mm256_xor_si256(x0, ssss6); //<<<18
    x0 = _mm256_xor_si256(x0, ssss0); //<<<24                                                                       
    //更新state（128bit）中第2列的内容（与原pt[16]-pt[23]的值做异或运算）
    //pt[16](low)-pt[23](high)
    pt[16] = _mm256_xor_si256(pt[16], x0);
    pt[17] = _mm256_xor_si256(pt[17], x1);
    pt[18] = _mm256_xor_si256(pt[18], x2);
    pt[19] = _mm256_xor_si256(pt[19], x3);
    pt[20] = _mm256_xor_si256(pt[20], x4);
    pt[21] = _mm256_xor_si256(pt[21], x5);
    pt[22] = _mm256_xor_si256(pt[22], x6);
    pt[23] = _mm256_xor_si256(pt[23], x7);

    return pt;
}

//ROUND3
__m256i* ENC_ROUND3(__m256i pt[32], __m256i k[8])
{
    //轮密钥在每一轮加密前可以任意赋值
    //第1、2、3列明文与轮密钥异或运算（轮密钥变量可以用来暂时存放异或运算的中间值）
    //第0比特（low）
    __m256i k0 = _mm256_xor_si256(k[0], pt[0]);
    k0 = _mm256_xor_si256(k0, pt[8]);
    k0 = _mm256_xor_si256(k0, pt[16]);
    //第1比特
    __m256i k1 = _mm256_xor_si256(k[1], pt[1]);
    k1 = _mm256_xor_si256(k1, pt[9]);
    k1 = _mm256_xor_si256(k1, pt[17]);
    //第2比特
    __m256i k2 = _mm256_xor_si256(k[2], pt[2]);
    k2 = _mm256_xor_si256(k2, pt[10]);
    k2 = _mm256_xor_si256(k2, pt[18]);
    //第3比特
    __m256i k3 = _mm256_xor_si256(k[3], pt[3]);
    k3 = _mm256_xor_si256(k3, pt[11]);
    k3 = _mm256_xor_si256(k3, pt[19]);
    //第4比特
    __m256i k4 = _mm256_xor_si256(k[4], pt[4]);
    k4 = _mm256_xor_si256(k4, pt[12]);
    k4 = _mm256_xor_si256(k4, pt[20]);
    //第5比特
    __m256i k5 = _mm256_xor_si256(k[5], pt[5]);
    k5 = _mm256_xor_si256(k5, pt[13]);
    k5 = _mm256_xor_si256(k5, pt[21]);
    //第6比特
    __m256i k6 = _mm256_xor_si256(k[6], pt[6]);
    k6 = _mm256_xor_si256(k6, pt[14]);
    k6 = _mm256_xor_si256(k6, pt[22]);
    //第7比特（high）
    __m256i k7 = _mm256_xor_si256(k[7], pt[7]);
    k7 = _mm256_xor_si256(k7, pt[15]);
    k7 = _mm256_xor_si256(k7, pt[23]);
    //S(x)=I(A1×x+C1)×A2+C2=B[M(T×A1×x+T×C1)]+C2
    //( 按位逻辑操作：XOR/AND )
    //变量k0-k7的值进入S盒（保留k0-k7的值不覆盖），中间值暂存入不同的变量
    //I(1.linear XOR/28+22)
    //T×A1×x (XOR/28)
    __m256i y13 = k1;
    __m256i n1 = _mm256_xor_si256(k4, k2);
    __m256i n2 = _mm256_xor_si256(k3, k0);
    __m256i n3 = _mm256_xor_si256(n2, k1);
    __m256i y10 = _mm256_xor_si256(n1, k7);
    __m256i y3 = _mm256_xor_si256(n3, k5);
    __m256i n4 = _mm256_xor_si256(y10, k5);
    __m256i y5 = _mm256_xor_si256(k6, k2);
    __m256i n5 = _mm256_xor_si256(k7, k4);
    __m256i y21 = _mm256_xor_si256(y3, n1);
    __m256i y14 = _mm256_xor_si256(n4, k6);
    __m256i n6 = _mm256_xor_si256(n5, k6);
    __m256i n7 = _mm256_xor_si256(k1, k0);
    __m256i y6 = _mm256_xor_si256(y21, n7);
    __m256i y17 = _mm256_xor_si256(n3, k7);
    __m256i y0 = _mm256_xor_si256(y17, k0);
    __m256i y1 = _mm256_xor_si256(y10, y6);
    __m256i y2 = _mm256_xor_si256(y5, k1);
    __m256i y7 = _mm256_xor_si256(y10, k0);
    __m256i y8 = _mm256_xor_si256(y5, y3);
    __m256i y9 = _mm256_xor_si256(n5, n3);
    __m256i y4 = _mm256_xor_si256(y9, y2);
    __m256i y11 = _mm256_xor_si256(n3, n1);
    __m256i y12 = _mm256_xor_si256(y14, k1);
    __m256i y15 = _mm256_xor_si256(n4, k1);
    __m256i y16 = _mm256_xor_si256(y21, k1);
    __m256i y18 = _mm256_xor_si256(n7, n6);
    __m256i y19 = _mm256_xor_si256(y3, n6);
    __m256i y20 = _mm256_xor_si256(k7, k2);
    //T×C1=(0100 0011 0001 0001 011010)
    //初始化2个常量（4×64bit的全1和全0）
    __m256i c0 = _mm256_set_epi64x(0, 0, 0, 0);//全0
    __m256i c1 = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);//全1
    //T×A1×x+T×C1 (y0-y21与立即数异或运算)
    //y0 = _mm256_xor_si256(c0, y0);
    y1 = _mm256_xor_si256(c1, y1);
    //y2 = _mm256_xor_si256(c0, y2);
    //y3 = _mm256_xor_si256(c0, y3);
    //y4 = _mm256_xor_si256(c0, y4);
    //y5 = _mm256_xor_si256(c0, y5);
    y6 = _mm256_xor_si256(c1, y6);
    y7 = _mm256_xor_si256(c1, y7);
    //y8 = _mm256_xor_si256(c0, y8);
    //y9 = _mm256_xor_si256(c0, y9);
    //y10 = _mm256_xor_si256(c0, y10);
    y11 = _mm256_xor_si256(c1, y11);
    //y12 = _mm256_xor_si256(c0, y12);
    //y13 = _mm256_xor_si256(c0, y13);
    //y14 = _mm256_xor_si256(c0, y14);
    y15 = _mm256_xor_si256(c1, y15);
    //y16 = _mm256_xor_si256(c0, y16);
    y17 = _mm256_xor_si256(c1, y17);
    y18 = _mm256_xor_si256(c1, y18);
    //y19 = _mm256_xor_si256(c0, y19);
    y20 = _mm256_xor_si256(c1, y20);
    //y21 = _mm256_xor_si256(c0, y21);
    //I(2.non-linear XOR&AND/62)
    //(1)
    __m256i t2 = _mm256_and_si256(y12, y15);
    __m256i t3 = _mm256_and_si256(y3, y6);
    __m256i t4 = _mm256_xor_si256(t3, t2);
    __m256i t5 = _mm256_and_si256(y4, y0);
    __m256i t6 = _mm256_xor_si256(t5, t2);
    __m256i t7 = _mm256_and_si256(y13, y16);
    __m256i t8 = _mm256_and_si256(y5, y1);
    __m256i t9 = _mm256_xor_si256(t8, t7);
    __m256i t10 = _mm256_and_si256(y2, y7);
    __m256i t11 = _mm256_xor_si256(t10, t7);
    __m256i t12 = _mm256_and_si256(y9, y11);
    __m256i t13 = _mm256_and_si256(y14, y17);
    __m256i t14 = _mm256_xor_si256(t13, t12);
    __m256i t15 = _mm256_and_si256(y8, y10);
    __m256i t16 = _mm256_xor_si256(t15, t12);
    __m256i t17 = _mm256_xor_si256(t4, t14);
    __m256i t18 = _mm256_xor_si256(t6, t16);
    __m256i t19 = _mm256_xor_si256(t9, t14);
    __m256i t20 = _mm256_xor_si256(t11, t16);
    __m256i t21 = _mm256_xor_si256(t17, y20);
    __m256i t22 = _mm256_xor_si256(t18, y19);
    __m256i t23 = _mm256_xor_si256(t19, y21);
    __m256i t24 = _mm256_xor_si256(t20, y18);
    //(2)
    __m256i t25 = _mm256_xor_si256(t21, t22);
    __m256i t26 = _mm256_and_si256(t21, t23);
    __m256i t27 = _mm256_xor_si256(t24, t26);
    __m256i t28 = _mm256_and_si256(t25, t27);
    __m256i t29 = _mm256_xor_si256(t28, t22);
    __m256i t30 = _mm256_xor_si256(t23, t24);
    __m256i t31 = _mm256_xor_si256(t22, t26);
    __m256i t32 = _mm256_and_si256(t31, t30);
    __m256i t33 = _mm256_xor_si256(t32, t24);
    __m256i t34 = _mm256_xor_si256(t23, t33);
    __m256i t35 = _mm256_xor_si256(t27, t33);
    __m256i t36 = _mm256_and_si256(t24, t35);
    __m256i t37 = _mm256_xor_si256(t36, t34);
    __m256i t38 = _mm256_xor_si256(t27, t36);
    __m256i t39 = _mm256_and_si256(t29, t38);
    __m256i t40 = _mm256_xor_si256(t25, t39);
    //(3)
    __m256i t41 = _mm256_xor_si256(t40, t37);
    __m256i t42 = _mm256_xor_si256(t29, t33);
    __m256i t43 = _mm256_xor_si256(t29, t40);
    __m256i t44 = _mm256_xor_si256(t33, t37);
    __m256i t45 = _mm256_xor_si256(t42, t41);
    __m256i z0 = _mm256_and_si256(t44, y15);
    __m256i z1 = _mm256_and_si256(t37, y6);
    __m256i z2 = _mm256_and_si256(t33, y0);
    __m256i z3 = _mm256_and_si256(t43, y16);
    __m256i z4 = _mm256_and_si256(t40, y1);
    __m256i z5 = _mm256_and_si256(t29, y7);
    __m256i z6 = _mm256_and_si256(t42, y11);
    __m256i z7 = _mm256_and_si256(t45, y17);
    __m256i z8 = _mm256_and_si256(t41, y10);
    __m256i z9 = _mm256_and_si256(t44, y12);
    __m256i z10 = _mm256_and_si256(t37, y3);
    __m256i z11 = _mm256_and_si256(t33, y4);
    __m256i z12 = _mm256_and_si256(t43, y13);
    __m256i z13 = _mm256_and_si256(t40, y5);
    __m256i z14 = _mm256_and_si256(t29, y2);
    __m256i z15 = _mm256_and_si256(t42, y9);
    __m256i z16 = _mm256_and_si256(t45, y14);
    __m256i z17 = _mm256_and_si256(t41, y8);
    //到此只需要保留z0-z17的值以完成线性部分的运算（clear?）
    //I(3.linear XOR/30)
    //I(A1×x+C1)×A2
    __m256i g1 = _mm256_xor_si256(z9, z15);
    __m256i g2 = _mm256_xor_si256(z6, z10);
    __m256i g3 = _mm256_xor_si256(z13, z14);
    __m256i g4 = _mm256_xor_si256(g2, g3);
    __m256i g5 = _mm256_xor_si256(z0, z1);
    __m256i g6 = _mm256_xor_si256(g1, z17);
    __m256i g7 = _mm256_xor_si256(g4, z7);
    __m256i g8 = _mm256_xor_si256(z4, z5);
    __m256i g9 = _mm256_xor_si256(g5, z8);
    __m256i g10 = _mm256_xor_si256(g6, g7);
    __m256i g11 = _mm256_xor_si256(g8, z11);
    __m256i g12 = _mm256_xor_si256(g1, z16);
    __m256i g13 = _mm256_xor_si256(g9, g12);
    __m256i g14 = _mm256_xor_si256(g8, g13);
    __m256i g15 = _mm256_xor_si256(z3, z4);
    __m256i g16 = _mm256_xor_si256(z12, z13);
    __m256i g17 = _mm256_xor_si256(z15, z16);
    __m256i g18 = _mm256_xor_si256(z0, z2);
    __m256i g19 = _mm256_xor_si256(z7, z10);
    __m256i s7 = _mm256_xor_si256(g16, g17);
    __m256i g20 = _mm256_xor_si256(g11, g18);
    __m256i s6 = _mm256_xor_si256(g7, g20);
    __m256i s5 = _mm256_xor_si256(g5, g10);
    __m256i s4 = _mm256_xor_si256(g2, g14);
    __m256i s3 = _mm256_xor_si256(g6, z11);
    __m256i s2 = _mm256_xor_si256(g10, g15);
    __m256i g21 = _mm256_xor_si256(g9, g11);
    __m256i s1 = _mm256_xor_si256(g4, g21);
    __m256i g22 = _mm256_xor_si256(g14, g15);
    __m256i s0 = _mm256_xor_si256(g19, g22);
    //I(A1×x+C1)×A2+C2
    //C2=(11010011)
    //s0-s7与立即数异或运算，结果暂存于s0-s7
    s7 = _mm256_xor_si256(c1, s7);
    s6 = _mm256_xor_si256(c1, s6);
    //s5 = _mm256_xor_si256(c0, s5);
    s4 = _mm256_xor_si256(c1, s4);
    //s3 = _mm256_xor_si256(c0, s3);
    //s2 = _mm256_xor_si256(c0, s2);
    s1 = _mm256_xor_si256(c1, s1);
    s0 = _mm256_xor_si256(c1, s0);
    //循环移位
    //<<<64 (10 01 00 11)  147
    //__m256i ss7 = _mm256_permute4x64_epi64(s7, 147);
    //__m256i ss6 = _mm256_permute4x64_epi64(s6, 147);
    //__m256i ss5 = _mm256_permute4x64_epi64(s5, 147);
    //__m256i ss4 = _mm256_permute4x64_epi64(s4, 147);
    //__m256i ss3 = _mm256_permute4x64_epi64(s3, 147);
    //__m256i ss2 = _mm256_permute4x64_epi64(s2, 147);
    //__m256i ss1 = _mm256_permute4x64_epi64(s1, 147);
    //__m256i ss0 = _mm256_permute4x64_epi64(s0, 147);
    __m256i shuffle1 = _mm256_set_epi8(30, 29, 28, 31, 26, 25, 24, 27, 22, 21, 20, 23, 18, 17, 16, 19, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3);
    __m256i shuffle2 = _mm256_set_epi8(29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    __m256i shuffle3 = _mm256_set_epi8(28, 31, 30, 29, 24, 27, 26, 25, 20, 23, 22, 21, 16, 19, 18, 17, 12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1);
    __m256i ss7 = _mm256_shuffle_epi8(s7, shuffle1);
    __m256i ss6 = _mm256_shuffle_epi8(s6, shuffle1);
    __m256i ss5 = _mm256_shuffle_epi8(s5, shuffle1);
    __m256i ss4 = _mm256_shuffle_epi8(s4, shuffle1);
    __m256i ss3 = _mm256_shuffle_epi8(s3, shuffle1);
    __m256i ss2 = _mm256_shuffle_epi8(s2, shuffle1);
    __m256i ss1 = _mm256_shuffle_epi8(s1, shuffle1);
    __m256i ss0 = _mm256_shuffle_epi8(s0, shuffle1);
    //<<<128 (01 00 11 10)  78
    //__m256i sss7 = _mm256_permute4x64_epi64(s7, 78);
    //__m256i sss6 = _mm256_permute4x64_epi64(s6, 78);
    //__m256i sss5 = _mm256_permute4x64_epi64(s5, 78);
    //__m256i sss4 = _mm256_permute4x64_epi64(s4, 78);
    //__m256i sss3 = _mm256_permute4x64_epi64(s3, 78);
    //__m256i sss2 = _mm256_permute4x64_epi64(s2, 78);
    //__m256i sss1 = _mm256_permute4x64_epi64(s1, 78);
    //__m256i sss0 = _mm256_permute4x64_epi64(s0, 78);
    __m256i sss7 = _mm256_shuffle_epi8(s7, shuffle2);
    __m256i sss6 = _mm256_shuffle_epi8(s6, shuffle2);
    __m256i sss5 = _mm256_shuffle_epi8(s5, shuffle2);
    __m256i sss4 = _mm256_shuffle_epi8(s4, shuffle2);
    __m256i sss3 = _mm256_shuffle_epi8(s3, shuffle2);
    __m256i sss2 = _mm256_shuffle_epi8(s2, shuffle2);
    __m256i sss1 = _mm256_shuffle_epi8(s1, shuffle2);
    __m256i sss0 = _mm256_shuffle_epi8(s0, shuffle2);
    //<<<192 (00 11 10 01)  57
    //__m256i ssss7 = _mm256_permute4x64_epi64(s7, 57);
    //__m256i ssss6 = _mm256_permute4x64_epi64(s6, 57);
    //__m256i ssss5 = _mm256_permute4x64_epi64(s5, 57);
    //__m256i ssss4 = _mm256_permute4x64_epi64(s4, 57);
    //__m256i ssss3 = _mm256_permute4x64_epi64(s3, 57);
    //__m256i ssss2 = _mm256_permute4x64_epi64(s2, 57);
    //__m256i ssss1 = _mm256_permute4x64_epi64(s1, 57);
    //__m256i ssss0 = _mm256_permute4x64_epi64(s0, 57);
    __m256i ssss7 = _mm256_shuffle_epi8(s7, shuffle3);
    __m256i ssss6 = _mm256_shuffle_epi8(s6, shuffle3);
    __m256i ssss5 = _mm256_shuffle_epi8(s5, shuffle3);
    __m256i ssss4 = _mm256_shuffle_epi8(s4, shuffle3);
    __m256i ssss3 = _mm256_shuffle_epi8(s3, shuffle3);
    __m256i ssss2 = _mm256_shuffle_epi8(s2, shuffle3);
    __m256i ssss1 = _mm256_shuffle_epi8(s1, shuffle3);
    __m256i ssss0 = _mm256_shuffle_epi8(s0, shuffle3);
    //循环移位产生的中间值暂时存入x0(low)-x7(high)
    //x7
    __m256i x7 = _mm256_xor_si256(s7, s5); //<<<2
    x7 = _mm256_xor_si256(x7, ss5); //<<<10
    x7 = _mm256_xor_si256(x7, sss5); //<<<18
    x7 = _mm256_xor_si256(x7, ssss7); //<<<24
    //x6
    __m256i x6 = _mm256_xor_si256(s6, s4); //<<<2
    x6 = _mm256_xor_si256(x6, ss4); //<<<10
    x6 = _mm256_xor_si256(x6, sss4); //<<<18
    x6 = _mm256_xor_si256(x6, ssss6); //<<<24
    //x5
    __m256i x5 = _mm256_xor_si256(s5, s3); //<<<2
    x5 = _mm256_xor_si256(x5, ss3); //<<<10
    x5 = _mm256_xor_si256(x5, sss3); //<<<18
    x5 = _mm256_xor_si256(x5, ssss5); //<<<24
    //x4
    __m256i x4 = _mm256_xor_si256(s4, s2); //<<<2
    x4 = _mm256_xor_si256(x4, ss2); //<<<10
    x4 = _mm256_xor_si256(x4, sss2); //<<<18
    x4 = _mm256_xor_si256(x4, ssss4); //<<<24
    //x3
    __m256i x3 = _mm256_xor_si256(s3, s1); //<<<2
    x3 = _mm256_xor_si256(x3, ss1); //<<<10
    x3 = _mm256_xor_si256(x3, sss1); //<<<18
    x3 = _mm256_xor_si256(x3, ssss3); //<<<24
    //x2
    __m256i x2 = _mm256_xor_si256(s2, s0); //<<<2
    x2 = _mm256_xor_si256(x2, ss0); //<<<10
    x2 = _mm256_xor_si256(x2, sss0); //<<<18
    x2 = _mm256_xor_si256(x2, ssss2); //<<<24
    //x1
    __m256i x1 = _mm256_xor_si256(s1, ss7); //<<<2
    x1 = _mm256_xor_si256(x1, sss7); //<<<10
    x1 = _mm256_xor_si256(x1, ssss7); //<<<18
    x1 = _mm256_xor_si256(x1, ssss1); //<<<24
    //x0
    __m256i x0 = _mm256_xor_si256(s0, ss6); //<<<2
    x0 = _mm256_xor_si256(x0, sss6); //<<<10
    x0 = _mm256_xor_si256(x0, ssss6); //<<<18
    x0 = _mm256_xor_si256(x0, ssss0); //<<<24                                                                       
    //更新state（128bit）中第3列的内容（与原pt[24]-pt[31]的值做异或运算）
    //pt[24](low)-pt[31](high)
    pt[24] = _mm256_xor_si256(pt[24], x0);
    pt[25] = _mm256_xor_si256(pt[25], x1);
    pt[26] = _mm256_xor_si256(pt[26], x2);
    pt[27] = _mm256_xor_si256(pt[27], x3);
    pt[28] = _mm256_xor_si256(pt[28], x4);
    pt[29] = _mm256_xor_si256(pt[29], x5);
    pt[30] = _mm256_xor_si256(pt[30], x6);
    pt[31] = _mm256_xor_si256(pt[31], x7);

    return pt;
}

//参考carryless-multiplication文档（结合算法1和5）
//Figure5. Code Sample - Performing Ghash Using Algorithms 1 and 5 (C)
//(adam·evan) mod p
void gfmul(__m128i adam, __m128i evan, __m128i* res)
{
    __m128i mae0, mae1, mae2, mae3, mae4, mae5, mae6, mae7, mae8, mae9;

    //Algorithm 1
    mae3 = _mm_clmulepi64_si128(adam, evan, 0x00);
    mae4 = _mm_clmulepi64_si128(adam, evan, 0x10);
    mae5 = _mm_clmulepi64_si128(adam, evan, 0x01);
    mae6 = _mm_clmulepi64_si128(adam, evan, 0x11);

    mae4 = _mm_xor_si128(mae4, mae5);
    mae5 = _mm_slli_si128(mae4, 8);
    mae4 = _mm_srli_si128(mae4, 8);
    mae3 = _mm_xor_si128(mae3, mae5);
    mae6 = _mm_xor_si128(mae6, mae4);

    //Algorithm 5 --- Step1
    mae7 = _mm_srli_epi32(mae3, 31);
    mae8 = _mm_srli_epi32(mae6, 31);
    mae3 = _mm_slli_epi32(mae3, 1);
    mae6 = _mm_slli_epi32(mae6, 1);

    mae9 = _mm_srli_si128(mae7, 12);
    mae8 = _mm_slli_si128(mae8, 4);
    mae7 = _mm_slli_si128(mae7, 4);
    mae3 = _mm_or_si128(mae3, mae7);
    mae6 = _mm_or_si128(mae6, mae8);
    mae6 = _mm_or_si128(mae6, mae9);

    //Algorithm 5 --- Step2
    mae7 = _mm_slli_epi32(mae3, 31);
    mae8 = _mm_slli_epi32(mae3, 30);
    mae9 = _mm_slli_epi32(mae3, 25);

    mae7 = _mm_xor_si128(mae7, mae8);
    mae7 = _mm_xor_si128(mae7, mae9);
    mae8 = _mm_srli_si128(mae7, 4);
    mae7 = _mm_slli_si128(mae7, 12);
    mae3 = _mm_xor_si128(mae3, mae7);

    //Algorithm 5 --- Step3
    mae2 = _mm_srli_epi32(mae3, 1);
    mae4 = _mm_srli_epi32(mae3, 2);
    mae5 = _mm_srli_epi32(mae3, 7);

    //Algorithm 5 --- Step4
    mae2 = _mm_xor_si128(mae2, mae4);
    mae2 = _mm_xor_si128(mae2, mae5);
    mae2 = _mm_xor_si128(mae2, mae8);
    mae3 = _mm_xor_si128(mae3, mae2);
    mae6 = _mm_xor_si128(mae6, mae3);

    *res = mae6;
}

//参考carryless-multiplication文档（结合算法2和5）
//Figure8. Code Sample - Performing Ghash Using an Aggregated Reduction Method - Algorithms 2 and 5 (C)
void reduce4(__m128i H1, __m128i H2, __m128i H3, __m128i H4, __m128i X1, __m128i X2, __m128i X3, __m128i X4, __m128i* res)
{
    /*algorithm by Krzysztof Jankowski, Pierre Laurent - Intel*/
    __m128i H1_X1_lo, H1_X1_hi,
        H2_X2_lo, H2_X2_hi,
        H3_X3_lo, H3_X3_hi,
        H4_X4_lo, H4_X4_hi,
        lo, hi;
    __m128i mae0, mae1, mae2, mae3;
    __m128i mae4, mae5, mae6, mae7;
    __m128i mae8, mae9;

    //Algorithm 2
    H1_X1_lo = _mm_clmulepi64_si128(H1, X1, 0x00);
    H2_X2_lo = _mm_clmulepi64_si128(H2, X2, 0x00);
    H3_X3_lo = _mm_clmulepi64_si128(H3, X3, 0x00);
    H4_X4_lo = _mm_clmulepi64_si128(H4, X4, 0x00);
    lo = _mm_xor_si128(H1_X1_lo, H2_X2_lo);
    lo = _mm_xor_si128(lo, H3_X3_lo);
    lo = _mm_xor_si128(lo, H4_X4_lo);

    H1_X1_hi = _mm_clmulepi64_si128(H1, X1, 0x11);
    H2_X2_hi = _mm_clmulepi64_si128(H2, X2, 0x11);
    H3_X3_hi = _mm_clmulepi64_si128(H3, X3, 0x11);
    H4_X4_hi = _mm_clmulepi64_si128(H4, X4, 0x11);
    hi = _mm_xor_si128(H1_X1_hi, H2_X2_hi);
    hi = _mm_xor_si128(hi, H3_X3_hi);
    hi = _mm_xor_si128(hi, H4_X4_hi);

    mae0 = _mm_shuffle_epi32(H1, 78);
    mae4 = _mm_shuffle_epi32(X1, 78);
    mae0 = _mm_xor_si128(mae0, H1);
    mae4 = _mm_xor_si128(mae4, X1);
    mae1 = _mm_shuffle_epi32(H2, 78);
    mae5 = _mm_shuffle_epi32(X2, 78);
    mae1 = _mm_xor_si128(mae1, H2);
    mae5 = _mm_xor_si128(mae5, X2);
    mae2 = _mm_shuffle_epi32(H3, 78);
    mae6 = _mm_shuffle_epi32(X3, 78);
    mae2 = _mm_xor_si128(mae2, H3);
    mae6 = _mm_xor_si128(mae6, X3);
    mae3 = _mm_shuffle_epi32(H4, 78);
    mae7 = _mm_shuffle_epi32(X4, 78);
    mae3 = _mm_xor_si128(mae3, H4);
    mae7 = _mm_xor_si128(mae7, X4);

    mae0 = _mm_clmulepi64_si128(mae0, mae4, 0x00);
    mae1 = _mm_clmulepi64_si128(mae1, mae5, 0x00);
    mae2 = _mm_clmulepi64_si128(mae2, mae6, 0x00);
    mae3 = _mm_clmulepi64_si128(mae3, mae7, 0x00);
    mae0 = _mm_xor_si128(mae0, lo);
    mae0 = _mm_xor_si128(mae0, hi);
    mae0 = _mm_xor_si128(mae1, mae0);
    mae0 = _mm_xor_si128(mae2, mae0);
    mae0 = _mm_xor_si128(mae3, mae0);

    mae4 = _mm_slli_si128(mae0, 8);
    mae0 = _mm_srli_si128(mae0, 8);
    lo = _mm_xor_si128(mae4, lo);
    hi = _mm_xor_si128(mae0, hi);

    mae3 = lo;
    mae6 = hi;

    //Algorithm 5
    mae7 = _mm_srli_epi32(mae3, 31);
    mae8 = _mm_srli_epi32(mae6, 31);
    mae3 = _mm_slli_epi32(mae3, 1);
    mae6 = _mm_slli_epi32(mae6, 1);

    mae9 = _mm_srli_si128(mae7, 12);
    mae8 = _mm_slli_si128(mae8, 4);
    mae7 = _mm_slli_si128(mae7, 4);
    mae3 = _mm_or_si128(mae3, mae7);
    mae6 = _mm_or_si128(mae6, mae8);
    mae6 = _mm_or_si128(mae6, mae9);

    mae7 = _mm_slli_epi32(mae3, 31);
    mae8 = _mm_slli_epi32(mae3, 30);
    mae9 = _mm_slli_epi32(mae3, 25);

    mae7 = _mm_xor_si128(mae7, mae8);
    mae7 = _mm_xor_si128(mae7, mae9);
    mae8 = _mm_srli_si128(mae7, 4);
    mae7 = _mm_slli_si128(mae7, 12);
    mae3 = _mm_xor_si128(mae3, mae7);

    mae2 = _mm_srli_epi32(mae3, 1);
    mae4 = _mm_srli_epi32(mae3, 2);
    mae5 = _mm_srli_epi32(mae3, 7);
    mae2 = _mm_xor_si128(mae2, mae4);
    mae2 = _mm_xor_si128(mae2, mae5);
    mae2 = _mm_xor_si128(mae2, mae8);
    mae3 = _mm_xor_si128(mae3, mae2);
    mae6 = _mm_xor_si128(mae6, mae3);

    *res = mae6;
}

// bit reverse / bit reflection
__m128i Rbit128(__m128i x)
{
    __m128i shufbytes = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m128i luthigh = _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    __m128i lutlow = _mm_slli_epi16(luthigh, 4);
    __m128i lowmask = _mm_set1_epi8(15);
    __m128i rbytes = _mm_shuffle_epi8(x, shufbytes);
    __m128i high = _mm_shuffle_epi8(lutlow, _mm_and_si128(rbytes, lowmask));
    __m128i low = _mm_shuffle_epi8(luthigh, _mm_and_si128(_mm_srli_epi16(rbytes, 4), lowmask));
    __m128i result = _mm_or_si128(low, high);
    __m128 rresult = _mm_castsi128_ps(result);
    rresult = _mm_permute_ps(rresult, 27); //00 01 10 11
    result = _mm_castps_si128(rresult);
    return result;
}

//unsigned char ---> 8-bit
//unsigned long ---> 32-bit
//不考虑解密也不考虑轮密钥生成

//4字节无符号数组转无符号long型
void four_uCh2uLong(unsigned char* in, unsigned long* out)
{
    int i = 0;
    *out = 0;
    for (i = 0; i < 4; i++)
        *out = ((unsigned long)in[i] << (24 - i * 8)) ^ *out;
}


//无符号long型转4字节无符号数组
void uLong2four_uCh(unsigned long in, unsigned char* out)
{
    int i = 0;
    //从32位unsigned long的高位开始取
    for (i = 0; i < 4; i++)
        *(out + i) = (unsigned char)(in >> (24 - i * 8));
}

//左移，保留丢弃位放置尾部---circular shift循环移位
unsigned long move(unsigned long data, int length)
{
    unsigned long result = 0;
    result = (data << length) ^ (data >> (32 - length));

    return result;
}

//加解密数据处理函数
unsigned long func_data(unsigned long input)
{
    int i = 0;
    unsigned long ulTmp = 0; //32-bit
    unsigned char ucIndexList[4] = { 0 };
    unsigned char ucSboxValueList[4] = { 0 }; //4个8-bit的S-box并行
    uLong2four_uCh(input, ucIndexList);
    for (i = 0; i < 4; i++)
    {
        ucSboxValueList[i] = TBL_SBOX[ucIndexList[i]];
    }
    four_uCh2uLong(ucSboxValueList, &ulTmp);
    ulTmp = ulTmp ^ move(ulTmp, 2) ^ move(ulTmp, 10) ^ move(ulTmp, 18) ^ move(ulTmp, 24); //循环移位

    return ulTmp;
}

//加密函数
unsigned long* SM4_LUT(unsigned long ulKeyList[32], unsigned long ulInitialInput[4])
{
    int i = 0;
    unsigned long ulDataList[36] = { 0 };
    ulDataList[0] = ulInitialInput[0];
    ulDataList[1] = ulInitialInput[1];
    ulDataList[2] = ulInitialInput[2];
    ulDataList[3] = ulInitialInput[3];
    //加密
    for (i = 0; i < 32; i++)
        ulDataList[i + 4] = ulDataList[i] ^ func_data(ulDataList[i + 1] ^ ulDataList[i + 2] ^ ulDataList[i + 3] ^ ulKeyList[i]);
    //逆序输出
    unsigned long ulResult[4];
    ulResult[0] = ulDataList[35];
    ulResult[1] = ulDataList[34];
    ulResult[2] = ulDataList[33];
    ulResult[3] = ulDataList[32];
    //trick：避免编译器优化
    //unsigned int back0 = ulResult[0];

    return ulResult;
}

/*
unsigned long ulKeyList[32] =
{ 0, 1, 2, 3, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23,
  24, 25, 26, 27, 28, 29, 30, 31
};
unsigned long ulInitialInput[4] = { 65280, 65280, 65280, 65280 };
*/
